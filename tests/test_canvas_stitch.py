# -*- coding: utf-8 -*-
r"""离线验收：GPU 画布拼接（不需要 GPU、不需要模型权重）。

这几条路径以前完全没测：`flashvsr_gpu_canvas`（engine.py）与逐 tile 上采样等价性都是
纯 CPU 就能验的数学，却只靠"发几个视频肉眼看"。这里用**节点自己的实现**（从 nodes.py
ast 里抽出 5 个纯函数）做参考实现，验：

1. `engine._prepare_input_gpu` 与节点 `prepare_input_tensor` 逐位等价（同 bicubic、
   同 center-crop、同 dtype 顺序），且把裁剪偏移写进 `mod._fs_pi_crop`。
2. 常数输入 → 常数输出，且权重图无空洞（wsum 覆盖满，没有黑边）。
3. tile 64 / 128 两种切法输出几乎一致（拼接与 tile 尺寸解耦）。
4. tile*scale 不是 128 的倍数的非法配置会被警告（以前静默出黑边）。

跑法：
    <ComfyUI 便携包>\python_embeded\python.exe tests\test_canvas_stitch.py
"""
import ast
import contextlib
import io as _io
import math
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import engine as E  # noqa: E402

_PURE = ("largest_8n1_leq", "next_8n5", "compute_scaled_and_target_dims",
         "tensor_upscale_then_center_crop", "prepare_input_tensor",
         "calculate_tile_coords", "create_feather_mask")


def node_ns():
    """把 nodes.py 里那几个纯函数抽出来执行——不 import 整个 ComfyUI 节点包
    （它 import 了 src/models、huggingface_hub，跑起来又慢又可能连带初始化 CUDA）。"""
    path = os.path.join(E.NODE_DIR, "nodes.py")
    tree = ast.parse(_io.open(path, encoding="utf-8").read())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _PURE]
    ns = {"torch": torch, "F": F, "math": math, "clean_vram": lambda: None}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns


class FakeMod(object):
    """只填 flashvsr_gpu_canvas / prepare 真正会碰到的那几个属性。"""

    def __init__(self):
        ns = node_ns()
        for k in _PURE:
            setattr(self, k, ns[k])
        self.cqdm = staticmethod(lambda it, desc="": it)

    def clean_vram(self):
        pass

    @staticmethod
    def tensor2video(out):
        # 与节点 tensor2video 等价（rearrange "C F H W -> F H W C"）
        return (out.squeeze(0).permute(1, 2, 3, 0).float() + 1.0) / 2.0


class FakePipe(object):
    """把 LQ_video 原样当"预测"返回：等价于"模型完美复现输入"，
    这样画布拼接的正确性就可以用解析解检验。"""

    device = torch.device("cpu")
    torch_dtype = torch.float32

    def __init__(self, trim_to=None):
        self.trim_to = trim_to

    def __call__(self, **kw):
        v = kw["LQ_video"]
        return v if self.trim_to is None else v[:, :, :self.trim_to]


def run_canvas(frames, scale, tile, overlap, trim_to=None):
    mod = FakeMod()
    E._install_gpu_prepare(mod)
    pipe = FakePipe(trim_to=trim_to)
    out = E.flashvsr_gpu_canvas(mod, pipe, frames, scale, False, tile, overlap,
                                0.1, 0.1, 0.1, seed=0)
    E._uninstall_gpu_prepare(mod)
    return out


def main():
    fails = []
    torch.manual_seed(0)

    # ---- 1. prepare 等价性（GPU 实现 vs 节点实现，float 输入）
    for (h0, w0, scale) in ((37, 53, 4), (64, 64, 2), (45, 61, 1)):
        x = torch.rand(5, h0, w0, 3)
        mod = FakeMod()
        ref, th_r, tw_r, F_r = mod.prepare_input_tensor(x, "cpu", scale=scale,
                                                        dtype=torch.float32)
        E._install_gpu_prepare(mod)
        got, th_g, tw_g, F_g = mod.prepare_input_tensor(x, "cpu", scale=scale,
                                                        dtype=torch.float32)
        tag = f"prepare({h0}x{w0},scale{scale})"
        if (th_r, tw_r, F_r) != (th_g, tw_g, F_g):
            fails.append(f"{tag} 尺寸不一致 {ref.shape} vs {got.shape}")
            continue
        if ref.shape != got.shape:
            fails.append(f"{tag} 形状不一致 {tuple(ref.shape)} vs {tuple(got.shape)}")
            continue
        d = float((ref - got).abs().max())
        if d > 1e-5:
            fails.append(f"{tag} 数值偏差 {d:.2e}（应与节点逐位一致）")
        sW, sH, tW, tH = mod.compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
        want = (max(0, (sH - tH) // 2), max(0, (sW - tW) // 2))
        if tuple(getattr(mod, E._PI_CROP_KEY, (0, 0))) != want:
            fails.append(f"{tag} 裁剪偏移记录错：{getattr(mod, E._PI_CROP_KEY, None)} != {want}")

    # ---- 1b. uint8 输入路径（GPU 版先 /255，等价于先转 float 再喂节点版）
    xu = torch.randint(0, 255, (7, 40, 56, 3), dtype=torch.uint8)
    m1 = FakeMod()
    ref_u, _, _, _ = m1.prepare_input_tensor(xu.float() / 255.0, "cpu", scale=2,
                                             dtype=torch.float32)
    E._install_gpu_prepare(m1)
    got_u, _, _, _ = m1.prepare_input_tensor(xu, "cpu", scale=2, dtype=torch.float32)
    d = float((ref_u - got_u).abs().max())
    if d > 1e-5:
        fails.append(f"prepare(uint8) 偏差 {d:.2e}")

    # ---- 2. 常数输入 → 常数输出 + 无空洞
    frames = torch.full((2, 40, 48, 3), 0.5)
    out = run_canvas(frames, 2, 32, 8)
    if tuple(out.shape) != (2, 80, 96, 3):
        fails.append(f"常数画布尺寸错 {tuple(out.shape)}")
    else:
        dev = float((out - 0.5).abs().max())
        if dev > 2e-2:
            fails.append(f"常数输入不平：最大偏差 {dev:.3f}（>0.02 说明混合权重有洞）")
        if float(out.min()) < 0.4:
            fails.append(f"出现黑边：min={float(out.min()):.3f}")

    # ---- 3. tile 64 vs 128（同一输入，两种切法应几乎一致）
    grad = torch.zeros(2, 128, 128, 3)
    xs = torch.linspace(0, 1, 128).view(1, 1, 128)
    ys = torch.linspace(0, 1, 128).view(1, 128, 1)
    grad[..., 0] = xs
    grad[..., 1] = ys
    grad[..., 2] = 0.5
    big = run_canvas(grad, 2, 128, 32)
    small = run_canvas(grad, 2, 64, 16)
    if big.shape != small.shape:
        fails.append(f"tile 切法改变了输出尺寸 {tuple(big.shape)} vs {tuple(small.shape)}")
    else:
        d = float((big.float() - small.float()).abs().max())
        if d > 5e-2:
            fails.append(f"tile64 与 tile128 输出差异 {d:.3f}（>0.05 说明接缝不一致）")

    # ---- 4. 非法 tile（100×2=200 不是 128 的倍数）必须警告
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_canvas(torch.full((2, 100, 100, 3), 0.5), 2, 100, 20)
    text = buf.getvalue()
    if "不是 128 的倍数" not in text:
        fails.append("非法 tile 没有给出警告（以前会静默出黑边）")

    if fails:
        print("FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS: 画布拼接 / prepare 等价 / 覆盖完整 / 非法 tile 有警告")
    return 0


if __name__ == "__main__":
    sys.exit(main())
