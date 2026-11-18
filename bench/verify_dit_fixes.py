# -*- coding: utf-8 -*-
"""验证这轮修复：
  1. _local_mask_additive 两种 kv_double 与旧实现逐位一致（且新的是 [1,L,S] 广播形）
  2. DiTBlock 走 is_stream=False 不再抛 ValueError
  3. build_step_freqs 超 1024 个时间位置时能自动扩表，数值与手算一致
"""
import sys

sys.path.insert(0, r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes")
import importlib  # noqa: E402
import types  # noqa: E402

import torch  # noqa: E402
from einops import rearrange  # noqa: E402

pkg = types.ModuleType("fvnode")
pkg.__path__ = [r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast"]
sys.modules["fvnode"] = pkg
W = importlib.import_module("fvnode.src.models.wan_video_dit")


def old_mask(lm, rl, rn, rh, kv_double):
    m = lm.unsqueeze(1).unsqueeze(0).repeat(rl, 1, rn, 1)
    m = rearrange(m, 'x a y b -> (x a) (y b)')
    if kv_double:
        m = m.repeat_interleave(2, dim=1)
    m = m.unsqueeze(0).repeat(rh, 1, 1).to(torch.float32)
    return m.masked_fill(m == False, -float('inf')).masked_fill(m == True, 0)


class _DummyDit:
    patch_size = (1, 2, 2)

    def __init__(self, head_dim=128):
        self.freqs = W.precompute_freqs_cis_3d(head_dim, 1024)

    def parameters(self):
        yield torch.zeros(1, device="cuda")


def _dummy_freqs(f, h, w, idx=0):
    return W.build_step_freqs(_DummyDit(), f, h, w, idx, torch.device("cuda"))


def t1():
    dev = torch.device("cuda")
    for (bh, bw, rl, rn, rh) in [(8, 8, 3, 1, 12), (2, 3, 3, 4, 12), (1, 1, 3, 384, 12)]:
        lm = W.build_local_block_mask_shifted_vec_normal_slide(bh, bw, 9, 9, include_self=True, device=dev)
        for kv in (False, True):
            s_shape = (rh, rl * bh, (rn * bw) * (2 if kv else 1))
            new = W._local_mask_additive(lm, ("sig", bh, bw, 9, True), rl, rn, rh, kv, s_shape)
            old = old_mask(lm, rl, rn, rh, kv)
            ok_shape = tuple(new.shape) == (1,) + old.shape[-2:]
            ok_val = torch.equal(new.expand_as(old), old)
            print(f"  bh={bh} bw={bw} rl={rl} rn={rn} kv={kv}: 形状 {tuple(new.shape)} "
                  f"广播一致={ok_val} 形状正确={ok_shape}")
        W._LOCAL_MASK_CACHE.clear()


def t2():
    """DiTBlock(is_stream=False)：修复前这里 ValueError: too many values to unpack"""
    dim, heads, ffn = 256, 2, 512   # head_dim=128
    blk = W.DiTBlock(dim, heads, ffn).cuda().eval()
    f, h, w = 2, 8, 8
    x = torch.randn(1, f * h * w, dim, device="cuda")
    ctx = torch.randn(1, 5, dim, device="cuda")
    t_mod = torch.randn(1, 6, dim, device="cuda")
    blk.cross_attn.init_cache(ctx)
    fr = _dummy_freqs(f, h, w)
    with torch.no_grad():
        y = blk(x, ctx, t_mod, fr, f, h, w, 2, 8, is_stream=False)
    print(f"  is_stream=False: 返回 {type(y).__name__} {tuple(y.shape)}")


def t3():
    """build_step_freqs：cur_process_idx=520 时 off+f 超过 1024，旧代码会切出空片段"""
    dim, heads = 128, 12
    class Dummy:
        patch_size = (1, 2, 2)
        def __init__(self):
            self.freqs = W.precompute_freqs_cis_3d(dim, 1024)
        def parameters(self):
            yield torch.zeros(1, device="cuda")
    dit = Dummy()
    f, h, w, idx = 6, 8, 8, 520
    fr = W.build_step_freqs(dit, f, h, w, idx, torch.device("cuda"))
    off = 4 + idx * 2
    # 手算参考：直接按位置 i 生成 cos/sin
    theta = 10000.0
    assert fr.shape == (f * h * w, 1, dim // 2), fr.shape
    print(f"  idx={idx} off+f={off + f} > 1024 → 自动扩表，输出 {tuple(fr.shape)}，"
          f"扩后表长 {dit.freqs[0].shape[0]}")
    # 抽查最后一行第 0 个复数：frame 位置 = off + f-1
    pos = off + f - 1
    freqs_f = 1.0 / (theta ** (torch.arange(0, dim - 2 * (dim // 3), 2,
                                           dtype=torch.float64) / (dim - 2 * (dim // 3))))
    ang = float(pos * freqs_f[0])
    import math
    ref = complex(math.cos(ang), math.sin(ang))
    got = fr[(f - 1) * h * w + 0, 0, 0].item()
    print(f"  抽查 frame{pos} 第0个复数: got={got:.6f} ref={ref:.6f}")


if __name__ == "__main__":
    print("1) mask 展开（旧 vs 新）")
    t1()
    print("2) DiTBlock is_stream=False")
    t2()
    print("3) freqs 越界扩表")
    t3()
