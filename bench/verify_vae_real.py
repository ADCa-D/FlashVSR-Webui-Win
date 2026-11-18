# -*- coding: utf-8 -*-
"""真实 VAE 图的等价性验证：加载 Wan2.1_VAE.pth，用同一份权重分别跑
.orig 与当前实现，比较 encode / decode / stream_decode 的输出。

这一步覆盖：CausalConv3d 新实现、Upsample 去 float、outs 重构、feat_idx 改动、
以及 encode 的尾帧补齐（所以输入故意用非 1+4k 的帧数）。
"""
import importlib
import os
import sys
import types

MODEL = r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast\src\models"
CKPT = r"D:\ComfyUI_windows_portable\ComfyUI\models\FlashVSR-v1.1\Wan2.1_VAE.pth"

sys.path.insert(0, r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes")
pkg = types.ModuleType("fvnode")
pkg.__path__ = [r"D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast"]
sys.modules["fvnode"] = pkg

import torch  # noqa: E402

NEW = importlib.import_module("fvnode.src.models.wan_video_vae")


def load_orig():
    tmp = os.path.join(MODEL, "_vae_orig.py")
    with open(os.path.join(MODEL, "wan_video_vae.py.orig"), "r", encoding="utf-8") as f:
        s = f.read()
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(s)
    return importlib.import_module("fvnode.src.models._vae_orig")


def build(mod, sd, tag):
    vae = mod.WanVideoVAE(z_dim=16, dim=96)
    r = vae.model.load_state_dict(sd, strict=False)   # 权重是内层 VideoVAE_ 的（无 model. 前缀）
    print("  [%s] 载入 %d 键, missing %d, unexpected %d"
          % (tag, len(sd), len(r.missing_keys), len(r.unexpected_keys)))
    if r.missing_keys[:3]:
        print("    missing 例:", r.missing_keys[:3])
    return vae.cuda().eval()


def cmp(tag, a, b):
    same = torch.equal(a, b)
    d = float((a.float() - b.float()).abs().max())
    print("  %-28s 形状 %s/%s  maxdiff %.3e  %s"
          % (tag, tuple(a.shape), tuple(b.shape), d, "相同" if same else "**不同**"))


def main():
    # 关掉 TF32/降级：如果差异是"cuDNN 换了算法 + TF32"，关掉后应降到 1e-6 量级
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    orig = load_orig()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    v_new = build(NEW, sd, "new")
    v_old = build(orig, sd, "old")

    g = torch.Generator().manual_seed(0)
    # frame 数故意用 6（不是 1+4k），会触发尾帧补齐分支
    x = torch.randn(1, 3, 6, 64, 64, generator=g).cuda().clamp(-1, 1)
    z = torch.randn(1, 16, 3, 8, 8, generator=g).cuda()

    with torch.no_grad():
        print("encode（6 帧，非 1+4k → 走尾帧补齐）")
        try:
            ea = NEW.WanVideoVAE.encode.__get__(v_new)([x[0]], "cuda")
        except Exception as e:
            print("  new encode 失败:", type(e).__name__, str(e)[:100])
            ea = None
        try:
            eb = orig.WanVideoVAE.encode.__get__(v_old)([x[0]], "cuda")
            print("  （旧实现在 6 帧下会把尾部 1 帧丢掉）")
        except Exception as e:
            print("  old encode 失败:", type(e).__name__, str(e)[:100])
            eb = None
        if ea is not None and eb is not None:
            print("  new 形状", tuple(ea.shape), " old 形状", tuple(eb.shape),
                  "→ 前 %d 帧应逐位相同: %s"
                  % (eb.shape[2], torch.equal(ea[:, :, :eb.shape[2]], eb)))

        print("decode（3 帧 latent）")
        da = NEW.WanVideoVAE.decode.__get__(v_new)([z[0]], "cuda")
        db = orig.WanVideoVAE.decode.__get__(v_old)([z[0]], "cuda")
        db2 = orig.WanVideoVAE.decode.__get__(v_old)([z[0]], "cuda")
        cmp("噪声底 old vs old2", db, db2)
        cmp("decode new vs old", da, db)

        print("stream_decode（3 帧 latent）")
        sa = NEW.WanVideoVAE.stream_decode.__get__(v_new)([z[0]])
        sb = orig.WanVideoVAE.stream_decode.__get__(v_old)([z[0]])
        cmp("stream_decode", sa, sb)

    os.remove(os.path.join(MODEL, "_vae_orig.py"))


if __name__ == "__main__":
    main()
