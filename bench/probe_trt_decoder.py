# -*- coding: utf-8 -*-
"""TCDecoder（VAE 解码）TRT/ORT 可行性探测。

思路：把「单帧 + 9 个 MemBlock 的 past」这条真实热路径导成 ONNX，
用 onnxruntime CUDA EP 与 torch 逐帧耗时对比。
- torch 侧：真实 sequential 路径（parallel=False，带跨帧状态）
- ORT 侧：T=1 的 parallel 图（past 置零，计算量等价）
若 ORT 拿不到 ≥1.3×，就不值得为 TRT 改主线（还要处理 9 份显式状态 + 动态 shape）。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import engine as E

TILE = 192          # 输入 LQ 的 tile（scale=4 → 输出 768 → latent 96）
LAT = TILE // 2     # 8 倍下采样，但 tile 指的是 LQ 像素；latent = tile*4/8 = tile/2
CH = 16


def bench(fn, n=8, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


class DecWrap(torch.nn.Module):
    """T=1 的 parallel 路径（past=0），与 sequential 单帧的计算量一致。"""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, x5):
        N, T, C, H, W = x5.shape
        x = x5.reshape(N * T, C, H, W)
        for b in self.decoder:
            if type(b).__name__ == "MemBlock":
                NT, C2, H2, W2 = x.shape
                _x = x.reshape(N, NT // N, C2, H2, W2)
                past = torch.nn.functional.pad(_x, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, : NT // N]
                x = b(x, past.reshape(x.shape))
            else:
                x = b(x)
        return x.view(N, T, C, H, W)


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", quant="none")
    dec = eng.pipe.TCDecoder
    dec.clean_mem()

    x = torch.randn(1, 1, CH, LAT, LAT, device="cuda", dtype=torch.bfloat16)
    print(f"latent {tuple(x.shape)} -> 输出 {TILE * 4}x{TILE * 4}", flush=True)

    # --- torch：真实 sequential 单帧（带状态链）
    def torch_step():
        dec.clean_mem()
        return dec.decode_video(x, parallel=False, cond=None)

    with torch.no_grad():
        t_torch = bench(torch_step)
    print(f"[torch sequential] {t_torch:8.1f} ms/帧", flush=True)

    wrap = DecWrap(dec.decoder).eval()
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench", "tcdecoder_t1.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrap, (x,), onnx_path,
                input_names=["x"], output_names=["y"],
                opset_version=17, do_constant_folding=True, dynamo=False,
            )
        print("ONNX 导出:", os.path.getsize(onnx_path) / 1048576, "MB", flush=True)
    except Exception as e:  # noqa
        print("ONNX 导出失败:", type(e).__name__, str(e)[:300], flush=True)
        return

    try:
        import onnxruntime as ort
    except Exception as e:  # noqa
        print("onnxruntime 不可用:", e, flush=True)
        return
    print("ORT providers:", ort.get_available_providers(), flush=True)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    sess = ort.InferenceSession(onnx_path, so, providers=providers)
    xn = x.detach().cpu().numpy()

    def ort_step():
        return sess.run(None, {"x": xn})

    t_ort = bench(ort_step)
    print(f"[ORT {sess.get_providers()[0]}] {t_ort:8.1f} ms/帧", flush=True)
    print(f"加速比 torch/ORT = {t_torch / t_ort:.2f}×", flush=True)

    # 数值一致性（对齐 torch 的 T=1 parallel 结果）
    with torch.no_grad():
        y_ref = wrap(x).detach().float().cpu().numpy()
    y_ort = ort_step()[0].astype("float32")
    md = abs(y_ref - y_ort).max()
    print(f"maxdiff = {md:.5f}（bf16 下 <0.05 视为一致）", flush=True)


if __name__ == "__main__":
    main()
