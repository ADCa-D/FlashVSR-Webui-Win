# -*- coding: utf-8 -*-
"""TCDecoder（VAE 解码）TRT/ORT 可行性探测。

把「单帧解码」这条真实热路径导成 ONNX，用 onnxruntime（TRT / CUDA EP）对比 torch 单帧耗时。
- torch 侧：真实 sequential 路径（parallel=False，带跨帧状态）
- ORT 侧：T=1、past=0 的等价图（计算量一致，只是没有跨帧状态）
判据：拿不到 ≥1.3× 就不改主线（改造还要把 9 个 MemBlock 的 past 做成显式 I/O + 按形状建引擎）。

解码器第一层是 784 通道（16 latent + 3×4×8×8 cond 经 PixelShuffle3d），单独测必须按 784 喂。
"""
import tempfile
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
# latent 尺寸 → 输出边长：解码器 3 次 Upsample×2 = 8×
SHAPES = [48, 96]               # 384×384 / 768×768（1024² 会让 ORT 端 OOM，跳过）


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
    """单帧解码图：T=1，MemBlock 的 past 全零（与 sequential 首帧等价，计算量一致）。"""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, x):
        for b in self.decoder:
            if type(b).__name__ == "MemBlock":
                x = b(x, torch.zeros_like(x))
            else:
                x = b(x)
        return x


def run_shape(dec, ch, lat, ort, providers, tag=""):
    x = torch.randn(1, 1, ch, lat, lat, device="cuda", dtype=torch.bfloat16)

    def torch_step():
        dec.clean_mem()
        return dec.decode_video(x, parallel=False, cond=None)

    with torch.no_grad():
        t_torch = bench(torch_step)

    # ORT 不收 bf16 Conv → 临时转 fp16 导出（同代 tensor core，速度等价），导完转回 bf16
    dec.to(torch.float16)
    wrap = DecWrap(dec.decoder).eval()
    x4 = x.reshape(ch, lat, lat).unsqueeze(0).half()
    onnx_path = os.path.join(HERE, f"tcdecoder_{lat}.onnx")
    try:
        with torch.no_grad():
            torch.onnx.export(wrap, (x4,), onnx_path, input_names=["x"], output_names=["y"],
                              opset_version=17, do_constant_folding=True, dynamo=False)
            y_ref = wrap(x4).detach().float().cpu().numpy()
    except Exception as e:  # noqa
        print(f"latent {lat}: ONNX 导出失败 {type(e).__name__}: {str(e)[:200]}", flush=True)
        dec.to(torch.bfloat16)
        return
    finally:
        del wrap
    mb = round(os.path.getsize(onnx_path) / 1048576, 1)

    # 让出显存
    dec.to("cpu")
    dec.to(torch.bfloat16)
    torch.cuda.empty_cache()

    line = f"latent {lat:3d} ({lat*8}×{lat*8})  torch {t_torch:7.1f} ms"
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        provs = []
        # TRT 的 create_directory 不支持非 ASCII 路径，所以放系统临时目录
        cache_dir = os.path.join(tempfile.gettempdir(), "trt_cache")
        os.makedirs(cache_dir, exist_ok=True)   # TRT 的 create_directory 不支持非 ASCII 路径
        for p in providers:
            if p == "TensorrtExecutionProvider":
                provs.append((p, {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache_dir,
                    "trt_max_workspace_size": 2 << 30,
                    "trt_builder_optimization_level": 5,
                }))
            else:
                provs.append(p)
        t_build0 = time.perf_counter()
        sess = ort.InferenceSession(onnx_path, so, providers=provs)
        xn = x4.detach().cpu().numpy()

        def ort_step():
            return sess.run(None, {"x": xn})

        ort_step()                      # 首次可能含引擎构建
        t_build = time.perf_counter() - t_build0
        t_ort = bench(ort_step)
        md = float(abs(y_ref - ort_step()[0].astype("float32")).max())
        line += (f" | {sess.get_providers()[0].replace('ExecutionProvider','')} {t_ort:7.1f} ms"
                 f" ({t_torch / t_ort:.2f}×) maxdiff {md:.4f} 首跑 {t_build:.1f}s | onnx {mb} MB")
        del sess
    except Exception as e:  # noqa
        line += f" | ORT 失败 {type(e).__name__}: {str(e)[:160]}"
    print(line, flush=True)

    dec.to("cuda")
    torch.cuda.empty_cache()


def _add_trt_dlls():
    """让 ORT 的 TensorrtExecutionProvider 能找到 nvinfer_10.dll。"""
    import site
    dirs = list(site.getsitepackages() or []) + [site.getusersitepackages()]
    for d in dirs:
        p = os.path.join(d, "tensorrt_libs")
        if os.path.isdir(p):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
            print("TRT libs:", p, flush=True)


def main():
    _add_trt_dlls()
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", quant="none")
    dec = eng.pipe.TCDecoder
    dec.to("cuda")
    try:
        eng.pipe.dit.to("cpu")      # 5.4 GB DiT 挪走，给 768×768/1024×1024 的激活腾地方
    except Exception:
        pass
    torch.cuda.empty_cache()
    dec.clean_mem()

    ch = dec.decoder[1].in_channels          # 784 = 16 latent + 768 cond
    print(f"第一层 conv: {ch} -> {dec.decoder[1].out_channels}", flush=True)

    import onnxruntime as ort
    avail = ort.get_available_providers()
    print("ORT providers:", avail, flush=True)
    providers = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in avail]

    for lat in SHAPES:
        run_shape(dec, ch, lat, ort, providers)


if __name__ == "__main__":
    main()
