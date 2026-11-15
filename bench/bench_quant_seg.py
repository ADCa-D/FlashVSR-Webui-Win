import os, sys, time
sys.path.insert(0, os.getcwd())
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
import torch, engine as E
mod = E.load_node_module()
frames_u8, _ = E.decode_frames(r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long.mp4")
frames = torch.from_numpy(frames_u8).float().div_(255.0)
seg = frames[:32]

def bench(tag, quant):
    t0=time.time()
    pipe = mod.init_pipeline("FlashVSR-v1.1", "tiny", "cuda:0", torch.bfloat16)
    t_load=time.time()-t0
    t_q=0.0
    if quant:
        t1=time.time(); n=E.quantize_dit_w4a8(pipe); t_q=time.time()-t1
        print(f"  quantized {n} linears in {t_q:.1f}s", flush=True)
    # 预热一次
    with torch.no_grad():
        mod.flashvsr(pipe, seg, 4, True, True, True, 160, 24, False, 2.0, 2.0, 11, 0, False)
    torch.cuda.synchronize()
    t0=time.time()
    with torch.no_grad():
        out = mod.flashvsr(pipe, seg, 4, True, True, True, 160, 24, False, 2.0, 2.0, 11, 0, False)
    torch.cuda.synchronize()
    dt=time.time()-t0
    print(f"{tag:12s} load {t_load:5.1f}s quant {t_q:5.1f}s | 单段 {dt:6.1f}s | peak {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB", flush=True)
    return out.float().clone()

a = bench("bf16", False)
b = bench("w4a8", True)
print(f"w4a8 vs bf16 最大差 {float((a-b).abs().max()):.4f} 均值 {float((a-b).abs().mean()):.5f}", flush=True)
