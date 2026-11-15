import os, sys, time
sys.path.insert(0, r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web")
import torch
import engine as E  # engine 里已设置 expandable_segments

mod = E.load_node_module()
wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]

frames_u8, info = E.decode_frames(r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_real.mp4")
frames = torch.from_numpy(frames_u8).float().div_(255.0)
print("frames", frames.shape, info, flush=True)

t0 = time.time()
pipe = mod.init_pipeline("FlashVSR-v1.1", "tiny", "cuda:0", torch.bfloat16)
print(f"load {time.time()-t0:.1f}s", flush=True)


def bench(name, scale=4, tile=128, kv=3.0, sparse_ratio=2.0, local_range=11, dense=False):
    orig = wan.flash_attention
    if dense:
        def patched(q, k, v, num_heads, compatibility_mode=False, attention_mask=None, return_KV=False):
            return orig(q, k, v, num_heads, True, None, return_KV)
        wan.flash_attention = patched
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time()
        with torch.no_grad():
            out = mod.flashvsr(
                pipe, frames, scale, True, True, True, tile, 24, False,
                sparse_ratio, kv, local_range, 0, True,
            )
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"[{name}] {dt:6.1f}s  peak {peak:.2f} GiB  out {tuple(out.shape)}", flush=True)
    except Exception as e:
        print(f"[{name}] FAILED {type(e).__name__}: {str(e)[:160]}", flush=True)
    finally:
        wan.flash_attention = orig


bench("baseline  sr2.0 lr11", sparse_ratio=2.0, local_range=11)
bench("sparse    sr1.5 lr11", sparse_ratio=1.5, local_range=11)
bench("sparse    sr1.5 lr9 ", sparse_ratio=1.5, local_range=9)
bench("dense SDPA (无稀疏)", dense=True)
print("DONE", flush=True)
