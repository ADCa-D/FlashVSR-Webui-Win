import os, sys, time
sys.path.insert(0, os.getcwd())
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
import torch, torch.nn as nn, torch.nn.functional as F
import engine as E
torch.backends.cudnn.benchmark = True

def patch_static(module):
    n = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            w, b, p = m.weight, m.bias, m.padding[0]
            fn = torch.compile(lambda x, _w=w, _b=b, _p=p: F.conv2d(x, _w, _b, padding=_p), dynamic=False)
            m.forward = (lambda x, _f=fn: _f(x))
            n += 1
    return n

mod = E.load_node_module()
pipe = mod.init_pipeline("FlashVSR-v1.1", "tiny", "cuda:0", torch.bfloat16)
dec = pipe.TCDecoder.to("cuda")
B, C, Fl, H, W = 1, 16, 8, 64, 64
lat = torch.randn(B, C, Fl, H, W, device="cuda", dtype=torch.bfloat16)
cond = torch.randn(B, 3, Fl*4, H*8, W*8, device="cuda", dtype=torch.bfloat16).clamp(-1,1)
x = lat.transpose(1,2)

def run(tag, iters=3):
    best=1e9; out=None
    for _ in range(iters):
        dec.clean_mem(); torch.cuda.synchronize(); t0=time.time()
        with torch.no_grad(): out = dec.decode_video(x, parallel=False, show_progress_bar=False, cond=cond)
        torch.cuda.synchronize(); best=min(best, time.time()-t0)
    print(f"{tag:24s} best {best*1000:7.0f} ms", flush=True)
    return out.float().clone()

ref = run("baseline cudnn")
t0=time.time(); n = patch_static(dec); print(f"patched {n} convs (dynamic=False)", flush=True)
_ = run("compiled warmup(编译)", iters=1)
print(f"warmup {time.time()-t0:.1f}s", flush=True)
out = run("compiled steady")
print(f"maxdiff {float((out-ref).abs().max()):.4f}", flush=True)
