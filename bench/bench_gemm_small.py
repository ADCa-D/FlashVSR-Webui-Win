import torch, comfy_kitchen as ck, torch.nn.functional as F, time
DT=torch.bfloat16
def ts(t): return (t.abs().amax()/(6.0*448.0)).clamp(min=1e-8).float().reshape(1)
def bench(fn, it=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/it*1000
for M in (2304, 4608):
    print(f"--- M={M} ---", flush=True)
    for name,N,K in (("FFN up 8960x1536",8960,1536), ("FFN dn 1536x8960",1536,8960)):
        w = torch.randn(N,K,device="cuda",dtype=DT)*0.02
        x = torch.randn(M,K,device="cuda",dtype=DT)
        wq,wb = ck.quantize_mxfp8(w.contiguous()); xq,xb = ck.quantize_mxfp8(x.contiguous())
        w4,b4 = ck.quantize_nvfp4(w.contiguous(), ts(w), pad_16x=True)
        t_bf = bench(lambda: F.linear(x,w))
        t_mx = bench(lambda: (lambda a,b: ck.scaled_mm_mxfp8(a, wq, b, wb, out_dtype=DT))( *ck.quantize_mxfp8(x.contiguous()) ))
        t_nv = bench(lambda: (lambda a,b,s: ck.scaled_mm_nvfp4(a, w4, s, ts(w), b, wb if False else ck.quantize_nvfp4(w.contiguous(), ts(w), pad_16x=True)[1], out_dtype=DT))(*ck.quantize_nvfp4(x.contiguous(), ts(x), pad_16x=True), ts(x)))
        print(f"{name:18s} bf16 {t_bf:5.2f}  mxfp8 {t_mx:5.2f}  nvfp4 {t_nv:5.2f} ms", flush=True)
