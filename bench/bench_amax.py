import torch, comfy_kitchen as ck, time
M,K = 3200,1536
x = torch.randn(M,K,device="cuda",dtype=torch.bfloat16)
def ts(t): return (t.abs().amax()/(6.0*448.0)).clamp(min=1e-8).float().reshape(1)
sc = ts(x)   # 预先算好
def bench(fn, it=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/it*1000
t_amax = bench(lambda: x.abs().amax())
t_q_nv = bench(lambda: ck.quantize_nvfp4(x, ts(x), pad_16x=True))
t_q_nv_pre = bench(lambda: ck.quantize_nvfp4(x, sc, pad_16x=True))
t_q_mx = bench(lambda: ck.quantize_mxfp8(x))
print(f"amax 单独        {t_amax:6.3f} ms", flush=True)
print(f"nvfp4 含 amax    {t_q_nv:6.3f} ms", flush=True)
print(f"nvfp4 预置 scale {t_q_nv_pre:6.3f} ms", flush=True)
print(f"mxfp8            {t_q_mx:6.3f} ms", flush=True)
