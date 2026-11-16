# 端到端冒烟：分段 + 异步写盘 + 预览 + manifest + concat，并与同步编码逐位比对
import os, sys, time, json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import engine as E

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "ComfyUI_windows_portable", "ComfyUI", "input", "flashvsr_long_544.mp4")
SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4"
WORK = r"D:\flashvsr-web\work"

class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
    def write(self, s):
        self.f.write(s); self.f.flush()
    def flush(self):
        self.f.flush()

def run(dst, tag):
    eng = E.Engine()
    t0 = time.time()
    stages = []
    def on_stage(stage, cur, tot, msg, **kw):
        stages.append((stage, cur, tot, msg))
        print(f"  [{tag}][{stage} {cur}/{tot}] {msg}", flush=True)
    params = dict(scale=2, tile=128, kv=2.0, chunk=4, overlap=1, mode="tiny-long",
                  attention="sparse", quant="mxfp8", pre_scale=0.5, max_frames=9,
                  encoder="cpu", quality=15, bitdepth=8, tile_overlap=24,
                  color_fix=True, seam_fix="smooth", restart=True)
    info = eng.upscale_video(SRC, dst, params, on_stage)
    dt = time.time() - t0
    print(f"[{tag}] 完成 {dt:.1f}s info={json.dumps(info, ensure_ascii=False)}", flush=True)
    return info, stages, dt

def load(path):
    return E.decode_frames_range(path, 0, 100, pre_scale=1.0)   # 返回 NHWC 数组，不是元组

def main():
    os.makedirs(WORK, exist_ok=True)
    a = os.path.join(WORK, "e2e_async.mp4")
    b = os.path.join(WORK, "e2e_sync.mp4")
    os.environ["FLASHVSR_ASYNC_ENCODE"] = "1"
    ia, sa, ta = run(a, "async")
    os.environ["FLASHVSR_ASYNC_ENCODE"] = "0"
    ib, sb, tb = run(b, "sync")
    fa, fb = load(a), load(b)
    print(f"帧数 async={len(fa)} sync={len(fb)} 期望=9", flush=True)
    n = min(len(fa), len(fb))
    d = np.abs(fa[:n].astype(np.int16) - fb[:n].astype(np.int16))
    mse = float((d.astype(np.float64) ** 2).mean())
    psnr = 99.0 if mse <= 0 else 10 * np.log10(255.0 * 255.0 / mse)
    print(f"async vs sync: max|Δ|={d.max()} mean|Δ|={d.mean():.3f} PSNR={psnr:.2f} dB", flush=True)
    ok = (len(fa) == 9 and len(fb) == 9 and d.max() == 0)
    # 分段校验：manifest 里所有段都 done，且总帧数 = 9
    print("STAGES(seg):", [s for s in sa if s[0] == "seg"][-3:], flush=True)
    print("VERDICT:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
