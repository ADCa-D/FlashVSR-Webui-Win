# -*- coding: utf-8 -*-
"""hi_precision（浮点解码）A/B：同一段素材，只差 --hi-precision，比画质与耗时。

会先把源裁成 N 帧（默认 32）放进 work/ab_hi_prec/，用 2× 放大跑——这是大多数人的
实际用法，显存占用可控；4× + 不分段的整段解码会直接顶满 8 GB。

用法：
    python bench/ab_hi_precision.py [源视频] [帧数]
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import config  # noqa: E402
import engine  # noqa: E402

PY = sys.executable
WORK = os.path.join(ROOT, "work", "ab_hi_prec")


def clip(src, nframes):
    """裁出前 N 帧做固定素材，避免每次跑的起点/时长不一样。"""
    os.makedirs(WORK, exist_ok=True)
    out = os.path.join(WORK, f"src_{nframes}.mp4")
    if os.path.isfile(out):
        return out
    cmd = [config.FFMPEG, "-v", "error", "-y", "-i", src, "-frames:v", str(nframes),
           "-c:v", "libx264", "-crf", "12", "-preset", "veryfast", "-pix_fmt", "yuv420p", out]
    subprocess.run(cmd, check=True)
    return out


def run(tag, src, hi):
    out = os.path.join(WORK, f"{tag}.mp4")
    args = [PY, os.path.join(ROOT, "inference.py"), "-s", src, "-o", out,
            "--preset", "custom", "--scale", "2", "--tile", "128", "--kv", "3",
            "--overlap", "12", "--chunk", "0", "--quant", "mxfp8",
            "--encoder", "x265", "--crf", "10", "--bitdepth", "10"]
    if hi:
        args.append("--hi-precision")
    t0 = time.perf_counter()
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    ok = p.returncode == 0 and os.path.isfile(out)
    tail = [l for l in ((p.stdout or "") + (p.stderr or "")).strip().splitlines() if l.strip()]
    return ok, dt, out, (tail[-1][:140] if tail else "")


def frames(path, n):
    import numpy as np
    info = engine.probe(path)
    cmd = [config.FFMPEG, "-v", "error", "-i", path, "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    a = np.frombuffer(p.stdout, np.uint16).astype(np.float32) / 65535.0
    return a.reshape(-1, info["h"], info["w"], 3), info


def main():
    import numpy as np
    src0 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        config.BASE, "ComfyUI", "input", "flashvsr_long_544.mp4")
    nframes = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    src = clip(src0, nframes)

    res = {}
    for tag, hi in (("u8", False), ("fp32", True)):
        ok, dt, out, tail = run(tag, src, hi)
        print(f"{tag:5s} {'OK ' if ok else 'FAIL'} {dt:6.1f}s  {tail}")
        if not ok:
            return 1
        res[tag] = (out, dt)

    a, info = frames(res["u8"][0], nframes)
    b, _ = frames(res["fp32"][0], nframes)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mse = float(((a - b) ** 2).mean())
    psnr = 99.0 if mse <= 0 else 10 * np.log10(1.0 / mse)
    l1 = float(np.abs(a - b).mean()) * 255
    mx = float(np.abs(a - b).max()) * 255

    print("\n=== fp32 解码 vs 8bit 解码（其余参数完全相同）===")
    print(f"帧数 {n}  {info['w']}x{info['h']}  2× 放大")
    print(f"PSNR       {psnr:.2f} dB")
    print(f"平均绝对差  {l1:.3f} /255   最大 {mx:.1f} /255")
    print(f"耗时       uint8 {res['u8'][1]:.1f}s   fp32 {res['fp32'][1]:.1f}s "
          f"（{res['fp32'][1] - res['u8'][1]:+.1f}s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
