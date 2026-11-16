# -*- coding: utf-8 -*-
"""hi_precision（浮点解码）A/B：同一段素材，只差 --hi-precision，比画质与耗时。

用法（GPU 空闲时跑，两个进程串行，避免两个模型同时驻留）：
    python bench/ab_hi_precision.py [源视频] [帧数]
"""
import json
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


def run(tag, src, nframes, hi):
    out = os.path.join(WORK, f"{tag}.mp4")
    args = [PY, os.path.join(ROOT, "inference.py"), "-s", src, "-o", out,
            "--preset", "custom", "--scale", "4", "--tile", "128", "--kv", "3",
            "--overlap", "12", "--chunk", "0", "--quant", "mxfp8",
            "--encoder", "x265", "--crf", "12", "--bitdepth", "10"]
    if hi:
        args.append("--hi-precision")
    t0 = time.perf_counter()
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    ok = p.returncode == 0 and os.path.isfile(out)
    return ok, dt, out, (p.stdout or "") + (p.stderr or "")


def frames(path, n):
    """取前 n 帧的 rgb48le 原始数据，返回 numpy 数组。"""
    import numpy as np
    info = engine.probe(path)
    cmd = [config.FFMPEG, "-v", "error", "-i", path, "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    a = np.frombuffer(p.stdout, np.uint16).astype(np.float32) / 65535.0
    return a.reshape(-1, info["h"], info["w"], 3), info


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        config.BASE, "ComfyUI", "input", "flashvsr_long_544.mp4")
    nframes = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    os.makedirs(WORK, exist_ok=True)

    res = {}
    for tag, hi in (("u8", False), ("fp32", True)):
        ok, dt, out, log = run(tag, src, nframes, hi)
        print(f"{tag:5s} {'OK ' if ok else 'FAIL'} {dt:6.1f}s  {out if ok else log.strip().splitlines()[-1][:120]}")
        if not ok:
            return 1
        res[tag] = (out, dt)

    import numpy as np
    a, info = frames(res["u8"][0], nframes)
    b, _ = frames(res["fp32"][0], nframes)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mse = float(((a - b) ** 2).mean())
    psnr = 99.0 if mse <= 0 else 10 * np.log10(1.0 / mse)
    l1 = float(np.abs(a - b).mean()) * 255
    mx = float(np.abs(a - b).max()) * 255

    print("\n=== fp32 vs uint8（模型输入不同，输出差异）===")
    print(f"帧数 {n}  {info['w']}x{info['h']}")
    print(f"PSNR      {psnr:.2f} dB")
    print(f"平均绝对差 {l1:.3f} /255   最大 {mx:.1f} /255")
    print(f"耗时      uint8 {res['u8'][1]:.1f}s   fp32 {res['fp32'][1]:.1f}s   "
          f"差 {res['fp32'][1] - res['u8'][1]:+.1f}s")
    json.dump(dict(psnr=psnr, l1=l1, max=mx, t_u8=res["u8"][1], t_fp32=res["fp32"][1],
                   n=n, wh=[info["w"], info["h"]]),
              open(os.path.join(WORK, "result.json"), "w", encoding="utf-8"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
