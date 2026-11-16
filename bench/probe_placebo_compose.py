# -*- coding: utf-8 -*-
"""用正确语法（extra_opts=preset=high_quality）测"前缩放 + 后处理"能否复合。

关键：extra_opts 是 libplacebo 滤镜的一个 **dictionary 选项**，
`extra_opts=preset=high_quality` 才是把 preset 交给 libplacebo 自己的选项解析器。

变体（1080p → 900x540）：
  A   只降采样（ewa_lanczos4sharpest）
  B   A + extra_opts=preset=high_quality        ← 已知：preset 会盖掉缩放器
  B2  spline36 + extra_opts=preset=high_quality ← 控制组：与 B 比字节，验证"覆盖"
  C   链式：ewa 缩放到目标 → 再来一级 1:1 的 extra_opts=preset=high_quality

测量：可逆性（放大回 1080p 与原片比 PSNR/SSIM）、字节对照、耗时。
"""
import hashlib
import os
import subprocess
import time

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
SRC = os.path.join(ROOT, "work", "placebo", "src_1080_50f.mp4")
OUT = os.path.join(ROOT, "work", "chain3")
FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
N = 20
W, H = 960, 540

VARIANTS = {
    "A_ewa": f"libplacebo=w={W}:h={H}:downscaler=ewa_lanczos4sharpest",
    "B_ewa_hq": f"libplacebo=w={W}:h={H}:downscaler=ewa_lanczos4sharpest:extra_opts=preset=high_quality",
    "B2_spline_hq": f"libplacebo=w={W}:h={H}:downscaler=spline36:extra_opts=preset=high_quality",
    "C_chain": (f"libplacebo=w={W}:h={H}:downscaler=ewa_lanczos4sharpest,"
                f"libplacebo=w={W}:h={H}:extra_opts=preset=high_quality"),
}


def run(vf, dst):
    cmd = [FF, "-v", "error", "-y", "-init_hw_device", "vulkan=pl",
           "-filter_hw_device", "pl", "-i", SRC, "-frames:v", str(N),
           "-vf", vf + ",hwdownload,format=yuv420p",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "6", "-pix_fmt", "yuv420p", dst]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    ok = p.returncode == 0
    if not ok:
        err = (p.stderr or "").strip().splitlines()
        print(f"  {dst.split(chr(92))[-1]:20s} 失败: {err[-1][:110] if err else '?'}")
    return ok, dt


def upscale_back(path, dst):
    cmd = [FF, "-v", "error", "-y", "-init_hw_device", "vulkan=pl",
           "-filter_hw_device", "pl", "-i", path, "-frames:v", str(N),
           "-vf", "libplacebo=w=1920:h=1080:upscaler=ewa_lanczos,hwdownload,format=yuv420p",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "4", "-pix_fmt", "yuv420p", dst]
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").returncode == 0


def metrics(a, b):
    out = subprocess.run([FF, "-hide_banner", "-i", a, "-i", b, "-lavfi",
                          "[0:v][1:v]psnr;[0:v][1:v]ssim", "-f", "null", "NUL"],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace").stderr or ""
    ps = ssim = "?"
    for line in out.splitlines():
        if "average:" in line:
            ps = line.strip().split("average:")[-1].split()[0]
        if "All:" in line:
            ssim = line.strip().split("All:")[-1].split()[0]
    return ps, ssim


def digest(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"源 {os.path.basename(SRC)} → {W}x{H}，{N} 帧\n")
    made, times = {}, {}
    for tag, vf in VARIANTS.items():
        dst = os.path.join(OUT, f"{tag}.mp4")
        ok, dt = run(vf, dst)
        if ok:
            made[tag] = dst
            times[tag] = dt
            print(f"  {tag:14s} {dt*1000:6.0f} ms")

    print("\n可逆性（放大回 1080p 与原片比）：")
    for tag, path in made.items():
        rt = os.path.join(OUT, f"rt_{tag}.mp4")
        if upscale_back(path, rt):
            ps, ss = metrics(rt, SRC)
            print(f"  {tag:14s} PSNR {ps:>12s}   SSIM {ss:>9s}   ({times[tag]*1000:.0f} ms)")

    print("\n字节级对照：")
    d = {k: digest(v) for k, v in made.items()}
    pairs = [("A_ewa", "B_ewa_hq"), ("B_ewa_hq", "B2_spline_hq"),
             ("A_ewa", "C_chain"), ("B_ewa_hq", "C_chain")]
    for x, y in pairs:
        if x in d and y in d:
            print(f"  {x:12s} vs {y:12s} {'完全相同' if d[x] == d[y] else '不同'}")


if __name__ == "__main__":
    main()
