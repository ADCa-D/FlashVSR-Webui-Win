# -*- coding: utf-8 -*-
"""libplacebo 两级链：前级只缩放、后级 1:1 只做后处理（preset=high_quality）。

三种写法：
  A  单级  downscaler=ewa_lanczos4sharpest
  B  单级  downscaler=ewa_lanczos4sharpest + preset=high_quality   （已知：preset 会盖掉缩放器）
  C  两级  ewa 缩放到目标 → 再一次 libplacebo(preset=high_quality) 1:1 后处理

测量：把结果放大回原尺寸，与原始 1080p 比 PSNR/SSIM（越高说明保留的细节越多、混叠越少）。
另外：
  - A 与 B 是否逐字节相同（复核"preset 覆盖缩放器"）
  - C 与 A 的差异（后处理是否真的改了东西）
  - 各自的耗时（两级链的代价）
"""
import hashlib
import os
import subprocess
import time

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
SRC = os.path.join(ROOT, "work", "placebo", "src_1080_50f.mp4")
OUT = os.path.join(ROOT, "work", "chain")
FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
N = 20
W, H = 960, 540

VARIANTS = {
    "A_single_ewa": f"libplacebo=w={W}:h={H}:downscaler=ewa_lanczos4sharpest",
    "B_single_hq": f"libplacebo=w={W}:h={H}:downscaler=ewa_lanczos4sharpest:preset=high_quality",
    "C_chain": (f"libplacebo=w={W}:h={H}:downscaler=ewa_lanczos4sharpest,"
                f"libplacebo=w={W}:h={H}:preset=high_quality"),
    "D_single_hq_only": f"libplacebo=w={W}:h={H}:preset=high_quality",
}


def run(vf, dst):
    cmd = [FF, "-v", "error", "-y", "-init_hw_device", "vulkan=pl",
           "-filter_hw_device", "pl", "-i", SRC, "-frames:v", str(N),
           "-vf", vf + ",hwdownload,format=yuv420p",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "6", "-pix_fmt", "yuv420p", dst]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        print(f"  !! {os.path.basename(dst)} 失败: {(p.stderr or '')[-300:]}")
    return dt


def upscale_back(path, dst):
    cmd = [FF, "-v", "error", "-y", "-init_hw_device", "vulkan=pl",
           "-filter_hw_device", "pl", "-i", path, "-frames:v", str(N),
           "-vf", f"libplacebo=w=1920:h=1080:upscaler=ewa_lanczos,hwdownload,format=yuv420p",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "4", "-pix_fmt", "yuv420p", dst]
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").returncode == 0


def metrics(a, b):
    cmd = [FF, "-hide_banner", "-i", a, "-i", b, "-lavfi", "[0:v][1:v]psnr;[0:v][1:v]ssim",
           "-f", "null", "NUL"]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
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
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"源 {os.path.basename(SRC)} → {W}x{H}，{N} 帧\n")
    made = {}
    for tag, vf in VARIANTS.items():
        dst = os.path.join(OUT, f"{tag}.mp4")
        dt = run(vf, dst)
        made[tag] = dst
        print(f"  {tag:18s} {dt*1e3:6.0f} ms")

    print("\n可逆性（放大回 1080p 与原始比，越高越好）：")
    rows = []
    for tag, path in made.items():
        rt = os.path.join(OUT, f"rt_{tag}.mp4")
        if upscale_back(path, rt):
            ps, ss = metrics(rt, SRC)
            rows.append((tag, ps, ss))
            print(f"  {tag:18s} PSNR {ps:>7s}   SSIM {ss:>8s}")

    print("\n字节级对照：")
    da, db, dc = digest(made["A_single_ewa"]), digest(made["B_single_hq"]), digest(made["C_chain"])
    print(f"  A vs B  {'完全相同 → preset 确实盖掉了缩放器' if da == db else '不同'}")
    print(f"  A vs C  {'完全相同 → 二级后处理没产生任何变化' if da == dc else '不同 → 链式后处理确实改了像素'}")
    print(f"  B vs C  {'相同' if db == dc else '不同（说明 C 不等价于直接加 preset）'}")


if __name__ == "__main__":
    main()
