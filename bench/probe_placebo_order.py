# -*- coding: utf-8 -*-
"""前缩放 vs 后处理的先后顺序（libplacebo）。

这个 ffmpeg 的 libplacebo 滤镜**没有 preset 选项**，可用的后处理旋钮是
deband*/dithering/contrast。所以能测的真实问题是：
  deband 到底该在"缩放前（源分辨率）"还是"缩放后"做？

变体：
  A  只降采样（基准）
  B  降采样 + preset=high_quality      ← 复核旧结论（预期：直接报错）
  C  降采样 + deband（单级，后处理在缩放之后）
  D  链式：先在源分辨率 deband → 再降采样（后处理在缩放之前）

测量：放大回原尺寸与原片比 PSNR/SSIM；再比字节/耗时。
"""
import hashlib
import os
import subprocess
import time

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
SRC = os.path.join(ROOT, "work", "placebo", "src_1080_50f.mp4")
OUT = os.path.join(ROOT, "work", "chain2")
FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
N = 20
W, H = 960, 540

DOWN = f"libplacebo=w={W}:h={H}:downscaler=ewa_lanczos4sharpest"
VARIANTS = {
    "A_scale_only": DOWN,
    "B_with_preset": DOWN + ":preset=high_quality",
    "C_deband_after": DOWN + ":deband=1",
    "D_deband_before": (f"libplacebo=w=1920:h=1080:deband=1,"
                        f"libplacebo=w={W}:h={H}:downscaler=ewa_lanczos4sharpest"),
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
        msg = (p.stderr or "").strip().splitlines()
        print(f"  {os.path.basename(dst):22s} 失败: {msg[-1][:110] if msg else '?'}")
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
            print(f"  {tag:18s} {dt*1000:6.0f} ms")

    print("\n可逆性（放大回 1080p 与原片比）：")
    for tag, path in made.items():
        rt = os.path.join(OUT, f"rt_{tag}.mp4")
        if upscale_back(path, rt):
            ps, ss = metrics(rt, SRC)
            print(f"  {tag:18s} PSNR {ps:>12s}   SSIM {ss:>9s}")

    print("\n字节级对照：")
    if len(made) >= 2:
        base = digest(made["A_scale_only"])
        for tag, path in made.items():
            if tag == "A_scale_only":
                continue
            same = digest(path) == base
            print(f"  A vs {tag:18s} {'完全相同（该选项没起作用）' if same else '不同（真的改变了像素）'}")


if __name__ == "__main__":
    main()
