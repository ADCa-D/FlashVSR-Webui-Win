# -*- coding: utf-8 -*-
"""preset=high_quality 的真实价值：去色带 / 抖动 / 色度还原。

我的上一轮用"往返 PSNR"评价它，这个指标**天生惩罚抖动**（抖动就是有意加的噪声），
所以得出"HQ 更差"是错的度量。这次专门测色带：
  - 生成一张平滑渐变（8bit 下必然带色带）
  - 三种处理：spline36 / ewa_lanczos4sharpest / preset=high_quality
  - 指标：一行里"不同亮度值个数"、"相邻像素最大跳变"、"最长同值平台"
    —— 色带 = 平台长 + 跳变大；抖动 = 平台碎、跳变小
"""
import os
import subprocess

ROOT = r"C:\Users\MECHREVO\Documents\默认项目\flashvsr-web"
OUT = os.path.join(ROOT, "work", "hq_banding")
FF = r"D:\ffmpeg-9.0.1\ffmpeg.exe"
FFPROBE = r"D:\ffmpeg-9.0.1\ffprobe.exe"
N = 8

CASES = [
    ("spline36", "libplacebo=w=540:h=304:downscaler=spline36"),
    ("ewa_lanczos4sharpest", "libplacebo=w=540:h=304:downscaler=ewa_lanczos4sharpest"),
    ("preset=high_quality", "libplacebo=w=540:h=304:extra_opts=preset=high_quality"),
    ("HQ + scaler(被覆盖验证)", "libplacebo=w=540:h=304:downscaler=spline36:extra_opts=preset=high_quality"),
]


def make_gradient(path):
    """平滑横向渐变 + 极轻的垂直变化：8bit 落盘后必然出现色带。"""
    vf = ("geq=lum='128 + 60*sin(X/W*3.14159*2)':cb='128 + 20*sin(X/W*3.14159)':"
          "cr='128 - 20*sin(X/W*3.14159)',format=yuv420p")
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi",
                    "-i", f"color=c=gray:s=1920x1080:r=25:d={N/25:.2f}",
                    "-vf", vf, "-frames:v", str(N), "-c:v", "libx264",
                    "-crf", "0", "-pix_fmt", "yuv420p", path], check=True)


def run(src, filt, dst):
    cmd = [FF, "-v", "error", "-y", "-init_hw_device", "vulkan=pl",
           "-filter_hw_device", "pl", "-i", src, "-frames:v", str(N),
           "-vf", filt + ",hwdownload,format=yuv420p", "-an",
           "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode == 0, (r.stderr or "")


def luma_rows(path, rows=(100, 150, 200), width=540, height=304):
    """取几行亮度，返回 numpy 数组。"""
    import numpy as np
    p = subprocess.run([FF, "-v", "error", "-i", path, "-frames:v", "1",
                        "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.uint8).reshape(-1, height, width)
    return [a[0, r, :] for r in rows if r < height]


def band_metrics(row):
    import numpy as np
    d = np.abs(np.diff(row.astype(int)))
    # 最长同值平台
    longest = run = 1
    for i in range(1, len(row)):
        run = run + 1 if row[i] == row[i - 1] else 1
        longest = max(longest, run)
    return dict(distinct=int(len(np.unique(row))), max_step=int(d.max()),
                longest_flat=int(longest))


def crop(path, out, x=180, y=60, w=300, h=190, zoom=3):
    subprocess.run([FF, "-v", "error", "-y", "-i", path, "-vf",
                    f"crop={w}:{h}:{x}:{y},scale=iw*{zoom}:ih*{zoom}:flags=neighbor",
                    "-frames:v", "1", out], check=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    src = os.path.join(OUT, "gradient_1080.mp4")
    if not os.path.isfile(src):
        make_gradient(src)
    print(f"素材 {os.path.basename(src)}（平滑渐变，8bit 必有色带）\n")
    print(f"{'配置':26s} {'体积KB':>7s} {'不同值':>7s} {'最大跳变':>8s} {'最长平台':>8s}")
    for tag, filt in CASES:
        dst = os.path.join(OUT, tag.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "") + ".mp4")
        ok, err = run(src, filt, dst)
        if not ok:
            print(f"{tag:26s} FAIL {err.strip().splitlines()[-1][:70]}")
            continue
        rows = luma_rows(dst)
        m = [band_metrics(r) for r in rows]
        avg = {k: sum(x[k] for x in m) / len(m) for k in m[0]}
        print(f"{tag:26s} {os.path.getsize(dst)//1024:7d} {avg['distinct']:7.1f} "
              f"{avg['max_step']:8.1f} {avg['longest_flat']:8.1f}")
        crop(dst, os.path.join(OUT, tag.split("+")[0].strip().replace(" ", "_").replace("=", "") + "_crop.png"))
    print("\n裁图在", OUT)


if __name__ == "__main__":
    main()
