# -*- coding: utf-8 -*-
"""验证单趟拼接：不产生 concat_video.mp4，音轨/无音轨都能出成品。"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine as E

FF = E.FFMPEG
SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4"


def make_seg(path, n=6):
    """切一小段当分段。"""
    subprocess.run([FF, "-v", "error", "-y", "-i", SRC, "-frames:v", str(n),
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", path],
                   check=True)


def main():
    d = tempfile.mkdtemp(prefix="concat_test_")
    segs = []
    for i in range(3):
        p = os.path.join(d, f"seg_{i}.mp4")
        make_seg(p)
        segs.append(p)

    for tag, audio in (("无音轨", None), ("带音轨", SRC)):
        out = os.path.join(d, f"out_{tag}.mp4")
        E.concat_segments(segs, out, audio_src=audio, backend="cpu", quality=18)
        size = os.path.getsize(out) / 1024
        leftover = [f for f in os.listdir(d) if f.startswith("concat_video")]
        print(f"  {tag}: 成品 {size:.0f} KB   中间文件残留 {len(leftover)}")

        # 帧数应为 3×6
        r = subprocess.run([r"D:\ffmpeg-9.0.1\ffprobe.exe", "-v", "error", "-count_frames",
                            "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
                            "-of", "csv=p=0", out], capture_output=True, text=True)
        n = int("".join(c for c in r.stdout if c.isdigit()) or 0)
        print(f"           帧数 {n}（期望 18）  {'OK' if n == 18 else '**FAIL**'}")
        if audio:
            r = subprocess.run([r"D:\ffmpeg-9.0.1\ffprobe.exe", "-v", "error",
                                "-select_streams", "a:0", "-show_entries", "stream=codec_name",
                                "-of", "csv=p=0", out], capture_output=True, text=True)
            print(f"           音轨 {r.stdout.strip() or '无'}")

    print(f"\n落盘文件: {sorted(os.listdir(d))}")


if __name__ == "__main__":
    sys.exit(main())
