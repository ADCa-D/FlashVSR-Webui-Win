# -*- coding: utf-8 -*-
"""端到端验证 tune 参数真的生效（用合成帧直接走 encode_frames，不需要 GPU）。

关键点：SVT-AV1 传 `tune=vq` 这种**名字**会被拒并**静默退回 PSNR**，
所以除了"能出片"，还要抓它自己打印的 `preset / tune / pred struct` 行确认生效。

    <ComfyUI 便携包>\\python_embeded\\python.exe tests\\test_tune_encode.py
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import config  # noqa: E402
import engine as E  # noqa: E402

W, H, N = 320, 240, 16
OUT = os.path.join(ROOT, "work", "tune_test")
SVT_EXPECT = {"vq": "VQ", "psnr": "PSNR", "ssim": "SSIM", "ms_ssim": "MS_SSIM", "vmaf": "VMAF"}


def frames():
    x = np.linspace(0, 255, W, dtype=np.float32)[None, None, :, None]
    y = np.linspace(0, 255, H, dtype=np.float32)[None, :, None, None]
    a = np.repeat((x * 0.6 + y * 0.4), 3, axis=3)[:, :, :, :3]
    return np.repeat(a, N, axis=0).astype(np.uint8)


def nframes(path):
    r = subprocess.run([config.FFPROBE, "-v", "error", "-select_streams", "v:0",
                        "-count_frames", "-show_entries", "stream=nb_read_frames",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return int((r.stdout or "0").strip())
    except ValueError:
        return 0


def svt_effective_tune(tune):
    """直接问 SVT：1 帧、-v info，抓 `preset / tune / pred struct : 8 / VQ / ...`。"""
    args = E.encoder_args("svtav1", 32, 8, tune=tune)
    r = subprocess.run([config.FFMPEG, "-hide_banner", "-v", "info", "-f", "lavfi",
                        "-i", f"color=c=gray:s={W}x{H}:r=5:d=0.2", "-frames:v", "1"]
                       + args + ["-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"preset / tune / pred struct\s*:\s*\d+\s*/\s*(\w+)\s*/", r.stderr or "")
    return m.group(1) if m else "?"


CASES = [
    ("cpu", "ssim", None), ("cpu", "psnr", None), ("cpu", "film", None),
    ("x265", "psnr", None), ("x265", "ssim", None),
    ("svtav1", "vq", None), ("svtav1", "psnr", None), ("svtav1", "ssim", None),
    ("svtav1", "ms_ssim", None), ("svtav1", "vmaf", None),
    ("cpu", "", None), ("x265", "", None), ("svtav1", "", "8"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    a = frames()
    fails = []
    for be, tune, svt_p in CASES:
        tag = f"{be}_{tune or 'none'}{'_p' + svt_p if svt_p else ''}"
        dst = os.path.join(OUT, tag + ".mp4")
        try:
            E.encode_frames(a, 25.0, dst, None, quality=28, backend=be,
                            bitdepth=8, svt_preset=svt_p, tune=tune or None)
        except Exception as e:  # noqa
            fails.append(f"{tag}: {type(e).__name__}: {str(e)[:120]}")
            print(f"FAIL {tag}")
            continue
        got = nframes(dst)
        ok = got == N and os.path.getsize(dst) > 512
        print(f"{'OK  ' if ok else 'FAIL'} {tag:22s} {got} 帧  {os.path.getsize(dst)//1024} KB")
        if not ok:
            fails.append(f"{tag}: 帧数 {got} != {N}")

    # tune 是否真的生效（防"名字被拒后静默退回 PSNR"）
    print()
    for tune, want in SVT_EXPECT.items():
        got = svt_effective_tune(tune)
        ok = got.upper() == want
        print(f"{'OK  ' if ok else 'FAIL'} svtav1 tune={tune:8s} → 实际 {got}（期望 {want}）")
        if not ok:
            fails.append(f"svtav1 tune={tune} 实际生效为 {got}，期望 {want}")

    if fails:
        print("\n失败：")
        for f in fails:
            print("  -", f)
        return 1
    print("\nPASS: tune 组合可用，且 SVT 的 tune 确实生效（不是静默退回 PSNR）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
