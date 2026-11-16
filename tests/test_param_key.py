# -*- coding: utf-8 -*-
"""回归测试：分段缓存键必须「面板 == 引擎」，且影响画面的参数必须进键。

不需要 GPU：
    <ComfyUI 便携包>\\python_embeded\\python.exe tests\\test_param_key.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import app as A  # noqa: E402
import config  # noqa: E402
import engine as E  # noqa: E402

ROOT_SEG = os.path.join(ROOT, "work", "segments")
FIXTURE = os.path.join(ROOT, "work", "test_fixture.mp4")

BASE = dict(preset="custom", scale="2", tile="256", kv="3", chunk="16", mode="tiny-long",
            attention="sparse", quant="mxfp8", encoder="x265", quality="18",
            prescale="off", preview="0")


def ensure_fixture():
    """随便一个能 probe 的小视频就够——测试只看文件名里的指纹。"""
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    if not os.path.isfile(FIXTURE):
        r = subprocess.run([config.FFMPEG, "-v", "error", "-y", "-f", "lavfi",
                            "-i", "color=c=gray:s=64x64:r=5:d=0.4",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", FIXTURE],
                           capture_output=True)
        if r.returncode != 0:
            return None
    return FIXTURE


def keys(src, **over):
    P = E.resolve_params(dict(BASE, **over), src)
    params = dict(scale=P["scale"], tile=P["tile"], kv=P["kv"], chunk=P["chunk"], overlap=8,
                  mode=P["mode"], attention=P["attention"], encoder=P["encoder"],
                  quality=P["quality"], quant=P["quant"], budget=P["budget"],
                  pre_scale=P["pre_scale"], tile_overlap=P["tile_overlap"],
                  bitdepth=P["bitdepth"], hi_precision=P["hi_precision"],
                  max_frames=P["max_frames"], color_fix=P["color_fix"],
                  seam_fix=P["seam_fix"], tune=P["tune"], extra_args=P["extra_args"],
                  tile_h=0, tile_w=0, restart=False, seg_dir=ROOT_SEG)
    eng = os.path.basename(os.path.join(ROOT_SEG, f"{E.quick_hash(src)}_{E._param_key(params)}"))
    app = os.path.basename(A._seg_key_dir(src, P, ROOT_SEG))
    return eng, app


def main():
    src = ensure_fixture()
    if not src:
        print("SKIP: 没有可用的 ffmpeg 生成测试素材")
        return 0

    fails = []
    base_eng, base_app = keys(src)
    if base_eng != base_app:
        fails.append(f"面板键 {base_app} ≠ 引擎键 {base_eng}")

    e, _ = keys(src, bitdepth="10")
    if e != base_eng:
        fails.append("bitdepth=10 不该换目录，却换了")

    for tag, over in (("tune=psnr", {"tune": "psnr"}),
                      ("extra_args 非空", {"extra_args": "-rc vbr"}),
                      ("color_fix=0", {"color_fix": "0"}),
                      ("seam_fix=align", {"seam_fix": "align"}),
                      ("hi_precision=1", {"hi_precision": "1"}),
                      ("max_frames=48", {"max_frames": "48"}),
                      ("tile_overlap=12", {"tile_overlap": "12"}),
                      ("tile=192", {"tile": "192"})):
        e, a = keys(src, **over)
        if e == base_eng:
            fails.append(f"{tag} 必须换目录，却复用了旧缓存")
        if e != a:
            fails.append(f"{tag}：面板与引擎键不一致（{a} vs {e}）")

    # 矩形 tile 会改接缝位置；dither 是环境变量但会改像素 —— 都必须进键
    raw0 = dict(scale=2, tile=256, kv=3, chunk=16, overlap=8, mode="tiny-long",
                attention="sparse", encoder="x265", quality=18, quant="mxfp8",
                budget="auto", pre_scale=0.0, tile_overlap=24, bitdepth=8,
                hi_precision=0, max_frames=0, color_fix=1, seam_fix="off", tune="",
                extra_args="", tile_h=0, tile_w=0, restart=False, seg_dir=ROOT_SEG)
    k0 = E._param_key(dict(raw0))
    for tag, over in (("tile_h=64", {"tile_h": 64}), ("tile_w=96", {"tile_w": 96}),
                      ("budget=motion", {"budget": "motion"})):
        if E._param_key(dict(raw0, **over)) == k0:
            fails.append(f"{tag} 必须换目录，却复用了旧缓存（参数键漏字段）")

    os.environ["FLASHVSR_DITHER"] = "bayer"
    try:
        if E._param_key(dict(raw0)) == k0:
            fails.append("FLASHVSR_DITHER 必须进键（改了抖动却复用旧分段）")
    finally:
        os.environ.pop("FLASHVSR_DITHER", None)

    if fails:
        print("FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS: 分段缓存键一致，且影响画面的参数都已进键")
    return 0


if __name__ == "__main__":
    sys.exit(main())
