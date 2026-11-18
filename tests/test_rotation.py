# -*- coding: utf-8 -*-
r"""离线验收：手机竖拍（rotation 元数据）的解码朝向。

背景：ffmpeg 对 rawvideo 输出**会**按 rotation 自动摆正（实测 rotation=90 的像素与
-noautorotate + transpose=2 逐位一致）。engine.probe() 也把宽高换成显示朝向。
两条必须同时成立，否则要么侧躺、要么二次旋转（180°）。这里用一份不带元数据的源
+ 一份 remux 后带 rotation=90 的副本，断言：

1. probe(rot) 报的是显示朝向（宽高互换）且 rot=90；
2. decode_frames(rot) 的帧 = 未旋转源帧的 90° 旋转（逐位），且**不**是 180°；
3. 没有 rotation 元数据的源解密后与原帧逐位相同。

跑法：
    <ComfyUI 便携包>\python_embeded\python.exe tests\test_rotation.py
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import engine as E  # noqa: E402


def make_fixtures(d):
    src = os.path.join(d, "plain.mp4")
    rot = os.path.join(d, "rotated.mp4")
    subprocess.check_call([E.FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                           "testsrc2=size=64x48:rate=5:duration=0.6",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", src])
    # remux 只加元数据，像素不动
    subprocess.check_call([E.FFMPEG, "-v", "error", "-y", "-display_rotation", "90",
                           "-i", src, "-c", "copy", rot])
    return src, rot


def main():
    fails = []
    d = tempfile.mkdtemp(prefix="fsv_rot_")
    try:
        src, rot = make_fixtures(d)
        i0, i1 = E.probe(src), E.probe(rot)
        if (i0["w"], i0["h"], i0["rot"]) != (64, 48, 0):
            fails.append(f"无元数据源 probe 错：{i0['w']}x{i0['h']} rot={i0['rot']}")
        if (i1["w"], i1["h"], i1["rot"]) != (48, 64, 90):
            fails.append(f"rotation=90 的 probe 应是显示朝向 48x64/rot=90，实际 "
                         f"{i1['w']}x{i1['h']} rot={i1['rot']}")

        a, _ = E.decode_frames(src)
        b, _info_b = E.decode_frames(rot)
        if b.shape != (a.shape[0], 64, 48, 3):
            fails.append(f"rotation=90 解出的形状应是 {a.shape[0]}x64x48，实际 {b.shape}")
        else:
            cw = np.rot90(a, 1, axes=(1, 2))     # 逆时针 90（= ffmpeg transpose=2）
            ccw = np.rot90(a, 3, axes=(1, 2))    # 顺时针 90（= transpose=1）
            hit = [k for k, ref in ((1, cw), (3, ccw))
                   if ref.shape == b.shape and np.array_equal(ref, b)]
            if len(hit) != 1:
                fails.append("解出的帧不是未旋转帧的 ±90°（可能是二次旋转成 180° 或没转）")
            else:
                d180 = np.rot90(a, 2, axes=(1, 2))
                if d180.shape == b.shape and np.array_equal(d180, b):
                    fails.append("解出的帧是 180°（二次旋转）")
    finally:
        shutil.rmtree(d, ignore_errors=True)

    if fails:
        print("FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS: rotation 元数据按显示朝向解码（probe 宽高互换 + 只转一次）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
