# -*- coding: utf-8 -*-
"""编码器参数回归：不改硬件也能验证命令行是否合法/是否符合兼容性策略。

    <ComfyUI 便携包>\\python_embeded\\python.exe tests\\test_encoders.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import engine as E  # noqa: E402

PASS, FAIL = "PASS", "FAIL"


def args(be, bd):
    return " ".join(E.encoder_args(be, 20, bd))


def main():
    checks = [
        # QSV 通用路径：系统内存 + nv12，不许出现 hwupload/hwframes
        ("qsv 8bit 用 hevc_qsv + nv12",
         "hevc_qsv" in args("qsv", 8) and "nv12" in args("qsv", 8)),
        ("qsv 不带 hwupload", "hwupload" not in args("qsv", 8)),
        ("qsv 用 global_quality（ICQ）", "global_quality" in args("qsv", 8)),
        ("qsv 10bit 走 main10/p010le",
         "main10" in args("qsv", 10) and "p010le" in args("qsv", 10)),
        ("qsv_h264 用 h264_qsv", "h264_qsv" in args("qsv_h264", 8)),
        ("qsv_h264 不接受 10bit（仍是 nv12）",
         "nv12" in args("qsv_h264", 10) and "p010le" not in args("qsv_h264", 10)),
        ("x265 10bit 走 yuv420p10le", "yuv420p10le" in args("x265", 10)),
        ("amf 10bit 仍是 8bit",
         "yuv420p" in args("amf", 10) and "10le" not in args("amf", 10)),
        ("硬件编码器都在回退名单里",
         all(b in E.HW_ENCODERS for b in ("qsv", "qsv_h264", "nvenc", "av1_nvenc", "amf"))),
        ("全部编码器都在 choices 里",
         all(b in E.ENCODER_CHOICES for b in ("qsv", "qsv_h264", "x265", "svtav1"))),
        # 每个命令都必须带 -c:v，避免拼出裸 -vf
        ("各后端参数结构完整",
         all("-c:v" in args(b, 8) for b in ("cpu", "x265", "svtav1", "nvenc", "qsv",
                                            "qsv_h264", "amf", "av1_nvenc"))),
    ]

    fails = [tag for tag, ok in checks if not ok]
    if fails:
        print(FAIL)
        for f in fails:
            print("  -", f)
        return 1
    print(PASS + ": 编码器参数符合策略（QSV 走系统内存 + nv12 + ICQ）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
