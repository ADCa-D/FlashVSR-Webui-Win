# -*- coding: utf-8 -*-
r"""回归测试：分段计划 _seg_plans 的边界（不需要 GPU / 模型）。

覆盖：正常切段、total < chunk、total == chunk、chunk=1、overlap >= chunk（不能死循环）、
单帧源、以及"每个保留帧恰好出现一次"这个拼接正确性的前提。

    <ComfyUI 便携包>\python_embeded\python.exe tests\test_seg_plan.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import engine as E  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    if not cond:
        FAIL.append(f"{name} {detail}".strip())
        print(f"FAIL {name} {detail}".strip())


def verify(total, chunk, overlap, tag):
    plans = E._seg_plans(total, chunk, overlap)
    # 1) 每个保留帧恰好出现一次（成片不重不漏）
    kept = [(p[0], p[2]) for p in plans]
    covered = []
    for a, b in kept:
        covered.extend(range(a, b))
    check(tag + " 覆盖", covered == list(range(total)), f"covered={covered[:8]}… total={total}")
    # 2) 段内几何：保留区间落在解码区间内、起点接上一段终点、每段至少 1 帧
    prev_end = 0
    for i, (start, ws, we) in enumerate(plans):
        check(tag + f" 第{i}段", we > ws and ws <= start < we and we - start >= 1,
              f"plan=({start},{ws},{we})")
        check(tag + f" 第{i}段起点", start == prev_end, f"start={start} prev_end={prev_end}")
        check(tag + f" 第{i}段重叠", start - ws <= max(0, int(overlap)), "")
        prev_end = we
    check(tag + " 终点", not plans or plans[-1][2] == total, "")
    # 3) 窗口长度不超过 chunk
    check(tag + " 窗长", all(p[2] - p[1] <= max(1, int(chunk)) for p in plans), "")
    return plans


def main():
    check("空源", E._seg_plans(0, 16, 4) == [], "")
    check("total<chunk", verify(3, 16, 4, "total<chunk") == [[0, 0, 3]], "")
    check("total==chunk", verify(16, 16, 4, "total==chunk") == [[0, 0, 16]], "")
    check("经典 9/4/1",
          verify(9, 4, 1, "9/4/1") == [[0, 0, 4], [4, 3, 7], [7, 6, 9]], "")
    check("整段 4/4/0", verify(8, 4, 0, "8/4/0") == [[0, 0, 4], [4, 4, 8]], "")
    check("chunk=1", verify(4, 1, 0, "chunk=1") == [[0, 0, 1], [1, 1, 2], [2, 2, 3], [3, 3, 4]], "")
    # overlap >= chunk 以前会让 start 不前进（win_end == start）→ 死循环/无限段
    p = verify(10, 4, 4, "overlap==chunk")
    check("overlap==chunk 有界", len(p) <= 10, f"n={len(p)}")
    p = verify(10, 4, 99, "overlap>>chunk")
    check("overlap>>chunk 有界", len(p) <= 10, f"n={len(p)}")
    check("负数容错", E._seg_plans(-5, 4, 1) == [], "")
    check("chunk=0 归一", E._seg_plans(3, 0, 0) == [[0, 0, 1], [1, 1, 2], [2, 2, 3]], "")

    if FAIL:
        print(f"FAIL（{len(FAIL)} 项）")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
