# -*- coding: utf-8 -*-
"""编码器降级阶梯 + 能力选择的回归测试（不需要 GPU/Intel 设备）。

    <ComfyUI 便携包>\\python_embeded\\python.exe tests\\test_encoder_ladder.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import engine as E  # noqa: E402


def caps_all_ok():
    return {be: dict(ok=True, detail="") for be in E.ENCODER_CHOICES}


def caps_none():
    return {be: dict(ok=False, detail="x") for be in E.ENCODER_CHOICES}


def main():
    fails = []

    # 1) 阶梯必须覆盖所有编码器，且最终都收敛到 cpu（不能死循环/断链）
    for be in E.ENCODER_CHOICES:
        if be == "auto":
            continue
        chain, seen = [], set()
        cur = be
        for _ in range(10):
            if cur == "cpu":
                break
            nxt = (E.FALLBACK_CHAIN.get(cur) or ("cpu",))[0]
            if nxt in seen:
                fails.append(f"{be} 的降级链出现环：{chain + [nxt]}")
                break
            seen.add(nxt)
            chain.append(nxt)
            cur = nxt
        else:
            fails.append(f"{be} 的降级链没有在 10 步内到 cpu")

    # 2) 全部可用时：auto 应选 cpu（GPU 编码抢推理资源，实测不划算）
    if E.resolve_encoder("auto", caps_all_ok()) != "cpu":
        fails.append("auto 在全部可用时没有选 cpu")

    # 3) 显式选择：可用就用（不擅自改成别的）
    for be in ("qsv", "qsv_h264", "x265", "svtav1", "nvenc", "amf", "av1_nvenc"):
        got = E.resolve_encoder(be, caps_all_ok())
        if got != be:
            fails.append(f"显式选 {be} 却被改成 {got}")

    # 4) 显式选择不可用 → 落到 cpu（最后一层兜底）
    if E.resolve_encoder("qsv", caps_none()) != "cpu":
        fails.append("编码器全不可用时没有落到 cpu")

    # 5) qsv 的失败阶梯必须包含 qsv_h264（同族 H.264 兜底）
    if "qsv_h264" not in E.FALLBACK_CHAIN.get("qsv", ()):
        fails.append("qsv 的降级链里没有 qsv_h264")

    if fails:
        print("FAIL")
        for f in fails:
            print("  -", f)
        return 1
    print("PASS: 降级阶梯收敛到 cpu、auto 选 cpu、显式选择不被篡改")
    return 0


if __name__ == "__main__":
    sys.exit(main())
