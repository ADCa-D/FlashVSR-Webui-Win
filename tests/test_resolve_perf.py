# -*- coding: utf-8 -*-
"""resolve_params 的性能相关规则（2026-11-17）：自动 overlap / 2× 块大小 / 单块上限。

这些值直接决定处理面积（≈耗时），且改动会静默改画质，所以钉住。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine as E

FAIL = []


def check(desc, got, want):
    ok = got == want
    print(("  OK   " if ok else "  FAIL ") + desc + f"：got={got!r} want={want!r}")
    if not ok:
        FAIL.append(desc)


def rp(**fields):
    return E.resolve_params(fields, None)


print("1) 自动 overlap：≤160 用 24，≥192 才按 tile/4 加宽")
for tile, want_ov in (("128", 24), ("96", 24), ("160", 24)):
    p = rp(preset="custom", scale="4", tile=tile)
    check(f"4× tile={tile} 自动 overlap", p["tile_overlap"], want_ov)
for tile, want_ov in (("192", 48), ("256", 64)):
    # prescale=off 才是原生 2×（tile 不折半、也不走 4× 的单块上限）
    p = rp(preset="custom", scale="2", prescale="off", tile=tile)
    check(f"2× tile={tile} 自动 overlap", p["tile_overlap"], want_ov)

print("2) 预设解析出的块大小 / 重叠")
for preset, tile, ov in (("quality", 128, 24), ("balance", 160, 24), ("fast", 128, 24)):
    p = rp(preset=preset)
    check(f"preset={preset} tile", p["tile"], tile)
    check(f"preset={preset} overlap", p["tile_overlap"], ov)

print("3) 2× 路径（prescale=half）不再把 tile 折半")
p = rp(preset="custom", scale="2", tile="128")
check("scale=2 tile=128 → LQ tile", p["tile"], 128)
check("scale=2 走 4× 模型", p["scale"], 4)
check("scale=2 pre_scale", p["pre_scale"], 0.5)
p = rp(preset="custom", scale="2", tile="64")
check("scale=2 tile=64 保持", p["tile"], 64)
check("tile=64 的 overlap 退到 1/4", p["tile_overlap"], 16)

print("4) 单块输出上限 640 px（再大模型会静默崩坏/直接 OOM）")
p = rp(preset="custom", scale="4", tile="256")
check("4× tile=256 降到", p["tile"], 160)
check("4× 降档有说明", "已降到 160" in (p.get("auto_note") or ""), True)
p = rp(preset="custom", scale="2", tile="512")
check("2× tile=512 → LQ tile", p["tile"], 128)
p = rp(preset="custom", scale="4", tile="128")
check("4× tile=128 不动", p["tile"], 128)
check("4× tile=128 无说明", p.get("auto_note"), None)

print()
if FAIL:
    print("FAIL: " + "、".join(FAIL))
    sys.exit(1)
print("PASS: 自动 overlap / 2× 块大小 / 单块上限 三条性能规则都钉住了")