# -*- coding: utf-8 -*-
r"""回归测试：probe 缓存 / 羽化缓存淘汰 / 抽帧分块判据 / 编码线程关闭语义（不需要 GPU）。

    <ComfyUI 便携包>\python_embeded\python.exe tests\test_infra.py
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np
import torch

import engine as E  # noqa: E402

FAIL = []
WORK = os.path.join(ROOT, "work")
FIX = os.path.join(WORK, "test_infra_fixture.mp4")


def check(name, cond, detail=""):
    if not cond:
        FAIL.append(f"{name} {detail}".strip())
        print(f"FAIL {name} {detail}".strip())


def make_fixture():
    os.makedirs(WORK, exist_ok=True)
    if os.path.exists(FIX):
        return
    subprocess.check_call(
        [E.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
         "color=c=gray:s=64x64:r=5:d=0.6", "-c:v", "libx264", "-crf", "18", FIX])


# ---------------------------------------------------------------- probe 缓存
def test_probe_cache():
    make_fixture()
    orig = subprocess.check_output
    n = {"json": 0}

    def counting(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and "-of" in cmd and "json" in cmd and any(
                isinstance(c, str) and "stream=width" in c for c in cmd):
            n["json"] += 1
        return orig(cmd, *a, **k)

    subprocess.check_output = counting
    try:
        E._PROBE_CACHE.clear()
        i1 = E.probe(FIX)
        c1 = n["json"]
        i2 = E.probe(FIX)
        check("probe 命中缓存", n["json"] == c1, f"calls {c1}→{n['json']}")
        check("probe 结果一致", i1 == i2, f"{i1} vs {i2}")
        check("probe 有尺寸", i1["w"] == 64 and i1["h"] == 64, str(i1))

        i2["w"] = 1
        check("probe 返回副本", E.probe(FIX)["w"] == 64, str(E.probe(FIX)["w"]))

        st = os.stat(FIX)
        os.utime(FIX, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))
        c_before = n["json"]
        E.probe(FIX)
        check("probe 文件改动后失效", n["json"] > c_before, f"calls {c_before}→{n['json']}")
    finally:
        subprocess.check_output = orig
        E._PROBE_CACHE.clear()


# ---------------------------------------------------------------- 羽化缓存淘汰
def test_feather_cache():
    E._FEATHER_CACHE.clear()
    for i in range(26):
        E._FEATHER_CACHE[(1000 + i, 64, 24, "torch.float16", "cpu", True)] = None
    hot = (999, 64, 24, "torch.float16", "cpu", True)   # 新键，插在最末（最「热」）
    E._FEATHER_CACHE[hot] = "HOT"
    m = E._feather_mask_nhwc((72, 64), 8, torch.float16, "cpu")
    check("羽化 mask 形状", tuple(m.shape) == (1, 72, 64, 1), str(m.shape))
    check("羽化缓存不再全清", E._FEATHER_CACHE.get(hot) == "HOT", "热点条目被 clear 掉了")
    check("羽化缓存有上界", len(E._FEATHER_CACHE) <= 26, str(len(E._FEATHER_CACHE)))
    E._FEATHER_CACHE.clear()


# ---------------------------------------------------------------- 抽帧分块判据
def _ref_drop(frames, thr=0.5, blk_thr=2.0):
    """旧实现的逐块 Python 循环版本，用来对拍向量化后的结果。"""
    n = len(frames)
    if n < 8:
        return 1
    g = np.asarray(frames[:, ::4, ::4, :3], dtype=np.int16).mean(axis=-1)
    d1 = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))
    d = np.abs(np.diff(g, axis=0))
    h, w = g.shape[1], g.shape[2]
    ny, nx = min(8, h), min(8, w)
    ys = np.linspace(0, h, ny + 1).astype(int)
    xs = np.linspace(0, w, nx + 1).astype(int)
    bmax = np.zeros(d.shape[0], dtype=np.float64)
    for bi in range(ny):
        for bj in range(nx):
            blk = d[:, ys[bi]:ys[bi + 1], xs[bj]:xs[bj + 1]]
            if blk.size:
                np.maximum(bmax, blk.mean(axis=(1, 2)), out=bmax)
    changed = (d1 >= thr) | (bmax >= blk_thr)
    for p in (2, 3):
        if n <= p + 2:
            continue
        kept = set(range(0, n, p))
        lost = [changed[i - 1] for i in range(1, n) if i not in kept]
        if lost and float(np.mean(np.asarray(lost))) < 0.05:
            return p
    return 1


def test_drop_stride():
    rng = np.random.default_rng(0)
    for seed in range(4):
        rng = np.random.default_rng(seed)
        frames = rng.integers(0, 256, size=(12, 40, 60, 3)).astype(np.uint8)
        check(f"抽帧随机 {seed}", E._drop_stride(frames) == _ref_drop(frames),
              f"{E._drop_stride(frames)} vs {_ref_drop(frames)}")
    still = np.full((12, 40, 60, 3), 7, np.uint8)
    check("抽帧静止帧→2", E._drop_stride(still) == 2, str(E._drop_stride(still)))
    # 局部小方块在动（全局 MAE 很低）：必须判成"有变化"，不能抽
    loc = np.full((12, 40, 60, 3), 7, np.uint8)
    for i in range(12):
        loc[i, 10:18, 20:28] = 7 + 80 * (i % 2)
    check("抽帧局部微动不抽", E._drop_stride(loc) == 1, str(E._drop_stride(loc)))
    # hi_precision 的 float [0,1] 不能被截成 0
    fl = loc.astype(np.float32) / 255.0
    check("抽帧 float32 与 uint8 一致", E._drop_stride(fl) == E._drop_stride(loc),
          f"{E._drop_stride(fl)} vs {E._drop_stride(loc)}")


# ---------------------------------------------------------------- 编码线程
def test_seg_encoder():
    seen = []

    def fn(frames, path, meta):
        seen.append(path)

    se = E._SegEncoder(fn)
    se.submit(0, None, "a.mp4", {})
    time.sleep(0.2)
    se.close()
    se.close()                     # 二次 close 必须无害（try + finally 各收一次）
    check("编码线程编完已提交段", seen == ["a.mp4"], str(seen))
    se.submit(1, None, "b.mp4", {})  # 关闭后提交：不能静默进队列
    time.sleep(0.1)
    check("关闭后提交不丢段也不编码", seen == ["a.mp4"], str(seen))


# ---------------------------------------------------------------- TRT past 双缓冲池
def test_past_pool():
    import trt_decoder as T

    shapes = [(1, 9, 4, 4), (2, 9, 4, 4)]
    p = T._PastPool(device="cpu")
    a = p.get(shapes)
    b = p.get(shapes)
    c = p.get(shapes)
    check("pool 形状", [tuple(t.shape) for t in a] == [(1, 9, 4, 4), (2, 9, 4, 4)],
          str([tuple(t.shape) for t in a]))
    check("pool A/B 交替", all(x is not y for x, y in zip(a, b)), "相邻两帧复用了同一组缓冲")
    check("pool 第三帧回第一组", all(x is y for x, y in zip(a, c)), "没有真正复用")
    other = p.get([(1, 3, 2, 2)])
    check("pool 形状不同各自成对", tuple(other[0].shape) == (1, 3, 2, 2), str(other[0].shape))
    p.clear()
    fresh = p.get(shapes)
    check("pool clear 后重建", all(x is not y for x, y in zip(a, fresh)), "clear 没释放")


def main():
    test_probe_cache()
    test_feather_cache()
    test_drop_stride()
    test_seg_encoder()
    test_past_pool()
    if FAIL:
        print("VERDICT: FAIL")
        for f in FAIL:
            print(" -", f)
        return 1
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
