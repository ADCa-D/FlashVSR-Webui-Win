# -*- coding: utf-8 -*-
"""从导出的 ONNX 图标定「算力占用率」：判断解码器还有多少理论余量。

用法：python bench/probe_flops.py tcdecoder_96.onnx
输出：总 MACs / FLOPs、按算子的分布、以及在给定耗时下的达成 TFLOPS。
"""
import sys
from collections import defaultdict

import numpy as np
import onnx
from onnx import numpy_helper


def main(path, ms=None):
    m = onnx.load(path)
    from onnx import shape_inference
    m = shape_inference.infer_shapes(m)
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    shapes = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        try:
            shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        except Exception:
            pass

    per = defaultdict(lambda: [0, 0])   # op -> [macs, calls]
    total_macs = 0
    for n in g.node:
        if n.op_type != "Conv":
            continue
        w = init.get(n.input[1])
        if w is None:
            continue
        k = w.shape[2] * (w.shape[3] if w.ndim > 3 else 1)
        cin, cout = w.shape[1], w.shape[0]
        x = shapes.get(n.input[0])
        if not x or len(x) < 4:
            continue
        h, wd = x[2], x[3]
        macs = k * cin * cout * h * wd
        per["Conv"][0] += macs
        per["Conv"][1] += 1
        total_macs += macs

    print(f"{path}")
    print(f"  总 MACs {total_macs/1e9:.1f} G  →  FLOPs {2*total_macs/1e9:.1f} G")
    print(f"  Conv 层数 {per['Conv'][1]}")
    if ms:
        print(f"  耗时 {ms:.1f} ms → 达成 {2*total_macs/1e9/(ms/1000)/1e3:.2f} TFLOPS")
        # 若按不同的假定峰值反推可达目标
        for peak in (20, 40, 60, 90):
            print(f"    峰值 {peak:3d} TFLOPS 时，利用率 {2*total_macs/1e9/(ms/1000)/peak/1e3*100:5.1f}%")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "bench/tcdecoder_96.onnx"
    ms = float(sys.argv[2]) if len(sys.argv) > 2 else None
    main(p, ms)
