# -*- coding: utf-8 -*-
"""查 copy 引擎为什么一直忙：是不是每 tile 都把权重/画布搬来搬去。

在 flashvsr_gpu_canvas 的循环前后各打一行：
  - GPU 已分配显存（掉到低位又涨回来 = 每 tile 重新分配/上卡）
  - TCDecoder / LQ_proj_in 所在设备（若在 CPU 或每 tile 变 = 每 tile 搬运）
  - 画布所在设备
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import engine as E  # noqa: E402

SRC = r"D:\ComfyUI_windows_portable\ComfyUI\input\flashvsr_long_544.mp4"


def main():
    eng = E.Engine()
    eng.ensure_loaded("tiny-long", "sparse", lambda *a: None, quant="mxfp8")
    pipe = eng.pipe
    mod = eng.mod

    calls = {"n": 0}
    orig_call = pipe.__class__.__call__

    def probe_call(self, *a, **kw):
        md = getattr(self, "dit", None)
        dec = getattr(self, "TCDecoder", None)
        lq = getattr(getattr(self, "dit", None), "LQ_proj_in", None)

        def dev(m):
            if m is None:
                return "-"
            try:
                p = next(m.parameters())
                return str(p.device)
            except StopIteration:
                return "empty"

        calls["n"] += 1
        if calls["n"] <= 3:
            print(f"[probe] pipe 调用 #{calls['n']}: dit={dev(md)} dec={dev(dec)} "
                  f"LQproj={dev(lq)}  allocated={torch.cuda.memory_allocated()/2**20:.0f} MiB",
                  flush=True)
        out = orig_call(self, *a, **kw)
        if calls["n"] <= 3:
            print(f"[probe]   返回后: allocated={torch.cuda.memory_allocated()/2**20:.0f} MiB",
                  flush=True)
        return out

    pipe.__class__.__call__ = probe_call

    frames = E.decode_frames_range(SRC, 0, 25, pre_scale=1.0, fps=25.0)
    frames = torch.from_numpy(np.ascontiguousarray(frames)).float().div_(255.0)
    add = mod.next_8n5(len(frames)) - len(frames)
    if add > 0:
        frames = torch.cat([frames, frames[-1:].repeat([add, 1, 1, 1])], dim=0)
    out = eng._run_window(frames, 2, 128, 3, None, sparse=2.0, local=9,
                          tile_overlap=32, color_fix=False, seam_fix="smooth")
    res = out[0] if isinstance(out, tuple) else out
    print(f"[probe] 共 {calls['n']} 次 pipe 调用，输出 {tuple(res.shape)}")


if __name__ == "__main__":
    sys.exit(main())
