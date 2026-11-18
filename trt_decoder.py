# -*- coding: utf-8 -*-
"""TCDecoder 的 TensorRT 逐帧解码（可选加速，失败自动回退 torch）。

为什么是「逐帧 + 9 个显式 past」：
- 管线实际走的是 `decode_video(..., parallel=False)`，按帧深度优先穿过 27 层；
  9 个 MemBlock 各保留「上一帧在本层的输入」作为 past。
- TGrow 在逐帧路径里会把 1 帧裂成 4 个子帧，所以并行图（parallel=True）与逐帧图
  计算量不同，不能拿并行图的数据当收益。

实测（latent 48 / 输出 384²）：torch 38.8 ms/帧 → TRT 21.7 ms/帧 = 1.79×，
数值与 torch 逐帧路径完全一致（maxdiff 0.0，同 fp16 权重）。

用法（engine.py 里）：
    from trt_decoder import maybe_enable
    maybe_enable(pipe.TCDecoder, cache_dir=os.path.join(APP_DIR, "work", "trt_cache"))
任何异常都只打印一行并保持 torch 路径。
"""
import os
import site
import sys
import time

import numpy as np
import torch


def _add_trt_dlls():
    """让 onnxruntime 的 TensorrtExecutionProvider 找到 nvinfer_10.dll。"""
    for d in list(site.getsitepackages() or []):
        p = os.path.join(d, "tensorrt_libs")
        if os.path.isdir(p):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(p)
            except Exception:
                pass
            return p
    return None


# 模块导入时就挂上 DLL 目录：onnxruntime 在 import 阶段就会尝试注册 TRT 插件，
# 晚于 import 再改 PATH 已经来不及（之前 _add_trt_dlls 从未被调用，TRT 一直没生效）。
_add_trt_dlls()


def trt_available():
    """onnxruntime 有 TRT provider 且 nvinfer 能加载。"""
    if not _add_trt_dlls():
        return False
    try:
        import onnxruntime as ort
        return "TensorrtExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


class _Step(torch.nn.Module):
    """单帧走完整个解码器：入 1 帧 784 通道，出 4 个 RGB 帧 + 9 个新 past。"""

    def __init__(self, decoder, mem_layers):
        super().__init__()
        self.decoder = decoder
        self.mem_layers = list(mem_layers)

    def forward(self, x, *pasts):
        mem = {L: p for L, p in zip(self.mem_layers, pasts)}
        n0 = x.shape[0]
        queue = [(x, 0)]
        outs = []
        dec = self.decoder
        while queue:
            xt, i = queue.pop(0)
            if i == len(dec):
                outs.append(xt)
                continue
            b = dec[i]
            name = type(b).__name__
            if name == "MemBlock":
                xt_new = b(xt, mem[i])
                mem[i] = xt
                queue.insert(0, (xt_new, i + 1))
            elif name == "TGrow":
                xt = b(xt)
                _nt, c_, h_, w_ = xt.shape
                for sub in reversed(xt.view(n0, b.stride * c_, h_, w_).chunk(b.stride, 1)):
                    queue.insert(0, (sub, i + 1))
            else:
                xt = b(xt)
                queue.insert(0, (xt, i + 1))
        return (tuple(outs),) + tuple(mem[L] for L in self.mem_layers)


class _PastPool:
    """past 输出缓冲的 A/B 双缓冲池：省掉热循环里每帧 9 次 torch.empty。

    past 是「本帧输出、下一帧输入」，所以相邻两帧只要用不同的一组缓冲，就不会踩到
    正在被 TensorRT 当输入读的那组；第三帧再回到第一组时，第一组的内容早已作废。
    形状按 session 固定，键用形状元组（多分辨率/多 tile 尺寸会各占一对）。
    """

    def __init__(self, device="cuda"):
        self._device = device
        self._sets = {}
        self._i = 0

    def get(self, shapes):
        key = tuple(tuple(int(x) for x in s) for s in shapes)
        pair = self._sets.get(key)
        if pair is None:
            pair = [[torch.empty(tuple(s), device=self._device, dtype=torch.float16)
                     for s in shapes] for _ in range(2)]
            self._sets[key] = pair
        buf = pair[self._i]
        self._i ^= 1
        return buf

    def clear(self):
        self._sets.clear()
        self._i = 0

class TRTDecoder:
    def __init__(self, taehv, cache_dir, log=print):
        self.dec = taehv
        self.cache = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.log = log
        self.layers = [i for i, b in enumerate(taehv.decoder)
                       if type(b).__name__ == "MemBlock"]
        self.sessions = {}
        self.mem = None            # 9 个 past，fp16，留在显存
        self._pool = _PastPool()   # past 输出的 A/B 双缓冲（省每帧 9 次 torch.empty）
        self.off = False
        self.t_build = 0.0
        self.before_build = None      # 建引擎前腾显存（engine 里注入：把 DiT 挪到 CPU）
        self.after_build = None       # 建完搬回来

    # ---------------- 内部
    def _shape_probe(self, h, w):
        """按真实数据流记录每个 MemBlock 的输入形状（past 形状）。"""
        ch = self.dec.decoder[1].in_channels
        dt = next(self.dec.parameters()).dtype        # 与当前权重同 dtype，否则 conv 类型不匹配
        x = torch.zeros(1, ch, h, w, device="cuda", dtype=dt)
        shapes, queue = {}, [(x, 0)]
        dec = self.dec.decoder
        while queue:
            xt, i = queue.pop(0)
            if i == len(dec):
                continue
            b = dec[i]
            name = type(b).__name__
            if name == "MemBlock":
                shapes.setdefault(i, tuple(xt.shape))
                xt = b(xt, torch.zeros_like(xt))
                queue.insert(0, (xt, i + 1))
            elif name == "TGrow":
                xt = b(xt)
                _nt, c_, h_, w_ = xt.shape
                for sub in reversed(xt.view(1, b.stride * c_, h_, w_).chunk(b.stride, 1)):
                    queue.insert(0, (sub, i + 1))
            else:
                xt = b(xt)
                queue.insert(0, (xt, i + 1))
        return shapes

    def _session(self, h, w):
        key = (h, w)
        if key in self.sessions:
            return self.sessions[key]

        import onnxruntime as ort
        # 建引擎 + 建执行上下文都要占显存：整个过程先把常驻的 DiT 挪走
        freed = self.before_build() if self.before_build else False
        try:
            return self._session_locked(h, w, ort, key)
        finally:
            if self.after_build:
                self.after_build(freed)

    def _session_locked(self, h, w, ort, key):
        # 管线是懒加载的：预热时解码器可能还在 CPU，先搬上卡再做形状探测
        if next(self.dec.parameters()).device.type != "cuda":
            self.dec.to("cuda")
        onnx_path = os.path.join(self.cache, f"decstep_{h}x{w}.onnx")
        shapes = self._shape_probe(h, w)
        layers = sorted(shapes)
        in_ch = self.dec.decoder[1].in_channels
        out_frames = 4
        out_names = [f"frame{i}" for i in range(out_frames)] + \
                    [f"past_out{i}" for i in range(len(layers))]

        if not os.path.isfile(onnx_path):
            was_half = next(self.dec.parameters()).dtype == torch.float16
            self.dec.to(torch.float16)
            step = _Step(self.dec.decoder, layers).eval()
            x = torch.zeros(1, in_ch, h, w, device="cuda", dtype=torch.float16)
            zeros = [torch.zeros(shapes[L], device="cuda", dtype=torch.float16) for L in layers]
            with torch.no_grad():
                torch.onnx.export(step, (x, *zeros), onnx_path,
                                  input_names=["x"] + [f"past{i}" for i in range(len(layers))],
                                  output_names=out_names,
                                  opset_version=17, do_constant_folding=True, dynamo=False)
            if not was_half:
                self.dec.to(torch.bfloat16)
            del step, zeros
            torch.cuda.empty_cache()

        so = ort.SessionOptions()
        so.log_severity_level = 3
        t0 = time.perf_counter()
        sess = ort.InferenceSession(onnx_path, so, providers=[("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": os.path.join(self.cache, "engines"),
            "trt_max_workspace_size": int(os.environ.get("FLASHVSR_TRT_WS", "3072")) << 20,
            "trt_builder_optimization_level": 3,
        })])
        self.t_build = time.perf_counter() - t0
        outs = dict(
            frames=[(1, 3, h * 8, w * 8)] * out_frames,
            pasts=[shapes[L] for L in layers],
        )
        self.sessions[key] = (sess, outs, out_names)
        self.log(f"TRT 解码器就绪 {h}x{w}（{self.t_build:.0f}s，{len(layers)} 个 past）providers={sess.get_providers()}")
        return self.sessions[key]

    def prewarm(self, h, w):
        """在显存最空闲的时刻（模型加载完、DiT 还没上卡）把引擎建好/加载好。"""
        self._session(h, w)
        return self.sessions[(h, w)][0]

    # ---------------- 对外
    def clean_mem(self):
        self.mem = None
        self._pool.clear()          # 窗口之间把双缓冲也还回去（否则常驻 2×9 个 past）

    def wrap(self, taehv):
        """接管 decode_video / clean_mem，保留原实现用于回退。"""
        self._orig = taehv.decode_video
        self._orig_clean = taehv.clean_mem

        def clean_mem():
            self.clean_mem()
            self._orig_clean()          # 回退路径也要清 torch 的 mem

        taehv.decode_video = self.decode_video
        taehv.clean_mem = clean_mem

    def decode_video(self, x, parallel=True, show_progress_bar=False, cond=None):
        """签名与 TAEHV.decode_video 一致；不满足条件就回退 torch。"""
        if self.off or parallel or cond is None:
            return self._fallback(x, parallel, show_progress_bar, cond)

        n, t, c, hh, ww = x.shape
        try:
            sess, outs, out_names = self._session(hh, ww)
        except Exception as e:  # noqa
            free, total = torch.cuda.mem_get_info()
            self.log(f"TRT 建引擎失败，回退 torch（空闲 {free/2**30:.1f}/{total/2**30:.1f} GB）："
                     f"{type(e).__name__}: {str(e)[:300]}")
            self.off = True
            return self._fallback(x, parallel, show_progress_bar, cond)

        trim_flag = self.mem is None
        try:
            with torch.no_grad():
                cnd = self.dec.pixel_shuffle(cond).half()        # (N, T, 768, h, w)
                lat = x.half()
                if self.mem is None:
                    self.mem = [torch.zeros(s, device="cuda", dtype=torch.float16)
                                for s in outs["pasts"]]
                frames = []
                for k in range(t):
                    inp = torch.cat([cnd[:, k], lat[:, k]], dim=1).contiguous()
                    b = sess.io_binding()
                    b.bind_input("x", "cuda", 0, "float16", tuple(inp.shape), inp.data_ptr())
                    for i, m in enumerate(self.mem):
                        b.bind_input(f"past{i}", "cuda", 0, "float16",
                                     tuple(m.shape), m.data_ptr())
                    nf = len(outs["frames"])
                    # 帧输出必须各自独立（最后要 stack 成一整段），所以照旧每帧分配；
                    # past 输出走 A/B 池：形状固定，交替复用两组，省掉每帧 9 次 empty。
                    new_frames = []
                    for i in range(nf):
                        shape = outs["frames"][i]
                        buf = torch.empty(shape, device="cuda", dtype=torch.float16)
                        b.bind_output(out_names[i], "cuda", 0, "float16", shape, buf.data_ptr())
                        new_frames.append(buf)
                    new_mem = self._pool.get(outs["pasts"])
                    for i, buf in enumerate(new_mem):
                        b.bind_output(out_names[nf + i], "cuda", 0, "float16",
                                      tuple(buf.shape), buf.data_ptr())
                    sess.run_with_iobinding(b)
                    frames.extend(new_frames)
                    self.mem = new_mem
                y = torch.stack([f.reshape(1, 3, f.shape[-2], f.shape[-1]) for f in frames], dim=1)
        except Exception as e:  # noqa
            self.log(f"TRT 推理失败，回退 torch：{type(e).__name__}: {str(e)[:160]}")
            # 中途异常时 self.mem 可能已经指向半算的 past；现在靠 off=True 永久回退还不会
            # 用错帧，但只要将来允许「单帧失败不关 TRT」，残留的 mem 会让下一窗漏掉首帧 trim。
            self.mem = None
            self.off = True
            return self._fallback(x, parallel, show_progress_bar, cond)

        y = y.to(torch.bfloat16)
        if trim_flag:
            return y[:, self.dec.frames_to_trim:]
        return y

    def _fallback(self, x, parallel, show_progress_bar, cond):
        return self._orig(x, parallel=parallel, show_progress_bar=show_progress_bar, cond=cond)


class _Stage(torch.nn.Module):
    """解码器的一段（不含 TGrow）：入 1 张特征 → 出 1 张 + 该段所有 MemBlock 的新 past。"""

    def __init__(self, decoder, start, end, mem_layers):
        super().__init__()
        self.decoder = decoder
        self.start = int(start)
        self.end = int(end)
        self.mem_layers = list(mem_layers)

    def forward(self, x, *pasts):
        mem = {L: p for L, p in zip(self.mem_layers, pasts)}
        xt = x
        for i in range(self.start, self.end):
            b = self.decoder[i]
            if type(b).__name__ == "MemBlock":
                xt_new = b(xt, mem[i])
                mem[i] = xt
                xt = xt_new
            else:
                xt = b(xt)
        return (xt,) + tuple(mem[L] for L in self.mem_layers)


class StageTRTDecoder(TRTDecoder):
    """【实验性，实测失败，勿用】按 TGrow 切成多段小引擎。

    2026-11-16 实测（640×360 / 25 帧 / tile128）：
    - 耗时 56.5 s vs torch 52.4 s（**更慢**：每帧 8 次调用，调度开销吃掉收益）
    - 数值不对（PSNR 16 dB / SSIM 0.18，状态链或子帧顺序有 bug 未修）
    - 4 个引擎常驻 4.2 GB，比整图单引擎的 3.2 GB 还多
    保留代码仅作为「此路不通」的记录。
    """

    def _walk(self, h, w, dt):
        """走一遍真实数据流，返回 (层→输入形状, TGrow 索引)。"""
        dec = self.dec.decoder
        in_ch = dec[1].in_channels
        x = torch.zeros(1, in_ch, h, w, device="cuda", dtype=dt)
        ins, queue = {}, [(x, 0)]
        with torch.no_grad():
            while queue:
                xt, i = queue.pop(0)
                if i >= len(dec):
                    continue
                b = dec[i]
                nm = type(b).__name__
                ins.setdefault(i, tuple(xt.shape))
                if nm == "MemBlock":
                    xt = b(xt, torch.zeros_like(xt))
                    queue.insert(0, (xt, i + 1))
                elif nm == "TGrow":
                    xt = b(xt)
                    _nt, c_, hh, ww = xt.shape
                    for sub in reversed(xt.view(1, b.stride * c_, hh, ww).chunk(b.stride, 1)):
                        queue.insert(0, (sub, i + 1))
                else:
                    xt = b(xt)
                    queue.insert(0, (xt, i + 1))
        tgs = [i for i, b in enumerate(dec) if type(b).__name__ == "TGrow"]
        return ins, tgs

    def prewarm(self, h, w):
        self._plan(h, w)
        return True

    def _plan(self, h, w):
        key = (h, w)
        if key in self.sessions:
            return self.sessions[key]

        import onnxruntime as ort
        freed = self.before_build() if self.before_build else False
        try:
            dt = torch.float16
            was = next(self.dec.parameters()).dtype
            if next(self.dec.parameters()).device.type != "cuda":
                self.dec.to("cuda")
            ins, tgs = self._walk(h, w, was)

            starts, s = [], 0
            for t in tgs:
                starts.append((s, t))
                s = t + 1
            starts.append((s, len(self.dec.decoder)))

            self.dec.to(dt)
            stages = []
            for (a, b) in starts:
                mems = [i for i in range(a, b)
                        if type(self.dec.decoder[i]).__name__ == "MemBlock"]
                in_shape = ins[a]
                past_shapes = [ins[L] for L in mems]
                onnx_path = os.path.join(self.cache,
                                         f"stage_{a}_{b}_{in_shape[2]}x{in_shape[3]}.onnx")
                if not os.path.isfile(onnx_path):
                    mod = _Stage(self.dec.decoder, a, b, mems).eval()
                    xx = torch.zeros(in_shape, device="cuda", dtype=dt)
                    zeros = [torch.zeros(ps, device="cuda", dtype=dt) for ps in past_shapes]
                    with torch.no_grad():
                        torch.onnx.export(mod, (xx, *zeros), onnx_path,
                                          input_names=["x"] + [f"past{i}" for i in range(len(mems))],
                                          output_names=["y"] + [f"p{i}" for i in range(len(mems))],
                                          opset_version=17, do_constant_folding=True, dynamo=False)
                    del mod, zeros, xx
                    torch.cuda.empty_cache()
                so = ort.SessionOptions()
                so.log_severity_level = 3
                ws = int(os.environ.get("FLASHVSR_TRT_WS_STAGE", "1024")) << 20
                sess = ort.InferenceSession(onnx_path, so, providers=[("TensorrtExecutionProvider", {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": os.path.join(self.cache, "engines_stage"),
                    "trt_max_workspace_size": ws,
                    "trt_builder_optimization_level": 3,
                })])
                stages.append(dict(sess=sess, mems=mems, past_shapes=past_shapes,
                                   out_shape=tuple(sess.get_outputs()[0].shape)))
                self.log(f"段引擎 {a}..{b} 就绪（{in_shape[2]}x{in_shape[3]}，{len(mems)} past）")
            if was != dt:
                self.dec.to(was)
            self.sessions[key] = (stages, tgs)
            return self.sessions[key]
        finally:
            if self.after_build:
                self.after_build(freed)

    def _run_stage(self, st, x, mem):
        b = st["sess"].io_binding()
        xin = x.half().contiguous()
        b.bind_input("x", "cuda", 0, "float16", tuple(xin.shape), xin.data_ptr())
        for i, m in enumerate(mem):
            b.bind_input(f"past{i}", "cuda", 0, "float16", tuple(m.shape), m.data_ptr())
        y = torch.empty(st["out_shape"], device="cuda", dtype=torch.float16)
        b.bind_output("y", "cuda", 0, "float16", st["out_shape"], y.data_ptr())
        new_mem = []
        for i, ps in enumerate(st["past_shapes"]):
            buf = torch.empty(ps, device="cuda", dtype=torch.float16)
            b.bind_output(f"p{i}", "cuda", 0, "float16", ps, buf.data_ptr())
            new_mem.append(buf)
        st["sess"].run_with_iobinding(b)
        return y, new_mem

    def decode_video(self, x, parallel=True, show_progress_bar=False, cond=None):
        if self.off or parallel or cond is None:
            return self._fallback(x, parallel, show_progress_bar, cond)
        n, t, c, hh, ww = x.shape
        first = self.mem is None
        try:
            stages, tgs = self._plan(hh, ww)
            if first:
                self.mem = [[torch.zeros(ps, device="cuda", dtype=torch.float16)
                             for ps in st["past_shapes"]] for st in stages]
            with torch.no_grad():
                cnd = self.dec.pixel_shuffle(cond).half()
                lat = x.half()
                frames = []
                for k in range(t):
                    cur = torch.cat([cnd[:, k], lat[:, k]], dim=1).contiguous()
                    queue, outs = [(cur, 0)], []
                    while queue:
                        xt, si = queue.pop(0)
                        if si == len(stages):
                            outs.append(xt)
                            continue
                        y, self.mem[si] = self._run_stage(stages[si], xt, self.mem[si])
                        if si < len(tgs):
                            tg = self.dec.decoder[tgs[si]]
                            # TGrow 是 torch 模块（bf16 权重），TRT 输出是 fp16 → 过一次 bf16
                            nt = tg(y.to(torch.bfloat16))
                            _nt, c_, h_, w_ = nt.shape
                            subs = list(nt.view(1, tg.stride * c_, h_, w_).chunk(tg.stride, 1))
                            for s_ in reversed(subs):
                                queue.insert(0, (s_.half(), si + 1))
                        else:
                            queue.insert(0, (y, si + 1))
                    frames.extend(outs)
                out = torch.stack(frames, dim=1).to(torch.bfloat16)
        except Exception as e:  # noqa
            free, total = torch.cuda.mem_get_info()
            self.log(f"分段 TRT 失败，回退 torch（空闲 {free/2**30:.1f}/{total/2**30:.1f} GB）："
                     f"{type(e).__name__}: {str(e)[:260]}")
            self.mem = None     # 同上：别把半算的 past 留到下一次
            self.off = True
            return self._fallback(x, parallel, show_progress_bar, cond)
        if first:
            return out[:, self.dec.frames_to_trim:]
        return out


class CompiledDecoder:
    def __init__(self, taehv, log=print, mode="default"):
        self.dec = taehv
        self.log = log
        self.mode = mode
        self.layers = [i for i, b in enumerate(taehv.decoder)
                       if type(b).__name__ == "MemBlock"]
        self.steps = {}
        self.mem = None
        self.off = False

    def _probe(self, h, w, dt):
        ch = self.dec.decoder[1].in_channels
        x = torch.zeros(1, ch, h, w, device="cuda", dtype=dt)
        shapes, queue = {}, [(x, 0)]
        dec = self.dec.decoder
        with torch.no_grad():
            while queue:
                xt, i = queue.pop(0)
                if i == len(dec):
                    continue
                b = dec[i]
                name = type(b).__name__
                if name == "MemBlock":
                    shapes.setdefault(i, tuple(xt.shape))
                    xt = b(xt, torch.zeros_like(xt))
                    queue.insert(0, (xt, i + 1))
                elif name == "TGrow":
                    xt = b(xt)
                    _nt, c_, h_, w_ = xt.shape
                    for sub in reversed(xt.view(1, b.stride * c_, h_, w_).chunk(b.stride, 1)):
                        queue.insert(0, (sub, i + 1))
                else:
                    xt = b(xt)
                    queue.insert(0, (xt, i + 1))
        return shapes

    def _get(self, h, w, dt):
        key = (h, w, str(dt))
        if key in self.steps:
            return self.steps[key]
        t0 = time.perf_counter()
        shapes = self._probe(h, w, dt)   # {layer_idx: shape}，是 dict 不是列表
        layers = sorted(shapes)
        fn = torch.compile(_Step(self.dec.decoder, layers).eval(),
                           mode=self.mode, dynamic=False)
        # 存**形状列表**（按层序），不是 dict：decode_video 里直接 for s in shapes 建
        # mem，遍历 dict 拿到的是 layer_idx 整数 → torch.zeros(3) 建出 1D 张量，首帧必崩。
        self.steps[key] = (fn, [shapes[L] for L in layers])
        self.log(f"torch.compile 解码器 {h}x{w}（编译 {time.perf_counter()-t0:.1f}s）")
        return self.steps[key]

    def clean_mem(self):
        self.mem = None

    def wrap(self, taehv):
        self._orig = taehv.decode_video
        self._orig_clean = taehv.clean_mem

        def clean_mem():
            self.clean_mem()
            self._orig_clean()

        taehv.decode_video = self.decode_video
        taehv.clean_mem = clean_mem

    def decode_video(self, x, parallel=True, show_progress_bar=False, cond=None):
        if self.off or parallel or cond is None:
            return self._orig(x, parallel=parallel, show_progress_bar=show_progress_bar, cond=cond)
        n, t, c, hh, ww = x.shape
        trim = self.mem is None
        try:
            fn, shapes = self._get(hh, ww, x.dtype)
            with torch.no_grad():
                cnd = self.dec.pixel_shuffle(cond)
                if self.mem is None:
                    self.mem = [torch.zeros(s, device=x.device, dtype=x.dtype) for s in shapes]
                outs = []
                for k in range(t):
                    r = fn(torch.cat([cnd[:, k], x[:, k]], dim=1), *self.mem)
                    outs.extend(r[0])
                    self.mem = list(r[1:])
                y = torch.stack(outs, dim=1)
        except Exception as e:  # noqa
            self.log(f"compile 解码失败，回退 torch：{type(e).__name__}: {str(e)[:160]}")
            self.off = True
            return self._orig(x, parallel=parallel, show_progress_bar=show_progress_bar, cond=cond)
        if trim:
            return y[:, self.dec.frames_to_trim:]
        return y


def maybe_enable(taehv, cache_dir, log=print):
    """把 TAEHV 实例的解码换成加速版；不可用时什么都不做。

    后端选择：FLASHVSR_DEC_BACKEND=compile（默认）| trt | torch
    - compile：torch.compile 融合，1.16×，不占额外显存
    - trt：TensorRT，1.8×，但要 3 GB workspace（8 GB 卡上 4× 尺寸装不下）
    """
    backend = (os.environ.get("FLASHVSR_DEC_BACKEND") or "torch").lower()
    if backend in ("0", "off", "false", "torch", ""):
        return None
    try:
        if backend in ("trt", "stage"):
            if not trt_available():
                log("TRT 不可用（缺 onnxruntime-gpu / TensorRT 库），回退 torch")
                return None
            dec = (StageTRTDecoder if backend == "stage" else TRTDecoder)(taehv, cache_dir, log=log)
        else:
            dec = CompiledDecoder(taehv, log=log)
        dec.wrap(taehv)
        return dec
    except Exception as e:  # noqa
        log(f"解码加速未启用（{type(e).__name__}: {str(e)[:120]}）")
        return None
