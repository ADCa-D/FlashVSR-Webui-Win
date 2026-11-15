# -*- coding: utf-8 -*-
"""FlashVSR 推理引擎（独立运行，不经 ComfyUI 前端/服务端）

把 ComfyUI 节点包当库用：桩掉 folder_paths / comfy.utils，以合成包名导入 nodes.py，
复用 init_pipeline / flashvsr。这一层补：ffmpeg 编解码、进度回调、长视频分段落盘+续跑、
运动感知预算、DiT 权重量化（MXFP8 / NVFP4）。
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import types

# 必须在 import torch 之前（Windows 不支持 expandable_segments）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as _cfg  # noqa: E402

BASE = _cfg.BASE
NODE_DIR = _cfg.NODE_DIR
MODELS_DIR = _cfg.MODELS_DIR
FFMPEG = _cfg.FFMPEG
FFPROBE = _cfg.FFPROBE

MAX_PIXELS = 700_000_000

ENCODER_CHOICES = ("auto", "cpu", "nvenc", "amf")
ENCODER_LABEL = {
    "auto": "自动（CPU 优先，GPU 不可用则回退）",
    "cpu": "CPU · libx264",
    "nvenc": "GPU · NVENC（NVIDIA 独显）",
    "amf": "GPU · AMF（AMD 核显）",
}
QUANT_CHOICES = ("none", "nvfp4", "mxfp8")


def encoder_args(backend, quality):
    q = str(int(quality))
    if backend == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq",
                "-rc", "vbr", "-cq", q, "-b:v", "0", "-pix_fmt", "yuv420p"]
    if backend == "amf":
        return ["-c:v", "h264_amf", "-quality", "high_quality",
                "-rc", "hqvbr", "-qvbr_quality_level", q, "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", q, "-pix_fmt", "yuv420p"]


def probe_encoders():
    src = ["-f", "lavfi", "-i", "color=c=black:s=256x256:r=5:d=0.2", "-frames:v", "1"]
    out = {}
    for be in ("cpu", "nvenc", "amf"):
        try:
            r = subprocess.run(
                [FFMPEG, "-hide_banner", "-v", "error"] + src + encoder_args(be, 18) + ["-f", "null", "-"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
            if r.returncode == 0:
                out[be] = dict(ok=True, detail="")
            else:
                # 从 stderr 里挑出真正有信息量的那行（编码器初始化失败比 "no packets" 有用）
                lines = [l.strip() for l in (r.stderr or "").splitlines() if l.strip()]
                key = [l for l in lines if any(k in l.lower() for k in
                                               ("error", "invalid", "not support", "failed", "cannot", "unable", "api"))]
                pick = (key or lines or ["未知错误"])
                out[be] = dict(ok=False, detail=" / ".join(pick[:2])[:220])
        except Exception as e:  # noqa
            out[be] = dict(ok=False, detail=f"{type(e).__name__}: {e}")
    if out["cpu"]["ok"]:
        out["auto"] = dict(ok=True, detail="CPU 优先")
    elif out.get("nvenc", {}).get("ok"):
        out["auto"] = dict(ok=True, detail="回退 NVENC")
    elif out.get("amf", {}).get("ok"):
        out["auto"] = dict(ok=True, detail="回退 AMF")
    else:
        out["auto"] = dict(ok=False, detail="没有可用的 H.264 编码器")
    return out


_CAPS = None


def get_caps(force=False):
    global _CAPS
    if _CAPS is None or force:
        _CAPS = probe_encoders()
    return _CAPS


def resolve_encoder(requested, caps=None):
    caps = caps or get_caps()
    be = requested if requested in ENCODER_CHOICES else "auto"
    if be != "auto" and caps.get(be, {}).get("ok"):
        return be
    if caps.get("cpu", {}).get("ok"):
        return "cpu"
    for alt in ("nvenc", "amf"):
        if caps.get(alt, {}).get("ok"):
            return alt
    return "cpu"


def _set_cudnn_benchmark(on: bool):
    try:
        torch.backends.cudnn.benchmark = bool(on)
    except Exception:
        pass


# ---------------------------------------------------------------- 权重量化
def _should_quant(name: str) -> bool:
    """只量化主干（blocks.* / LQ_proj_in.*）。time_embedding / head 等层 token 极少（M 甚至 =1），
    满足不了 fp4/fp8 的行对齐，而且耗时占比接近 0。"""
    return name.startswith("blocks.") or name.startswith("LQ_proj_in.")


def dit_to_cuda(pipe):
    """init_pipeline() 结尾会 offload 到 CPU；量化前必须搬上来，否则所有层被静默跳过。"""
    try:
        pipe.load_models_to_device(["dit"])
    except Exception:
        pass
    try:
        if next(pipe.dit.parameters()).device.type != "cuda":
            pipe.dit.to("cuda")
    except StopIteration:
        pass
    return pipe


def _iter_linears(model):
    for name, m in list(model.named_modules()):
        if isinstance(m, torch.nn.Linear):
            yield name, m


def _take_weight(m):
    w = m.weight
    if w is None or w.numel() == 0 or w.device.type != "cuda":
        return None
    return w.detach().contiguous()


def _drop_weight(m):
    m._parameters.pop("weight", None)
    m.register_parameter("weight", None)


def quantize_dit_nvfp4(pipe, log=None):
    """W4A4（NVFP4）。Blackwell 原生 FP4；单层微基准 FFN up 3.78×，误差 0.134。
    注意：激活/权重量化都要 pad_16x=True，否则 CUDA 内核越界（0xC0000005）。"""
    import comfy_kitchen as ck

    def _ts(t):
        return (t.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1)

    def _call(x, wq, wb, tw, bias):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).contiguous()
        m = x2.shape[0]
        tx = _ts(x2)
        xq, xb = ck.quantize_nvfp4(x2, tx, pad_16x=True)
        y = ck.scaled_mm_nvfp4(xq, wq, tx, tw, xb, wb, bias=bias, out_dtype=x.dtype)
        return y[:m].reshape(*shp[:-1], y.shape[-1])

    n = skipped = 0
    for name, m in _iter_linears(pipe.dit):
        if not _should_quant(name):
            skipped += 1
            continue
        w = _take_weight(m)
        if w is None or w.shape[1] % 16 != 0:
            skipped += 1
            continue
        try:
            tw = _ts(w)
            wq, wb = ck.quantize_nvfp4(w, tw, pad_16x=True)
        except Exception as e:  # noqa
            if log:
                log(f"nvfp4 量化失败 {name}: {e}")
            skipped += 1
            continue
        bias = None if m.bias is None else m.bias.detach()
        m.forward = (lambda x, _q=wq, _s=wb, _t=tw, _b=bias: _call(x, _q, _s, _t, _b))
        _drop_weight(m)
        n += 1
    torch.cuda.empty_cache()
    if log:
        log(f"NVFP4(pad16): {n} 个 Linear 已量化，跳过 {skipped}")
    return n


def quantize_dit_mxfp8(pipe, log=None):
    """MXFP8（e4m3 + 每 32 元素 e8m0）。Ada/Hopper/Blackwell 都有原生 FP8 张量核。

    实测：单段 77.0→65.4 s（−15%），峰值 6.19→4.93 GiB，PSNR 46.49 dB。
    """
    import comfy_kitchen as ck

    def _call(x, wq, wb, bias):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).contiguous()
        xq, xb = ck.quantize_mxfp8(x2)
        y = ck.scaled_mm_mxfp8(xq, wq, xb, wb, bias=bias, out_dtype=x.dtype)
        return y.reshape(*shp[:-1], y.shape[-1])

    n = skipped = 0
    for name, m in _iter_linears(pipe.dit):
        if not _should_quant(name):
            skipped += 1
            continue
        w = _take_weight(m)
        if w is None or w.shape[1] % 32 != 0:
            skipped += 1
            continue
        try:
            wq, wb = ck.quantize_mxfp8(w)
        except Exception as e:  # noqa
            if log:
                log(f"mxfp8 量化失败 {name}: {e}")
            skipped += 1
            continue
        bias = None if m.bias is None else m.bias.detach()
        m.forward = (lambda x, _q=wq, _s=wb, _b=bias: _call(x, _q, _s, _b))
        _drop_weight(m)
        n += 1
    torch.cuda.empty_cache()
    if log:
        log(f"MXFP8: {n} 个 Linear 已量化，跳过 {skipped}")
    return n


# ---------------------------------------------------------------- 指纹 / manifest
def quick_hash(path, chunk=1 << 20):
    h = hashlib.sha1()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > 2 * chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


def _param_key(params):
    keys = ("scale", "tile", "kv", "chunk", "overlap", "mode", "attention", "quant", "pre_scale")
    return hashlib.sha1(json.dumps({k: params.get(k) for k in keys}, sort_keys=True).encode()).hexdigest()[:8]


def seg_root_dir(params):
    return params.get("seg_dir") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "work", "segments")


def _manifest_path(seg_dir):
    return os.path.join(seg_dir, "manifest.json")


def load_manifest(seg_dir):
    try:
        with open(_manifest_path(seg_dir), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_manifest(seg_dir, data):
    p = _manifest_path(seg_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def probe_frames(path):
    try:
        out = subprocess.check_output(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path],
            stderr=subprocess.DEVNULL).decode().strip()
        if out and out.isdigit() and int(out) > 0:
            return int(out)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
            stderr=subprocess.DEVNULL).decode().strip()
        return int(out) if out.isdigit() else -1
    except Exception:
        return -1


# ---------------------------------------------------------------- 桩 + 导入
def _install_stubs():
    fp = types.ModuleType("folder_paths")
    fp.models_dir = MODELS_DIR
    fp.get_filename_list = lambda *a, **k: []
    fp.get_full_path_or_raise = lambda *a, **k: None
    sys.modules["folder_paths"] = fp

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total=0):
            self.total = total

        def update(self, n=1):
            pass

    utils.ProgressBar = ProgressBar
    comfy.utils = utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = utils


def load_node_module():
    if "flashvsr_node.nodes" in sys.modules:
        return sys.modules["flashvsr_node.nodes"]
    _install_stubs()
    pkg = types.ModuleType("flashvsr_node")
    pkg.__path__ = [NODE_DIR]
    sys.modules["flashvsr_node"] = pkg
    spec = importlib.util.spec_from_file_location(
        "flashvsr_node.nodes", os.path.join(NODE_DIR, "nodes.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 视频 IO
def probe(path):
    out = subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
         "-show_entries", "format=duration", "-of", "json", path])
    j = json.loads(out.decode("utf-8", "ignore"))
    st = (j.get("streams") or [{}])[0]
    w, h = int(st.get("width", 0)), int(st.get("height", 0))
    fr = st.get("avg_frame_rate") or st.get("r_frame_rate") or "25/1"
    try:
        num, den = fr.split("/")
        fps = (float(num) / float(den)) if float(den) else 25.0
    except Exception:
        fps = 25.0
    dur = float((j.get("format") or {}).get("duration") or 0)
    has_audio = bool(json.loads(subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "json", path]).decode("utf-8", "ignore")).get("streams"))
    return dict(w=w, h=h, fps=fps, duration=dur, has_audio=has_audio)


def decode_frames(path, on_progress=None, pre_scale=1.0):
    """解成 uint8 NHWC。pre_scale<1 用 libplacebo(spline36) 在核显上降采样。"""
    info = probe(path)
    w, h = info["w"], info["h"]
    if w <= 0 or h <= 0:
        raise RuntimeError("读不到视频尺寸，文件可能不是视频")
    vf = None
    if abs(pre_scale - 1.0) > 1e-6:
        tw = max(2, int(round(w * pre_scale / 2)) * 2)
        th = max(2, int(round(h * pre_scale / 2)) * 2)
        vf = f"libplacebo=w={tw}:h={th}:downscaler=spline36,hwdownload,format=yuv420p"
        info["w"], info["h"] = tw, th
        w, h = tw, th
    cmd = [FFMPEG, "-v", "error"]
    if vf:
        cmd += ["-init_hw_device", "vulkan=pl", "-filter_hw_device", "pl"]
    cmd += ["-i", path]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = w * h * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunks, n = [], 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if not buf or len(buf) < frame_bytes:
            break
        chunks.append(buf)
        n += 1
        if on_progress and n % 10 == 0:
            on_progress(n)
        if n * w * h > MAX_PIXELS:
            proc.kill()
            raise RuntimeError(f"帧太多（>{MAX_PIXELS // (w * h)} 帧 @ {w}x{h}），请先切段再跑")
    proc.wait()
    if not chunks:
        raise RuntimeError("没解出任何帧")
    return np.frombuffer(b"".join(chunks), np.uint8).reshape(-1, h, w, 3), info


def encode_frames(frames_u8, fps, out_path, audio_src=None, quality=15, backend="cpu"):
    h, w = frames_u8.shape[1], frames_u8.shape[2]
    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", f"{fps:.6f}", "-i", "-"]
    if audio_src:
        cmd += ["-i", audio_src, "-map", "0:v:0", "-map", "1:a?", "-c:a", "copy", "-shortest"]
    cmd += encoder_args(backend, quality) + ["-fps_mode", "cfr", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(0, len(frames_u8), 8):
        proc.stdin.write(frames_u8[i:i + 8].tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", "ignore")
    if proc.wait() != 0:
        raise RuntimeError(f"{backend} 编码失败：" + err.strip()[-400:])


def concat_segments(seg_paths, out_path, audio_src=None, backend="cpu", quality=15):
    if not seg_paths:
        raise RuntimeError("没有可拼接的分段")
    tmp_dir = os.path.dirname(seg_paths[0])
    list_path = os.path.join(tmp_dir, "segments.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in seg_paths:
            f.write("file '%s'\n" % p.replace("\\", "/").replace("'", "'\\''"))

    silent = os.path.join(tmp_dir, "concat_video.mp4")
    base = [FFMPEG, "-y", "-v", "error", "-fflags", "+genpts",
            "-f", "concat", "-safe", "0", "-i", list_path]
    r = subprocess.run(base + ["-c", "copy", silent],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not os.path.exists(silent) or os.path.getsize(silent) < 1024:
        r = subprocess.run(base + encoder_args(backend, quality) + [silent],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError("分段拼接失败：" + (r.stderr or "")[-400:])

    if audio_src:
        r = subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-i", silent, "-i", audio_src,
             "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-shortest", out_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            shutil.move(silent, out_path)
        else:
            os.remove(silent)
    else:
        shutil.move(silent, out_path)


# ---------------------------------------------------------------- 引擎
PRESETS = {
    "quality": dict(scale=4, tile=128, kv=3.0, label="画质优先"),
    "balance": dict(scale=4, tile=160, kv=2.0, label="平衡"),
    "fast": dict(scale=2, tile=256, kv=3.0, label="快速"),
}


class Engine:
    def __init__(self):
        self.mod = None
        self.pipe = None
        self.loaded_key = None
        self._orig_attn = None

    def ensure_loaded(self, mode="tiny", attention="sparse", on_stage=None, quant="none"):
        key = (mode, attention, quant)
        if self.pipe is not None and self.loaded_key == key:
            return
        if on_stage:
            on_stage("load", 0, 1, f"加载模型（mode={mode}, attn={attention}, quant={quant}）")
        _set_cudnn_benchmark(True)
        self.mod = load_node_module()
        wan = sys.modules["flashvsr_node.src.models.wan_video_dit"]

        if attention == "dense":
            if self._orig_attn is None:
                self._orig_attn = wan.flash_attention

            def patched(q, k, v, num_heads, compatibility_mode=False, attention_mask=None, return_KV=False):
                return self._orig_attn(q, k, v, num_heads, True, None, return_KV)

            wan.flash_attention = patched
        elif self._orig_attn is not None:
            wan.flash_attention = self._orig_attn

        if self.pipe is not None:
            self.pipe = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        self.pipe = self.mod.init_pipeline("FlashVSR-v1.1", mode, "cuda:0", torch.bfloat16)

        import gc as _gc
        state = {"n": 0}

        def _fast_clean_vram():
            state["n"] += 1
            if state["n"] % 8 == 0:
                _gc.collect()
            torch.cuda.empty_cache()

        self.mod.clean_vram = _fast_clean_vram

        # VAE 解码走 NHWC：实测 625 → 540 ms（−13.6%），输出逐位一致（maxdiff 0）
        try:
            dec = self.pipe.TCDecoder
            dec.to(memory_format=torch.channels_last)
            _orig_decode = dec.decode_video

            def _decode_cl(x, parallel=True, show_progress_bar=False, cond=None):
                x = x.contiguous(memory_format=torch.channels_last_3d)
                if cond is not None:
                    cond = cond.contiguous(memory_format=torch.channels_last_3d)
                return _orig_decode(x, parallel=parallel, show_progress_bar=show_progress_bar, cond=cond)

            dec.decode_video = _decode_cl
        except Exception as e:  # noqa
            print(f"[FlashVSR] channels_last 未启用: {e}", flush=True)

        if quant in ("nvfp4", "mxfp8"):
            dit_to_cuda(self.pipe)
            if on_stage:
                on_stage("load", 0, 1, f"量化 DiT 权重（{quant.upper()}）…")
            fn = quantize_dit_nvfp4 if quant == "nvfp4" else quantize_dit_mxfp8
            print(f"[FlashVSR] {fn(self.pipe, log=print)} 个 Linear 已量化", flush=True)

        self.loaded_key = key
        torch.cuda.empty_cache()
        if on_stage:
            on_stage("load", 1, 1, "模型就绪")

    def _run_window(self, frames_u8, scale, tile, kv, on_tile=None, sparse=2.0, local=11):
        """force_offload=False：每 tile 结束把 2.84 GB DiT 卸到 CPU 再搬回要多花 ~2.9 s/tile。"""
        mod = self.mod
        orig = mod.cqdm

        class Hook:
            def __init__(self, iterable=None, total=None, desc=""):
                self.total = total if total is not None else (len(iterable) if iterable is not None else 1)
                self.it = iter(iterable) if iterable is not None else None
                self.n = 0

            def __iter__(self):
                return self

            def __next__(self):
                v = next(self.it)
                self.n += 1
                if on_tile:
                    on_tile(self.n, self.total)
                return v

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def __len__(self):
                return self.total

        mod.cqdm = Hook
        try:
            frames = torch.from_numpy(np.ascontiguousarray(frames_u8)).float().div_(255.0)
            out = mod.flashvsr(
                self.pipe, frames, int(scale), True, True, True,
                int(tile), 24, False, float(sparse), float(kv), int(local), 0, False)
            return (out.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        finally:
            mod.cqdm = orig

    def upscale_video(self, src, dst, params, on_stage):
        frames, info = decode_frames(
            src, on_progress=lambda n: on_stage("decode", n, 0, f"解码中 {n} 帧"),
            pre_scale=float(params.get("pre_scale") or 1.0))
        total = len(frames)
        on_stage("decode", total, total, f"共 {total} 帧 {info['w']}x{info['h']} @{info['fps']:.2f}fps")
        self.ensure_loaded(params.get("mode") or "tiny-long", params.get("attention") or "sparse",
                           on_stage, quant=params.get("quant") or "none")

        chunk = max(0, int(params.get("chunk") or 0))
        overlap = max(0, min(int(params.get("overlap") or 8), max(0, chunk // 4)))
        audio = src if info["has_audio"] else None
        quality = int(params.get("quality") or 15)
        backend = resolve_encoder(params.get("encoder"))
        budget_mode = params.get("budget") or "auto"

        def budget_for(win_u8, kv):
            if budget_mode != "auto" or len(win_u8) < 2:
                return 2.0, 11, kv
            small = win_u8[:, ::8, ::8, 1].astype("int16")
            motion = float(np.abs(np.diff(small, axis=0)).mean())
            if motion <= 3.0:
                return 1.5, 9, min(kv, 2.0)
            if motion <= 9.0:
                return 2.0, 11, kv
            return 2.0, 11, max(kv, 2.5)

        def run_win(win, tile, kv, cb=None):
            sp, lr, kk = budget_for(win, kv)
            res = self._run_window(win, params["scale"], tile, kk, cb, sparse=sp, local=lr)
            return res, sp, lr, kk

        if not chunk or total <= chunk:
            out_t, sp, lr, kk = run_win(
                frames, params["tile"], params["kv"],
                lambda c, t: on_stage("infer", c, t, f"tile {c}/{t}"))
            frames_out = out_t[:total]
            on_stage("encode", 0, 1, f"编码中（sparse={sp} kv={kk} local={lr}）")
            encode_frames(frames_out, info["fps"], dst, audio, quality=quality, backend=backend)
            on_stage("encode", 1, 1, f"编码完成（{backend}）")
            return dict(frames=total, fps=info["fps"], in_wh=[info["w"], info["h"]],
                        out_wh=[int(frames_out.shape[2]), int(frames_out.shape[1])], segments=0)

        src_sig = dict(size=os.path.getsize(src), hash=quick_hash(src))
        pkey = _param_key(params)
        root = seg_root_dir(params)
        seg_dir = os.path.join(root, f"{src_sig['hash']}_{pkey}")
        if params.get("restart"):
            shutil.rmtree(seg_dir, ignore_errors=True)
        os.makedirs(seg_dir, exist_ok=True)

        man = load_manifest(seg_dir)
        expect = dict(src=src_sig, params_scale=params.get("scale"), params_tile=params.get("tile"),
                      params_kv=params.get("kv"), params_chunk=params.get("chunk"),
                      params_overlap=params.get("overlap"), params_mode=params.get("mode"),
                      params_attention=params.get("attention"), params_quant=params.get("quant"),
                      params_pre_scale=params.get("pre_scale"), total=total)
        same = bool(man) and all(man.get(k) == v for k, v in expect.items())
        if not same:
            shutil.rmtree(seg_dir, ignore_errors=True)
            os.makedirs(seg_dir, exist_ok=True)
            man = dict(expect)
            man["done"] = {}
            save_manifest(seg_dir, man)
        done = man.get("done", {})

        plans, start = [], 0
        while start < total:
            win_start = max(0, start - overlap)
            win_end = min(total, win_start + chunk)
            plans.append([start, win_start, win_end])
            start = win_end
        n_seg = len(plans)
        man["plans"] = plans
        save_manifest(seg_dir, man)

        resumed = 0
        for k in list(done.keys()):
            try:
                i = int(k)
            except ValueError:
                done.pop(k, None)
                continue
            if not (0 <= i < n_seg):
                done.pop(k, None)
                continue
            keep = plans[i][2] - plans[i][0]
            seg = os.path.join(seg_dir, f"seg_{i:04d}.mp4")
            if not os.path.exists(seg) or probe_frames(seg) != keep:
                done.pop(k, None)
            else:
                resumed += 1
        man["done"] = done
        save_manifest(seg_dir, man)
        if resumed:
            on_stage("seg", resumed, n_seg, f"检测到上次进度：{resumed}/{n_seg} 段已完成，从这里续跑")

        out_wh = None
        seg_paths = []
        for i, (start, win_start, win_end) in enumerate(plans):
            seg_path = os.path.join(seg_dir, f"seg_{i:04d}.mp4")
            seg_paths.append(seg_path)
            keep_from = start - win_start
            keep = win_end - start
            if str(i) in done:
                on_stage("seg", i + 1, n_seg, f"段 {i + 1}/{n_seg} 已完成，跳过")
                continue
            win = frames[win_start:win_end]
            on_stage("seg", i, n_seg, f"段 {i + 1}/{n_seg}：{win_start}–{win_end} 帧")

            def tile_cb(c, t, i=i):
                on_stage("seg", i, n_seg, f"段 {i + 1}/{n_seg} tile {c}/{t}")

            res, sp, lr, kk = run_win(win, params["tile"], params["kv"], tile_cb)
            out_frames = res[keep_from:keep_from + keep]
            out_wh = [int(out_frames.shape[2]), int(out_frames.shape[1])]
            on_stage("encode", 0, 1, f"段 {i + 1}/{n_seg} 写盘（{backend}）")
            encode_frames(out_frames, info["fps"], seg_path, None, quality=quality, backend=backend)
            del res, out_frames, win

            got = probe_frames(seg_path)
            if got != keep:
                raise RuntimeError(f"第 {i + 1} 段校验失败：期望 {keep} 帧，实际 {got} 帧")
            done[str(i)] = dict(frames=keep, bytes=os.path.getsize(seg_path), at=int(time.time()))
            man["done"] = done
            save_manifest(seg_dir, man)
            on_stage("seg", i + 1, n_seg, f"段 {i + 1}/{n_seg} 完成（sparse={sp} kv={kk} local={lr}）")

        on_stage("encode", 0, 1, f"拼接 {n_seg} 段 + 混入音轨")
        concat_segments(seg_paths, dst, audio_src=audio, backend=backend, quality=quality)
        got = probe_frames(dst)
        if got and got != total:
            on_stage("encode", 1, 1,
                     f"拼接完成，但输出 {got} 帧 ≠ 期望 {total} 帧（音画可能差 {(total - got) / info['fps']:.2f}s）")
        else:
            on_stage("encode", 1, 1, f"拼接完成（{backend}，{got} 帧）")
        if not params.get("keep_seg"):
            shutil.rmtree(seg_dir, ignore_errors=True)
        return dict(frames=total, fps=info["fps"], in_wh=[info["w"], info["h"]],
                    out_wh=out_wh, segments=n_seg, resumed=resumed,
                    frames_out=got if got and got > 0 else None)

    def unload(self):
        self.pipe = None
        self.mod = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
