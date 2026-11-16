# -*- coding: utf-8 -*-
"""FlashVSR 推理引擎（独立运行，不经 ComfyUI 前端/服务端）

把 ComfyUI 节点包当库用：桩掉 folder_paths / comfy.utils，以合成包名导入 nodes.py，
复用 init_pipeline / flashvsr。这一层补：ffmpeg 编解码、进度回调、长视频分段落盘+续跑、
运动感知预算、DiT 权重量化（MXFP8 / NVFP4）。
"""
import hashlib
import importlib.util
import json
import math
import os
import random
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

ENCODER_CHOICES = ("auto", "cpu", "nvenc", "qsv", "qsv_h264", "amf", "svtav1",
                    "av1_nvenc", "x265")
ENCODER_LABEL = {
    "auto": "自动（CPU 优先，GPU 不可用则回退）",
    "cpu": "CPU · libx264（H.264）",
    "nvenc": "GPU · NVENC H.264",
    "qsv": "GPU · Intel QSV · HEVC 8bit（4K，6 代+ 核显）",
    "qsv_h264": "GPU · Intel QSV · H.264 8bit（4K，兼容最老，4 代+）",
    "amf": "GPU · AMF（AMD 核显）",
    "svtav1": "CPU · SVT-AV1（同画质体积最小，慢）",
    "av1_nvenc": "GPU · NVENC AV1（40/50 系）",
    "x265": "CPU · libx265（H.265，均衡）",
}
QUANT_CHOICES = ("none", "nvfp4", "mxfp8", "mixed", "w8")


def encoder_args(backend, quality, bitdepth=8):
    q = str(int(quality))
    # 10-bit：只有 x265 在本机验证可用（x264 是 8bit 构建、SVT-AV1 在 4K 下 10bit 会 OOM）
    pix = "yuv420p10le" if (int(bitdepth) == 10 and backend == "x265") else "yuv420p"
    if backend == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq",
                "-rc", "vbr", "-cq", q, "-b:v", "0", "-pix_fmt", "yuv420p"]
    if backend == "av1_nvenc":
        return ["-c:v", "av1_nvenc", "-preset", "p7", "-tune", "hq",
                "-rc", "vbr", "-cq", q, "-b:v", "0", "-pix_fmt", "yuv420p"]
    if backend in ("qsv", "qsv_h264"):
        # Intel Quick Sync 的"通用可行"写法：**系统内存 + nv12 + ICQ**。
        # 不用 hwupload/hwframes —— 那套依赖 D3D11 纹理池，老驱动/老核显经常初始化失败；
        # 系统内存输入由驱动内部拷贝，兼容性最好（代价是多一次上传）。
        #   qsv      → hevc_qsv（8bit main；6 代 Skylake 起支持 4K 编码，9 代起 main10）
        #   qsv_h264 → h264_qsv（8bit high；4-5 代起支持 4K，兼容最老）
        preset = os.environ.get("FLASHVSR_QSV_PRESET", "medium")
        if backend == "qsv":
            if int(bitdepth) == 10:
                return ["-c:v", "hevc_qsv", "-preset", preset, "-global_quality", q,
                        "-profile:v", "main10", "-vf", "format=p010le",
                        "-pix_fmt", "p010le"]
            return ["-c:v", "hevc_qsv", "-preset", preset, "-global_quality", q,
                    "-profile:v", "main", "-vf", "format=nv12", "-pix_fmt", "nv12"]
        return ["-c:v", "h264_qsv", "-preset", preset, "-global_quality", q,
                "-profile:v", "high", "-vf", "format=nv12", "-pix_fmt", "nv12"]
    if backend == "amf":
        # AMD AMF：只有 8 bit（没有 main10），且同码率画质明显弱于 x264/x265，
        # 只当“显卡不支持 NVENC / CPU 又太慢”时的兜底；要画质请用 x265 或 svtav1。
        return ["-c:v", "h264_amf", "-quality", "high_quality",
                "-rc", "hqvbr", "-qvbr_quality_level", q, "-pix_fmt", "yuv420p"]
    if backend == "svtav1":
        # SVT-AV1：preset 越低画质越好、同 CRF 体积越小（p4 比 p10 小 ~30%）。
        # 编码时间相对推理可忽略（4K/50 帧：p4≈7s、p2≈13s，而推理要 30–90 分钟），所以默认质量档。
        # lp 必须固定 4：默认按全部逻辑核并行，4K 下会 ENOMEM（本机 32 核实测 lp=8/12 失败）。
        preset = os.environ.get("FLASHVSR_SVT_PRESET", "4")
        return ["-c:v", "libsvtav1", "-crf", q, "-preset", preset,
                "-svtav1-params", "lp=4", "-g", "240", "-pix_fmt", "yuv420p"]
    if backend == "x265":
        # 共享功耗墙：CPU 编码线程开太多会抢走 GPU 的 boost 预算，而编码本身不是瓶颈
        return ["-c:v", "libx265", "-preset", "medium", "-crf", q,
                "-threads", "4", "-pix_fmt", pix]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", q,
            "-threads", "4", "-pix_fmt", "yuv420p"]


# 走 GPU 编码的引擎：初始化失败（驱动/核显不支持该 codec）时自动回退 CPU
HW_ENCODERS = ("nvenc", "av1_nvenc", "qsv", "qsv_h264", "amf")


def probe_encoders():
    src = ["-f", "lavfi", "-i", "color=c=black:s=256x256:r=5:d=0.2", "-frames:v", "1"]
    out = {}
    for be in ("cpu", "nvenc", "qsv", "qsv_h264", "amf", "svtav1", "av1_nvenc", "x265"):
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
    for alt in ("nvenc", "qsv", "qsv_h264", "amf"):
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


def _feather_mask(size, overlap, smooth=True):
    """生成羽化权重 (1,1,H,W)；smooth=True 用 smoothstep（C1 连续），否则线性。"""
    H, W = size
    n = max(2, int(overlap))
    t = torch.linspace(0.0, 1.0, n)
    ramp = t * t * (3.0 - 2.0 * t) if smooth else t
    mask = torch.ones(1, 1, H, W)
    mask[:, :, :, :n] = torch.minimum(mask[:, :, :, :n], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -n:] = torch.minimum(mask[:, :, :, -n:], ramp.flip(0).view(1, 1, 1, -1))
    mask[:, :, :n, :] = torch.minimum(mask[:, :, :n, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -n:, :] = torch.minimum(mask[:, :, -n:, :], ramp.flip(0).view(1, 1, -1, 1))
    return mask


def _install_smooth_feather(mod):
    """monkeypatch 节点的 create_feather_mask 为 smoothstep（幂等）。"""
    if getattr(mod.create_feather_mask, "_smooth", False):
        return
    orig = mod.create_feather_mask
    mod._fs_orig_feather = orig

    def create_feather_mask(size, overlap):
        if int(overlap) < 2:
            return orig(size, overlap)
        return _feather_mask(size, overlap, smooth=True)

    create_feather_mask._smooth = True
    mod.create_feather_mask = create_feather_mask


def _uninstall_smooth_feather(mod):
    """还原原版羽化：同一进程里批量跑不同 seam_fix 时必需，否则 off 会沿用 smooth。"""
    orig = getattr(mod, "_fs_orig_feather", None)
    if orig is not None and getattr(mod.create_feather_mask, "_smooth", False):
        mod.create_feather_mask = orig
        mod._fs_orig_feather = None


def flashvsr_aligned(mod, pipe, frames, scale, color_fix, tile, tile_overlap,
                     sparse, kv, local, seed=0):
    """逐 tile 推理 + **偏置对齐** 的拼接（对应节点的 tiled_dit 分支）。

    与原版的唯一区别：每个新 tile 先在"重叠区且已有内容"的像素上估计一个
    每通道 DC 偏差，把 tile 整体平移过去再加权累加，避免接缝两侧出现亮暗台阶。
    """
    pipe_cls = getattr(mod, "FlashVSRTinyLongPipeline", None)
    device = pipe.device
    dtype = pipe.torch_dtype
    n0 = frames.shape[0]
    add_n = mod.next_8n5(n0) - n0
    if add_n > 0:
        frames = torch.cat([frames, frames[-1:, :, :, :].repeat(add_n, 1, 1, 1)], dim=0)

    N, H, W, C = frames.shape
    canvas = torch.zeros((N, H * scale, W * scale, C), dtype=torch.float16, device="cpu")
    wsum = torch.zeros_like(canvas)
    coords = mod.calculate_tile_coords(H, W, tile, tile_overlap)

    for x1, y1, x2, y2 in mod.cqdm(coords, desc="Processing Tiles"):
        input_tile = frames[:, y1:y2, x1:x2, :]
        LQ_tile, th, tw, F = mod.prepare_input_tensor(input_tile, device, scale=scale, dtype=dtype)
        if pipe_cls is not None and not isinstance(pipe, pipe_cls):
            LQ_tile = LQ_tile.to(device)
        out_gpu = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
            tiled=True, LQ_video=LQ_tile, num_frames=F, height=th, width=tw,
            is_full_block=False, if_buffer=True,
            topk_ratio=sparse * 768 * 1280 / (th * tw), kv_ratio=kv, local_range=local,
            color_fix=color_fix, unload_dit=False, force_offload=False)
        tile_cpu = mod.tensor2video(out_gpu).to("cpu")

        mask = _feather_mask((tile_cpu.shape[1], tile_cpu.shape[2]),
                             int(tile_overlap) * scale, smooth=True).permute(0, 2, 3, 1)
        ox1, oy1 = x1 * scale, y1 * scale
        ox2, oy2 = ox1 + tile_cpu.shape[2], oy1 + tile_cpu.shape[1]

        patch = canvas[:, oy1:oy2, ox1:ox2, :]
        wpatch = wsum[:, oy1:oy2, ox1:ox2, :]

        have = wpatch[..., :1] > 1e-3
        band = mask < 0.999
        sel = (have & band).expand(-1, -1, -1, tile_cpu.shape[-1])
        if int(sel.sum()) > 512 * tile_cpu.shape[-1]:
            cur = patch / wpatch.clamp_min(1e-6)
            d = (cur - tile_cpu)[sel].reshape(-1, tile_cpu.shape[-1]).mean(dim=0)  # 每通道偏差
            tile_cpu = tile_cpu + d

        patch += tile_cpu * mask
        wpatch += mask

        del LQ_tile, out_gpu, tile_cpu, input_tile
        mod.clean_vram()

    wsum[wsum == 0] = 1.0
    out = canvas / wsum
    return out[:n0]


def _patch_tile_coords(mod):
    """节点的 calculate_tile_coords 只支持正方形 tile；这里换成支持 (th, tw)。

    只改坐标生成：flashvsr() 后面所有逻辑（裁剪、羽化、画布累加）都由坐标推导，
    所以替换它是安全的。羽化宽度仍由 tile_overlap 决定。
    """
    orig = getattr(mod, "calculate_tile_coords", None)
    if orig is None or getattr(orig, "_rect", False):
        return orig

    def coords(height, width, tile_size, overlap):
        if isinstance(tile_size, (tuple, list)):
            th, tw = int(tile_size[0]), int(tile_size[1])
        else:
            th = tw = int(tile_size)
        stride_h = max(1, th - overlap)
        stride_w = max(1, tw - overlap)
        rows = math.ceil((height - overlap) / stride_h)
        cols = math.ceil((width - overlap) / stride_w)
        out = []
        for r in range(rows):
            for c in range(cols):
                y1, x1 = r * stride_h, c * stride_w
                y2, x2 = min(y1 + th, height), min(x1 + tw, width)
                if y2 - y1 < th:
                    y1 = max(0, y2 - th)
                if x2 - x1 < tw:
                    x1 = max(0, x2 - tw)
                out.append((x1, y1, x2, y2))
        return out

    coords._rect = True
    mod.calculate_tile_coords = coords
    return coords


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


def quantize_dit_mixed(pipe, log=None):
    """混合精度：FFN 用 NVFP4（权重最省、M 最大），注意力/投影保 MXFP8，
    形状不兼容（列数不是 16/32 的倍数）的层自动退回 bf16。

    动机：纯 NVFP4 在小 M 下比 MXFP8 慢 27%，且常驻多 1.35 GiB；
    但 FFN 是最大的矩阵（8960×1536），也是 M 最大的地方，值得单独用 4bit。
    """
    import comfy_kitchen as ck

    def _ts(t):
        return (t.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1)

    def _call_fp4(x, wq, wb, tw, bias):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).contiguous()
        tx = _ts(x2)
        xq, xb = ck.quantize_nvfp4(x2, tx, pad_16x=True)
        y = ck.scaled_mm_nvfp4(xq, wq, tx, tw, xb, wb, bias=bias, out_dtype=x.dtype)
        return y[:x2.shape[0]].reshape(*shp[:-1], y.shape[-1])

    def _call_fp8(x, wq, wb, bias):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).contiguous()
        xq, xb = ck.quantize_mxfp8(x2)
        y = ck.scaled_mm_mxfp8(xq, wq, xb, wb, bias=bias, out_dtype=x.dtype)
        return y.reshape(*shp[:-1], y.shape[-1])

    n_fp4 = n_fp8 = skipped = 0
    for name, m in _iter_linears(pipe.dit):
        if not _should_quant(name):
            skipped += 1
            continue
        w = _take_weight(m)
        if w is None:
            skipped += 1
            continue
        bias = None if m.bias is None else m.bias.detach()
        want_fp4 = ".ffn." in name and w.shape[1] % 16 == 0
        try:
            if want_fp4:
                tw = _ts(w)
                wq, wb = ck.quantize_nvfp4(w, tw, pad_16x=True)
                m.forward = (lambda x, _q=wq, _s=wb, _t=tw, _b=bias: _call_fp4(x, _q, _s, _t, _b))
                n_fp4 += 1
            elif w.shape[1] % 32 == 0:
                wq, wb = ck.quantize_mxfp8(w)
                m.forward = (lambda x, _q=wq, _s=wb, _b=bias: _call_fp8(x, _q, _s, _b))
                n_fp8 += 1
            else:
                skipped += 1
                continue
            _drop_weight(m)
        except Exception as e:  # noqa
            skipped += 1
            if log:
                log(f"mixed 量化失败 {name}: {e}")
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if log:
        log(f"混合精度：FFN {n_fp4} 层 NVFP4 + {n_fp8} 层 MXFP8，跳过 {skipped}")
    return n_fp4 + n_fp8


def quantize_dit_w8a16(pipe, log=None):
    """通用 weight-only FP8：权重存 E4M3（按输出通道缩放），前向反量化回 bf16 再乘。

    不依赖 fp8 张量核（sm89+），任何能跑 bf16 的卡都行（30/40 系、甚至 CPU 后端）。
    显存：权重减半（2.84 GB → ~1.4 GB）；速度约等于 bf16（多一次 cast）。
    """
    n = skipped = 0
    maxv = 448.0                       # e4m3 最大值
    for name, m in _iter_linears(pipe.dit):
        if not _should_quant(name):
            skipped += 1
            continue
        w = _take_weight(m)
        if w is None:
            skipped += 1
            continue
        try:
            scale = (w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / maxv).float()
            wq = (w.float() / scale).clamp(-maxv, maxv).to(torch.float8_e4m3fn)
        except Exception as e:  # noqa
            skipped += 1
            if log:
                log(f"w8 量化失败 {name}: {e}")
            continue
        bias = None if m.bias is None else m.bias.detach()
        m.forward = (lambda x, _q=wq, _s=scale, _b=bias:
                     torch.nn.functional.linear(x, _q.to(x.dtype) * _s.to(x.dtype), _b))
        _drop_weight(m)
        n += 1
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if log:
        log(f"W8A16（权重 fp8 + bf16 激活）: {n} 层，跳过 {skipped}")
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


# 分段缓存键的字段（面板与引擎共用这一份，别再各写一遍）
_PARAM_KEY_KEYS = ("scale", "tile", "kv", "chunk", "tile_overlap", "mode", "attention",
                   "quant", "pre_scale", "color_fix", "seam_fix", "hi_precision",
                   "max_frames")


def _param_key(params):
    """分段目录的指纹：**只要会改变画面或分段边界的参数都必须进这里**。

    历史上漏过 color_fix / seam_fix / hi_precision / max_frames，导致改参数后
    仍命中旧分段（输出与设置不符）以及试跑与全片互相清空缓存。
    """
    chunk = max(0, int(params.get("chunk") or 0))
    ov = int(params.get("overlap") or 8)
    if chunk:
        ov = min(ov, chunk // 4)
    d = {k: params.get(k) for k in _PARAM_KEY_KEYS}
    d["overlap"] = ov
    d["color_fix"] = bool(params.get("color_fix", True))
    d["seam_fix"] = str(params.get("seam_fix") or "smooth")
    d["hi_precision"] = bool(params.get("hi_precision"))
    d["max_frames"] = int(params.get("max_frames") or 0)
    return hashlib.sha1(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:8]


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
    try:
        frames = int(st.get("nb_frames") or 0)
    except (TypeError, ValueError):
        frames = 0
    has_audio = bool(json.loads(subprocess.check_output(
        [FFPROBE, "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "json", path]).decode("utf-8", "ignore")).get("streams"))
    return dict(w=w, h=h, fps=fps, duration=dur, frames=frames, has_audio=has_audio)


def _decode_cmd(path, w, h, pre_scale, start_frame=None, count=None, fps=25.0, fp32=False):
    """拼 ffmpeg 命令；start_frame/count 给定时只解这一段（输入侧 -ss + 精确 seek）。

    fp32=True 时走 libplacebo 的 gbrpf32le 输出：源精度一路保留到模型输入，
    不再经过 rgb24 的 8bit 量化（代价是每帧内存 4×）。
    """
    vf = None
    tw, th = w, h
    if abs(pre_scale - 1.0) > 1e-6:
        tw = max(2, int(round(w * pre_scale / 2)) * 2)
        th = max(2, int(round(h * pre_scale / 2)) * 2)
    if fp32:
        vf = (f"libplacebo=w={tw}:h={th}:downscaler=spline36:format=gbrpf32le,"
              f"hwdownload,format=gbrpf32le")
        w, h = tw, th
    elif abs(pre_scale - 1.0) > 1e-6:
        vf = f"libplacebo=w={tw}:h={th}:downscaler=spline36,hwdownload,format=yuv420p"
        w, h = tw, th
    cmd = [FFMPEG, "-v", "error"]
    if vf:
        cmd += ["-init_hw_device", "vulkan=pl", "-filter_hw_device", "pl"]
    # 解码线程同样限 4：共享功耗墙下，解码多开线程只会抢 GPU 的 boost 预算
    cmd += ["-threads", str(int(os.environ.get("FLASHVSR_DEC_THREADS", "4")))]
    if start_frame:
        # -ss 放 -i 前（快速 seek）+ accurate_seek 保证落在指定帧
        cmd += ["-ss", f"{start_frame / max(fps, 1e-6):.6f}", "-accurate_seek"]
    cmd += ["-i", path]
    if vf:
        cmd += ["-vf", vf]
    if count:
        cmd += ["-frames:v", str(int(count))]
    cmd += ["-f", "rawvideo", "-pix_fmt", "gbrpf32le" if fp32 else "rgb24", "-"]
    return cmd, w, h


def decode_frames_range(path, start, count, pre_scale=1.0, fps=25.0, on_progress=None,
                        fp32=False):
    """只解 [start, start+count) 这一段，返回 NHWC（uint8 或 float32，内存只跟段长有关）。"""
    info = probe(path)
    if info["w"] <= 0 or info["h"] <= 0:
        raise RuntimeError("读不到视频尺寸，文件可能不是视频")
    cmd, w, h = _decode_cmd(path, info["w"], info["h"], pre_scale, start, count, fps, fp32)
    frame_bytes = w * h * 3 * (4 if fp32 else 1)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunks, n = [], 0
    while n < count:
        buf = proc.stdout.read(frame_bytes)
        if not buf or len(buf) < frame_bytes:
            break
        chunks.append(buf)
        n += 1
        if on_progress and n % 10 == 0:
            on_progress(n)
    proc.wait()
    if not chunks:
        return np.zeros((0, h, w, 3), np.float32 if fp32 else np.uint8)
    if fp32:
        # gbrpf32le = 平面 G/B/R，转成 HWC 的 R/G/B 并夹到 [0,1]（spline36 会有过冲）
        a = np.frombuffer(b"".join(chunks), np.float32).reshape(-1, 3, h, w)
        a = a[:, [2, 0, 1]].transpose(0, 2, 3, 1)
        return np.clip(a, 0.0, 1.0, out=a)
    return np.frombuffer(b"".join(chunks), np.uint8).reshape(-1, h, w, 3)


def decode_frames(path, on_progress=None, pre_scale=1.0, fp32=False):
    """解成 NHWC（uint8 或 float32）。pre_scale<1 用 libplacebo(spline36) 在核显上降采样。"""
    info = probe(path)
    w, h = info["w"], info["h"]
    if w <= 0 or h <= 0:
        raise RuntimeError("读不到视频尺寸，文件可能不是视频")
    cmd, w, h = _decode_cmd(path, w, h, pre_scale, fp32=fp32)
    if abs(pre_scale - 1.0) > 1e-6:
        info["w"], info["h"] = w, h
    frame_bytes = w * h * 3 * (4 if fp32 else 1)
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
    if fp32:
        a = np.frombuffer(b"".join(chunks), np.float32).reshape(-1, 3, h, w)
        a = a[:, [2, 0, 1]].transpose(0, 2, 3, 1)
        return np.clip(a, 0.0, 1.0, out=a), info
    return np.frombuffer(b"".join(chunks), np.uint8).reshape(-1, h, w, 3), info


def extract_preview(path, idx, fps, out_path, width=640):
    """从分段文件里抽一帧存成 jpg（预览用，缩到 width 宽）。"""
    t = max(0.0, float(idx) / max(float(fps), 1e-6))
    cmd = [FFMPEG, "-y", "-v", "error", "-ss", f"{t:.4f}", "-i", path,
           "-frames:v", "1", "-vf", f"scale={int(width)}:-2", "-q:v", "4", out_path]
    subprocess.run(cmd, capture_output=True, timeout=120)
    return os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def encode_frames(frames_u8, fps, out_path, audio_src=None, quality=15, backend="cpu",
                  bitdepth=8):
    h, w = frames_u8.shape[1], frames_u8.shape[2]
    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", f"{fps:.6f}", "-i", "-"]
    if audio_src:
        # 音轨和字幕流都无损 copy（字幕没有就静默跳过）
        cmd += ["-i", audio_src, "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
                "-c:a", "copy", "-c:s", "copy", "-shortest"]
    cmd += encoder_args(backend, quality, bitdepth) + ["-fps_mode", "cfr", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(0, len(frames_u8), 8):
        proc.stdin.write(frames_u8[i:i + 8].tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", "ignore")
    if proc.wait() != 0:
        raise RuntimeError(f"{backend} 编码失败：" + err.strip()[-400:])


def concat_segments(seg_paths, out_path, audio_src=None, backend="cpu", quality=15, bitdepth=8):
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
        r = subprocess.run(base + encoder_args(backend, quality, bitdepth) + [silent],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError("分段拼接失败：" + (r.stderr or "")[-400:])

    if audio_src:
        r = subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-i", silent, "-i", audio_src,
             "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
             "-c", "copy", "-shortest", out_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            # 字幕/音轨封装失败时退回只带音轨（很多 mp4 没有字幕流，-map 1:s? 本身不会报错）
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


# ---------------------------------------------------------------- 参数解析
def resolve_params(fields, src_path=None):
    """把表单字段解析成 worker 认识的一组参数（含 tile 对齐 / 2× 降采样换算）。"""
    preset = fields.get("preset", "quality")
    if preset in PRESETS:
        p = dict(PRESETS[preset])
        scale, tile, kv = p["scale"], p["tile"], p["kv"]
        # 量化后理论上能上 tile192（更快），但 8 GB 实测会 OOM（2026-11-15）。
        # 除非显式要求（fields["tile_up"]），否则按预设的保守值走。
        if preset in ("quality", "balance") and str(fields.get("tile_up") or "") in ("1", "true", "on"):
            tile = 192 if (fields.get("quant") or "none") != "none" else p["tile"]
    else:
        scale = int(fields.get("scale", 4))
        tile = int(fields.get("tile", 128))
        kv = float(fields.get("kv", 3.0))

    # 节点内部会把每个 tile 的输出裁到 128 的倍数（compute_scaled_and_target_dims），
    # 若 tile*scale 不是 128 的倍数，裁掉的部分会让拼贴几何错位 → 接缝。这里提前对齐。
    step = 128 // math.gcd(128, max(1, int(scale)))
    if tile % step:
        tile = max(step, (tile // step) * step)

    # 输入预缩放：控制「源先 /2 再 4×」这条路径，自定义档也能用
    #   auto → 目标是 2× 时自动 /2（模型走 4×）
    #   half → 强制 源/2 + 4×模型（不管上面填的 scale 是多少，输出恒为 2×）
    #   off  → 不预缩放（scale=2 就是原生 2×，由模型自己放大）
    pre_scale = 1.0
    prescale = (fields.get("prescale") or "").strip()
    if prescale not in ("auto", "half", "off"):
        prescale = "auto"
    if prescale == "auto":
        # 目标 2× 就先 /2（模型走 4×，留在训练分布内）；其它倍率不缩放
        prescale = "half" if int(scale) == 2 else "off"
    if prescale == "half":
        scale, pre_scale = 4, 0.5
        tile = max(64, tile // 2)          # 输入减半，tile 相应缩小才是同一块画面
        step = 128 // math.gcd(128, int(scale))
        if tile % step:
            tile = max(step, (tile // step) * step)

    chunk = int(fields.get("chunk", 0) or 0)
    try:
        tile_overlap = int(fields.get("tile_overlap") or 0)
    except ValueError:
        tile_overlap = 0
    if tile_overlap <= 0:
        # 自动：羽化带随 tile 走。4K 实测 tile256 用 12 时接缝处清晰度掉 13%，
        # 提到 64（=tile/4）后缝处/非缝比值 0.866 → 0.997，肉眼即消失。
        tile_overlap = max(24, tile // 4)
    tile_overlap = max(4, min(tile_overlap, max(8, tile // 2)))

    # 自动分段：整段解码会缓存「帧数 × 输出分辨率」的 LQ 视频（bf16），
    # 8 GB 上这是 OOM 的主因（640×360/100 帧就已 2.2 GB，960×544 直接 5 GB）。
    auto_note = None
    if src_path:
        try:
            info = probe(src_path)
            ow = max(2, int(round(info["w"] * pre_scale))) * int(scale)
            oh = max(2, int(round(info["h"] * pre_scale))) * int(scale)
            frames = int(info.get("frames") or round(info["duration"] * info["fps"]))
            if frames > 0 and ow > 0 and oh > 0:
                per = ow * oh * 3 * 2                     # bf16 的 LQ 缓冲，每帧字节数
                # 预算随 tile 变：tile192 的 per-tile 工作集比 tile128 多 ~2.4 GB
                # （实测 960×544：tile192 chunk40 峰值 7450 OOM；chunk20 才 6672 稳）
                budget = (1.0 if tile >= 192 else 3.2) * 1024 ** 3
                limit = max(8, min(400, int(budget / per)))
                if chunk <= 0 and frames * per > budget:
                    chunk = limit
                    auto_note = (f"已自动分段 {chunk} 帧/段：{frames} 帧 × {ow}×{oh} "
                                 f"整段要 {frames * per / 2 ** 30:.1f} GB，超出 8 GB 预算")
                elif chunk > limit:
                    was = chunk
                    chunk = limit
                    auto_note = (f"分段已从 {was} 下调到 {chunk} 帧/段："
                                 f"{ow}×{oh} 下每帧 LQ 占 {per / 2 ** 20:.0f} MB")
        except Exception:
            pass
    mode = fields.get("mode") or "tiny-long"
    if mode not in ("tiny", "tiny-long"):
        mode = "tiny-long"
    attention = fields.get("attention") or "sparse"
    if attention not in ("sparse", "dense"):
        attention = "sparse"
    encoder = fields.get("encoder") or "auto"
    if encoder not in ENCODER_CHOICES:
        encoder = "auto"
    quant = fields.get("quant") or "none"
    if quant not in QUANT_CHOICES:
        quant = "none"
    budget = fields.get("budget") or "auto"
    if budget not in ("auto", "fixed"):
        budget = "auto"
    preview = str(fields.get("preview", "1")).lower() not in ("0", "false", "off", "no")
    try:
        quality = int(float(fields.get("quality") or 15))
    except ValueError:
        quality = 15
    quality = max(10, min(28, quality))

    bitdepth = 10 if str(fields.get("bitdepth") or "8").strip() == "10" else 8
    hi_precision = str(fields.get("hi_precision") or "0").strip() in ("1", "true", "on", "yes")
    try:
        max_frames = max(0, int(fields.get("max_frames") or 0))
    except (TypeError, ValueError):
        max_frames = 0
    color_fix = str(fields.get("color_fix") or "1").strip() not in ("0", "false", "off", "no")
    seam_fix = str(fields.get("seam_fix") or "smooth").strip().lower()
    if seam_fix not in ("off", "smooth", "align"):
        seam_fix = "smooth"
    return dict(scale=scale, tile=tile, kv=kv, chunk=chunk, mode=mode,
                attention=attention, encoder=encoder, quality=quality,
                quant=quant, budget=budget, pre_scale=pre_scale,
                tile_overlap=tile_overlap, bitdepth=bitdepth,
                hi_precision=hi_precision, color_fix=color_fix, seam_fix=seam_fix,
                max_frames=max_frames,
                preview=preview, auto_note=auto_note)


class Engine:
    def __init__(self):
        self.mod = None
        self.pipe = None
        self.loaded_key = None
        self._orig_attn = None
        self.trt = None

    def ensure_loaded(self, mode="tiny", attention="sparse", on_stage=None, quant="none", prewarm=None):
        key = (mode, attention, quant)
        if self.pipe is not None and self.loaded_key == key:
            return
        if on_stage:
            on_stage("load", 0, 1, f"加载模型（mode={mode}, attn={attention}, quant={quant}）")
        _set_cudnn_benchmark(True)
        self.mod = load_node_module()
        _patch_tile_coords(self.mod)          # 支持矩形 tile（tile_h × tile_w）
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

        # VAE 解码换 TensorRT（逐帧 9 个 past，实测 ~1.8×；任何异常都回退 torch）
        try:
            import trt_decoder
            self.trt = trt_decoder.maybe_enable(
                dec, os.path.join(os.environ.get("TEMP", "."), "flashvsr_trt_cache"),
                log=lambda s: print(f"[TRT] {s}", flush=True))
        except Exception as e:  # noqa
            self.trt = None
            print(f"[FlashVSR] TRT 解码未启用: {type(e).__name__}: {e}", flush=True)

        # 建 TRT 引擎需要 3 GB workspace：临时把常驻的 DiT 挪到 CPU，建完搬回。
        # 引擎会缓存到磁盘（TEMP\flashvsr_trt_cache），只有第一次要付这个代价。
        if self.trt is not None and getattr(self.pipe, "dit", None) is not None:
            dit = self.pipe.dit

            def _offload_dit():
                try:
                    if any(p.is_cuda for p in dit.parameters()):
                        before = torch.cuda.mem_get_info()[0]
                        dit.to("cpu")
                        torch.cuda.empty_cache()
                        after = torch.cuda.mem_get_info()[0]
                        print(f"[TRT] DiT 临时卸到 CPU：空闲 {before/2**30:.1f} → {after/2**30:.1f} GB",
                              flush=True)
                        return True
                    print(f"[TRT] DiT 已在 CPU，空闲 {torch.cuda.mem_get_info()[0]/2**30:.1f} GB", flush=True)
                except Exception as e:  # noqa
                    print(f"[TRT] 卸 DiT 失败：{e}", flush=True)
                return False

            def _restore_dit(was_moved):
                if was_moved:
                    try:
                        dit.to("cuda:0")
                        print(f"[TRT] DiT 搬回显存，空闲 {torch.cuda.mem_get_info()[0]/2**30:.1f} GB",
                              flush=True)
                    except Exception as e:  # noqa
                        print(f"[TRT] DiT 搬回显存失败：{e}", flush=True)

            self.trt.before_build = _offload_dit
            self.trt.after_build = _restore_dit

        # 在 DiT 上卡之前把解码引擎建好（此时显存最空；引擎同时落盘，后续 ~1 s 加载）
        if self.trt is not None and prewarm and hasattr(self.trt, "prewarm"):
            try:
                self.trt.prewarm(int(prewarm[0]), int(prewarm[1]))
            except Exception as e:  # noqa
                print(f"[TRT] 预热失败（会在首次解码时重试）：{type(e).__name__}: {str(e)[:200]}",
                      flush=True)

        if quant in ("nvfp4", "mxfp8", "mixed", "w8"):
            dit_to_cuda(self.pipe)
            if on_stage:
                on_stage("load", 0, 1, f"量化 DiT 权重（{quant.upper()}）…")
            fn = {"nvfp4": quantize_dit_nvfp4, "mxfp8": quantize_dit_mxfp8,
                  "mixed": quantize_dit_mixed, "w8": quantize_dit_w8a16}[quant]
            print(f"[FlashVSR] {fn(self.pipe, log=print)} 个 Linear 已量化", flush=True)

        self.loaded_key = key
        torch.cuda.empty_cache()
        if on_stage:
            on_stage("load", 1, 1, "模型就绪")

    def _run_window(self, frames_u8, scale, tile, kv, on_tile=None, sparse=2.0, local=11,
                    tile_overlap=24, color_fix=True, seam_fix="smooth"):
        """tile 可以是 int（正方形）或 (th, tw)（矩形）。"""
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
            _f = torch.from_numpy(np.ascontiguousarray(frames_u8))
            # uint8 → [0,1] 需要缩放；float32 已是 [0,1]（不能原地除，否则会把 numpy 缓冲改坏）
            frames = _f.float().div_(255.0) if _f.dtype == torch.uint8 else _f.float()
            if seam_fix == "align":
                _install_smooth_feather(mod)
                out = flashvsr_aligned(
                    mod, self.pipe, frames, int(scale), bool(color_fix), tile,
                    int(tile_overlap), float(sparse), float(kv), int(local), 0)
            else:
                if seam_fix != "off":
                    _install_smooth_feather(mod)
                else:
                    _uninstall_smooth_feather(mod)
                out = mod.flashvsr(
                    self.pipe, frames, int(scale), bool(color_fix), True, True,
                    tile, int(tile_overlap), False, float(sparse), float(kv), int(local), 0, False)
            return (out.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
        finally:
            mod.cqdm = orig

    def upscale_video(self, src, dst, params, on_stage):
        pre_scale = float(params.get("pre_scale") or 1.0)
        chunk = max(0, int(params.get("chunk") or 0))
        overlap = max(0, min(int(params.get("overlap") or 8), max(0, chunk // 4)))

        # 分段时只解当前窗口需要的帧（原来会把整段视频一次解完，30 帧/段也照样吃满内存）
        info = probe(src)
        total = int(info.get("frames") or 0) or int(round(info.get("duration", 0) * info.get("fps", 25)))
        # 试跑：只处理前 N 帧（0/缺省 = 全片）。分段、续跑、指标都跟着这个上限走。
        max_frames = max(0, int(params.get("max_frames") or 0))
        if max_frames and 0 < max_frames < total:
            total = max_frames
        # 浮点解码：YUV→RGB 全程 float32（libplacebo），不再经过 rgb24 的 8bit 量化。
        # 不做预缩放时也有效（省掉 swscale 的四舍五入与抖动），代价是每帧内存 4×。
        fp32 = bool(params.get("hi_precision"))
        if fp32:
            tw = max(2, int(round(info["w"] * pre_scale / 2)) * 2)
            th = max(2, int(round(info["h"] * pre_scale / 2)) * 2)
            win_f = min(total, chunk) if chunk else total
            gb = win_f * tw * th * 3 * 4 / 2 ** 30
            if gb > 2.0:
                on_stage("decode", 0, total,
                         f"浮点解码需要 {gb:.1f} GB 内存，超过 2 GB 上限 → 回退 8bit")
                fp32 = False
            else:
                on_stage("decode", 0, total, f"浮点解码（gbrpf32le，窗口约 {gb:.2f} GB）")
        frames = None
        if chunk <= 0 or total <= chunk or total <= 0:
            if max_frames:
                # 试跑路径：只解前 total 帧，不把整段视频读进内存
                frames = decode_frames_range(
                    src, 0, total, pre_scale=pre_scale, fps=info["fps"], fp32=fp32,
                    on_progress=lambda n: on_stage("decode", n, total, f"解码中 {n}/{total} 帧"))
                if abs(pre_scale - 1.0) > 1e-6:
                    info["w"] = max(2, int(round(info["w"] * pre_scale / 2)) * 2)
                    info["h"] = max(2, int(round(info["h"] * pre_scale / 2)) * 2)
            else:
                frames, info = decode_frames(
                    src, on_progress=lambda n: on_stage("decode", n, 0, f"解码中 {n} 帧"),
                    pre_scale=pre_scale, fp32=fp32)
            total = len(frames)
            on_stage("decode", total, total, f"共 {total} 帧 {info['w']}x{info['h']} @{info['fps']:.2f}fps")
        else:
            if abs(pre_scale - 1.0) > 1e-6:
                info["w"] = max(2, int(round(info["w"] * pre_scale / 2)) * 2)
                info["h"] = max(2, int(round(info["h"] * pre_scale / 2)) * 2)
            on_stage("decode", 0, total, f"分段解码：共 {total} 帧 {info['w']}x{info['h']} @{info['fps']:.2f}fps")

        self.ensure_loaded(params.get("mode") or "tiny-long", params.get("attention") or "sparse",
                           on_stage, quant=params.get("quant") or "none",
                           prewarm=(max(16, int(params["tile"]) * int(params["scale"]) // 8),) * 2)
        audio = src if info["has_audio"] else None
        quality = int(params.get("quality") or 15)
        bitdepth = int(params.get("bitdepth") or 8)
        backend = resolve_encoder(params.get("encoder"))
        if bitdepth == 10 and backend not in ("x265", "qsv"):
            # x265 与 hevc_qsv 能出 main10；其余（含 AMF/NVENC）只有 8bit
            on_stage("encode", 0, 0,
                     f"{backend} 不支持 10 bit，本次按 8 bit 输出（要 10 bit 请用 x265 或 qsv）")
            bitdepth = 8
        budget_mode = params.get("budget") or "auto"

        def encode_win(frames_in, path, audio_src=None):
            """统一编码入口：硬件编码器初始化失败 → 自动回退 x265（并说明原因）。"""
            nonlocal backend
            try:
                encode_frames(frames_in, info["fps"], path, audio_src,
                              quality=quality, backend=backend, bitdepth=bitdepth)
            except RuntimeError as e:
                if backend not in HW_ENCODERS:
                    raise
                msg = str(e).strip().splitlines()[-1][:140] if str(e).strip() else "未知错误"
                on_stage("encode", 0, 0,
                         f"{backend} 编码失败（多半是核显/驱动不支持该 codec），自动回退 x265：{msg}")
                backend = "x265"
                encode_frames(frames_in, info["fps"], path, audio_src,
                              quality=quality, backend=backend, bitdepth=bitdepth)

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
            base_h = tile[0] if isinstance(tile, (tuple, list)) else tile
            base_w = tile[1] if isinstance(tile, (tuple, list)) else tile
            th_t = int(params.get("tile_h") or base_h)
            tw_t = int(params.get("tile_w") or base_w)
            tshape = (th_t, tw_t) if th_t != tw_t else th_t
            res = self._run_window(win, params["scale"], tshape, kk, cb, sparse=sp, local=lr,
                                   tile_overlap=int(params.get("tile_overlap") or 24),
                                   color_fix=params.get("color_fix", True),
                                   seam_fix=str(params.get("seam_fix") or "smooth"))
            return res, sp, lr, kk

        if not chunk or total <= chunk:
            out_t, sp, lr, kk = run_win(
                frames, params["tile"], params["kv"],
                lambda c, t: on_stage("infer", c, t, f"tile {c}/{t}"))
            frames_out = out_t[:total]
            on_stage("encode", 0, 1, f"编码中（sparse={sp} kv={kk} local={lr}）")
            encode_win(frames_out, dst, audio)
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
                      params_pre_scale=params.get("pre_scale"), total=total,
                      params_tile_overlap=params.get("tile_overlap"),
                      params_ctrl=dict(color_fix=bool(params.get("color_fix", True)),
                                       seam_fix=str(params.get("seam_fix") or "smooth"),
                                       hi_precision=bool(params.get("hi_precision")),
                                       max_frames=int(params.get("max_frames") or 0)))
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
            if frames is not None:
                win = frames[win_start:win_end]
            else:
                on_stage("decode", win_start, total, f"解码 {win_start}–{win_end} 帧")
                win = decode_frames_range(src, win_start, win_end - win_start,
                                          pre_scale=pre_scale, fps=info["fps"], fp32=fp32,
                                          on_progress=lambda n, ws=win_start: on_stage(
                                              "decode", ws + n, total, f"解码 {ws + n}/{total} 帧"))
                if len(win) != win_end - win_start:
                    # 帧数与探测不符（典型：VFR 源）。以前只修正本段的 keep、
                    # 不动后续段起点，结果是成片中间**静默缺帧**。宁可失败：
                    got_n = len(win)
                    raise RuntimeError(
                        f"解码帧数与探测不符：第 {i + 1}/{n_seg} 段期望 "
                        f"{win_end - win_start} 帧，实际 {got_n} 帧。\n"
                        f"这是 VFR（可变帧率）源的典型表现。先转成恒定帧率再跑：\n"
                        f'  ffmpeg -i "输入.mp4" -fps_mode cfr -r 30 -c:v libx264 -crf 12 "输入_cfr.mp4"')
            on_stage("seg", i, n_seg, f"段 {i + 1}/{n_seg}：{win_start}–{win_end} 帧")

            def tile_cb(c, t, i=i):
                on_stage("seg", i, n_seg, f"段 {i + 1}/{n_seg} tile {c}/{t}")

            res, sp, lr, kk = run_win(win, params["tile"], params["kv"], tile_cb)
            out_frames = res[keep_from:keep_from + keep]
            out_wh = [int(out_frames.shape[2]), int(out_frames.shape[1])]
            on_stage("encode", 0, 1, f"段 {i + 1}/{n_seg} 写盘（{backend}）")
            encode_win(out_frames, seg_path, None)
            del res, out_frames, win

            got = probe_frames(seg_path)
            if got != keep:
                raise RuntimeError(f"第 {i + 1} 段校验失败：期望 {keep} 帧，实际 {got} 帧")
            done[str(i)] = dict(frames=keep, bytes=os.path.getsize(seg_path), at=int(time.time()))
            man["done"] = done
            save_manifest(seg_dir, man)

            # 预览帧：每段随机抽一帧（分段专属，方便边跑边看效果）
            pv = None
            pdir = params.get("preview_dir")
            if pdir and keep > 0:
                try:
                    os.makedirs(pdir, exist_ok=True)
                    name = f"seg_{i:04d}.jpg"
                    if extract_preview(seg_path, random.randrange(keep), info["fps"],
                                       os.path.join(pdir, name)):
                        pv = name
                except Exception as e:  # noqa
                    print(f"[preview] 第 {i + 1} 段抽帧失败: {e}", flush=True)

            on_stage("seg", i + 1, n_seg,
                     f"段 {i + 1}/{n_seg} 完成（sparse={sp} kv={kk} local={lr}）",
                     **({"preview": pv, "pv_cur": i + 1, "pv_total": n_seg} if pv else {}))

        on_stage("encode", 0, 1, f"拼接 {n_seg} 段 + 混入音轨")
        concat_segments(seg_paths, dst, audio_src=audio, backend=backend, quality=quality, bitdepth=bitdepth)
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
        self.trt = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
