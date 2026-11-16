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

ENCODER_CHOICES = ("auto", "cpu", "nvenc", "qsv", "qsv_h264", "amf", "vulkan",
                    "svtav1", "av1_nvenc", "x265")
ENCODER_LABEL = {
    "auto": "自动（CPU 优先，GPU 不可用则回退）",
    "cpu": "CPU · libx264（H.264）",
    "nvenc": "GPU · NVENC H.264",
    "qsv": "GPU · Intel QSV · HEVC 8bit（4K，6 代+ 核显）",
    "qsv_h264": "GPU · Intel QSV · H.264 8bit（4K，兼容最老，4 代+）",
    "vulkan": "GPU · Vulkan Video encode（HEVC，需驱动支持 VK_KHR_video_encode_queue）",
    "amf": "GPU · AMF（AMD 核显）",
    "svtav1": "CPU · SVT-AV1（同画质体积最小，慢）",
    "av1_nvenc": "GPU · NVENC AV1（40/50 系）",
    "x265": "CPU · libx265（H.265，均衡）",
}
# NVFP4 与混合档（FFN 用 NVFP4）都已停用：画质比 MXFP8 低 6 dB（44.65 dB vs 50.75 dB），
# 显存还更大（3.40 GiB vs 2.05 GiB），速度只在超大 token 才占优。
# 对外只保留三档：bf16 基准 / MXFP8 推荐 / W8A16 老卡兼容。
QUANT_CHOICES = ("none", "mxfp8", "w8")


TUNE_CHOICES = {
    "cpu": ("ssim", "psnr", "film", "animation", "grain", "stillimage",
            "fastdecode", "zerolatency"),
    "x265": ("ssim", "psnr", "grain", "fastdecode", "zerolatency"),
    "svtav1": ("vq", "psnr", "ssim", "ms_ssim", "vmaf"),
}
# SVT-AV1 的 tune 只认**数值**（名字会静默退回 PSNR）：
# 0=VQ 1=PSNR 2=SSIM 3=IQ(仅 all-intra) 4=MS_SSIM 5=VMAF
SVT_TUNE_NUM = {"vq": 0, "psnr": 1, "ssim": 2, "ms_ssim": 4, "vmaf": 5}
# libplacebo 缩放器（本机实测可用；jinc 不存在）
SCALERS = ("ewa_lanczos4sharpest", "ewa_lanczos", "lanczos", "spline36", "sinc",
           "bicubic", "bilinear", "nearest")
DEFAULT_SCALER = "ewa_lanczos4sharpest"
# libplacebo 的抖动方式（本机枚举，没有 error_diffusion）
# 实际可用/可推荐的只有 blue 与 ordered_fixed（ordered / white 虽能跑，属实现变体，不暴露）
DITHERS = ("", "blue", "ordered_fixed")

TUNE_ALL = ("", "ssim", "psnr", "vq", "vmaf", "ms_ssim", "film", "animation",
            "grain", "stillimage", "fastdecode", "zerolatency")


def tunes_for(backend):
    """某个编码器**真正支持**的 tune 列表（UI 按它渲染下拉，避免列一堆无效项）。"""
    return TUNE_CHOICES.get(backend, ())


def encoder_global_args(backend):
    """某些后端需要**输入前**的全局参数（Vulkan 要建 hw device）。"""
    if backend == "vulkan":
        return ["-init_hw_device", "vulkan=vk:0", "-filter_hw_device", "vk"]
    return []


def _encoder_base(backend, quality, bitdepth=8, svt_preset=None, tune=None):
    q = str(int(quality))
    # 10-bit：只有 x265 在本机验证可用（x264 是 8bit 构建、SVT-AV1 在 4K 下 10bit 会 OOM）
    pix = "yuv420p10le" if (int(bitdepth) == 10 and backend == "x265") else "yuv420p"
    if backend == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq",
                "-rc", "vbr", "-cq", q, "-b:v", "0", "-pix_fmt", "yuv420p"]
    if backend == "av1_nvenc":
        return ["-c:v", "av1_nvenc", "-preset", "p7", "-tune", "hq",
                "-rc", "vbr", "-cq", q, "-b:v", "0", "-pix_fmt", "yuv420p"]
    if backend == "vulkan":
        # Vulkan Video encode：只吃 vulkan hwframe，所以必须 hwupload。
        # 质量用 cqp（常数 QP，语义接近 CRF）；老驱动/老卡没有 VK_KHR_video_encode_queue 会直接失败。
        return ["-c:v", "hevc_vulkan", "-rc_mode", "cqp", "-qp", q,
                "-tune", "hq", "-usage", "record",
                "-vf", "format=nv12,hwupload=extra_hw_frames=16"]
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
        # 用户主动选 svtav1 → 用 FLASHVSR_SVT_PRESET（默认 4，质量档）；
        # 作为**兜底**时由 encode_win 传 svt_preset="8"（4K 实测 p8 只要 p4 的 1/2.8 时间）
        preset = str(svt_preset or os.environ.get("FLASHVSR_SVT_PRESET", "4"))
        # 默认必须是 VQ（0）：不传 tune 时 ffmpeg 封装给 SVT 的是 **PSNR**，
        # 而 VQ 才是 SVT 的主观质量档（心理视觉优化）；PSNR/SSIM/VMAF 只是指标对齐用。
        tune_num = SVT_TUNE_NUM.get(tune, 0)
        svt = ["lp=4", f"tune={tune_num}"]
        # 只能传数值：`tune=vq` 会被拒（Error parsing option tune: vq）然后静默退回 PSNR；
        # IQ(3) 仅 all-intra/low-delay，视频直接失败，故不提供。
        return (["-c:v", "libsvtav1", "-crf", q, "-preset", preset,
                 "-svtav1-params", ":".join(svt), "-g", "240", "-pix_fmt", "yuv420p"])
    if backend == "x265":
        # 共享功耗墙：CPU 编码线程开太多会抢走 GPU 的 boost 预算。
        # 注意：`-threads N` 对 libx265 **不生效** —— 实测 -threads 4 时它照样建 32 线程池，
        # 并发 16.1×；真正限住要写 `-x265-params pools=N`（实测 4.0×，代价 wall 3.9→8.9s）。
        pools = os.environ.get("FLASHVSR_X265_POOLS", "4")
        x265_params = []
        if pools and str(pools).isdigit() and int(pools) > 0:
            x265_params.append(f"pools={pools}")
        th = pools if (pools and str(pools).isdigit() and int(pools) > 0) else "4"
        args = ["-c:v", "libx265", "-preset", "medium", "-crf", q,
                "-threads", str(th), "-pix_fmt", pix]
        if x265_params:
            args += ["-x265-params", ":".join(x265_params)]
        if tune and tune in TUNE_CHOICES["x265"]:
            args += ["-tune", tune]
        return args
    args = ["-c:v", "libx264", "-preset", "slow", "-crf", q,
            "-threads", "4", "-pix_fmt", "yuv420p"]
    if tune and tune in TUNE_CHOICES["cpu"]:
        args += ["-tune", tune]
    return args


def encoder_args(backend, quality, bitdepth=8, svt_preset=None, tune=None, extra=None):
    """统一出口：所有后端的参数 + 开发模式附加参数（extra 一律追加在末尾，覆盖前值）。"""
    return _encoder_base(backend, quality, bitdepth, svt_preset, tune) + list(extra or [])


_SVT_TUNE_CAPS = None


def svt_tune_caps(force=False):
    """本机这个 SVT-AV1 版本实际支持哪些 tune（试编 1 帧，按退出码判断）。

    版本差异真实存在：v4.2 是 0=VQ 1=PSNR 2=SSIM 3=IQ 4=MS_SSIM 5=VMAF，
    更老的版本只有 0..2。写死映射迟早出错，所以每次装新 ffmpeg 都重新问它。
    """
    global _SVT_TUNE_CAPS
    if _SVT_TUNE_CAPS is not None and not force:
        return _SVT_TUNE_CAPS
    ok = []
    for name, num in SVT_TUNE_NUM.items():
        try:
            r = subprocess.run(
                [FFMPEG, "-hide_banner", "-v", "error", "-f", "lavfi",
                 "-i", "color=c=gray:s=160x120:r=5:d=0.2", "-frames:v", "1",
                 "-c:v", "libsvtav1", "-crf", "32",
                 "-svtav1-params", f"lp=2:tune={num}", "-f", "null", "-"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=90)
            if r.returncode == 0:
                ok.append(name)
        except Exception:  # noqa
            pass
    _SVT_TUNE_CAPS = ok
    return ok


# 走 GPU 编码的引擎：初始化失败（驱动/核显不支持该 codec，或撑不住目标分辨率）时自动降级
HW_ENCODERS = ("nvenc", "av1_nvenc", "qsv", "qsv_h264", "amf", "vulkan")

# 降级阶梯：先同族（HEVC→H.264），再 CPU 高质量（x265），最后最兼容（x264）
FALLBACK_CHAIN = {
    "qsv": ("qsv_h264", "x265", "cpu"),
    "qsv_h264": ("x265", "cpu"),
    "nvenc": ("x265", "cpu"),
    "av1_nvenc": ("x265", "cpu"),
    "amf": ("x265", "cpu"),
    "vulkan": ("x265", "cpu"),
    "svtav1": ("x265", "cpu"),
    "x265": ("cpu",),
    "cpu": (),
}

CAPS_TTL = 7 * 86400

# 编码参数语义变更时 +1：让旧的分段缓存整体失效（否则会用旧语义的成片续跑）
ENCODER_ARGS_REV = 2


def fallback_chain(backend):
    """取某个编码器失败后的降级顺序（可用 FLASHVSR_FALLBACK_CHAIN 覆盖）。

    覆盖格式：`x265>svtav1>cpu`（对所有后端生效），或 `qsv=qsv_h264>x265>cpu`
    只覆盖某一个后端。末尾总是补 cpu，保证一定收得住。
    """
    raw = (os.environ.get("FLASHVSR_FALLBACK_CHAIN") or "").strip()
    chain = None
    if raw:
        for part in raw.split(";"):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                if k.strip() == backend:
                    chain = [x.strip() for x in v.split(">") if x.strip()]
            elif chain is None:
                chain = [x.strip() for x in part.split(">") if x.strip()]
    if chain is None:
        chain = list(FALLBACK_CHAIN.get(backend, ()))
    chain = [x for x in chain if x in ENCODER_CHOICES and x != backend]
    if "cpu" not in chain:
        chain.append("cpu")
    return tuple(chain)


def _caps_file():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "work", "encoder_caps.json")


def gpu_info():
    """给 UI 判断架构用：返回 {name, cap}（如 {'RTX 5060 Laptop GPU', '12.0'}）。"""
    try:
        import torch
        if torch.cuda.is_available():
            return dict(name=torch.cuda.get_device_name(0),
                        cap=".".join(str(x) for x in torch.cuda.get_device_capability(0)))
    except Exception:  # noqa
        pass
    return dict(name="", cap="")


def _gpu_tag():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa
        pass
    return "unknown"


def _load_caps_cache():
    try:
        with open(_caps_file(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa
        return {}


def _save_caps_cache(data):
    try:
        os.makedirs(os.path.dirname(_caps_file()), exist_ok=True)
        tmp = _caps_file() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, _caps_file())
    except Exception:  # noqa
        pass


def probe_encoders(w=256, h=256):
    """在**实际目标分辨率**上各编 1 帧，判断这个编码器在本机+本尺寸下能不能用。

    为什么不用 256×256：老核显可能支持 H.264，但 4K（level 5.1+）超规格 —
    小尺寸探测通过、真跑 4K 就失败，等于白等几小时。
    """
    w = max(64, int(w) // 2 * 2)
    h = max(64, int(h) // 2 * 2)
    src = ["-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r=5:d=0.2", "-frames:v", "1"]
    out = {}
    for be in ("cpu", "nvenc", "qsv", "qsv_h264", "amf", "vulkan", "svtav1",
               "av1_nvenc", "x265"):
        try:
            r = subprocess.run(
                [FFMPEG, "-hide_banner", "-v", "error"] + encoder_global_args(be) + src
                + encoder_args(be, 18) + ["-f", "null", "-"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
            if r.returncode == 0:
                out[be] = dict(ok=True, detail="")
            else:
                # 从 stderr 里挑出真正有信息量的那行（初始化失败比 "no packets" 有用）
                lines = [l.strip() for l in (r.stderr or "").splitlines() if l.strip()]
                key = [l for l in lines if any(k in l.lower() for k in
                                               ("error", "invalid", "not support", "failed",
                                                "cannot", "unable", "api", "mfx"))]
                pick = (key or lines or ["未知错误"])
                out[be] = dict(ok=False, detail=" / ".join(pick[:2])[:220])
        except Exception as e:  # noqa
            out[be] = dict(ok=False, detail=f"{type(e).__name__}: {e}")
    if out["cpu"]["ok"]:
        out["auto"] = dict(ok=True, detail="CPU 优先")
    elif out.get("x265", {}).get("ok"):
        out["auto"] = dict(ok=True, detail="回退 x265")
    else:
        out["auto"] = dict(ok=False, detail="没有可用的编码器")
    return out


_CAPS = {}


def get_caps(force=False, w=256, h=256):
    """按 (GPU, w×h) 取能力表；有 7 天内的缓存就直接用，省掉每次开任务的探测时间。"""
    w = max(64, int(w) // 2 * 2)
    h = max(64, int(h) // 2 * 2)
    key = f"{_gpu_tag()}|{w}x{h}"
    if not force and key in _CAPS:
        return _CAPS[key]
    cache = _load_caps_cache()
    ent = cache.get(key)
    if not force and ent and time.time() - float(ent.get("ts") or 0) < CAPS_TTL:
        _CAPS[key] = ent["caps"]
        return _CAPS[key]
    caps = probe_encoders(w, h)
    _CAPS[key] = caps
    cache[key] = dict(ts=int(time.time()), caps=caps)
    _save_caps_cache(cache)
    return caps


def resolve_encoder(requested, caps=None, w=256, h=256):
    """把用户选的编码器解析成"本机真能跑"的那个（不可用就走 auto 顺序）。"""
    caps = caps or get_caps(w=w, h=h)
    be = requested if requested in ENCODER_CHOICES else "auto"
    if be != "auto" and caps.get(be, {}).get("ok"):
        return be
    # auto / 不可用的显式选择 → CPU 优先（实测：GPU 编码会抢推理的显存与功耗预算，
    # AMF 画质还明显更差，所以硬件编码器只排在后面当兜底）
    for alt in ("cpu", "x265", "nvenc", "qsv", "qsv_h264", "amf", "vulkan"):
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


def _gpu_canvas_fits(n_frames, h, w, scale, limit_gb=0.8):
    """画布（fp16 NHWC）+ 权重图要多少显存；太大就回退 CPU 画布。"""
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:  # noqa
        return False
    need = n_frames * (h * scale) * (w * scale) * 3 * 2 * 1.5   # canvas + wsum + 余量
    return need < min(limit_gb * 2 ** 30, free * 0.5)


def flashvsr_gpu_canvas(mod, pipe, frames, scale, color_fix, tile, tile_overlap,
                        sparse, kv, local, seed=0):
    """和节点的 tiled_dit 分支等价，但**画布与混合都在 GPU 上**。

    省掉的成本：每 tile 一次 `.to("cpu")`（8 帧 256² ≈ 3 MB PCIe）+ CPU 上的
    float16 乘加（15 tile ≈ 0.5–0.8 s，且与 GPU 串行）。
    """
    device = pipe.device
    dtype = pipe.torch_dtype
    n0 = frames.shape[0]
    add_n = mod.next_8n5(n0) - n0
    if add_n > 0:
        frames = torch.cat([frames, frames[-1:, :, :, :].repeat(add_n, 1, 1, 1)], dim=0)

    N, H, W, C = frames.shape
    canvas = torch.zeros((N, H * scale, W * scale, C), dtype=torch.float16, device=device)
    wsum = torch.zeros((N, H * scale, W * scale, 1), dtype=torch.float16, device=device)
    coords = mod.calculate_tile_coords(H, W, tile, tile_overlap)
    pipe_cls = getattr(mod, "FlashVSRTinyLongPipeline", None)

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
        tile_gpu = mod.tensor2video(out_gpu)                    # (F,H,W,C)
        # 实测：tiny-long 的 pipe(...) 输出会落到 CPU（流式解码内部就搬了），
        # 所以这里先对齐设备——若是 CPU 就 H2D（3 MB/tile ≈ 0.25 ms，可忽略），
        # 换取混合在显存里做（省掉 CPU 上的 float16 乘加）。
        if tile_gpu.device.type == "cpu":
            tile_gpu = tile_gpu.to(device, non_blocking=True)
        mask = _feather_mask((tile_gpu.shape[1], tile_gpu.shape[2]),
                             int(tile_overlap) * scale, smooth=True).to(tile_gpu.device)
        mask = mask.permute(0, 2, 3, 1).to(tile_gpu.dtype)
        ox1, oy1 = x1 * scale, y1 * scale
        ox2, oy2 = ox1 + tile_gpu.shape[2], oy1 + tile_gpu.shape[1]
        canvas[:, oy1:oy2, ox1:ox2, :] += (tile_gpu * mask).to(torch.float16)
        wsum[:, oy1:oy2, ox1:ox2, :] += mask.to(torch.float16)
        del LQ_tile, out_gpu, tile_gpu
        mod.clean_vram()

    out = canvas / wsum.clamp_min_(1e-6)
    return out[:n0]


def _drop_stride(frames, thr=0.5):
    """检测"按 p 拍画"（p=2/3）。返回 p 或 1（不抽）。

    判据：规整保留 i%p==0 时，**被丢掉的帧**是否都是前一张的复制：
    |Δ(i, i−1)| < thr。丢错率 ≥5% 就认为这条内容不是按 p 拍的，不抽。

    只允许 phase=0：这样插回就是 np.repeat(out, p)[:n]，不需要 keep 表。
    """
    n = len(frames)
    if n < 8:
        return 1
    g = np.asarray(frames[:, ::4, ::4, :3], dtype=np.int16).mean(axis=-1)  # 低分辨率灰度
    d1 = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))                     # d1[i-1] = |f(i)-f(i-1)|
    for p in (2, 3):
        if n <= p + 2:
            continue
        kept = set(range(0, n, p))
        lost = [d1[i - 1] for i in range(1, n) if i not in kept]
        if lost and float(np.mean(np.asarray(lost) >= thr)) < 0.05:
            return p
    return 1


def _expand(out, p, n):
    return out if p <= 1 else np.repeat(out, p, axis=0)[:n]


def install_stream_loader():
    """把节点的 safetensors 加载改成"逐个张量读 → 立刻转 dtype → 立刻上卡"。

    原实现（utils.load_state_dict_from_safetensors）会攒一个完整的 CPU state_dict，
    实测 commit 峰值 **12.6 GB**（f32 视图 5.4 + bf16 副本 2.8 + 分配器缓存 + 显存），
    在 16 GB 内存的机器上加载时浏览器都得关掉。
    流式之后 CPU 侧同时只持有一个张量，峰值降到"一个最大的张量"量级。

    只影响加载路径，不改模型结构；失败就退回原实现。
    """
    try:
        utils = sys.modules.get("flashvsr_node.src.models.utils")
        if utils is None or getattr(utils, "_stream_patched", False):
            return False
        orig = utils.load_state_dict_from_safetensors

        def patched(file_path, torch_dtype=None):
            """不用 safe_open（它 mmap 整个文件，5.4 GB 文件页会常驻工作集）。

            』 这里自己解析头 + read()，一次只读一个张量：
              - 工作集峰值 ≈ 最大的那个张量（~50 MB），而不是整个 checkpoint
              - 读出来 → 转 dtype → 立刻上卡 → 立刻释放
            等价于 cpp-port 分支里 load_dit.cpp 的做法（那边 host 峰值 542 MiB）。
            """
            import json
            import struct

            import numpy as np
            import torch as _t

            dev = "cuda" if _t.cuda.is_available() else "cpu"
            np_dtype = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16,
                        "I64": np.int64, "I32": np.int32, "I16": np.int16,
                        "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}
            out = {}
            with open(file_path, "rb") as f:
                hlen = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(hlen))
                base = 8 + hlen
                for k, info in hdr.items():
                    if k == "__metadata__" or not isinstance(info, dict):
                        continue
                    beg, end = info["data_offsets"]
                    f.seek(base + beg)
                    raw = f.read(end - beg)
                    nd = np_dtype.get(info["dtype"])
                    if nd is None:
                        raise RuntimeError(f"不支持的 dtype {info['dtype']} ({k})")
                    arr = np.frombuffer(raw, dtype=nd)
                    if info["dtype"] == "BF16":
                        # bf16 在 numpy 里没有原生类型：按 uint16 读，再用 view 转 torch
                        ten = arr.copy().view(_t.bfloat16).reshape(info["shape"])
                    else:
                        ten = _t.from_numpy(arr.copy()).reshape(info["shape"])
                    if torch_dtype is not None:
                        ten = ten.to(torch_dtype)
                    out[k] = ten.to(dev)
                    del ten, arr, raw
            return out

        def guarded(file_path, torch_dtype=None):
            try:
                return patched(file_path, torch_dtype)
            except Exception as e:  # noqa
                print(f"[stream-load] 流式加载失败，回退原实现: {e}", flush=True)
                return orig(file_path, torch_dtype)

        utils.load_state_dict_from_safetensors = guarded
        utils._stream_patched = True
        print("[stream-load] 已启用：逐张量读→上卡，不再攒 CPU state_dict", flush=True)
        return True
    except Exception as e:  # noqa
        print(f"[stream-load] 启用失败（不影响运行）: {e}", flush=True)
        return False


def _quant8(out):
    """float [0,1] → uint8。**这里是整条链路唯一的 8bit 量化点**。

    之前是纯 round()，没有抖动 → 渐变/大面积平坦区在 8bit 输出上会出色带。
    预缩放阶段的 libplacebo 抖动输出的是 gbrpf32le（浮点），没有 8bit 目标可抖，
    所以那些抖动对本项目无效——必须抖在这一步。

    用 FLASHVSR_DITHER 选：
      off     纯四舍五入（默认，保持历史行为）
      ordered 8×8 Bayer 有序抖动（确定性、逐帧不变、几乎不加码率）
      tpdf    三角分布随机抖动（最平滑，但会略微增加码率）
    """
    mode = (os.environ.get("FLASHVSR_DITHER") or "off").lower()
    x = out.clamp(0, 1) * 255.0
    if mode == "ordered":
        bayer = torch.tensor(
            [[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
             [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
             [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
             [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]],
            dtype=torch.float32, device=x.device) / 64.0 - 0.5
        h, w = x.shape[-3], x.shape[-2]
        tile = bayer.repeat((h // 8) + 1, (w // 8) + 1)[:h, :w]
        x = x + tile.view(1, h, w, 1)
    elif mode == "tpdf":
        n = torch.rand_like(x) - torch.rand_like(x)     # 三角分布，幅度 ±1 LSB
        x = x + n
    return x.round().clamp_(0, 255).to(torch.uint8).cpu().numpy()


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
    # 【已停用】NVFP4 画质太差（混合档实测 44.65 dB），仅保留代码备查。
    """W4A4（NVFP4）。Blackwell 原生 FP4；单层微基准 FFN up 3.78×，误差 0.134。
    注意：激活/权重量化都要 pad_16x=True，否则 CUDA 内核越界（0xC0000005）。"""
    import comfy_kitchen as ck

    def _ts(t):
        return (t.abs().amax() / (6.0 * 448.0)).clamp(min=1e-8).float().reshape(1)

    def _call(x, wq, wb, tw, bias, st=None):
        shp = x.shape
        x2 = x.reshape(-1, shp[-1]).contiguous()
        m = x2.shape[0]
        # 每层重算 abs+amax 占 fp4 每层开销的 42%（实测 0.78 → 0.46 ms）。
        # 这里缓存标定值、每 N 次刷新一次，并留 25% 余量防止后续 batch 溢出。
        every = int(os.environ.get("FLASHVSR_FP4_AMAX_EVERY", "16"))
        if st is None or every <= 0:
            tx = _ts(x2)
        else:
            if st.get("scale") is None or st["n"] % every == 0:
                st["scale"] = _ts(x2) * 1.25
            st["n"] = st.get("n", 0) + 1
            tx = st["scale"]
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
        _st = {"scale": None, "n": 0}
        m.forward = (lambda x, _q=wq, _s=wb, _t=tw, _b=bias, _st=_st:
                     _call(x, _q, _s, _t, _b, _st))
        _drop_weight(m)
        n += 1
    torch.cuda.empty_cache()
    if log:
        log(f"NVFP4(pad16): {n} 个 Linear 已量化，跳过 {skipped}")
    return n


def quantize_dit_mixed(pipe, log=None):
    # 【已停用】FFN 用 NVFP4，实测 44.65 dB（比 MXFP8 低 6 dB），仅保留代码备查。
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


QCACHE_REV = 1            # 权重格式/量化实现变了就 +1，旧缓存自动失效


def _qcache_dir():
    d = os.environ.get("FLASHVSR_QCACHE_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "work", "qcache")
    os.makedirs(d, exist_ok=True)
    return d


def _qcache_path(pipe, quant):
    """缓存键：DiT 前 8 个 Linear 的形状 + 权重切片哈希 + 量化档 + 格式版本。

    刻意不 hash 5.4 GB（那本身就慢）；8×256 个数足以区分不同 checkpoint，
    换模型/换量化实现都会变。
    """
    try:
        import hashlib
        h = hashlib.sha1()
        cnt = 0
        for name, m in _iter_linears(pipe.dit):
            w = getattr(m, "weight", None)
            if w is None:
                continue
            h.update(f"{name}{tuple(w.shape)}".encode())
            h.update(w.detach().flatten()[:256].float().cpu().numpy().tobytes())
            cnt += 1
            if cnt >= 8:
                break
        if cnt == 0:
            return None
        key = f"dit_{h.hexdigest()[:16]}_{quant}_v{QCACHE_REV}"
        return os.path.join(_qcache_dir(), key + ".safetensors")
    except Exception:
        return None


def _qcache_load(path, log=None):
    """读缓存 → {layer: (wq, wb)}；格式不对就当没有。"""
    if not path or not os.path.exists(path):
        return None
    try:
        from safetensors.torch import load_file
        d = load_file(path, device="cuda")
        out = {}
        for k in d:
            if not k.endswith("|q"):
                continue
            name = k[:-2]
            b = d.get(name + "|b")
            if b is None:
                return None
            out[name] = (d[k].view(torch.float8_e4m3fn), b.view(torch.float8_e8m0fnu))
        if log:
            log(f"量化缓存命中：{len(out)} 层（{os.path.getsize(path) / 2**30:.2f} GB）")
        return out or None
    except Exception as e:  # noqa
        if log:
            log(f"量化缓存读取失败（忽略，重新量化）: {e}")
        return None


def _qcache_save(path, items, log=None):
    """原子写：先写 .tmp 再 rename，中断不会留半截文件。"""
    if not path or not items:
        return
    try:
        from safetensors.torch import save_file
        blob = {}
        for name, (wq, wb) in items.items():
            blob[name + "|q"] = wq.contiguous().view(torch.uint8).cpu()
            blob[name + "|b"] = wb.contiguous().view(torch.uint8).cpu()
        tmp = path + ".tmp"
        save_file(blob, tmp)
        os.replace(tmp, path)
        if log:
            log(f"量化缓存已写盘：{len(items)} 层（{os.path.getsize(path) / 2**30:.2f} GB）")
    except Exception as e:  # noqa
        if log:
            log(f"量化缓存落盘失败（不影响本次运行）: {e}")


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

    # 权重是**确定的产物**：算一次就落盘，之后直接复用。
    # 除了省一次量化，更重要的是把量化期间的 commit 峰值削掉——那一段会把机器逼进
    # 页文件交换（实测 commit 峰值 12.5 GB），页文件写入才是真伤 SSD 的东西。
    use_cache = os.environ.get("FLASHVSR_QCACHE", "1") != "0"
    cache_path = _qcache_path(pipe, "mxfp8") if use_cache else None
    cached = _qcache_load(cache_path, log) if cache_path else None
    fresh = {}

    n = skipped = n_hit = 0
    for name, m in _iter_linears(pipe.dit):
        if not _should_quant(name):
            skipped += 1
            continue
        hit = cached.get(name) if cached else None
        if hit is not None:
            wq, wb = hit
            n_hit += 1
        else:
            w = _take_weight(m)
            if w is None or w.shape[1] % 32 != 0:
                skipped += 1
                continue
            try:
                wq, wb = ck.quantize_mxfp8(w)
                fresh[name] = (wq, wb)
            except Exception as e:  # noqa
                if log:
                    log(f"mxfp8 量化失败 {name}: {e}")
                skipped += 1
                continue
        bias = None if m.bias is None else m.bias.detach()
        m.forward = (lambda x, _q=wq, _s=wb, _b=bias: _call(x, _q, _s, _b))
        _drop_weight(m)
        n += 1
    if fresh and cache_path:
        _qcache_save(cache_path, fresh, log)
    torch.cuda.empty_cache()
    if log:
        log(f"MXFP8: {n} 个 Linear（缓存命中 {n_hit}，本次新量化 {len(fresh)}），跳过 {skipped}")
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
                   "max_frames", "reuse")


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
    # 空串与缺省必须等价，否则面板（resolve_params 给 ""）和引擎（没传）会算出不同键
    d["tune"] = str(params.get("tune") or "")
    d["extra_args"] = str(params.get("extra_args") or "").strip()
    d["enc_rev"] = ENCODER_ARGS_REV
    d["prescale_scaler"] = str(params.get("prescale_scaler") or DEFAULT_SCALER)
    # 面板侧没有 reuse 字段时按 "off" 归一（否则两边指纹不一致 → 续跑混用旧分段）
    d["reuse"] = str(params.get("reuse") or "off")
    d["prescale_deband"] = bool(params.get("prescale_deband"))
    d["deband_grain"] = params.get("deband_grain")
    d["prescale_dither"] = str(params.get("prescale_dither") or "")
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


def _decode_cmd(path, w, h, pre_scale, start_frame=None, count=None, fps=25.0, fp32=False,
                scaler=None, deband=False, deband_grain=None, dither=""):
    """拼 ffmpeg 命令；start_frame/count 给定时只解这一段（输入侧 -ss + 精确 seek）。

    fp32=True 时走 libplacebo 的 gbrpf32le 输出：源精度一路保留到模型输入，
    不再经过 rgb24 的 8bit 量化（代价是每帧内存 4×）。
    """
    vf = None
    tw, th = w, h
    scaler = scaler if scaler in SCALERS else DEFAULT_SCALER
    extra = ""                      # 附加的 libplacebo 选项（去色带 / 抖动）
    if deband:
        extra += ":deband=1"
        if deband_grain is not None:
            extra += f":deband_grain={float(deband_grain):g}"
    if dither and dither in DITHERS:
        extra += f":dithering={dither}"
    if abs(pre_scale - 1.0) > 1e-6:
        tw = max(2, int(round(w * pre_scale / 2)) * 2)
        th = max(2, int(round(h * pre_scale / 2)) * 2)
    if fp32:
        vf = (f"libplacebo=w={tw}:h={th}:downscaler={scaler}{extra}:format=gbrpf32le,"
              f"hwdownload,format=gbrpf32le")
        w, h = tw, th
    elif abs(pre_scale - 1.0) > 1e-6 or extra:
        vf = (f"libplacebo=w={tw}:h={th}:downscaler={scaler}{extra},"
              f"hwdownload,format=yuv420p")
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
                        fp32=False, scaler=None, deband=False, deband_grain=None,
                        dither=""):
    """只解 [start, start+count) 这一段，返回 NHWC（uint8 或 float32，内存只跟段长有关）。"""
    info = probe(path)
    if info["w"] <= 0 or info["h"] <= 0:
        raise RuntimeError("读不到视频尺寸，文件可能不是视频")
    cmd, w, h = _decode_cmd(path, info["w"], info["h"], pre_scale, start, count, fps, fp32,
                            scaler, deband, deband_grain, dither)
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


def decode_frames(path, on_progress=None, pre_scale=1.0, fp32=False, scaler=None,
                  deband=False, deband_grain=None, dither=""):
    """解成 NHWC（uint8 或 float32）。pre_scale<1 用 libplacebo(spline36) 在核显上降采样。"""
    info = probe(path)
    w, h = info["w"], info["h"]
    if w <= 0 or h <= 0:
        raise RuntimeError("读不到视频尺寸，文件可能不是视频")
    cmd, w, h = _decode_cmd(path, w, h, pre_scale, fp32=fp32, scaler=scaler,
                            deband=deband, deband_grain=deband_grain, dither=dither)
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


PREVIEW_WIDTH = int(os.environ.get("FLASHVSR_PREVIEW_WIDTH", "1280"))
PREVIEW_FORMAT = os.environ.get("FLASHVSR_PREVIEW_FORMAT", "avif").lower()
_PREVIEW_ENC = None


def preview_encoder(fmt=None):
    """返回 (ffmpeg 参数, 扩展名)；本机不支持就依次回退。结果缓存。"""
    global _PREVIEW_ENC
    fmt = (fmt or PREVIEW_FORMAT).lower()
    order = [fmt] + [f for f in ("avif", "webp", "jpg") if f != fmt]

    def args_for(f):
        if f == "avif":
            return (["-c:v", "libsvtav1", "-crf", "28", "-still-picture", "1"], ".avif")
        if f == "webp":
            return (["-c:v", "libwebp", "-quality", "90"], ".webp")
        if f == "png":
            return (["-c:v", "png"], ".png")
        return (["-c:v", "mjpeg", "-q:v", "2"], ".jpg")

    if _PREVIEW_ENC is not None and _PREVIEW_ENC[0] == fmt:
        return _PREVIEW_ENC[1], _PREVIEW_ENC[2]
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work", "_pv_probe")
    os.makedirs(probe, exist_ok=True)
    for f in order:
        a, ext = args_for(f)
        dst = os.path.join(probe, "t" + ext)
        try:
            r = subprocess.run(
                [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x64:d=0.1",
                 "-frames:v", "1"] + a + [dst], capture_output=True, timeout=60)
            # 64x64 的 webp 只有 60 字节，阈值给 0 就行（只看能否编出来）
            if r.returncode == 0 and os.path.isfile(dst) and os.path.getsize(dst) > 0:
                _PREVIEW_ENC = (fmt, a, ext)
                return a, ext
        except Exception:  # noqa
            pass
        finally:
            try:
                os.remove(dst)
            except OSError:
                pass
    _PREVIEW_ENC = (fmt, ["-c:v", "mjpeg", "-q:v", "2"], ".jpg")
    return _PREVIEW_ENC[1], _PREVIEW_ENC[2]


def preview_ext():
    return preview_encoder()[1]


def extract_preview(path, idx, fps, out_path, width=None, fmt=None):
    """从分段文件里抽一帧存成 jpg（预览用）。

    宽度默认 1280 且不超过原片宽度：以前是 640，而页面用 width:100% 铺满卡片，
    宽屏下等于把 640 的图放大插值 → 看着糊。
    """
    w = int(width or PREVIEW_WIDTH)
    t = max(0.0, float(idx) / max(float(fps), 1e-6))
    enc, _ext = preview_encoder(fmt)
    cmd = [FFMPEG, "-y", "-v", "error", "-ss", f"{t:.4f}", "-i", path,
           "-frames:v", "1", "-vf", f"scale='min({w},iw)':-2"] + enc + [out_path]
    subprocess.run(cmd, capture_output=True, timeout=120)
    return os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def encode_frames(frames_u8, fps, out_path, audio_src=None, quality=15, backend="cpu",
                  bitdepth=8, svt_preset=None, tune=None, extra=None):
    h, w = frames_u8.shape[1], frames_u8.shape[2]
    cmd = ([FFMPEG, "-y", "-v", "error"] + encoder_global_args(backend)
           + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
              "-r", f"{fps:.6f}", "-i", "-"])
    if audio_src:
        # 音轨和字幕流都无损 copy（字幕没有就静默跳过）
        cmd += ["-i", audio_src, "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?",
                "-c:a", "copy", "-c:s", "copy", "-shortest"]
    cmd += encoder_args(backend, quality, bitdepth, svt_preset, tune, extra) + \
        ["-fps_mode", "cfr", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(0, len(frames_u8), 8):
        proc.stdin.write(frames_u8[i:i + 8].tobytes())
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", "ignore")
    if proc.wait() != 0:
        raise RuntimeError(f"{backend} 编码失败：" + err.strip()[-400:])


def concat_segments(seg_paths, out_path, audio_src=None, backend="cpu", quality=15,
                    bitdepth=8, tune=None, extra=None):
    """拼接分段，**一次成型**：concat 与音轨封装放在同一条 ffmpeg 命令里。

    改之前是"分段 → concat_video.mp4 → 再封装音轨"：一个任务写盘量 = 输出大小的 3 倍
    （分段 1x + 中间文件 1x + 成品 1x），中间文件还留在 work/ 里。现在压到 2x
    （分段 + 成品），不再产生中间文件。对机械盘 / QLC SSD 的寿命影响明显。
    """
    if not seg_paths:
        raise RuntimeError("没有可拼接的分段")
    tmp_dir = os.path.dirname(seg_paths[0])
    list_path = os.path.join(tmp_dir, "segments.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for p in seg_paths:
            f.write("file '%s'\n" % p.replace("\\", "/").replace("'", "'\\''"))

    base = ([FFMPEG, "-y", "-v", "error"] + encoder_global_args(backend)
            + ["-fflags", "+genpts", "-f", "concat", "-safe", "0", "-i", list_path])
    if audio_src:
        base += ["-i", audio_src]

    def _ok(*args):
        return subprocess.run(args, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")

    def _done():
        return os.path.exists(out_path) and os.path.getsize(out_path) > 1024

    if audio_src:
        for maps in (["-map", "0:v:0", "-map", "1:a?", "-map", "1:s?"],
                     ["-map", "0:v:0", "-map", "1:a?"]):
            r = _ok(*(base + maps + ["-c", "copy", "-shortest", out_path]))
            if r.returncode == 0 and _done():
                return
        r = _ok(*(base + ["-map", "0:v:0", "-map", "1:a?"]
                  + encoder_args(backend, quality, bitdepth, None, tune, extra)
                  + ["-c:a", "copy", "-shortest", out_path]))
        if r.returncode == 0 and _done():
            return
    else:
        r = _ok(*(base + ["-c", "copy", out_path]))
        if r.returncode == 0 and _done():
            return
        r = _ok(*(base + encoder_args(backend, quality, bitdepth, None, tune, extra) + [out_path]))
        if r.returncode == 0 and _done():
            return

    # 全失败才退回"中间文件"路径，并且立刻删掉中间文件
    silent = os.path.join(tmp_dir, "concat_video.mp4")
    r = _ok(*([FFMPEG, "-y", "-v", "error"] + encoder_global_args(backend)
              + ["-fflags", "+genpts", "-f", "concat", "-safe", "0", "-i", list_path,
                 "-c", "copy", silent]))
    if r.returncode != 0 or not os.path.exists(silent):
        raise RuntimeError("分段拼接失败：" + (r.stderr or "")[-400:])
    if audio_src:
        r = _ok(*([FFMPEG, "-y", "-v", "error", "-i", silent, "-i", audio_src,
                   "-map", "0:v:0", "-map", "1:a?", "-c", "copy", "-shortest", out_path]))
        if r.returncode != 0:
            shutil.move(silent, out_path)
        else:
            try:
                os.remove(silent)
            except OSError:
                pass
    else:
        shutil.move(silent, out_path)


def prune_segments(root, keep_days=None, max_dirs=None):
    """清理过期的分段缓存（续跑用）。默认保留 7 天 / 最近 8 个任务。

    分段文件 = 整个输出大小，不清会一直涨；反复写盘对 SSD 寿命也不友好。
    """
    if keep_days is None:
        keep_days = float(os.environ.get("FLASHVSR_SEG_KEEP_DAYS", "7"))
    if max_dirs is None:
        max_dirs = int(os.environ.get("FLASHVSR_SEG_KEEP_DIRS", "8"))
    removed = 0
    try:
        if not os.path.isdir(root):
            return 0
        paths = [os.path.join(root, d) for d in os.listdir(root)
                 if os.path.isdir(os.path.join(root, d))]
        paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        now = time.time()
        for i, p in enumerate(paths):
            if ((keep_days > 0 and (now - os.path.getmtime(p)) > keep_days * 86400)
                    or (max_dirs > 0 and i >= max_dirs)):
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
    except Exception:
        return removed
    return removed


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
        # 预设只是"默认值"：表单里显式给了 tile/kv 就以表单为准
        # （面板上 tile/kv 常驻可编辑；CLI 只有 --preset custom 时才带这两个字段）
        try:
            if fields.get("tile"):
                tile = int(fields["tile"])
            if fields.get("kv"):
                kv = float(fields["kv"])
        except (TypeError, ValueError):
            pass
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

    # 按**输入**分辨率自动选 tile：输入小 → 192（GEMM 的 M 更大、tile 更少）；
    # 输入大 → 保持 128（1080p 上 tile192 峰值 6636 MiB，余量只剩 1.5 GB）。
    # 输出分辨率影响的是 LQ 缓冲与画布，由下面的自动分段（chunk）兜。
    if (os.environ.get("FLASHVSR_AUTO_TILE", "0") == "1" and src_path
            and preset in ("quality", "balance") and not fields.get("tile")):
        try:
            _info = probe(src_path)
            _iw = int(round(_info["w"] * pre_scale))
            _ih = int(round(_info["h"] * pre_scale))
            if _iw * _ih <= 960 * 600:
                tile = 192 if tile < 192 else tile
                if tile % step:
                    tile = max(64, int(round(tile / step)) * step)
        except Exception:
            pass

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
    note_nv = ""
    if quant in ("nvfp4", "mixed"):
        # 老配置兜底：落到 MXFP8（显存接近、画质高 6 dB），不要静默变 bf16（更容易 OOM）
        note_nv = f"{quant.upper() if quant == 'nvfp4' else '混合档'} 已停用（画质差），本次改用 MXFP8"
        quant = "mxfp8"
    if quant not in QUANT_CHOICES:
        quant = "none"
    budget = fields.get("budget") or "auto"
    if budget not in ("auto", "fixed"):
        budget = "auto"
    if note_nv:
        auto_note = ((auto_note + "；") if auto_note else "") + note_nv
    preview = str(fields.get("preview", "1")).lower() not in ("0", "false", "off", "no")
    # 默认 CRF 按编码器定：AV1 实测 23 已过肉眼阈值（再往下只增码率），其余保持 15
    qdef = 23 if (fields.get("encoder") in ("svtav1", "av1_nvenc")) else 15
    try:
        quality = int(float(fields.get("quality") or 0))
    except ValueError:
        quality = 0
    if not quality:
        quality = qdef
    quality = max(10, min(40, quality))

    bitdepth = 10 if str(fields.get("bitdepth") or "8").strip() == "10" else 8
    hi_precision = str(fields.get("hi_precision") or "0").strip() in ("1", "true", "on", "yes")
    scaler = str(fields.get("prescale_scaler") or "").strip()
    if scaler not in SCALERS:
        scaler = os.environ.get("FLASHVSR_PRESCALE_SCALER", DEFAULT_SCALER)
        if scaler not in SCALERS:
            scaler = DEFAULT_SCALER
    prescale_deband = str(fields.get("prescale_deband") or "0").strip() in ("1", "true", "on", "yes")
    deband_grain = fields.get("deband_grain")
    try:
        deband_grain = None if deband_grain in (None, "") else float(deband_grain)
    except (TypeError, ValueError):
        deband_grain = None
    prescale_dither = str(fields.get("prescale_dither") or "").strip()
    if prescale_dither not in DITHERS:
        prescale_dither = ""
    tune = str(fields.get("tune") or "").strip().lower()
    if tune not in TUNE_ALL:
        tune = ""
    extra_args = str(fields.get("extra_args") or "").strip()[:300]
    try:
        max_frames = max(0, int(fields.get("max_frames") or 0))
    except (TypeError, ValueError):
        max_frames = 0
    color_fix = str(fields.get("color_fix") or "1").strip() not in ("0", "false", "off", "no")
    seam_fix = str(fields.get("seam_fix") or "smooth").strip().lower()
    if seam_fix not in ("off", "smooth", "align"):
        seam_fix = "smooth"
    # 复用模式：off / tile（静止 tile 跳过解码，54.9 dB）/ window（整窗跳过 DiT，50.6 dB）
    reuse = str(fields.get("reuse") or os.environ.get("FLASHVSR_REUSE", "off")).strip().lower()
    if reuse not in ("off", "tile", "window"):
        reuse = "off"
    return dict(scale=scale, tile=tile, kv=kv, chunk=chunk, mode=mode,
                attention=attention, encoder=encoder, quality=quality,
                quant=quant, budget=budget, pre_scale=pre_scale,
                tile_overlap=tile_overlap, bitdepth=bitdepth,
                hi_precision=hi_precision, color_fix=color_fix, seam_fix=seam_fix,
                max_frames=max_frames, tune=tune, extra_args=extra_args,
                prescale_scaler=scaler, prescale_deband=prescale_deband,
                deband_grain=deband_grain, prescale_dither=prescale_dither,
                reuse=reuse, preview=preview, auto_note=auto_note)


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

        install_stream_loader()
        self.pipe = self.mod.init_pipeline("FlashVSR-v1.1", mode, "cuda:0", torch.bfloat16)

        import gc as _gc
        state = {"n": 0}

        def _fast_clean_vram():
            # 实测（8 帧 / tile128 / 2×）：每个 tile 做 gc+empty_cache 会让整轮慢 7%
            # （去掉它 6.29s → 5.84s）；只做 gc.collect() 更慢 60%（它要遍历整个对象图）。
            # empty_cache 会与驱动同步、把缓存还给系统，随后又得重新申请，纯负收益。
            # 因此默认空操作；真要回收只在段与段之间做（见 upscale_video 的段循环）。
            state["n"] += 1
            if os.environ.get("FLASHVSR_AGGRESSIVE_GC") == "1" and state["n"] % 8 == 0:
                _gc.collect()
                torch.cuda.empty_cache()

        self.mod.clean_vram = _fast_clean_vram

        # VAE 解码走 NHWC：实测 625 → 540 ms（−13.6%），输出逐位一致（maxdiff 0）
        try:
            if os.environ.get("FLASHVSR_VAE_CL", "1") == "0":
                raise RuntimeError("channels_last disabled by FLASHVSR_VAE_CL=0")
            dec = self.pipe.TCDecoder
            dec.to(memory_format=torch.channels_last)
            _orig_decode = dec.decode_video

            # ---- 静止 tile 复用（FLASHVSR_TILE_REUSE=1 开启）------------------
            # 依据（实测）：静止块解码后的 RGB 跨窗差 0.26–0.41/255（< 1 LSB），
            # 运动块 12.6–18.7/255。判据必须用 cond（模型输入侧的 LQ），不能用 latent
            # —— 静止块的 latent 也差 3.5–4%，噪声结构被解码器抹掉了。
            _ru = {
                "on": os.environ.get("FLASHVSR_TILE_REUSE", "0") == "1",
                "thr": float(os.environ.get("FLASHVSR_REUSE_THR", "0.002")),
                "max_run": int(os.environ.get("FLASHVSR_REUSE_MAX_RUN", "8")),
                "prev": None, "out": None, "run": 0, "skip": 0, "n": 0,
            }
            self._reuse_stats = _ru

            def _decode_cl(x, parallel=True, show_progress_bar=False, cond=None):
                x = x.contiguous(memory_format=torch.channels_last_3d)
                if cond is not None:
                    cond = cond.contiguous(memory_format=torch.channels_last_3d)
                _ru["n"] += 1
                if os.environ.get("FLASHVSR_WIN_DEBUG") == "1":
                    print(f"[win-debug] decode #{_ru['n']} latent {tuple(x.shape)} "
                          f"cond {tuple(cond.shape) if cond is not None else None}",
                          flush=True)
                wst = getattr(self, "_win", None)
                if wst is not None and wst.get("flag"):
                    wst["flag"] = False
                    if _ru["out"] is not None and _ru.get("out_t") == x.size(1):
                        wst["reused_frames"] = wst.get("reused_frames", 0) + 1
                        return _ru["out"]
                    # 形状对不上就不冒险：老老实实解码这一窗
                    print("[win-reuse] 形状不匹配，本窗退回真实解码", flush=True)
                if (getattr(self, "_reuse_tile", False) and _ru["prev"] is not None
                        and cond is not None
                        and cond.shape == _ru["prev"].shape and _ru["run"] < _ru["max_run"]):
                    d = (cond.float() - _ru["prev"].float()).abs().mean().item()
                    if d < _ru["thr"]:
                        _ru["skip"] += 1
                        _ru["run"] += 1
                        return _ru["out"]
                _ru["run"] = 0
                out = _orig_decode(x, parallel=parallel, show_progress_bar=show_progress_bar,
                                   cond=cond)
                _ru["prev"] = cond
                _ru["out"] = out
                _ru["out_t"] = x.size(1)
                return out

            dec.decode_video = _decode_cl
        except Exception as e:  # noqa
            print(f"[FlashVSR] channels_last 未启用: {e}", flush=True)

        # 抽帧复用：噪声必须按**完整时间轴**生成再抽样，否则帧数一变整段随机数全变
        # （A/B 会比出 30 dB 的假损失）。这里包一层 generate_noise。
        try:
            _orig_noise = self.pipe.generate_noise
            self._noise_stride = 1

            def _noise(shape, seed=None, device=None, dtype=None):
                p = int(getattr(self, "_noise_stride", 1) or 1)
                if p <= 1:
                    return _orig_noise(shape, seed=seed, device=device, dtype=dtype)
                full_f = (shape[2] - 1) * p + 1
                full = _orig_noise((shape[0], shape[1], full_f) + tuple(shape[3:]),
                                   seed=seed, device=device, dtype=dtype)
                return full[:, :, ::p][:, :, :shape[2]].contiguous()

            self.pipe.generate_noise = _noise
        except Exception as e:  # noqa
            print(f"[FlashVSR] 抽帧噪声对齐未启用: {e}", flush=True)

        # ---- 整窗复用：保持时间轴，只跳过 30 层 DiT 的计算 -------------------
        # 判据用 LQ_latents（LQ_proj_in 的输出，随窗口更新），命中则：
        #   ① 不跑 blocks；② kv cache 复制最后一块（时间轴不变）；③ noise_pred 返回 0
        #   ④ decode 包装器看到 flag 直接返回上一窗的 RGB
        # ③④ 必须共用同一个 flag，否则会把纯噪声解码成画面。
        try:
            pl_mod = sys.modules.get("flashvsr_node.src.pipelines.flashvsr_tiny_long")
            self._win = {"prev": None, "n": 0, "skip": 0, "flag": False,
                         "thr": float(os.environ.get("FLASHVSR_WIN_THR", "0.001")),
                         "on": os.environ.get("FLASHVSR_WIN_REUSE", "0") == "1"}
            if pl_mod is not None and not getattr(pl_mod, "_win_wrapped", False):
                _orig_fn = pl_mod.model_fn_wan_video
                st = self._win

                def _win_fn(dit, x, timestep, context, **kw):
                    lat = kw.get("LQ_latents")
                    dup = False
                    if not getattr(self, "_reuse_win", False):
                        st["n"] += 1
                        st["prev"] = [t.detach().clone() for t in lat] if lat is not None else None
                        st["flag"] = False
                        return _orig_fn(dit, x, timestep, context, **kw)
                    dmax = -1.0
                    if st["prev"] is not None and lat is not None and len(lat) == len(st["prev"]):
                        # 首窗与后续窗的 LQ_latents 时间长度不同（512 vs 1536），
                        # 形状不一致就不比、也不复用（安全优先）
                        same_shape = all(a.shape == b.shape for a, b in zip(lat, st["prev"]))
                        if same_shape:
                            dmax = 0.0
                            for a, b in zip(lat, st["prev"]):
                                dmax = max(dmax, float((a.float() - b.float()).abs().mean()))
                            dup = dmax < st["thr"]
                    if not dup:
                        st["prev"] = [t.detach().clone() for t in lat] if lat is not None else None
                        st["n"] += 1
                        out = _orig_fn(dit, x, timestep, context, **kw)
                        st["flag"] = False
                        return out

                    # --- 跳过 30 层：cache 复制最后一块，再裁到 kv_len ---
                    st["n"] += 1
                    st["skip"] += 1
                    st["flag"] = True
                    one = max(1, int(x.size(2)) // 2)
                    kvl = int(kw.get("kv_ratio") or 3)
                    for key in ("pre_cache_k", "pre_cache_v"):
                        lst = kw.get(key)
                        if not lst:
                            continue
                        for bi in range(len(lst)):
                            ck = lst[bi]
                            if ck is None:
                                continue
                            new = torch.cat([ck, ck[-one:]], dim=0)
                            if new.size(0) > kvl:
                                new = new[-kvl:]
                            lst[bi] = new
                    if st["n"] % 5 == 1:
                        print(f"[win-reuse] 命中（LQ 差 {dmax:.4f} < {st['thr']}）"
                              f" 累计 {st['skip']}/{st['n']}", flush=True)
                    return torch.zeros_like(x), kw.get("pre_cache_k"), kw.get("pre_cache_v")

                pl_mod.model_fn_wan_video = _win_fn
                pl_mod._win_wrapped = True
                print(f"[FlashVSR] 整窗复用已启用（阈值 {self._win['thr']}）", flush=True)
        except Exception as e:  # noqa
            print(f"[FlashVSR] 整窗复用未启用: {e}", flush=True)

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
            use_gpu_canvas = (os.environ.get("FLASHVSR_GPU_CANVAS", "1") != "0")
            if use_gpu_canvas:
                n_f, h_f, w_f = frames.shape[0], frames.shape[1], frames.shape[2]
                if not _gpu_canvas_fits(n_f, h_f, w_f, int(scale)):
                    use_gpu_canvas = False
            if use_gpu_canvas:
                out = flashvsr_gpu_canvas(
                    mod, self.pipe, frames, int(scale), bool(color_fix), tile,
                    int(tile_overlap), float(sparse), float(kv), int(local), 0)
            elif seam_fix == "align":
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
            return _quant8(out)
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
                    scaler=params.get("prescale_scaler"),
                    deband=bool(params.get("prescale_deband")),
                    deband_grain=params.get("deband_grain"),
                    dither=str(params.get("prescale_dither") or ""),
                    on_progress=lambda n: on_stage("decode", n, total, f"解码中 {n}/{total} 帧"))
                if abs(pre_scale - 1.0) > 1e-6:
                    info["w"] = max(2, int(round(info["w"] * pre_scale / 2)) * 2)
                    info["h"] = max(2, int(round(info["h"] * pre_scale / 2)) * 2)
            else:
                frames, info = decode_frames(
                    src, on_progress=lambda n: on_stage("decode", n, 0, f"解码中 {n} 帧"),
                    pre_scale=pre_scale, fp32=fp32, scaler=params.get("prescale_scaler"),
                    deband=bool(params.get("prescale_deband")),
                    deband_grain=params.get("deband_grain"),
                    dither=str(params.get("prescale_dither") or ""))
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
        drop_on = os.environ.get("FLASHVSR_FRAME_DROP", "0") == "1"
        reuse_mode = str(params.get("reuse") or "off")
        self._reuse_tile = reuse_mode in ("tile", "window")
        self._reuse_win = reuse_mode == "window"
        if reuse_mode != "off":
            on_stage("infer", 0, 0,
                     "复用已开启：" + ("整窗跳过 DiT（≈50.6 dB）" if reuse_mode == "window"
                                     else "静止 tile 跳过解码（≈54.9 dB）"))
        audio = src if info["has_audio"] else None
        quality = int(params.get("quality") or 15)
        bitdepth = int(params.get("bitdepth") or 8)
        # 能力探测用**输出分辨率**：老核显能编 H.264 不代表能编 4K
        caps = get_caps(w=int(info["w"]) * int(params["scale"]),
                        h=int(info["h"]) * int(params["scale"]))
        backend = resolve_encoder(params.get("encoder"), caps)
        if bitdepth == 10 and backend not in ("x265", "qsv"):
            # x265 与 hevc_qsv 能出 main10；其余（含 AMF/NVENC）只有 8bit
            on_stage("encode", 0, 0,
                     f"{backend} 不支持 10 bit，本次按 8 bit 输出（要 10 bit 请用 x265 或 qsv）")
            bitdepth = 8
        budget_mode = params.get("budget") or "auto"
        # 开发模式：用户自传编码参数（GPU 编码器的 -rc/-cq/-tune 之类没有统一入口）
        extra_args = []
        raw_extra = str(params.get("extra_args") or "").strip()
        if raw_extra:
            import shlex
            if len(raw_extra) > 300 or "\n" in raw_extra:
                on_stage("encode", 0, 0, "附加参数过长或含换行，已忽略")
            else:
                try:
                    extra_args = shlex.split(raw_extra, posix=False)
                    extra_args = [a.strip('"') for a in extra_args]
                    on_stage("encode", 0, 0, "已附加自定义编码参数：" + " ".join(extra_args))
                except ValueError as e:
                    on_stage("encode", 0, 0, f"附加参数解析失败，已忽略：{e}")

        def encode_win(frames_in, path, audio_src=None):
            """统一编码入口：沿 FALLBACK_CHAIN 降级（qsv→qsv_h264→x265→cpu），并记住能用的那个。

            这样"哪代核显支持什么"不需要我们预先知道——本机当场试，失败就换下一个，
            且只在第一次失败时付出探测成本（之后 backend 已经改成可用的那个）。
            """
            nonlocal backend, bitdepth
            primary = backend
            chain = (backend,) + fallback_chain(backend)
            seen, last = set(), "未知错误"
            for be in chain:
                if be in seen:
                    continue
                seen.add(be)
                bd = bitdepth
                if bd == 10 and be not in ("x265", "qsv"):
                    bd = 8                 # 只有 x265 / hevc_qsv 能出 10bit
                try:
                    # svtav1 若不是用户的选择（而是兜底），用 p8：4K 实测 2.7 s/50 帧、
                    # VMAF 96.22，p4 要 7.7 s 才 96.90 —— 兜底不值得为 0.7 dB 多等 3 倍
                    svt_p = "8" if (be == "svtav1" and be != primary) else None
                    encode_frames(frames_in, info["fps"], path, audio_src,
                                  quality=quality, backend=be, bitdepth=bd,
                                  svt_preset=svt_p, tune=params.get("tune"),
                                  extra=extra_args)
                    if be != backend:
                        on_stage("encode", 0, 0,
                                 f"编码器 {backend} 不可用，已改用 {be}（后续段沿用）")
                        backend, bitdepth = be, bd
                    return
                except RuntimeError as e:
                    last = (str(e).strip().splitlines() or ["未知错误"])[-1][:160]
                    on_stage("encode", 0, 0, f"{be} 编码失败：{last}")
            raise RuntimeError(f"所有编码器都失败，最后一次：{last}")

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
            p_drop = _drop_stride(frames) if drop_on else 1
            fin = frames[::p_drop] if p_drop > 1 else frames
            if p_drop > 1:
                on_stage("decode", 0, total,
                         f"抽帧复用：检测到 {p_drop} 拍，{total} → {len(fin)} 帧"
                         f"（省 {(1-len(fin)/max(1,total))*100:.0f}% 计算）")
            self._noise_stride = p_drop
            out_t, sp, lr, kk = run_win(
                fin, params["tile"], params["kv"],
                lambda c, t: on_stage("infer", c, t, f"tile {c}/{t}"))
            self._noise_stride = 1
            frames_out = _expand(out_t[:len(fin)], p_drop, total)
            on_stage("encode", 0, 1, f"编码中（sparse={sp} kv={kk} local={lr}）")
            encode_win(frames_out, dst, audio)
            on_stage("encode", 1, 1, f"编码完成（{backend}）")
            return dict(frames=total, fps=info["fps"], in_wh=[info["w"], info["h"]],
                        out_wh=[int(frames_out.shape[2]), int(frames_out.shape[1])], segments=0)

        src_sig = dict(size=os.path.getsize(src), hash=quick_hash(src))
        pkey = _param_key(params)
        root = seg_root_dir(params)
        # 顺手清掉过期分段缓存（默认 7 天 / 最近 8 个任务，可用环境变量调）
        pruned = prune_segments(root)
        if pruned:
            on_stage("seg", 0, 0, f"已清理 {pruned} 个过期分段缓存（省盘）")
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
        step = max(1, chunk - overlap)
        on_stage("seg", 0, n_seg,
                 f"共 {n_seg} 段：{total} 帧，{chunk} 帧/段"
                 f"（含 {overlap} 帧段间重叠 → 每段实际推进 {step} 帧）")
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
                                          scaler=params.get("prescale_scaler"),
                                          deband=bool(params.get("prescale_deband")),
                                          deband_grain=params.get("deband_grain"),
                                          dither=str(params.get("prescale_dither") or ""),
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

            n_in = len(win)
            p_drop = _drop_stride(win) if drop_on else 1
            win_in = win[::p_drop] if p_drop > 1 else win
            if p_drop > 1:
                on_stage("seg", i, n_seg,
                         f"段 {i + 1}/{n_seg} 抽帧：{p_drop} 拍，"
                         f"{n_in} → {len(win_in)} 帧（省 {(1-len(win_in)/n_in)*100:.0f}%）")

            def tile_cb(c, t, i=i):
                on_stage("seg", i, n_seg, f"段 {i + 1}/{n_seg} tile {c}/{t}")

            self._noise_stride = p_drop
            res, sp, lr, kk = run_win(win_in, params["tile"], params["kv"], tile_cb)
            self._noise_stride = 1
            res = _expand(res[:len(win_in)], p_drop, n_in)
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
                    name = f"seg_{i:04d}{preview_ext()}"
                    if extract_preview(seg_path, random.randrange(keep), info["fps"],
                                       os.path.join(pdir, name)):
                        pv = name
                except Exception as e:  # noqa
                    print(f"[preview] 第 {i + 1} 段抽帧失败: {e}", flush=True)

            on_stage("seg", i + 1, n_seg,
                     f"段 {i + 1}/{n_seg} 完成（sparse={sp} kv={kk} local={lr}）",
                     **({"preview": pv, "pv_cur": i + 1, "pv_total": n_seg} if pv else {}))

        on_stage("encode", 0, 1, f"拼接 {n_seg} 段 + 混入音轨")
        concat_segments(seg_paths, dst, audio_src=audio, backend=backend, quality=quality,
                        bitdepth=bitdepth, tune=params.get("tune"), extra=extra_args)
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
