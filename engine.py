# -*- coding: utf-8 -*-
"""FlashVSR 推理引擎（独立运行，不经 ComfyUI 前端/服务端）

把 ComfyUI 节点包当库用：桩掉 folder_paths / comfy.utils，以合成包名导入 nodes.py，
复用 init_pipeline / flashvsr。这一层补：ffmpeg 编解码、进度回调、长视频分段落盘+续跑、
运动感知预算、DiT 权重量化（MXFP8 / NVFP4）。
"""
import collections
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
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
# 只有三档：bf16 基准 / MXFP8 推荐 / W8A16 老卡兼容。
# NVFP4 与混合档（FFN 用 NVFP4）实测画质比 MXFP8 低 6 dB（44.65 dB vs 50.75 dB），
# 显存还更大（3.40 GiB vs 2.05 GiB），速度只在超大 token 才占优，已在 2026-11 删除实现；
# 旧配置里的 nvfp4/mixed 由 resolve_params 与 ensure_loaded 自动降级到 MXFP8。
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


def _inject_bt709_vf(enc_args, enabled=None):
    """rgb24 → yuv 编码前按 BT.709 做全范围→有限范围转换。

    历史问题：ffmpeg 对 rgb24 输入默认按 BT.601 转 yuv，1080p/4K 母版基本都是 BT.709，
    结果成片偏绿/发灰（改之前没有 -colorspace/-color_primaries/-color_trc 标记）。
    qsv/vulkan 的 -vf 是编码器必需的格式转换（format=nv12,hwupload…），不能整条替换，
    只能把色彩转换拼到它前面。
    """
    out = list(enc_args or [])
    if enabled is None:
        enabled = os.environ.get("FLASHVSR_ENC_BT709", "1") != "0"
    if not enabled:
        return out
    # setparams 必须带上：光写 -colorspace/-color_primaries 只有矩阵标记生效，
    # 实测 primaries/transfer 仍是 unknown（libx264 的 VUI 从帧侧数据取，不从编码器
    # 上下文取），播放器于是按 BT.601 解释 → 依旧偏色。
    conv = ("scale=in_range=full:out_color_matrix=bt709:out_range=tv,"
            "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv")
    idx = -1
    for i in range(len(out) - 1):
        if out[i] in ("-vf", "-filter:v", "-filter:v:0"):
            idx = i
            break
    if idx >= 0:
        out[idx + 1] = f"{conv},{out[idx + 1]}"
    else:
        out = ["-vf", conv] + out
    return out + ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


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
# 3: 注入 BT.709 色彩矩阵/范围转换（rgb24 → yuv 不再按默认 BT.601 偏色）
ENCODER_ARGS_REV = 3


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
    # 用 (i+0.5)/n 而不是 linspace(0,1)：ramp 首尾不再是 0。
    # 画布最外圈像素只被"这一块"覆盖，权重若为 0 就是除以 1e-6 → 一条黑边。
    t = (torch.arange(n, dtype=torch.float32) + 0.5) / n
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


# ---------------------------------------------------------------- 逐 tile 搬运优化
# 节点自带的 prepare_input_tensor（nodes.py:75-100）每帧要走一次
# CPU → GPU 上采样 → **D2H 回 CPU** → CPU 上 stack → 再被 engine 搬回 GPU。
# 220 个 tile × 9 帧 ≈ 1980 次设备同步/窗，且整条链路上多搬 2 倍字节。
# 这里换成全程留在 GPU 的等价实现（数值一致：同一 bicubic、同一 dtype 转换顺序）。
_PI_CROP_KEY = "_fs_pi_crop"          # 记录中心裁剪偏移 (top, left)，拼接时补回来
_FEATHER_CACHE = {}
_TILE_HK = [0]


def _feather_mask_nhwc(size, overlap, dtype, device, smooth=True):
    """羽化权重按 (h, w, band, dtype, device) 缓存：每 tile 现建+转置+dtype 转换是纯开销。"""
    key = (int(size[0]), int(size[1]), int(overlap), str(dtype), str(device), bool(smooth))
    m = _FEATHER_CACHE.get(key)
    if m is None:
        m = _feather_mask(size, overlap, smooth=smooth)
        m = m.permute(0, 2, 3, 1).to(dtype=dtype, device=device).contiguous()
        if len(_FEATHER_CACHE) > 24:
            # 逐条淘汰最老的，不要 clear()：tile 尺寸/窗口帧数来回变时全清会把热点
            # （同一尺寸的 mask 每个 tile 都要用）一起丢掉，紧接着重建 → 周期性抖动。
            for _old in list(_FEATHER_CACHE)[:8]:
                _FEATHER_CACHE.pop(_old, None)
        _FEATHER_CACHE[key] = m
    return m


def _tile_housekeeping(every=10):
    """每 tile 都 gc+empty_cache 实测慢 7%（HANDOFF §4）；这里只每 N 个 tile 清一次缓存。"""
    _TILE_HK[0] += 1
    if every and _TILE_HK[0] % every == 0:
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa
            pass


_PI_STAGE = {}          # pinned 双缓冲暂存：{(shape, device): {...}}
_PI_STAGE_MAX = 16


def _stage_uint8_to_gpu(image_tensor, device):
    """uint8 → float32（/255）并异步搬到 device，走 pinned 双缓冲。

    /255 必须在 CPU 上做：GPU 的 float32 除法与 CPU 有 1 ULP 差异，经 fp8 量化放大后
    整段输出 PSNR 只有 ~37 dB。但以前用 image_tensor.float() 临时分配的是 pageable 内存，
    H2D 的 non_blocking=True 形同虚设（pageable 拷贝是阻塞实现，还要内部再转一次 staging）。
    这里把 /255 写进 pinned 缓冲再做 H2D，并用两块缓冲轮换：

        填 B（CPU）的同时，GPU 在搬 A —— canvas 逐 tile 调用，天然有下一个 tile 可填。
        复用某块之前先等它上一次的拷贝事件，避免 CPU 覆盖 GPU 还在读的内存。

    形状一直不变（逐 tile 切片），所以两块缓冲能反复用，省掉每 tile 的分配/页锁开销。
    """
    dev = device if isinstance(device, torch.device) else torch.device(str(device))
    key = (tuple(image_tensor.shape), str(dev))
    if os.environ.get("FLASHVSR_PIN_STAGE", "1") == "0":
        # 退路：万一某台机器的 pinned 池有副作用（锁页内存配额、老驱动），一行关掉
        return image_tensor.float().div_(255.0).to(device=dev, non_blocking=True)
    st = _PI_STAGE.get(key)
    if st is None:
        bufs, evs = [], []
        for _ in range(2):
            try:
                b = torch.empty(image_tensor.shape, dtype=torch.float32, pin_memory=True)
            except Exception:  # noqa  锁定内存不足 → 退回 pageable（只是不再异步）
                b = torch.empty(image_tensor.shape, dtype=torch.float32)
            bufs.append(b)
            evs.append(torch.cuda.Event() if dev.type == "cuda" else None)
        if len(_PI_STAGE) >= _PI_STAGE_MAX:
            # 逐条淘汰：clear() 会把正在用的热点缓冲一起丢掉（下一窗立刻重建 + 重新锁页）。
            # 丢弃前先等它上一次的 H2D 拷贝做完，否则释放 pinned 内存时 GPU 还在读。
            for _old in list(_PI_STAGE)[:4]:
                _st = _PI_STAGE.pop(_old, None)
                if _st:
                    for _ev in _st.get("evs") or ():
                        try:
                            if _ev is not None:
                                _ev.synchronize()
                        except Exception:  # noqa
                            pass
        st = {"bufs": bufs, "evs": evs, "i": 0}
        _PI_STAGE[key] = st
    i = st["i"]
    st["i"] = 1 - i
    buf, ev = st["bufs"][i], st["evs"][i]
    if ev is not None:
        ev.synchronize()          # 上一次从这块缓冲搬的拷贝必须做完，才能覆盖它
    torch.div(image_tensor, 255.0, out=buf)
    dst = buf.to(device=dev, non_blocking=True)
    if ev is not None:
        ev.record()
    return dst


def _prepare_input_gpu(mod, image_tensor, device, scale=4, dtype=torch.bfloat16):
    """prepare_input_tensor 的 GPU 版：输入可以是 uint8 NHWC 或 float NHWC（[0,1]）。

    与节点版逐位一致的部分：F = largest_8n1_leq(N0+4)、bicubic(align_corners=False)、
    居中裁剪到 128 的倍数、先 to(dtype) 再 *2-1、输出 (1,3,F,tH,tW)。
    不同处：不做 D2H、不在 CPU 上 stack、不在末尾 clean_vram（由 canvas 每 N 个 tile 清一次）。
    """
    N0, h0, w0, _ = image_tensor.shape
    sW, sH, tW, tH = mod.compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=128)
    F = mod.largest_8n1_leq(N0 + 4)
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {N0 + 4}.")

    if image_tensor.dtype == torch.uint8:
        # 必须**在 CPU 上**做 /255：GPU 的 float32 除法与 CPU 有 1 ULP 差异
        # （实测一半元素差 5.96e-8），经 fp8 量化放大后整段输出 PSNR 只有 ~37 dB。
        # 只对"当前 tile 的切片"做，512 KB 级别，代价可忽略；整窗仍保持 uint8。
        # _stage_uint8_to_gpu 用 pinned 双缓冲，让 H2D 真异步（见其 docstring）。
        src = _stage_uint8_to_gpu(image_tensor, device)
    else:
        src = image_tensor.to(device=device, non_blocking=True)
        if src.dtype != torch.float32:
            src = src.float()

    idx = torch.arange(F, device=src.device).clamp_(max=max(N0 - 1, 0))
    x = src.index_select(0, idx).permute(0, 3, 1, 2)          # (F,3,h0,w0)
    x = torch.nn.functional.interpolate(x, size=(sH, sW), mode="bicubic", align_corners=False)
    left, top = max(0, (sW - tW) // 2), max(0, (sH - tH) // 2)
    x = x[:, :, top:top + tH, left:left + tW]
    vid = (x.to(dtype) * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
    setattr(mod, _PI_CROP_KEY, (top, left))
    return vid, tH, tW, F


def _install_gpu_prepare(mod):
    """幂等替换 prepare_input_tensor。FLASHVSR_GPU_PREPARE=0 可关。"""
    if getattr(mod.prepare_input_tensor, "_fs_gpu", False):
        return True
    orig = getattr(mod, "prepare_input_tensor", None)
    if orig is None:
        return False

    def prepare_input_tensor(image_tensor, device, scale=4, dtype=torch.bfloat16):
        return _prepare_input_gpu(mod, image_tensor, device, scale=scale, dtype=dtype)

    prepare_input_tensor._fs_gpu = True
    mod.prepare_input_tensor = prepare_input_tensor
    mod._fs_pi_orig = orig
    return True


def _uninstall_gpu_prepare(mod):
    orig = getattr(mod, "_fs_pi_orig", None)
    if orig is not None and getattr(mod.prepare_input_tensor, "_fs_gpu", False):
        mod.prepare_input_tensor = orig
        del mod._fs_pi_orig


# 当前正在解码的 tile 坐标（静止块复用按"位置"缓存而不是按"上一次 decode"）
_TILE_POS = [None]


def flashvsr_gpu_canvas(mod, pipe, frames, scale, color_fix, tile, tile_overlap,
                        sparse, kv, local, seed=0, seam_fix="smooth"):
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
    # 小视频的两个坑（都只在"窗口比 tile 小"的轴上出现）：
    #   1) prepare_input_tensor 的目标边长是 max(128, floor(s/128)*128)，s < 128 时它
    #      不补边、直接输出更小的 tile，却仍把 128 当 height/width 传给 pipe → 出图全黑；
    #   2) 只有一个 tile 时，居中裁掉的边距没有邻居 tile 来补 → 右/下留黑边。
    # 处理：把这条轴用边缘复制补到 128/scale 的整数倍，拼完再裁回原尺寸。
    # 窗口 >= tile 的轴上每块都正好是 tile 大小、边距为零，走原来的逻辑（不改 544p 等常规路径）。
    if isinstance(tile, (tuple, list)):
        _th_t, _tw_t = int(tile[0]), int(tile[1])
    else:
        _th_t = _tw_t = int(tile)
    unit = 128 // math.gcd(128, max(1, int(scale)))
    Hp = ((H + unit - 1) // unit) * unit if H < _th_t else H
    Wp = ((W + unit - 1) // unit) * unit if W < _tw_t else W
    if Hp != H or Wp != W:
        import torch.nn.functional as _F
        _x = frames.permute(0, 3, 1, 2)
        _x = _F.pad(_x, (0, Wp - W, 0, Hp - H), mode="replicate")
        frames = _x.permute(0, 2, 3, 1)
    # 累加精度：默认 fp32。实测同帧 fp16 vs fp32 只有 mean|d|=0.18/255（9% 像素），
    # 耗时差 <1%，但 fp16 在重叠区最多叠 4 块时会丢尾数，hi_precision 下更敏感。
    # FLASHVSR_CANVAS_FP32=0 可退回 fp16（显存占用减半，4K 输出省约 50 MiB/帧）。
    _acc = torch.float32 if os.environ.get("FLASHVSR_CANVAS_FP32", "1") != "0" else torch.float16
    canvas = torch.zeros((N, Hp * scale, Wp * scale, C), dtype=_acc, device=device)
    wsum = torch.zeros((N, Hp * scale, Wp * scale, 1), dtype=_acc, device=device)
    # run_win 可能传矩形 tile：(tile_h, tile_w)。以前这里直接 int(tile) → TypeError。
    if isinstance(tile, (tuple, list)):
        th_t, tw_t = int(tile[0]), int(tile[1])
    else:
        th_t = tw_t = int(tile)
    # 节点的 prepare_input_tensor 会把 tile 居中裁剪到 128 的倍数（nodes.py:53-71）。
    # tile*scale 不是 128 的倍数时，每块输出比自己的覆盖范围小 → 画布必然有空洞。
    sc = int(scale)
    if (th_t * sc) % 128 != 0 or (tw_t * sc) % 128 != 0:
        step = 128 // math.gcd(128, max(1, sc))
        if th_t == tw_t:
            print(f"[canvas] 警告：tile={tile}×scale={scale}={th_t * sc} 不是 128 的倍数，"
                  f"接缝处会有 %128 的错位/空洞（建议 tile 取 {step} 的倍数）", flush=True)
        else:
            print(f"[canvas] 警告：tile={th_t}×{tw_t}×scale={scale} 时 "
                  f"{th_t * sc}×{tw_t * sc} 不是 128 的倍数，"
                  f"接缝处会有 %128 的错位/空洞（建议 tile_h/tile_w 取 {step} 的倍数）", flush=True)
    coords = mod.calculate_tile_coords(Hp, Wp, tile, tile_overlap)
    if not coords:
        # calculate_tile_coords 在 H/W <= overlap 时返回空表（num_rows=0）→ 从这里出去就是整帧全黑
        coords = [(0, 0, Wp, Hp)]
    pipe_cls = getattr(mod, "FlashVSRTinyLongPipeline", None)
    _TILE_HK[0] = 0

    for x1, y1, x2, y2 in mod.cqdm(coords, desc="Processing Tiles"):
        input_tile = frames[:, y1:y2, x1:x2, :]
        LQ_tile, th, tw, F = mod.prepare_input_tensor(input_tile, device, scale=scale, dtype=dtype)
        if pipe_cls is not None and not isinstance(pipe, pipe_cls):
            LQ_tile = LQ_tile.to(device)
        # 静止块复用的"位置键"：必须在 pipe(...) 之前设，解码发生在其内部
        _TILE_POS[0] = (int(x1), int(y1), int(x2), int(y2))
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
        if tile_gpu.shape[0] != canvas.shape[0]:
            # pipe 内部还能再补帧（8n+1 对齐），多出来的尾帧不属于本窗
            tile_gpu = tile_gpu[:canvas.shape[0]]
        th_o, tw_o = int(tile_gpu.shape[1]), int(tile_gpu.shape[2])
        # seam_fix 在 GPU 画布上也要生效（以前只有 CPU 回退路径才看得到这个参数）：
        # off=硬边（不加羽化过渡），smooth=平滑羽化，align=羽化 + 逐 tile 亮度偏置对齐。
        mask_hw = _feather_mask_nhwc((th_o, tw_o), int(tile_overlap) * scale,
                                     tile_gpu.dtype, tile_gpu.device,
                                     smooth=(seam_fix != "off"))
        # prepare_input_tensor 会把 tile 居中裁剪到 128 的倍数，裁掉的偏移必须补回来，
        # 否则 tile 尺寸不是 128/scale 的倍数时每块都会整体偏移 + 右/下留黑边。
        ct, cl = getattr(mod, _PI_CROP_KEY, (0, 0))
        ox1, oy1 = x1 * scale + int(cl), y1 * scale + int(ct)
        ox2, oy2 = ox1 + tw_o, oy1 + th_o
        if seam_fix == "align":
            # 只在"重叠区且已有内容、且不在本 tile 的实心区"的像素上估每通道 DC 偏差，
            # 把整块 tile 平移到已有画布上，接缝两侧就没有亮暗台阶。
            _wp = wsum[:, oy1:oy2, ox1:ox2, :]
            _have = (_wp > 1e-3) & (mask_hw < 0.999)
            if int(_have.sum()) > 512:
                _cur = canvas[:, oy1:oy2, ox1:ox2, :] / _wp.clamp_min(1e-6)
                _sel = _have.expand(-1, -1, -1, canvas.shape[3])
                _d = (_cur.to(tile_gpu.dtype) - tile_gpu)[_sel].reshape(
                    -1, canvas.shape[3]).mean(dim=0)
                tile_gpu = tile_gpu + _d
        canvas[:, oy1:oy2, ox1:ox2, :] += (tile_gpu * mask_hw).to(_acc)
        wsum[:, oy1:oy2, ox1:ox2, :] += mask_hw.to(_acc)
        del LQ_tile, out_gpu, tile_gpu, mask_hw
        _TILE_POS[0] = None
        _tile_housekeeping()

    _debug_check_tensor("canvas", canvas)
    _debug_check_tensor("wsum", wsum)
    if float(wsum.min()) <= 1e-6:
        # 以前这里是静默的：没覆盖到的像素除完就是黑边，只有肉眼能发现
        print("[canvas] 警告：有像素未被任何 tile 覆盖（会是黑边），检查 tile/overlap/裁剪偏移",
              flush=True)
    out = canvas / wsum.clamp_min_(1e-6)
    out = out[:n0]
    if out.shape[1] != H * scale or out.shape[2] != W * scale:
        out = out[:, :H * scale, :W * scale, :]      # 去掉补边（复制的边缘像素）
    return out


def _drop_stride(frames, thr=0.5, blk_thr=None):
    """检测"按 p 拍画"（p=2/3）。返回 p 或 1（不抽）。

    判据：规整保留 i%p==0 时，**被丢掉的帧**是否都是前一张的复制：
    |Δ(i, i−1)| < thr。丢错率 ≥5% 就认为这条内容不是按 p 拍的，不抽。

    只允许 phase=0：这样插回就是 np.repeat(out, p)[:n]，不需要 keep 表。

    thr 是**全局平均** MAE：暗场里一小块局部微动（人物手指、火星、水面小波纹）
    被整幅图摊薄后很容易低于 thr，那一帧就被当成重复丢掉 → 微动消失。所以再加
    一条网格分块判据：任一 ≤8×8 分块的平均 MAE ≥ blk_thr 就算"这一帧变了"。
    两条判据（全局 / 局部）都不满足才算复制帧。
    """
    n = len(frames)
    if n < 8:
        return 1
    sub = frames[:, ::4, ::4, :3]
    if np.issubdtype(np.asarray(sub).dtype, np.floating):
        # hi_precision 解码给的是 [0,1] float32：astype(int16) 会把每个像素截成 0，
        # 帧差恒为 0 → 所有帧都算"复制帧"，抽帧会误删一半正常帧。
        g = np.asarray(sub, dtype=np.float32).mean(axis=-1) * 255.0
    else:
        g = np.asarray(sub, dtype=np.int16).mean(axis=-1)  # 低分辨率灰度
    d1 = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))                     # d1[i-1] = |f(i)-f(i-1)|
    if blk_thr is None:
        try:
            blk_thr = float(os.environ.get("FLASHVSR_DROP_BLK_THR", "2.0"))
        except ValueError:
            blk_thr = 2.0
    bmax = None
    h, w = g.shape[1], g.shape[2]
    if blk_thr > 0 and h >= 4 and w >= 4 and n > 1:
        d = np.abs(np.diff(g, axis=0))
        ny, nx = min(8, h), min(8, w)
        ys = np.linspace(0, h, ny + 1).astype(int)
        xs = np.linspace(0, w, nx + 1).astype(int)
        # 分块均值向量化：两次 reduceat（先按行段、再按列段）代替 8×8 次 Python
        # 切片循环，帧多/分辨率高时省的是解释器开销而不是数学。
        _sum = np.add.reduceat(np.add.reduceat(d, ys[:-1], axis=1), xs[:-1], axis=2)
        _cnt = np.maximum(np.outer(np.diff(ys), np.diff(xs)).astype(np.float64), 1.0)
        bmax = (_sum / _cnt).max(axis=(1, 2))
    if bmax is not None:
        # 全局差得少、但局部块变化明显 → 也算非复制帧
        changed = (d1 >= thr) | (bmax >= blk_thr)
    else:
        changed = d1 >= thr
    for p in (2, 3):
        if n <= p + 2:
            continue
        kept = set(range(0, n, p))
        lost = [changed[i - 1] for i in range(1, n) if i not in kept]
        if lost and float(np.mean(np.asarray(lost))) < 0.05:
            return p
    return 1


def _expand(out, p, n):
    return out if p <= 1 else np.repeat(out, p, axis=0)[:n]


# 流式加载时"哪些权重可以不读盘"的配置：必须跟着 (checkpoint, quant) 变。
# 历史 bug：install_stream_loader 初始化时把 mxfp8 的 skip 集合写死进闭包，
# 切成 quant="none"/"w8" 后仍然跳过这些权重、只填未初始化的 torch.empty → 花屏。
_STREAM_SKIP_CFG = {"ckpt": None, "keys": set()}


def set_stream_skip(quant="none"):
    """按当前 (checkpoint, quant) 重算流式加载的跳过集合（模块级，patched 每次读它）。"""
    ckpt, keys = None, set()
    try:
        if str(quant or "none").lower() == "mxfp8" and                 os.environ.get("FLASHVSR_QCACHE", "1") != "0":
            _ck = _dit_checkpoint()
            if _ck:
                ckpt = os.path.normcase(os.path.abspath(_ck))
                keys = _qcache_keys(_qcache_path_static("mxfp8"))
    except Exception as e:  # noqa
        print(f"[stream-load] 计算跳过集合失败（改为全部读盘）: {e}", flush=True)
        ckpt, keys = None, set()
    _STREAM_SKIP_CFG["ckpt"] = ckpt
    _STREAM_SKIP_CFG["keys"] = keys
    if keys:
        print(f"[stream-load] quant=mxfp8：{len(keys)} 个层命中量化缓存，加载时跳过读盘",
              flush=True)
    return len(keys)


def install_stream_loader(quant="none"):
    """把节点的 safetensors 加载改成"逐个张量读 → 立刻转 dtype → 立刻上卡"。

    原实现（utils.load_state_dict_from_safetensors）会攒一个完整的 CPU state_dict，
    实测 commit 峰值 **12.6 GB**（f32 视图 5.4 + bf16 副本 2.8 + 分配器缓存 + 显存），
    在 16 GB 内存的机器上加载时浏览器都得关掉。
    流式之后 CPU 侧同时只持有一个张量，峰值降到"一个最大的张量"量级。

    只影响加载路径，不改模型结构；失败就退回原实现。
    """
    # 即使已经 patch 过也要重算：切 quant（none/w8）后旧集合会让权重被错误跳过
    set_stream_skip(quant)
    try:
        utils = sys.modules.get("flashvsr_node.src.models.utils")
        if utils is None or getattr(utils, "_stream_patched", False):
            return False
        orig = utils.load_state_dict_from_safetensors

        # 缓存里已有的层：不读盘、不转 bf16，直接在显存里占位（真张量，不是 meta，
        # 这样节点的 model.to(dtype, device) 不会崩）。随后 quantize_dit_mxfp8 会用
        # 缓存里的 fp8 覆盖 forward 并把占位丢掉，所以这些空张量只活几秒。
        # 集合按文件判断：非本 DiT checkpoint 一律不跳过（否则别的模型也吃空张量）。
        dev_ph = "cuda" if torch.cuda.is_available() else "cpu"

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
            _tgt = _STREAM_SKIP_CFG["ckpt"]
            _is_dit = _tgt is not None and \
                os.path.normcase(os.path.abspath(file_path)) == _tgt
            skip = _STREAM_SKIP_CFG["keys"] if _is_dit else set()
            skipped_n = 0
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
                    _base = k[:-7] if k.endswith(".weight") else k
                    if _base in skip:
                        out[k] = torch.empty(info["shape"],
                                             dtype=torch_dtype or torch.bfloat16,
                                             device=dev_ph)
                        skipped_n += 1
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
            if skipped_n:
                print(f"[stream-load] 跳过 {skipped_n} 个已在缓存里的权重（不读盘、不转 bf16）",
                      flush=True)
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


_BAYER8 = {}


def _bayer8(device):
    """8×8 Bayer 矩阵：以前每窗都重建一次（tensor + repeat + view），提成常量。"""
    key = str(device)
    m = _BAYER8.get(key)
    if m is None:
        m = torch.tensor(
            [[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
             [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
             [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
             [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]],
            dtype=torch.float32, device=device) / 64.0 - 0.5
        _BAYER8[key] = m
    return m


def _debug_check_tensor(name, t):
    """FLASHVSR_DEBUG_CHECK=1 时打印张量的形状/设备/dtype/范围/NaN/Inf（不建梯度）。

    整套流程里"花屏/黑屏/发灰"可能出在解码、拼缝、量化任何一步，肉眼只能看到最终
    结果。插一行这个，就知道是哪一步把数据搞坏的（NaN 会一路传到输出成全黑）。
    """
    if os.environ.get("FLASHVSR_DEBUG_CHECK", "0") in ("", "0"):
        return
    if t is None:
        print(f"[check] {name}: None", flush=True)
        return
    try:
        with torch.no_grad():
            f = t.detach()
            finfo = torch.isfinite(f)
            nan = int((torch.isnan(f)).sum()) if f.is_floating_point() else 0
            inf = int((torch.isinf(f)).sum()) if f.is_floating_point() else 0
            fin = f[finfo]
            if fin.numel():
                rng = f"min={float(fin.min()):.6g} max={float(fin.max()):.6g} mean={float(fin.mean()):.6g}"
            else:
                rng = "min/max/mean=空"
            print(f"[check] {name}: shape={tuple(f.shape)} dtype={f.dtype} dev={f.device} "
                  f"{rng} nan={nan} inf={inf}", flush=True)
    except Exception as e:  # noqa
        print(f"[check] {name}: 检查失败 {type(e).__name__}: {str(e)[:200]}", flush=True)


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
        bayer = _bayer8(x.device)
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
        if tile_cpu.shape[0] != canvas.shape[0]:
            tile_cpu = tile_cpu[:canvas.shape[0]]      # 8n+1 补齐多出来的尾帧不属本窗

        mask = _feather_mask((tile_cpu.shape[1], tile_cpu.shape[2]),
                             int(tile_overlap) * scale, smooth=True).permute(0, 2, 3, 1)
        # prepare_input_tensor 的居中裁剪偏移：和 flashvsr_gpu_canvas 用同一套补偿，
        # 否则 GPU 画布关掉（显存不足回退）时每块整体偏 (cl, ct) 且右/下留黑边。
        _ct, _cl = getattr(mod, _PI_CROP_KEY, (0, 0))
        ox1, oy1 = x1 * scale + int(_cl), y1 * scale + int(_ct)
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


def _dit_checkpoint():
    """定位 DiT 的 safetensors（缓存键要用它的大小/时间，必须加载前就能拿到）。"""
    try:
        import config as C
        base = getattr(C, "MODEL_PATH", None)
        files = list(getattr(C, "MODEL_FILES", []) or [])
        if base:
            cand = [f for f in files if "streaming_dmd" in f or f.endswith(".safetensors")]
            for f in cand:
                p = os.path.join(base, f)
                if os.path.isfile(p):
                    return p
            for f in sorted(os.listdir(base)):
                if f.endswith(".safetensors"):
                    return os.path.join(base, f)
    except Exception:
        pass
    return os.environ.get("FLASHVSR_DIT_PATH") or None


def _qcache_path_static(quant):
    path = _dit_checkpoint()
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = f"dit_{st.st_size}_{int(st.st_mtime)}_{quant}_v{QCACHE_REV}"
    return os.path.join(_qcache_dir(), key + ".safetensors")


def _qcache_keys(path):
    """只读 safetensors 头，拿到缓存里有哪些层（不加载张量）。"""
    if not path or not os.path.exists(path):
        return set()
    try:
        import json
        import struct
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        return {k[:-2] for k in hdr if k.endswith("|q")}
    except Exception:
        return set()


def _qcache_path(pipe, quant):
    return _qcache_path_static(quant)


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
    # 流式加载已经把缓存里那些层换成**未初始化**的 torch.empty（省读盘）。此时若缓存
    # 读不出来（文件截断 / 显存不够 / 版本不符），下面会把随机垃圾量化再落盘，之后每次
    # 运行都从这份垃圾起跳 → 永久花屏。宁可报错并删掉坏缓存，让它下次从原模型重量化。
    _skip_keys = set(_STREAM_SKIP_CFG.get("keys") or ())
    if _skip_keys:
        _have = set((cached or {}).keys())
        _miss = _skip_keys - _have
        if _miss:
            if cache_path and os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    print(f"[FlashVSR] 量化缓存不完整，已删除 {os.path.basename(cache_path)}"
                          f"（下次运行将从原模型重新量化）", flush=True)
                except OSError as e:
                    print(f"[FlashVSR] 删除损坏的量化缓存失败：{e}", flush=True)
            raise RuntimeError(
                f"MXFP8 量化缓存不完整（缺 {len(_miss)} 层），而权重是按「跳过读盘」"
                f"加载的，继续跑会把未初始化显存量化后写回缓存。已清理缓存，请重新运行。")
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
    # 编码器/画质/位深同样改产物：换编码器后旧分段 concat -c copy 会直接失败，
    # 画质不同则前后撕裂。历史上这三项漏在键外，改完参数仍命中旧分段。
    d["encoder"] = str(params.get("encoder") or "auto")
    d["quality"] = int(params.get("quality") or 15)
    d["bitdepth"] = int(params.get("bitdepth") or 8)
    # 矩形 tile 会改接缝位置；budget=auto 会按运动量改 sparse/local/kv ——
    # 这两样历史上都漏在键外，切档后续跑会混用旧分段。
    d["tile_h"] = int(params.get("tile_h") or 0)
    d["tile_w"] = int(params.get("tile_w") or 0)
    d["budget"] = str(params.get("budget") or "auto")
    # 抖动和抽帧是**环境变量**，但都会改画面/分段边界 → 必须进键，否则改完还命中旧缓存
    d["dither"] = (os.environ.get("FLASHVSR_DITHER") or "off").lower()
    d["drop"] = os.environ.get("FLASHVSR_FRAME_DROP", "0") == "1"
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
def _probe_uncached(path):
    """读容器/流信息（未缓存版）。w/h 是**显示朝向**尺寸（手机竖拍的 rotation 已换算）。"""
    try:
        out = subprocess.check_output(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames",
             "-show_entries", "stream_side_data=rotation",
             "-show_entries", "format=duration", "-of", "json", path],
            stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or b"").decode("utf-8", "ignore").strip().splitlines()
        raise RuntimeError(
            f"无法读取视频 {os.path.basename(str(path))}：文件损坏、不是视频，或编码不受支持"
            + (f"（{tail[-1][:160]}）" if tail else "")) from e
    except FileNotFoundError as e:
        raise RuntimeError(f"找不到 ffprobe（{FFPROBE}），请检查 ffmpeg 安装路径") from e
    j = json.loads(out.decode("utf-8", "ignore"))
    st = (j.get("streams") or [{}])[0]
    w, h = int(st.get("width", 0)), int(st.get("height", 0))
    # Display Matrix 的 rotation：ffmpeg 自己的 autorotate 语义是 90→transpose=2、
    # 270→transpose=1、180→hflip,vflip（本机实测与 -vf 逐像素一致）。默认 0。
    rot = 0
    for sd in (st.get("side_data_list") or []):
        try:
            if sd.get("rotation") is not None:
                rot = int(round(float(sd["rotation"]))) % 360
                break
        except (TypeError, ValueError):
            continue
    if rot not in (0, 90, 180, 270):
        rot = 0
    if rot in (90, 270) and w > 0 and h > 0:
        # 编码朝向 64×48 + rotation 90 → 显示是 48×64：处处按显示尺寸处理，
        # 否则 in_wh/画布/预览全是侧躺的，成片方向也错。
        w, h = h, w
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
    return dict(w=w, h=h, fps=fps, duration=dur, frames=frames, has_audio=has_audio,
                rot=rot)


# probe 在一条分段流式任务里会被反复调用（源头一次、每个窗口一次、每段产物校验一次），
# 每次起 1~2 个 ffprobe 子进程（正片 30 段 ≈ 60 次进程启动）。按 (realpath, size,
# mtime_ns) 做进程内 memoize：文件一旦被重写（原子改名、续跑重编）size/mtime 就变，
# 键自然失效，不会把旧几何当成新的。返回副本，避免调用方改到缓存里的 dict。
_PROBE_CACHE = {}
_PROBE_CACHE_MAX = 32


def probe(path):
    """读容器/流信息（带缓存）；坏文件照旧抛 RuntimeError。"""
    try:
        st = os.stat(path)
        key = (os.path.realpath(str(path)), int(st.st_size), int(st.st_mtime_ns))
    except OSError:
        return _probe_uncached(path)
    hit = _PROBE_CACHE.get(key)
    if hit is not None:
        return dict(hit)
    info = _probe_uncached(path)
    if len(_PROBE_CACHE) >= _PROBE_CACHE_MAX:
        for _old in list(_PROBE_CACHE)[:8]:
            _PROBE_CACHE.pop(_old, None)
    _PROBE_CACHE[key] = dict(info)
    return info


# 非 fp32 解码时 libplacebo 的输出/下载格式。bgra = 着色器里直接转 rgb（4:4:4、全精度），
# 比旧写法的 yuv420p 中转少一次色度对半砍。本机 ffmpeg 9.0.1 实测 bgra/rgba/yuv444p 可用，
# 但 hwdownload,format=rgb24 不行（vulkan 没有 24 位格式），所以用 bgra 再交给输出 pix_fmt rgb24。
# 万一某个驱动不支持，可用 FLASHVSR_DEC_DL=yuv420p 退回旧行为。
_DEC_DL = os.environ.get("FLASHVSR_DEC_DL", "bgra").lower()
if _DEC_DL not in ("bgra", "rgba", "yuv444p", "yuv420p"):
    _DEC_DL = "bgra"

# 解码时的 fps 模式。默认 passthrough（按真实帧出图）。
# 历史问题：分段 mp4 在"各段帧数不等"时用 -c copy 拼接后，容器里个别帧的
# duration 会被写成 3 帧（0.04s → 0.12s，见 HANDOFF §11）。ffmpeg 默认按 CFR 补帧，
# 会把这个空档当丢帧补回来——9 帧的成片能解出 11 帧，引擎校验就会报"帧数不符"。
# 设 passthrough 后按真实帧出图；真要 CFR 行为可设 FLASHVSR_DEC_FPS=cfr（设空值关掉）。
_DEC_FPS = os.environ.get("FLASHVSR_DEC_FPS", "passthrough").strip()


def _start_stderr_drain(pipe, maxlen=40):
    """后台线程排空子进程 stderr，只保留最后 maxlen 行。

    历史 bug：decode/encode 只读 stdout/stdin，最后才 read() stderr。ffmpeg 的警告/错误
    写满管道缓冲（Windows 4–64 KB）后，ffmpeg 自己阻塞在写 stderr 上，父进程则阻塞在
    读 stdout（或写 stdin）→ 双方永久卡死，表现是任务"无声挂住"。
    """
    buf = collections.deque(maxlen=max(1, int(maxlen)))

    def _pump():
        try:
            for line in iter(pipe.readline, b""):
                buf.append(line.decode("utf-8", "ignore").rstrip())
        except Exception:  # noqa
            pass
        finally:
            try:
                pipe.close()
            except Exception:  # noqa
                pass

    th = threading.Thread(target=_pump, name="ffmpeg-stderr", daemon=True)
    th.start()
    return th, buf


def _stderr_tail(buf, n=8):
    return "\n".join([x for x in list(buf)[-n:] if x])


def _decode_cmd(path, w, h, pre_scale, start_frame=None, count=None, fps=25.0, fp32=False,
                scaler=None, deband=False, deband_grain=None, dither="", fps_mode=None):
    """拼 ffmpeg 命令；start_frame/count 给定时只解这一段（输入侧 -ss + 精确 seek）。

    fp32=True 时走 libplacebo 的 gbrpf32le 输出：源精度一路保留到模型输入，
    不再经过 rgb24 的 8bit 量化（代价是每帧内存 4×）。

    w/h 传**显示朝向**尺寸（probe 已按 rotation 互换过）。ffmpeg 对 rawvideo 输出
    默认也会 autorotate（本机实测：rotation=90 的输出与 transpose=2 逐像素一致），
    所以这里不能再加 transpose，否则会二次旋转；显式写 -autorotate 1 把这个隐式
    依赖挑明。autorotate 的滤镜插在用户滤镜链之前，libplacebo 拿到的是显示朝向。
    """
    vf = None
    scaler = scaler if scaler in SCALERS else DEFAULT_SCALER
    extra = ""                      # 附加的 libplacebo 选项（去色带 / 抖动）
    if deband:
        extra += ":deband=1"
        if deband_grain is not None:
            extra += f":deband_grain={float(deband_grain):g}"
    if dither and dither in DITHERS:
        extra += f":dithering={dither}"
    tw, th = w, h
    if abs(pre_scale - 1.0) > 1e-6:
        tw = max(2, int(round(w * pre_scale / 2)) * 2)
        th = max(2, int(round(h * pre_scale / 2)) * 2)
    if fp32:
        vf = (f"libplacebo=w={tw}:h={th}:downscaler={scaler}{extra}:format=gbrpf32le,"
              f"hwdownload,format=gbrpf32le")
    elif abs(pre_scale - 1.0) > 1e-6 or extra:
        # 旧写法是 hwdownload,format=yuv420p 再转 rgb24：10bit / 4:2:2 / 4:4:4 源先被
        # 砍成 8bit 4:2:0，彩边和细纹理发灰。改成让 libplacebo 在着色器里直接输出 rgb。
        vf = (f"libplacebo=w={tw}:h={th}:downscaler={scaler}{extra}:format={_DEC_DL},"
              f"hwdownload,format={_DEC_DL}")
    w, h = tw, th
    cmd = [FFMPEG, "-v", "error"]
    if vf and "libplacebo" in vf:
        cmd += ["-init_hw_device", "vulkan=pl", "-filter_hw_device", "pl"]
    # 手机竖拍等带 rotation 元数据的源：ffmpeg 默认会转，显式写出来更可靠
    # （注意 -autorotate 是布尔开关，不能跟 "1"，否则 "1" 会被当成输出文件名）
    cmd += ["-autorotate"]
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
    fm = _DEC_FPS if fps_mode is None else str(fps_mode)
    if fm:
        cmd += ["-fps_mode", fm]
    cmd += ["-f", "rawvideo", "-pix_fmt", "gbrpf32le" if fp32 else "rgb24", "-"]
    return cmd, w, h


def _decode_range_cmd(path, info, start, count, pre_scale, fps, on_progress, fp32, scaler,
                      deband, deband_grain, dither, fps_mode=None):
    """解一次 [start, start+count)，返回 NHWC；可能短于 count（源真实帧数不够）。"""
    cmd, w, h = _decode_cmd(path, info["w"], info["h"], pre_scale, start, count, fps, fp32,
                            scaler, deband, deband_grain, dither, fps_mode=fps_mode)
    frame_bytes = w * h * 3 * (4 if fp32 else 1)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err_th, err_buf = _start_stderr_drain(proc.stderr)
    # 预分配一次而不是 chunks + b"".join()：整窗多一次全量 memcpy（4K 下 100+ MB/窗）
    raw = bytearray(frame_bytes * count)
    mv = memoryview(raw)
    n = 0
    while n < count:
        b = proc.stdout.read(frame_bytes)
        if not b or len(b) < frame_bytes:
            break
        mv[n * frame_bytes:(n + 1) * frame_bytes] = b
        n += 1
        if on_progress and n % 10 == 0:
            on_progress(n)
    proc.wait()
    err_th.join(timeout=2.0)
    del mv
    if n == 0:
        if proc.returncode:
            raise RuntimeError("解码失败：" + (_stderr_tail(err_buf) or f"ffmpeg 退出码 {proc.returncode}"))
        return np.zeros((0, h, w, 3), np.float32 if fp32 else np.uint8)
    buf = raw[:n * frame_bytes]
    if fp32:
        # gbrpf32le = 平面 G/B/R，转成 HWC 的 R/G/B 并夹到 [0,1]（spline36 会有过冲）
        a = np.frombuffer(buf, np.float32).reshape(-1, 3, h, w)
        a = a[:, [2, 0, 1]].transpose(0, 2, 3, 1)
        return np.clip(a, 0.0, 1.0, out=a)
    return np.frombuffer(buf, np.uint8).reshape(-1, h, w, 3)


def decode_frames_range(path, start, count, pre_scale=1.0, fps=25.0, on_progress=None,
                        fp32=False, scaler=None, deband=False, deband_grain=None,
                        dither=""):
    """只解 [start, start+count) 这一段，返回 NHWC（uint8 或 float32，内存只跟段长有关）。

    默认按真实帧出图（fps_mode=passthrough）。以前的默认是 CFR 补帧：容器时间戳有一点
    空档就会补出重复帧，成片帧数比探测值多。反过来，VFR 源在 passthrough 下这一段可能
    凑不满 count 帧，这时自动退回 cfr 再解一遍，保持旧行为（VFR 检查/提示照旧工作）。
    """
    info = probe(path)
    if info["w"] <= 0 or info["h"] <= 0:
        raise RuntimeError("读不到视频尺寸，文件可能不是视频")
    out = _decode_range_cmd(path, info, start, count, pre_scale, fps, on_progress, fp32,
                            scaler, deband, deband_grain, dither)
    total = int(info.get("frames") or 0)
    # 只有"请求区间落在片长之内却还是读不满"才怀疑 VFR。请求本来就超片长（末窗按
    # 探测帧数给足 count，实际文件少一帧）属于正常短读，不能拿 CFR 去补——那正好会
    # 把 concat 时长异常补成重复帧。
    if count and len(out) < count and _DEC_FPS and (total <= 0 or start + count <= total):
        alt = _decode_range_cmd(path, info, start, count, pre_scale, fps, None, fp32,
                                scaler, deband, deband_grain, dither, fps_mode="cfr")
        if len(alt) > len(out):
            return alt
    return out


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
    err_th, err_buf = _start_stderr_drain(proc.stderr)
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
    err_th.join(timeout=2.0)
    err = _stderr_tail(err_buf)
    if not chunks:
        raise RuntimeError("没解出任何帧" + (("：" + err) if err else ""))

    if proc.returncode:
        print(f"[decode] 警告：ffmpeg 退出码 {proc.returncode}，已解出 {n} 帧"
              + (("；" + err) if err else ""), flush=True)
    if fp32:
        a = np.frombuffer(b"".join(chunks), np.float32).reshape(-1, 3, h, w)
        a = a[:, [2, 0, 1]].transpose(0, 2, 3, 1)
        return np.clip(a, 0.0, 1.0, out=a), info
    return np.frombuffer(b"".join(chunks), np.uint8).reshape(-1, h, w, 3), info


PREVIEW_WIDTH = int(os.environ.get("FLASHVSR_PREVIEW_WIDTH", "1280"))
PREVIEW_FORMAT = os.environ.get("FLASHVSR_PREVIEW_FORMAT", "avif").lower()
_PREVIEW_ENC = None


def _shares(a, b):
    """两块 ndarray 是否共用同一片底层内存。

    不能用 is 比：read() 返回的是 arr[:n]（基本切片 → 新 ndarray 对象、同一片内存），
    于是 ent[2] is not avoid 恒为 True，池子会把主线程正在读的那块交给预取线程覆盖。
    """
    if a is None or b is None:
        return False
    if a is b:
        return True
    try:
        return bool(np.shares_memory(a, b))
    except Exception:  # noqa
        return False


class _WindowStream(object):
    """常驻一条 ffmpeg 解码进程，按窗口顺序连续取帧（替代每个窗口 Popen 一次）。

    老做法 decode_frames_range 每窗口重开 ffmpeg：起进程 + vulkan 设备初始化 +
    libplacebo 着色器编译在 Windows 上单次 0.3–0.6 s，而每段只有 4–9 帧
    （重复软解本身 <5%，起进程才是大头）。这里一条管道顺着往下读：上一窗的尾巴
    （overlap 部分）直接从上一窗缓冲切片，既不重复软解也不重开进程。

    双缓冲：窗口帧直接读进 pinned 内存的池（默认 2 块）。主线程还在用第 N 块时，
    预取线程往另一块里读，两块互不覆盖；pinned 也让后续 H2D 能真异步
    （pageable 内存的 non_blocking=True 其实是同步拷贝）。池里 ndarray 与 tensor
    互相持有引用，避免 pinned 块被回收。

    只处理 uint8（8bit）主路径；浮点解码（hi_precision）仍走 decode_frames_range。
    """

    def __init__(self, path, fps=25.0, start=0, pre_scale=1.0, scaler=None,
                 deband=False, deband_grain=None, dither="", blocks=2):
        self.path = path
        self.fps = float(fps or 25.0)
        self.pre_scale = pre_scale
        self.scaler = scaler
        self.deband = bool(deband)
        self.deband_grain = deband_grain
        self.dither = str(dither or "")
        info = probe(path)
        if info["w"] <= 0 or info["h"] <= 0:
            raise RuntimeError("读不到视频尺寸，文件可能不是视频")
        # probe 给的已是显示朝向尺寸；ffmpeg 解码时自动按 rotation 摆正，尺寸一致
        self.raw_wh = (int(info["w"]), int(info["h"]))
        _, self.w, self.h = _decode_cmd(path, self.raw_wh[0], self.raw_wh[1], pre_scale,
                                        fps=self.fps)
        self.frame_bytes = self.w * self.h * 3
        self._lock = threading.RLock()
        self._nblocks = max(1, int(blocks))
        self._blocks = []       # [(count, tensor, ndarray, pinned)]
        self._cur = None        # 上一窗返回的 ndarray（池里的一块，别覆盖）
        self._cur_start = 0
        self._next = int(start)  # 下一条从管道读出的绝对帧号
        self._raw = None
        self._proc = None
        self._err_th = None
        self._err_buf = None
        self.opened = 0
        self.frames_read = 0
        self._open(int(start))

    # ---------------------------------------------------------------- 进程管理
    def _open(self, start):
        self._close_proc()
        cmd, _, _ = _decode_cmd(self.path, self.raw_wh[0], self.raw_wh[1], self.pre_scale,
                                int(start), None, self.fps, False, self.scaler,
                                self.deband, self.deband_grain, self.dither)
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._err_th, self._err_buf = _start_stderr_drain(self._proc.stderr)
        self._next = int(start)
        self.opened += 1

    def _close_proc(self):
        p = self._proc
        self._proc = None
        if p is not None:
            try:
                if p.poll() is None:
                    p.kill()
            except Exception:  # noqa
                pass
            try:
                p.wait(timeout=10)
            except Exception:  # noqa
                pass
        th, self._err_th = self._err_th, None
        if th is not None:
            # 先等排空线程自己读到 EOF 退出，再去关管道（反过来会从线程里抛 ValueError）
            try:
                th.join(timeout=3.0)
            except Exception:  # noqa
                pass
        if p is not None:
            for s in (p.stdout, p.stderr):
                try:
                    if s is not None:
                        s.close()
                except Exception:  # noqa
                    pass

    def close(self):
        with self._lock:
            self._close_proc()
            self._blocks = []
            self._cur = None
            self._raw = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa
            pass

    # ---------------------------------------------------------------- 缓冲池
    def _get_block(self, count, avoid):
        """取一块 count 帧的缓冲（优先复用；避开 avoid 那块）。"""
        for ent in self._blocks:
            if ent[0] == count and not _shares(ent[2], avoid):
                return ent[1], ent[2]
        try:
            ten = torch.empty(count * self.frame_bytes, dtype=torch.uint8, pin_memory=True)
        except Exception:  # noqa  显存/锁定内存不足时退回 pageable（只是 H2D 不再异步）
            ten = torch.empty(count * self.frame_bytes, dtype=torch.uint8)
        arr = ten.numpy().reshape(count, self.h, self.w, 3)
        self._blocks.append((count, ten, arr, bool(ten.is_pinned())))
        if len(self._blocks) > self._nblocks:
            # 池超限：丢掉最老的、且不是上一窗在用的那块
            for k, ent in enumerate(self._blocks):
                if not _shares(ent[2], avoid) and not _shares(ent[2], self._cur):
                    del self._blocks[k]
                    break
        return ten, arr

    # ---------------------------------------------------------------- 读帧
    def _read_into(self, mv):
        """恰好读满一帧返回 True；EOF 返回 False。"""
        total = len(mv)
        n = 0
        while n < total:
            got = self._proc.stdout.readinto(mv[n:])
            if not got:
                return False
            n += got
        return True

    def _skip(self, k):
        """丢掉 k 帧（续跑/跳段时才会走到），边丢边推进 _next。"""
        if k <= 0:
            return
        if self._raw is None:
            self._raw = bytearray(self.frame_bytes * 8)
        per = max(1, len(self._raw) // self.frame_bytes)
        left = int(k)
        while left > 0:
            m = min(per, left)
            mv = memoryview(self._raw)[:m * self.frame_bytes]
            if not self._read_into(mv):
                return
            left -= m
            self._next += m

    def read(self, start, count):
        """取 [start, start+count) 帧，返回 NHWC uint8（pinned 视图）；不足时返回短的。"""
        count = int(count)
        if count <= 0:
            return np.zeros((0, self.h, self.w, 3), np.uint8)
        with self._lock:
            pre = None
            if self._cur is not None:
                cs = self._cur_start
                ce = cs + len(self._cur)
                if cs <= start < ce:
                    pre = self._cur[start - cs:]
                    if len(pre) >= count:      # 整窗都在上一窗缓冲里（窗口不前移时）
                        return pre[:count]
            _, arr = self._get_block(count, avoid=self._cur)
            have = 0
            if pre is not None:
                have = len(pre)
                arr[:have] = pre               # overlap 复用：省掉重复软解
            need0 = int(start) + have
            if self._proc is None or self._next != need0:
                # 正常流水线不会走到这里（窗口只前移、overlap 已复用）；只有续跑跳段、
                # 或者上一窗短读造成错位才会。向前小幅错位就丢弃，向后就重开进程。
                if self._proc is not None and need0 > self._next:
                    # 跨过多个已完成分段时逐帧软解毫无意义（GOP 中间每帧都要真解再丢），
                    # 大跳直接输入侧 -ss 重开；只有小幅错位才顺序读走。
                    if need0 - self._next > 32:
                        self._open(need0)
                    else:
                        self._skip(need0 - self._next)
                else:
                    self._open(need0)
            n = have
            mv = memoryview(arr).cast("B")
            while n < count:
                if not self._read_into(mv[n * self.frame_bytes:(n + 1) * self.frame_bytes]):
                    break
                n += 1
            self.frames_read += n - have
            out = arr[:n]
            self._cur, self._cur_start = out, int(start)
            self._next = int(start) + n
            if n <= have and self._proc is not None and self._proc.returncode:
                raise RuntimeError("解码失败：" + (_stderr_tail(self._err_buf) or
                                                  f"ffmpeg 退出码 {self._proc.returncode}"))
            return out


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
    # BT.709 必须在 rgb24 进入编码器之前做，且要和后端自带的 -vf 合并
    cmd += _inject_bt709_vf(encoder_args(backend, quality, bitdepth, svt_preset, tune, extra)) \
        + ["-fps_mode", "cfr", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    err_th, err_buf = _start_stderr_drain(proc.stderr)
    try:
        for i in range(0, len(frames_u8), 8):
            # 直接写 buffer：.tobytes() 每 8 帧多拷 400 MB（4K），这里零拷贝
            chunk = np.ascontiguousarray(frames_u8[i:i + 8])
            if chunk.dtype != np.uint8:
                chunk = chunk.astype(np.uint8)
            try:
                proc.stdin.write(memoryview(chunk).cast("B"))
            except (BrokenPipeError, OSError, ValueError):
                # ffmpeg 已提前退出（编码器初始化失败/磁盘满）：继续喂数据会永远阻塞
                break
    finally:
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
    err_th.join(timeout=2.0)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{backend} 编码失败：" +
                           (_stderr_tail(err_buf) or f"ffmpeg 退出码 {rc}"))


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
        # 面板 tile 是「输出像素」的块大小，源/2 之后同一个输出块只需一半的 LQ tile
        # —— 但一味折半会把 LQ 块缩小 4 倍、处理面积反而涨 40%：544p 输出 1920×1088
        # 实测 LQ tile 64 → 82.8 s、96 → 61.2 s、128 → 58.9 s（面积模型见 HANDOFF）。
        # 而单个 tile 输出超过约 400k px 模型会静默崩坏（geometry 错位），
        # 4× 下就是 tile ≤ 128（512×512）；顺带挡掉「面板 tile 512 + 2× → 1024² 单块」
        # 的 OOM（实测直接 CUDA out of memory）。所以这里只夹住上限，不再折半。
        tile = max(64, min(tile, 128))
        step = 128 // math.gcd(128, int(scale))
        if tile % step:
            tile = max(step, (tile // step) * step)

    # 单块输出有硬上限。实测（2× 路径，相位相关对齐 + NCC 判定）：LQ 块 320²/输出 640²
    # 正常（shift (0,0)、NCC 0.95），LQ 131k px（256×512，输出 512×1024）起就静默崩坏
    # —— 几何错位 + 画面胡编，PSNR/肉眼在暗场里都未必看得出来；4× 下 tile 256
    # （输出 1024²）更是直接 OOM。所以按输出块 640 px 夹：4× → 160、2× → 320。
    _tile_cap = max(64, 640 // max(1, int(scale)))
    _tile_note = None
    if tile > _tile_cap:
        _tile_note = (f"tile {tile} 在 {int(scale)}× 下单块输出 {tile * int(scale)}×"
                      f"{tile * int(scale)} px，超出模型稳定上限，已降到 {_tile_cap}")
        tile = _tile_cap
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
        # 但那条结论是在 tile256/4K 上量的；tile ≤ 160 时 tile/4 会给出 32，比文档
        # 默认的 24 多一圈重叠，网格凭空变密：544p 4× 下 tile128 从 45 块涨到 60 块
        # （实测 58.5 s → 83.0 s，+42%），而两者输出 NCC 0.9956、mean|d| 0.88/255
        # （相位相关 shift (0,0)），肉眼无差别。所以 ≤160 用 24，≥192 才按 1/4 加宽。
        tile_overlap = max(24, tile // 4) if tile >= 192 else min(tile // 4, 24)
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
    for _extra_note in (_tile_note, note_nv):
        if _extra_note:
            auto_note = ((auto_note + "；") if auto_note else "") + _extra_note
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


# 整窗复用（model_fn_wan_video 包装器）的共享状态。
# 历史 bug：ensure_loaded 每次都新建 self._win 字典，但包装器只装一次、闭包捕获的是
# 第一次那个 dict → 命中标记写在旧 dict、_decode_cl 读新 dict，复用静默失效/错帧。
# 现在所有地方共用这一个对象，ensure_loaded 只做原地 update。
_WIN_STATE = {"prev": None, "n": 0, "skip": 0, "flag": False, "thr": 0.001, "on": False}


class _SegEncoder(object):
    """单线程异步写盘：GPU 在跑下一窗时，上一段在 CPU 上编码。

    编码（x265 / AV1）占整条流水线的 10–15%，而这 10–15% 以前是**串行**的，
    GPU 在 ffmpeg 编码期间纯空转。现在只把 encode_win 挪到后台线程：
    提交点=每段推理完；join 点=①每段开头 drain（校验/预览/manifest）
    ②concat_segments 之前。内存上限按「排队段数 + 正在编的段」算，默认 2 段
    （= 在编 1 + 排队 1，也是能真正重叠的配置）；调小到 1 就退化成串行，
    1080p/4K 段内存紧张时可用。
    FLASHVSR_ASYNC_ENCODE=0 可关。
    """

    def __init__(self, fn, max_pending=2):
        self._fn = fn                    # encode_win：定义在 upscale_video 内部，必须显式传入
        self._max = int(max_pending)
        self._cv = threading.Condition()
        self._q = []
        self._res = []
        self._inflight = False
        self._closed = False
        self._done = False          # close() 幂等标记：正常路径 try 里收一次、finally 再收一次
        self._th = threading.Thread(target=self._run, name="seg-encode", daemon=True)
        self._th.start()

    def submit(self, idx, frames, path, meta):
        with self._cv:
            # 正在编的那段也要计入：worker 一 pop 队列就空了，只数队列等于允许
            # 「在编 1 段 + 排队 1 段 + 主线程正在推理的 1 段」= 3 段输出帧同时驻留。
            while (len(self._q) + int(bool(self._inflight))) >= self._max and not self._closed:
                self._cv.wait(0.5)          # 背压：别把整条成片都堆在内存里
            if self._closed:
                # 收尾后才提交：编码线程已经走了，再塞队列就是静默丢段。
                print(f"[encode] 警告：编码线程已关闭，段 {idx} 未提交（不会被编码）", flush=True)
                return
            self._q.append((idx, frames, path, meta))
            self._cv.notify_all()

    def _push(self, row):
        with self._cv:
            self._res.append(row)
            self._cv.notify_all()

    def _run(self):
        while True:
            with self._cv:
                while not self._q and not self._closed:
                    self._cv.wait(0.5)
                if not self._q:
                    return
                idx, frames, path, meta = self._q.pop(0)
                self._inflight = True
                self._cv.notify_all()
            try:
                self._fn(frames, path, None)
                self._push((idx, path, meta, None))
            except BaseException as e:      # noqa: BLE001 - 原样抛回主线程
                self._push((idx, path, meta, e))
            finally:
                with self._cv:
                    self._inflight = False
                    self._cv.notify_all()

    def poll(self):
        with self._cv:
            rows, self._res = self._res, []
            return rows

    def close(self):
        """幂等：upcale_video 的正常路径 try 里收一次、finally 里必须再收一次。

        第二次直接返回。注意 _run 只在队列空时才退出，所以第一次 close() 会先把
        队列里已提交的段编完（异常路径下也不会静默丢段），只是要等它们写完盘。
        """
        with self._cv:
            if self._done:
                return
            self._done = True
            self._closed = True
            self._cv.notify_all()
        self._th.join()


def _tmp_name(path):
    """x.mp4 → x.tmp.mp4：先写临时名，成功后再原子改名。

    保留扩展名（ffmpeg 靠扩展名选封装器）。被强杀/断电只会留下 .tmp，
    不会把半截文件当成"已完成"的产物。
    """
    root, ext = os.path.splitext(str(path))
    return f"{root}.tmp{ext or '.mp4'}"


def _seg_plans(total, chunk, overlap):
    """把 total 帧切成 [(start, win_start, win_end), …]（纯函数，方便单测边界）。

    start 是本段第一个**保留**帧；win_start/win_end 是实际解码=模型输入的区间，
    前面多带 overlap 帧给模型当上下文。overlap 夹到 chunk-1：否则
    win_end = start-overlap+chunk <= start，start 不再前进 → 死循环。
    """
    total = max(0, int(total))
    chunk = max(1, int(chunk))
    overlap = max(0, min(int(overlap), chunk - 1))
    plans, start = [], 0
    while start < total:
        win_start = max(0, start - overlap)
        win_end = min(total, win_start + chunk)
        plans.append([start, win_start, win_end])
        start = win_end
    return plans


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
        _set_cudnn_benchmark(os.environ.get("FLASHVSR_CUDNN_BENCH", "1") != "0")
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

        # quant 决定"哪些权重已在量化缓存里、加载时可跳过读盘"，必须传进去
        install_stream_loader(quant)
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

        # 解码加速后端（FLASHVSR_DEC_BACKEND=trt|compile|stage）必须在 _decode_cl 之前
        # 套上：它替换的正是 TCDecoder.decode_video，晚装就会把 channels_last 与
        # tile/窗口复用整层盖掉（复用统计看不到真实解码，TRT 也会绕过 NHWC 路径）。
        # 反过来先装它、再让 _decode_cl 包在外面，两条路互不干扰。
        try:
            import trt_decoder
            self.trt = trt_decoder.maybe_enable(
                self.pipe.TCDecoder,
                os.path.join(os.environ.get("TEMP", "."), "flashvsr_trt_cache"),
                log=lambda s: print(f"[TRT] {s}", flush=True))
        except Exception as e:  # noqa
            self.trt = None
            print(f"[FlashVSR] TRT 解码未启用: {type(e).__name__}: {e}", flush=True)

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
                # key=seq：跟"上一次 decode"比（旧行为，实际比的是扫描序相邻 tile）
                # key=pos：跟"同一坐标的上一次 decode"比（跨窗同位置，才能吃到静止块）
                "key": os.environ.get("FLASHVSR_REUSE_KEY", "seq"),
                "cache_cap": max(1, int(os.environ.get("FLASHVSR_REUSE_CACHE", "192"))),
                "pos": {},
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
                        return _ru["out"].clone()
                    # 形状对不上就不冒险：老老实实解码这一窗
                    print("[win-reuse] 形状不匹配，本窗退回真实解码", flush=True)
                reuse_tile = getattr(self, "_reuse_tile", False)
                if reuse_tile and cond is not None and _ru["key"] == "pos":
                    pos = _TILE_POS[0]
                    # decode_video 是按时轴 chunk 调用的（一个 tile 9 次左右），
                    # 所以"位置键"要带 chunk 序号；判定只在 tile 的首个 chunk 做一次，
                    # 避免每个 chunk 都 .item() 同步（实测每 chunk 同步净亏 2%）。
                    if pos != _ru.get("cur_pos"):
                        _ru["cur_pos"] = pos
                        _ru["chunk_i"] = -1
                        _ru["verdict"] = None
                    _ru["chunk_i"] = _ru.get("chunk_i", -1) + 1
                    pk = (pos, _ru["chunk_i"]) if pos is not None else None
                    if _ru["verdict"] is None and pk is not None:
                        ent0 = _ru["pos"].get(pk)
                        if ent0 is not None and ent0[2] < _ru["max_run"]:
                            d = (cond.float() - ent0[0]).abs().mean().item()
                            _ru["verdict"] = "skip" if d < _ru["thr"] else "dyn"
                        else:
                            _ru["verdict"] = "miss"
                    if _ru["verdict"] == "skip" and pk in _ru["pos"]:
                        ent = _ru["pos"].pop(pk)
                        _ru["pos"][pk] = (ent[0], ent[1], ent[2] + 1)
                        _ru["skip"] += 1
                        # 必须 clone：调用方会对返回值做 transpose(1,2).mul_(2).sub_(1) 原地缩放
                        return ent[1].clone()
                    out = _orig_decode(x, parallel=parallel,
                                       show_progress_bar=show_progress_bar, cond=cond)
                    if pk is not None:
                        # 缓存留在 GPU（省掉每 chunk 一次 D2H+H2D）；cond 也 clone 一份，
                        # 否则会钉住整段 LQ_video。
                        _ru["pos"][pk] = (cond.detach().clone(), out.detach().clone(), 0)
                        while len(_ru["pos"]) > _ru["cache_cap"]:
                            _ru["pos"].pop(next(iter(_ru["pos"])))
                    _ru["out_t"] = x.size(1)
                    return out
                if (reuse_tile and _ru["prev"] is not None
                        and cond is not None
                        and cond.shape == _ru["prev"].shape and _ru["run"] < _ru["max_run"]):
                    d = (cond.float() - _ru["prev"].float()).abs().mean().item()
                    if d < _ru["thr"]:
                        _ru["skip"] += 1
                        _ru["run"] += 1
                        # 必须 clone：调用方拿到的是 transpose 视图，会原地 mul_(2).sub_(1)
                        return _ru["out"].clone()
                _ru["run"] = 0
                out = _orig_decode(x, parallel=parallel, show_progress_bar=show_progress_bar,
                                   cond=cond)
                _ru["prev"] = cond
                # 存一份私有副本：调用方马上会通过 transpose 视图原地乘 2 减 1，
                # 直接存 out 会让缓存变成 [-1,1]，下次复用再缩一次 → 画面越来越黑。
                _ru["out"] = out.detach().clone()
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
            # 原地 update：包装器闭包捕获的是 _WIN_STATE 这个对象本身，换对象就断线
            _WIN_STATE.update({"prev": None, "n": 0, "skip": 0, "flag": False,
                               "thr": float(os.environ.get("FLASHVSR_WIN_THR", "0.001")),
                               "on": os.environ.get("FLASHVSR_WIN_REUSE", "0") == "1"})
            self._win = _WIN_STATE
            if pl_mod is not None and not getattr(pl_mod, "_win_wrapped", False):
                _orig_fn = pl_mod.model_fn_wan_video
                st = _WIN_STATE

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

        # （TRT/compile 后端的启用已前移到 _decode_cl 安装之前，见上面的注释）

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
                # prewarm 可以是 (h, w)，也可以是 [(h, w), …]（矩形 tile 每种尺寸都要预热）
                pw = prewarm
                if isinstance(pw, (tuple, list)) and len(pw) == 2 and \
                        all(isinstance(v, (int, float)) for v in pw):
                    pw = [tuple(pw)]
                for _h, _w in pw:
                    if int(_h) > 0 and int(_w) > 0:
                        self.trt.prewarm(int(_h), int(_w))
            except Exception as e:  # noqa
                print(f"[TRT] 预热失败（会在首次解码时重试）：{type(e).__name__}: {str(e)[:200]}",
                      flush=True)

        if quant in ("nvfp4", "mixed"):
            # 旧脚本可能直接 ensure_loaded(quant="nvfp4")：实现已删，静默降级到 MXFP8
            print(f"[FlashVSR] quant={quant} 已移除（画质差），改用 mxfp8", flush=True)
            quant = "mxfp8"
        if quant in ("mxfp8", "w8"):
            dit_to_cuda(self.pipe)
            if on_stage:
                on_stage("load", 0, 1, f"量化 DiT 权重（{quant.upper()}）…")
            fn = {"mxfp8": quantize_dit_mxfp8, "w8": quantize_dit_w8a16}[quant]
            print(f"[FlashVSR] {fn(self.pipe, log=print)} 个 Linear 已量化", flush=True)

        self.loaded_key = key
        torch.cuda.empty_cache()
        if on_stage:
            on_stage("load", 1, 1, "模型就绪")

    def _run_window(self, frames_u8, scale, tile, kv, on_tile=None, sparse=2.0, local=11,
                    tile_overlap=24, color_fix=True, seam_fix="smooth"):
        """tile 可以是 int（正方形）或 (th, tw)（矩形）。

        force_offload=False：每 tile 结束把 2.84 GB DiT 卸到 CPU 再搬回要多花 ~2.9 s/tile，因此不做。
        """
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
            n_f, h_f, w_f = int(_f.shape[0]), int(_f.shape[1]), int(_f.shape[2])
            use_gpu_canvas = (os.environ.get("FLASHVSR_GPU_CANVAS", "1") != "0")
            if use_gpu_canvas and not _gpu_canvas_fits(n_f, h_f, w_f, int(scale)):
                use_gpu_canvas = False
            want_gpu_prep = use_gpu_canvas and os.environ.get("FLASHVSR_GPU_PREPARE", "1") != "0"
            if want_gpu_prep:
                patched = _install_gpu_prepare(mod)
            else:
                # 关掉时必须还原：否则上一次窗口留下的 GPU 版 prepare 会一直生效
                _uninstall_gpu_prepare(mod)
                patched = False
            if patched and _f.dtype == torch.uint8:
                # 整窗保持 uint8：每个 tile 只把用到的切片搬上卡再转 float。
                # 以前是整窗先 float32（1080p/8 帧 ≈ 199 MB + 全量 CPU 遍历），而下游只用切片。
                frames = _f
            else:
                # uint8 → [0,1] 需要缩放；float32 已是 [0,1]（不能原地除，否则会把 numpy 缓冲改坏）
                frames = _f.float().div_(255.0) if _f.dtype == torch.uint8 else _f.float()
            if use_gpu_canvas:
                out = flashvsr_gpu_canvas(
                    mod, self.pipe, frames, int(scale), bool(color_fix), tile,
                    int(tile_overlap), float(sparse), float(kv), int(local), 0,
                    seam_fix=str(seam_fix or "smooth"))
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
            _debug_check_tensor("window_out", out)
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

        # 预热尺寸必须和 run_win 实际用的 tile 一致：矩形 tile 下以前按方形算 → 白预热一次
        _bt = params.get("tile") or 0
        _bh = _bt[0] if isinstance(_bt, (tuple, list)) else _bt
        _bw = _bt[1] if isinstance(_bt, (tuple, list)) else _bt
        _pth = max(1, int(params.get("tile_h") or _bh))
        _ptw = max(1, int(params.get("tile_w") or _bw))
        _psc = max(1, int(params["scale"]))
        _prewarm = [(max(16, _pth * _psc // 8), max(16, _ptw * _psc // 8))]
        self.ensure_loaded(params.get("mode") or "tiny-long", params.get("attention") or "sparse",
                           on_stage, quant=params.get("quant") or "none",
                           prewarm=_prewarm)
        drop_on = os.environ.get("FLASHVSR_FRAME_DROP", "0") == "1"
        reuse_mode = str(params.get("reuse") or "off")
        self._reuse_tile = reuse_mode in ("tile", "window")
        self._reuse_win = reuse_mode == "window"
        # 复用缓存是"上一支视频"的：pos 里每项都持有 cond/out 两份 GPU 张量（最多
        # 192 项 ≈ 几百 MB），不清就是跨任务显存泄漏，首窗还可能撞上一支的同坐标。
        _ru0 = getattr(self, "_reuse_stats", None)
        if isinstance(_ru0, dict):
            _ru0["pos"].clear()
            _ru0["prev"] = None
            _ru0["out"] = None
            _ru0["run"] = 0
            _ru0["skip"] = 0
            _ru0["verdict"] = None
            if reuse_mode in ("tile", "window") and not os.environ.get("FLASHVSR_REUSE_KEY"):
                # reuse=tile 的静止块判定必须「同坐标跨窗比」；默认 seq 比的是扫描序
                # 相邻的不同 tile，永远不可能命中。
                _ru0["key"] = "pos"
        _wst0 = getattr(self, "_win", None)
        if isinstance(_wst0, dict):
            _wst0["prev"] = None
            _wst0["flag"] = False
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

        # 编码器状态：后台编码线程会改、主线程会读，不能用 nonlocal 变量（赋值/读取跨线程
        # 无同步，主线程可能读到旧值，on_stage 与 concat 的 backend 参数就会和实际产物不符）。
        enc_lock = threading.Lock()
        enc_state = {"backend": backend, "bitdepth": bitdepth}

        def _cur_enc():
            with enc_lock:
                return enc_state["backend"], enc_state["bitdepth"]

        def encode_win(frames_in, path, audio_src=None):
            """统一编码入口：沿 FALLBACK_CHAIN 降级（qsv→qsv_h264→x265→cpu），并记住能用的那个。

            这样"哪代核显支持什么"不需要我们预先知道——本机当场试，失败就换下一个，
            且只在第一次失败时付出探测成本（之后 enc_state 已经改成可用的那个）。
            """
            primary, primary_bd = _cur_enc()
            chain = (primary,) + fallback_chain(primary)
            seen, last = set(), "未知错误"
            for be in chain:
                if be in seen:
                    continue
                seen.add(be)
                bd = primary_bd
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
                    if be != primary:
                        on_stage("encode", 0, 0,
                                 f"编码器 {primary} 不可用，已改用 {be}（后续段沿用）")
                        with enc_lock:
                            enc_state["backend"], enc_state["bitdepth"] = be, bd
                    return
                except RuntimeError as e:
                    last = (str(e).strip().splitlines() or ["未知错误"])[-1][:160]
                    on_stage("encode", 0, 0, f"{be} 编码失败：{last}")
            raise RuntimeError(f"所有编码器都失败，最后一次：{last}")

        def budget_for(win_u8, kv):
            if budget_mode != "auto" or len(win_u8) < 2:
                return 2.0, 11, kv
            sub = win_u8[:, ::8, ::8, 1]
            if np.issubdtype(np.asarray(sub).dtype, np.floating):
                # hi_precision：[0,1] float32；astype(int16) 会全变 0 → 永远判成"静止"，
                # 每窗都被压到最低预算（sparse 1.5 / local 9 / kv 2.0）。
                small = np.asarray(sub, dtype=np.float32) * 255.0
            else:
                small = np.asarray(sub, dtype=np.int16)
            motion = float(np.abs(np.diff(small.astype(np.float32), axis=0)).mean())
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
            _tmp_out = _tmp_name(dst)
            encode_win(frames_out, _tmp_out, audio)
            try:
                os.replace(_tmp_out, dst)      # 编码成功才落正式名
            except OSError as e:
                raise RuntimeError(f"编码完成但无法改名为 {dst}：{e}") from e
            on_stage("encode", 1, 1, f"编码完成（{_cur_enc()[0]}）")
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
        try:
            os.makedirs(seg_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(
                f"无法创建分段缓存目录 {seg_dir}（磁盘满或没有写权限？）：{e}") from e

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

        # 上次被强杀/断电可能留下没改名成功的半成品段（*.tmp.mp4）。它们从没进过
        # manifest，留着只占盘，直接清掉。
        try:
            for _name in os.listdir(seg_dir):
                if _name.endswith(".tmp.mp4"):
                    try:
                        os.remove(os.path.join(seg_dir, _name))
                    except OSError:
                        pass
        except OSError:
            pass

        plans = _seg_plans(total, chunk, overlap)
        n_seg = len(plans)
        step = max(1, chunk - overlap)
        # 磁盘预检（只提示不阻断）：分段缓存 + 成片 ≈ 0.4 bpp（CRF 15 实测 ~0.2 bpp，
        # 再留一倍余量）。不够时至少让人先知道，而不是中途 ENOSPC 崩在 ffmpeg 里。
        _disk_note = ""
        try:
            _need = total * (int(info["w"]) * int(params["scale"])) \
                * (int(info["h"]) * int(params["scale"])) * 0.4 / 8
            _free = shutil.disk_usage(seg_dir).free
            if _free < _need:
                _disk_note = (f"；磁盘可能不够：预计约 {_need / 2**30:.1f} GB，"
                              f"剩余 {_free / 2**30:.1f} GB")
        except Exception:  # noqa  预检失败不影响主流程
            pass
        on_stage("seg", 0, n_seg,
                 f"共 {n_seg} 段：{total} 帧，{chunk} 帧/段"
                 f"（含 {overlap} 帧段间重叠 → 每段实际推进 {step} 帧）{_disk_note}")
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
        async_encode = os.environ.get("FLASHVSR_ASYNC_ENCODE", "1") != "0"
        enc = _SegEncoder(encode_win) if async_encode else None

        # 连续流式解码：一条常驻 ffmpeg 顺着窗口往下读，替代每段 Popen 一次
        # （Windows 起进程 + vulkan/libplacebo 初始化 0.3–0.6 s/段，比软解本身贵）。
        # 整段已在内存（frames is not None）和浮点解码（fp32）路径不走这里。
        stream = None
        dec_pool = None
        pending = None          # (段号, future)：下一段的解码已经在线程里跑（双缓冲）

        def _next_open(after):
            """after（含）之后第一个还没完成的段号。"""
            for j in range(after, n_seg):
                if str(j) not in done:
                    return j
            return None

        def _decode_win(ws, we, prog=None):
            """逐段解码一个窗口（常驻流读短了时的兜底，内含 CFR 兜底）。"""
            return decode_frames_range(src, ws, we - ws, pre_scale=pre_scale,
                                       fps=info["fps"], fp32=fp32,
                                       scaler=params.get("prescale_scaler"),
                                       deband=bool(params.get("prescale_deband")),
                                       deband_grain=params.get("deband_grain"),
                                       dither=str(params.get("prescale_dither") or ""),
                                       on_progress=prog)

        if frames is None and not fp32 and os.environ.get("FLASHVSR_STREAM_DEC", "1") != "0":
            _first = _next_open(0)
            if _first is not None:
                try:
                    stream = _WindowStream(src, info["fps"], plans[_first][1],
                                           pre_scale=pre_scale,
                                           scaler=params.get("prescale_scaler"),
                                           deband=bool(params.get("prescale_deband")),
                                           deband_grain=params.get("deband_grain"),
                                           dither=str(params.get("prescale_dither") or ""))
                    dec_pool = concurrent.futures.ThreadPoolExecutor(
                        max_workers=1, thread_name_prefix="dec")
                except Exception as _e:  # noqa
                    print(f"[stream-dec] 常驻解码不可用，退回逐段解码: {_e}", flush=True)
                    stream = None
                    dec_pool = None

        def _finish_seg(idx, seg_path, meta):
            """写盘之后的收尾：校验帧数 → 原子改名 → 记 manifest → 抽预览帧。"""
            keep_n = int(meta.get("keep") or 0)
            got_n = probe_frames(seg_path)
            if got_n != keep_n:
                raise RuntimeError(f"第 {idx + 1} 段校验失败：期望 {keep_n} 帧，实际 {got_n} 帧")
            # 校验通过才把 *.tmp.mp4 改成正式名：进程被杀时只会留下 .tmp，
            # 续跑逻辑永远看不到"半个段"，也就不会拼出坏成片。
            final = seg_path
            if ".tmp" in os.path.basename(seg_path):
                final = os.path.join(os.path.dirname(seg_path),
                                     os.path.basename(seg_path).replace(".tmp", "", 1))
                os.replace(seg_path, final)
            done[str(idx)] = dict(frames=keep_n, bytes=os.path.getsize(final),
                                  at=int(time.time()))
            man["done"] = done
            save_manifest(seg_dir, man)
            pv = None
            pdir = meta.get("pdir")
            if pdir and keep_n > 0:
                try:
                    os.makedirs(pdir, exist_ok=True)
                    name = f"seg_{idx:04d}{preview_ext()}"
                    if extract_preview(final, random.randrange(keep_n), info["fps"],
                                       os.path.join(pdir, name)):
                        pv = name
                except Exception as e:  # noqa
                    print(f"[preview] 第 {idx + 1} 段抽帧失败: {e}", flush=True)
            extra = {"preview": pv, "pv_cur": idx + 1, "pv_total": n_seg} if pv else {}
            on_stage("seg", idx + 1, n_seg,
                     f"段 {idx + 1}/{n_seg} 完成（sparse={meta.get('sp')} "
                     f"kv={meta.get('kv')} local={meta.get('local')}）", **extra)

        def _drain():
            """收掉后台线程已编码完的段；错误在这里抛回主线程。"""
            if enc is None:
                return
            for idx, spath, meta, err in enc.poll():
                if err is not None:
                    raise RuntimeError(f"第 {idx + 1} 段编码失败：{err}")
                _finish_seg(idx, spath, meta)

        try:
            for i, (start, win_start, win_end) in enumerate(plans):
                _drain()
                seg_final = os.path.join(seg_dir, f"seg_{i:04d}.mp4")
                seg_paths.append(seg_final)     # 拼接用正式名（改名由 _finish_seg 完成）
                seg_path = _tmp_name(seg_final)
                keep_from = start - win_start
                keep = win_end - start
                if str(i) in done:
                    on_stage("seg", i + 1, n_seg, f"段 {i + 1}/{n_seg} 已完成，跳过")
                    continue
                if frames is not None:
                    win = frames[win_start:win_end]
                elif stream is not None:
                    on_stage("decode", win_start, total, f"解码 {win_start}–{win_end} 帧（常驻解码）")
                    if pending is not None and pending[0] == i:
                        win = pending[1].result()
                        pending = None
                    else:
                        # 预取错位（上一段被跳过 / 上次异常）→ 丢掉旧结果，同步读本窗
                        if pending is not None:
                            try:
                                pending[1].result()
                            except Exception:  # noqa
                                pass
                            pending = None
                        win = stream.read(win_start, win_end - win_start)
                    if len(win) != win_end - win_start:
                        # 常驻流水读短了（源已到尾，或 VFR 源按帧号取不满这一段）：
                        # 这一窗退回逐段解码，那里有 CFR 兜底与明确的 VFR 报错。
                        win = _decode_win(win_start, win_end, lambda n, ws=win_start: on_stage(
                            "decode", ws + n, total, f"解码 {ws + n}/{total} 帧"))
                    # 本窗已到手，立刻让后台线程去读下一窗：GPU 跑推理时解码在重叠
                    if pending is None:
                        _j = _next_open(i + 1)
                        if _j is not None:
                            _ws, _we = plans[_j][1], plans[_j][2]
                            pending = (_j, dec_pool.submit(stream.read, _ws, _we - _ws))
                else:
                    on_stage("decode", win_start, total, f"解码 {win_start}–{win_end} 帧")
                    win = _decode_win(win_start, win_end, lambda n, ws=win_start: on_stage(
                        "decode", ws + n, total, f"解码 {ws + n}/{total} 帧"))
                if frames is None and len(win) != win_end - win_start:
                    # 帧数与探测不符（典型：VFR 源，passthrough 下源的真实帧数不够）。
                    # 以前只修正本段的 keep、不动后续段起点，结果是成片中间**静默缺帧**。
                    # 宁可失败：
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
                on_stage("encode", 0, 1, f"段 {i + 1}/{n_seg} 写盘（{_cur_enc()[0]}）")
                meta = dict(keep=keep, pdir=params.get("preview_dir"), sp=sp, kv=kk, local=lr)
                if enc is None:
                    encode_win(out_frames, seg_path, None)
                    _finish_seg(i, seg_path, meta)
                else:
                    # 异步写盘：GPU 立刻去跑下一窗，编码在后台线程里与推理重叠
                    enc.submit(i, out_frames, seg_path, meta)
                del res, out_frames, win

            if enc is not None:
                enc.close()      # 等最后一两段写完，再拼接
                _drain()
        finally:
            # 异常路径也必须收掉后台编码线程（否则线程还持着整段输出帧在写盘）
            if enc is not None:
                enc.close()
            if dec_pool is not None:
                dec_pool.shutdown(wait=True)     # 先等预取线程退出，再关解码管道
            if stream is not None:
                stream.close()

        # 全段续跑（resumed == n_seg）时主循环一次都没解，out_wh 还是 None：
        # 直接从第一段探一下尺寸，别把 None 丢给前端。
        if out_wh is None and seg_paths and os.path.exists(seg_paths[0]):
            _sp = probe(seg_paths[0])
            if _sp.get("w") and _sp.get("h"):
                out_wh = [int(_sp["w"]), int(_sp["h"])]

        be_now, bd_now = _cur_enc()      # 后台线程可能已降级过，不能再用旧值
        on_stage("encode", 0, 1, f"拼接 {n_seg} 段 + 混入音轨")
        concat_segments(seg_paths, dst, audio_src=audio, backend=be_now, quality=quality,
                        bitdepth=bd_now, tune=params.get("tune"), extra=extra_args)
        got = probe_frames(dst)
        if got and got != total:
            on_stage("encode", 1, 1,
                     f"拼接完成，但输出 {got} 帧 ≠ 期望 {total} 帧（音画可能差 {(total - got) / info['fps']:.2f}s）")
        else:
            on_stage("encode", 1, 1, f"拼接完成（{be_now}，{got} 帧）")
        if not params.get("keep_seg"):
            shutil.rmtree(seg_dir, ignore_errors=True)
        return dict(frames=total, fps=info["fps"], in_wh=[info["w"], info["h"]],
                    out_wh=out_wh, segments=n_seg, resumed=resumed,
                    frames_out=got if got and got > 0 else None)

