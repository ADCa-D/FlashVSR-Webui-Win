# -*- coding: utf-8 -*-
"""路径与外部依赖的集中配置（全部可用环境变量覆盖，方便换机器部署）

环境变量：
  FLASHVSR_BASE     ComfyUI_windows_portable 目录（内含 ComfyUI/ 与 python_embeded/）
  FLASHVSR_NODES    自定义节点目录（默认 <BASE>/ComfyUI/custom_nodes）
  FLASHVSR_MODELS   模型根目录（默认 <BASE>/ComfyUI/models）
  FLASHVSR_FFMPEG   ffmpeg.exe 路径（默认自动探测）
  FLASHVSR_FFPROBE  ffprobe.exe 路径（默认与 ffmpeg 同目录）
"""
import os
import shutil
import sys

# safetensors 用 rayon 按“逻辑核数”并行读盘/解压，32 核同时上只会拉高功耗和温度，
# 吞吐早被磁盘/PCIe 卡住。统一限到 4 线程（可在外部覆盖）。
os.environ.setdefault("RAYON_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- ComfyUI 便携包根目录
_CANDIDATES = [
    os.environ.get("FLASHVSR_BASE"),
    r"D:\ComfyUI_windows_portable",
    r"C:\ComfyUI_windows_portable",
    os.path.join(os.path.expanduser("~"), "ComfyUI_windows_portable"),
    os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI_windows_portable"),
]
BASE = next((p for p in _CANDIDATES if p and os.path.isdir(p)), _CANDIDATES[1])
BASE = os.path.abspath(BASE)

COMFY_DIR = os.path.join(BASE, "ComfyUI")
NODES_DIR = os.environ.get("FLASHVSR_NODES") or os.path.join(COMFY_DIR, "custom_nodes")
MODELS_DIR = os.environ.get("FLASHVSR_MODELS") or os.path.join(COMFY_DIR, "models")

# FlashVSR 节点包（本工具把它的推理代码当库调用）
NODE_DIR = os.path.join(NODES_DIR, "ComfyUI-FlashVSR_Ultra_Fast")


def _find_bin(name: str, env_key: str) -> str:
    p = os.environ.get(env_key)
    if p and os.path.isfile(p):
        return p
    w = shutil.which(name)
    if w:
        return w
    for d in (r"D:\ffmpeg-9.0.1", r"C:\ffmpeg", r"C:\ffmpeg\bin", r"D:\ffmpeg\bin"):
        c = os.path.join(d, name + ".exe")
        if os.path.isfile(c):
            return c
    return name  # 交给 PATH 解析


FFMPEG = _find_bin("ffmpeg", "FLASHVSR_FFMPEG")
FFPROBE = _find_bin("ffprobe", "FLASHVSR_FFPROBE")

# 模型目录名（节点代码里按 repo 名拼路径，不能改）
MODEL_DIR_NAME = "FlashVSR-v1.1"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_DIR_NAME)
MODEL_FILES = (
    "diffusion_pytorch_model_streaming_dmd.safetensors",
    "LQ_proj_in.ckpt",
    "TCDecoder.ckpt",
    "Wan2.1_VAE.pth",
)


def check_env(verbose=True):
    """启动自检：返回 [(名称, 是否就绪, 说明)]，便于 UI/日志展示。"""
    rows = []

    def add(name, ok, note=""):
        rows.append((name, ok, note))

    add("ComfyUI 目录", os.path.isdir(COMFY_DIR), COMFY_DIR)
    add("FlashVSR 节点", os.path.isdir(NODE_DIR), NODE_DIR)
    missing = [f for f in MODEL_FILES if not os.path.isfile(os.path.join(MODEL_PATH, f))]
    add("模型权重", not missing, MODEL_PATH if not missing else "缺少：" + ", ".join(missing))
    add("ffmpeg", os.path.isfile(FFMPEG) or shutil.which(FFMPEG) is not None, FFMPEG)
    try:
        import torch
        add("torch/CUDA", torch.cuda.is_available(),
            f"{torch.__version__} / {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'}")
    except Exception as e:  # noqa
        add("torch/CUDA", False, str(e)[:80])
    if verbose:
        for n, ok, note in rows:
            print(("  [OK]   " if ok else "  [MISS] ") + f"{n:12s} {note}", file=sys.stderr)
    return rows
