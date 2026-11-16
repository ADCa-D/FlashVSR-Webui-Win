# -*- coding: utf-8 -*-
"""部署自检：一条命令看清「还差什么」。

    <ComfyUI 便携包>\\python_embeded\\python.exe check_env.py

退出码 0 = 可以启动；1 = 有硬性缺失（会说明怎么修）。
"""
import os
import shutil
import socket
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import config  # noqa: E402

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_rows = []


def add(level, name, note=""):
    _rows.append((level, name, note))


def _run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except Exception as e:  # noqa
        return -1, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- Python / torch
def check_python():
    v = sys.version_info
    exe = sys.executable
    if (v.major, v.minor) < (3, 10):
        add(FAIL, "Python 版本", f"{v.major}.{v.minor} 太旧，需要 3.10+（推荐 ComfyUI 便携包自带的 3.13）")
    else:
        add(OK, "Python 版本", f"{sys.version.split()[0]}  {exe}")

    # 便携版 python 被裁剪过：缺 Include/ 和 libs/ 会让 triton 编译内核时崩
    root = os.path.dirname(exe)
    inc = os.path.join(root, "Include", "Python.h")
    lib_dir = os.path.join(root, "libs")
    libs = [f for f in os.listdir(lib_dir)] if os.path.isdir(lib_dir) else []
    if os.path.isfile(inc) and libs:
        add(OK, "Python 头文件/导入库", f"Include/Python.h + libs/{libs[0]}")
    else:
        add(FAIL, "Python 头文件/导入库",
            "缺 Include/Python.h 或 libs/*.lib → triton 首次编译会崩，见 README「补 Python 头文件」")

    try:
        import torch
    except Exception as e:  # noqa
        add(FAIL, "torch", f"导入失败：{str(e)[:100]}")
        return
    if not torch.cuda.is_available():
        add(FAIL, "torch/CUDA", f"torch {torch.__version__} 但 CUDA 不可用（驱动或 torch 版本不匹配）")
        return
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    cap = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    level = OK if total >= 7.5 else WARN
    add(level, "GPU", f"{name}  {total:.1f} GB  算力 {cap}")
    add(OK, "torch", f"{torch.__version__}  CUDA {torch.version.cuda}")

    try:
        import triton
        add(OK, "triton", triton.__version__)
    except Exception as e:  # noqa
        add(WARN, "triton", f"未安装/导入失败：{str(e)[:80]}（稀疏注意力需要 triton-windows）")

    try:
        import comfy_kitchen  # noqa
        add(OK, "comfy_kitchen", "mxfp8 / nvfp4 量化内核可用")
    except Exception:  # noqa
        add(WARN, "comfy_kitchen", "未安装 → 只能用 quant=none(bf16)/w8，MXFP8/NVFP4 不可用")

    # 实测：480p/32 帧的峰值占用；小于 7.5 GB 会频繁降档
    if total < 7.5:
        add(WARN, "显存", "低于 8 GB：建议 tile 96–128、kv 2–3，并开分段")


# ---------------------------------------------------------------- 外部程序
def check_ffmpeg():
    ff = config.FFMPEG
    if not (os.path.isfile(ff) or shutil.which(ff)):
        add(FAIL, "ffmpeg", f"找不到 {ff} → 设 FLASHVSR_FFMPEG 或用 winget 装 ffmpeg")
        return
    code, out = _run([ff, "-hide_banner", "-version"])
    ver = out.splitlines()[0] if code == 0 and out else "版本未知"
    add(OK, "ffmpeg", ver[:88])

    code, out = _run([ff, "-hide_banner", "-encoders"])
    encs = out or ""
    for enc, why in (("libx264", "CPU H.264（默认编码器）"),
                     ("libx265", "H.265 / 唯一支持 10 bit"),
                     ("libsvtav1", "SVT-AV1（同画质体积最小）")):
        add(OK if enc in encs else WARN, f"编码器 {enc}", why if enc in encs else "该 ffmpeg 构建里没有")

    code, out = _run([ff, "-hide_banner", "-filters"])
    has_lp = "libplacebo" in (out or "")
    add(OK if has_lp else WARN, "滤镜 libplacebo",
        "预缩放/浮点解码走核显 Vulkan" if has_lp else "缺 libplacebo → 预缩放会退化成软件 swscale")

    code, out = _run([ff, "-hide_banner", "-init_hw_device", "vulkan=pl",
                      "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1:d=0.1",
                      "-frames:v", "1", "-f", "null", "-"], timeout=60)
    add(OK if code == 0 else WARN, "Vulkan 设备",
        "可用于 libplacebo" if code == 0 else "不可用 → 只能用 8bit 软件缩放（会慢 3 倍）")


def check_models():
    add(OK if os.path.isdir(config.COMFY_DIR) else FAIL, "ComfyUI 目录", config.COMFY_DIR)
    add(OK if os.path.isdir(config.NODE_DIR) else FAIL, "FlashVSR 节点", config.NODE_DIR)
    miss = [f for f in config.MODEL_FILES
            if not os.path.isfile(os.path.join(config.MODEL_PATH, f))]
    if miss:
        add(FAIL, "模型权重", f"{config.MODEL_PATH} 缺少：{', '.join(miss)}（见 README 第 4 步）")
    else:
        size = sum(os.path.getsize(os.path.join(config.MODEL_PATH, f)) for f in config.MODEL_FILES)
        add(OK, "模型权重", f"{size / 2 ** 30:.1f} GB  {config.MODEL_PATH}")


def check_ports():
    for port in (8288, 8289):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
            add(OK, f"端口 {port}", "空闲")
        except OSError:
            add(WARN, f"端口 {port}", "已被占用（面板可能已经在跑）")
        finally:
            s.close()


def main():
    print("FlashVSR WebUI 部署自检")
    print("=" * 72)
    check_python()
    check_ffmpeg()
    check_models()
    check_ports()

    print()
    width = max(len(n) for _, n, _ in _rows)
    for level, name, note in _rows:
        mark = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}[level]
        print(f"{mark} {name.ljust(width)}  {note}")

    bad = [r for r in _rows if r[0] == FAIL]
    warn = [r for r in _rows if r[0] == WARN]
    print()
    if bad:
        print(f"有 {len(bad)} 项硬性缺失，先按上面的提示补齐再启动。")
        return 1
    if warn:
        print(f"可以启动，但有 {len(warn)} 项提醒（不影响跑起来）。")
    else:
        print("全部就绪，直接运行 start.bat 或 python app.py。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
