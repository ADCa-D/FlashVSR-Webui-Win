# FlashVSR WebUI

一个**只为 FlashVSR 一件事**的本地网页面板：拖视频 → 选档位 → 出超分视频。
不经过 ComfyUI 前端和工作流，直接把 [ComfyUI-FlashVSR_Ultra_Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast) 的推理代码当库调用。

> 在 8 GB 显存的 RTX 5060 Laptop 上实测：**640×360 → 2560×1440（4×），32 帧/段 48.5 s**；
> 512 帧、960×544 的 17 秒片子约 **30 分钟**（同配置未优化时是 80 分钟）。

---

## 特性

- **单功能面板**：上传 / 档位 / 进度（含 ETA）/ 结果预览 / 历史清理，没有多余的节点图
- **分段处理 + 断点续跑**：长视频按 48 帧切段，逐段落盘，完成后 `ffmpeg concat -c copy` 拼接（不二次编码）；崩溃/取消后重提同参数自动跳过已完成段
- **崩溃隔离**：每次推理跑在独立子进程里，崩了只影响本次任务，显存自动归还
- **自动降档**：显存不足时按 `tile/kv` 阶梯自动重试（原参数 → tile≤128/kv≤2 → tile96 → tile64）
- **权重量化**：MXFP8（推荐）/ bf16 / NVFP4（实验），量化后自动把 tile 提到 192
- **运动自适应预算**：静止镜头自动降 `sparse_ratio/local_range` 省算力
- **显存回收守卫**：任务运行时写 `pause.flag`，避免系统内存回收脚本打断推理（见 FAQ）

## 实测性能（RTX 5060 Laptop 8 GB）

单段 = 32 帧 / 4× / 源 640×360：

| 配置 | 单段耗时 | 峰值显存 | 相对基线 |
|---|---|---|---|
| bf16 + tile160 + kv2 | 75.7 s | 6.19 GiB | 基线 |
| MXFP8 + tile160 + kv2 | 65.4 s | 4.93 GiB | −14% |
| **MXFP8 + tile192 + kv2** | **49.7 s** | 6.18 GiB | **−35%** |
| ＋channels_last 解码 | 48.5 s | 6.01 GiB | −36% |

其他实测：

| 项 | 结果 |
|---|---|
| 端到端 64 帧（bf16 t160 → mxfp8 t192） | 136.6 s → **90.7 s**，PSNR 39.3 dB / SSIM 0.998 |
| 量化本身（同 tile 对比 bf16） | PSNR **46.2 dB**，峰值 −1.26 GiB |
| 时序一致性（512 帧成片） | 段边界异常度 z 均值 **−0.07**，成片/源帧间差分比 **0.95** |
| `tiny` vs `tiny-long` | 50.1 vs 50.5 s，峰值 6.01 vs **5.86 GiB**，输出**逐位一致** |

## 环境要求

- Windows 10/11
- NVIDIA GPU（推荐 ≥8 GB；量化路径在 Blackwell 上收益最大）
- [ComfyUI Windows 便携包](https://github.com/comfyanonymous/ComfyUI/releases)（自带 Python 与 torch）
- ffmpeg（建议 7.x+；用到 `libx264`、可选 `libplacebo`）

本项目**只用** ComfyUI 的 Python 环境和模型目录，不启动 ComfyUI 主程序。

---

## 部署流程

### 1. 准备 ComfyUI 便携包

解压到例如 `D:\ComfyUI_windows_portable`，确认存在：

```
D:\ComfyUI_windows_portable\
├─ ComfyUI\               ← 模型与自定义节点都在这里
└─ python_embeded\python.exe
```

### 2. 装 FlashVSR 节点（代码会被当库调用）

```powershell
$ProgressPreference='SilentlyContinue'   # 加速 Invoke-WebRequest
Invoke-WebRequest https://codeload.github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast/zip/refs/heads/main -OutFile "$env:TEMP\fv.zip"
Expand-Archive "$env:TEMP\fv.zip" -DestinationPath "D:\ComfyUI_windows_portable\ComfyUI\custom_nodes" -Force
Rename-Item "D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast-main" "ComfyUI-FlashVSR_Ultra_Fast"
```

> 直接 `git clone` 在国内常被 reset，用 codeload 下 zip 更稳。

### 3. 补 Python 头文件 + 装 triton-windows（**关键**）

便携包的 `python_embeded` 是裁剪版，**缺 `Include` 和 `libs`**。triton 会在第一次跑内核时用自带 tcc 编译一个 C 小模块，缺头文件就会崩（`Python.h not found`）。

```powershell
$src="C:\Users\<你>\AppData\Local\Programs\Python\Python313"   # 官方 Python 3.13
$dst="D:\ComfyUI_windows_portable\python_embeded"
Copy-Item "$src\include\*" "$dst\Include\" -Recurse -Force
New-Item -ItemType Directory -Force "$dst\libs" | Out-Null
Copy-Item "$src\libs\*.lib" "$dst\libs\" -Force

D:\ComfyUI_windows_portable\python_embeded\python.exe -m pip install -U triton-windows
```

验证（必须跑一个真内核，只 `import triton` 不算）：

```python
import torch, triton, triton.language as tl
@triton.jit
def add_k(X, Y, OUT, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0); off = pid*BLOCK + tl.arange(0, BLOCK); m = off < N
    tl.store(OUT+off, tl.load(X+off, mask=m) + tl.load(Y+off, mask=m), mask=m)
x = torch.randn(4096, device="cuda"); y = torch.randn(4096, device="cuda"); o = torch.empty(4096, device="cuda")
add_k[(16,)](x, y, o, 4096, BLOCK=256)
print(triton.__version__, torch.allclose(o, x+y))   # 期望 True
```

### 4. 下载模型（约 6.6 GB）

从 HuggingFace [`JunhaoZhuang/FlashVSR-v1.1`](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1) 下载以下 4 个文件到
`D:\ComfyUI_windows_portable\ComfyUI\models\FlashVSR-v1.1\`（**目录名不要改**，节点代码按 repo 名拼路径）：

| 文件 | 大小 |
|---|---|
| `diffusion_pytorch_model_streaming_dmd.safetensors` | 5.4 GB |
| `LQ_proj_in.ckpt` | 549 MB |
| `Wan2.1_VAE.pth` | 484 MB |
| `TCDecoder.ckpt` | 180 MB |

```powershell
D:\ComfyUI_windows_portable\python_embeded\python.exe -m pip install -U huggingface_hub
$env:HTTPS_PROXY="http://127.0.0.1:7890"; $env:HF_HUB_DISABLE_XET="1"   # 按你的代理端口改
D:\ComfyUI_windows_portable\python_embeded\python.exe -c "from huggingface_hub import snapshot_download as d; d('JunhaoZhuang/FlashVSR-v1.1', local_dir=r'D:\ComfyUI_windows_portable\ComfyUI\models\FlashVSR-v1.1', allow_patterns=['*.ckpt','*.pth','*.safetensors'], max_workers=4)"
```

> 主模型建议校验 sha256：`bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be`。
> 国内直连 HF 不通时：hf-mirror 只有 ~0.1 MB/s，走本地代理 8 线程分段能到 ~5 MB/s。

### 5. 启动

```powershell
D:\ComfyUI_windows_portable\python_embeded\python.exe app.py
```

浏览器打开 <http://127.0.0.1:8288>。也可以双击 `start.bat`（内部就是上面这条命令）。

路径不是默认位置时用环境变量覆盖（见 `config.py`）：

```powershell
$env:FLASHVSR_BASE="E:\ComfyUI_windows_portable"
$env:FLASHVSR_FFMPEG="C:\ffmpeg\bin\ffmpeg.exe"
```

启动时终端会打印自检结果（节点 / 模型 / ffmpeg / CUDA）。

---

## 使用

1. 拖视频进左侧（**音轨会自动保留**，帧率沿用源）
2. 选档位：`画质优先` / `平衡` / `快速`，或勾「自定义」填 `scale/tile/kv`
3. 长视频勾「长视频分段」（默认 48 帧/段），失败后可勾「续跑」从断点继续
4. 开始超分 → 进度含「已用 / 预计剩余」→ 结果在右侧播放，历史可单删/清空
5. 成品在 `flashvsr-web\out\`，中间分段在 `work\segments\<指纹>\`（成功后自动删除）

## 档位与参数

| 档位 | scale | tile（bf16） | tile（开量化） | kv | 说明 |
|---|---|---|---|---|---|
| 画质优先 | 4 | 128 | **192** | 3.0 | 默认推荐（配合 MXFP8） |
| 平衡 | 4 | 160 | **192** | 2.0 | 显存更紧时 |
| 快速 | 2 | 256 | 256 | 3.0 | 只要 2× 时用 |

- `tile_size` 与 `kv_ratio` **互斥吃显存**：8 GB 上 `160+kv3`、`192+kv2` 是上限附近
- 显存不够会自动降档重试，进度里会写「自动降档重试（第 N 次）：tile=… kv=…」
- `权重`：MXFP8（推荐，−35% 且 PSNR 46 dB）/ bf16（基准）/ NVFP4（实验，小 token 下更慢）

## 目录结构

```
flashvsr-web/
├─ app.py          # aiohttp：上传/队列/SSE 进度/Range 播放/历史
├─ engine.py       # 把 ComfyUI 节点当库调 + ffmpeg 编解码 + 分段/续跑/量化
├─ worker.py       # 单任务子进程（崩溃隔离 + 显存自动归还）
├─ config.py       # 路径集中配置（环境变量可覆盖）
├─ ui.html         # 单页界面
├─ temporal_check.py   # 时序一致性检查（帧间差分 / 段边界 z 分数）
├─ compare_videos.py   # 两段视频 PSNR/SSIM 对比
└─ bench_*.py      # 各项优化的基准脚本（保留实测依据）
```

## FAQ

**Q: 报 `OSError 1455 页面文件太小`**
多半是**上一个 worker 没死透**，抱着几 GB 提交内存。执行 `Get-Process python | Sort-Object WorkingSet64 -Descending`，把异常大的那个杀掉。

**Q: 任务中途进程静默消失，没有 traceback（`0xC0000005` / `3221225477`）**
这是 **CUDA 原生崩溃**（越界），Windows 上不会抛 Python 异常。本项目已做两处规避：
- 子进程隔离：崩了只影响本次任务
- NVFP4 量化路径必须 `pad_16x=True`（否则 block-scale 行不对齐 → 内核越界）

**Q: 显存不足 / 自动降档后还是 OOM**
把 `tile` 降到 96、`kv` 降到 1.0，或改用「快速」档（2×）。

**Q: 输出比源少几帧**
已修：分段编码加 `-fps_mode cfr`、拼接加 `-fflags +genpts`，并在拼接后用 ffprobe 校验帧数（不符会在进度里报警）。

**Q: 系统内存回收脚本（如 `mem-reclaim`）会不会打断推理**
会 —— 全局 `EmptyWorkingSets` 会把解码帧踢进页面文件。本项目在任务运行时写 `pause.flag`（内容=PID），此类脚本校验 PID 存活即跳过；另有 `nvidia-smi` 利用率兜底。

## 已知限制

- 只验证 `tiny` / `tiny-long`；`full`（Wan VAE 解码）未适配
- 8 GB 卡上 `tile 224` 会 OOM；4K 源需要更小的 tile
- NVFP4 虽已修好，但在模型实际的 token 数（2300–4600）下**比 MXFP8 慢**，不推荐
- 时间维帧批处理解码实测**更慢**（工作集撑爆 L2），已放弃

## 致谢

- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) / [lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 的 `comfy_kitchen`（MXFP8/NVFP4 融合内核）

## License

MIT
