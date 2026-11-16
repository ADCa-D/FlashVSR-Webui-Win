# FlashVSR WebUI

一个**只为 FlashVSR 一件事**的本地网页面板：拖视频 → 选档位 → 出超分视频。
不经过 ComfyUI 前端和工作流，直接把 [ComfyUI-FlashVSR_Ultra_Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast) 的推理代码当库调用。

> 在 8 GB 显存的 RTX 5060 Laptop 上实测：**640×360 → 2560×1440（4×），32 帧/段 48.5 s**；
> 512 帧、960×544 的 17 秒片子约 **30 分钟**（同配置未优化时是 80 分钟）。

---

## 快速开始（3 步）

```bat
REM 1. 自检：一条命令看清缺什么（模型 / triton / 头文件 / ffmpeg / GPU）
python check_env.py

REM 2. 缺什么按提示补（见下面「部署流程」）

REM 3. 启动（会自动再自检一次，硬性缺失直接停住并说明原因）
start.bat
```

跑起来后浏览器打开 <http://127.0.0.1:8288>，左边选档位 → 拖视频 → 点开始。单机默认走回环地址，**不需要登录**；从别的机器访问才要密码（首次启动会把密码写进 `work/首次登录密码.txt`）。

## 特性

- **单功能面板**：上传 / 档位 / 进度（含 ETA）/ 结果预览 / 历史清理，没有多余的节点图
- **分段处理 + 断点续跑**：长视频按 48 帧切段，逐段落盘，完成后 `ffmpeg concat -c copy` 拼接（不二次编码）；崩溃/取消后重提同参数会先问「继续上次 / 从头开始 / 忘记历史缓存」
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

### 累计加速（性能模式实测，同一 640×360 素材 / 4×）

| 阶段 | 单段 32 帧 | 相对最初 |
|---|---|---|
| 最初（每 tile 搬 2.84 GB DiT、bf16、tile160/kv3、overlap24） | 121.2 s | 1.00× |
| ＋`force_offload=False` | 77.6 s | 1.56× |
| ＋MXFP8 量化（tile160/kv2） | 65.4 s | 1.85× |
| ＋tile192/kv2 | 49.7 s | 2.44× |
| ＋channels_last 解码 | 48.5 s | **2.50×** |
| ＋tile_overlap 24→12（叠在稳定档 tile128 上：65 → 52 s / 25 帧） | 2.08 s/帧 | **1.82×（默认档）** |

- **极限**（tile192，仅低分辨率源安全）：2.50×
- **当前默认**（tile128/kv3/overlap12，高分辨率也不会 OOM）：约 **1.8×**，代价换来的是稳定


### 解码器（TCDecoder）为什么默认不走 TensorRT

VAE 解码占 kernel 时间约 65%，理论上最值得加速。实测确实能加速，但**8 GB 卡装不下**：

| 方案 | 解码单帧（latent 96 / 768² 输出） | 端到端 4× 25 帧 | 显存代价 | 结果 |
|---|---|---|---|---|
| torch eager（默认） | 164 ms | 62 s | — | ✅ |
| torch.compile（融合 cat/conv/add/relu） | 141 ms（1.16×） | OOM | inductor 额外吃 ~2 GB | ❌ |
| TensorRT（fp16 + 9 个显式 past） | ~90 ms（约 1.8×） | OOM | 构建 3 GB workspace，会话常驻 3.2 GB | ❌ |
| fp16 替代 bf16 | 162 ms（1.3%） | — | 0 | 无意义 |

- TRT 版解码器**数值与 torch 逐帧路径完全一致**（同 fp16 时 maxdiff 0.0；bf16 权重下 0.012），代码在 `trt_decoder.py`，`FLASHVSR_DEC_BACKEND=trt` 可开（16 GB 以上显卡值得一试）。
- 卡住的原因是显存，不是速度：DiT(MXFP8) 1.5 GB + 推理激活 ~2.4 GB + TRT 会话 3.2 GB + CUDA context 0.6 GB > 8 GB。
- 2× 路径（latent 48）能装下并跑通，但端到端只有 **1.06×**（33.9 → 32.0 s）——解码在 2× 任务里占比只有 ~8 s，被固定开销摊薄，不值得多装 1.9 GB 依赖。


## 笔记本功耗墙（为什么到处限 4 线程）

笔记本的 CPU 和 GPU 共用一份功耗/散热预算：CPU 多吃 20 W，GPU 就少 boost 一档。
所以本项目在**所有非瓶颈环节**都把线程限到 4：

| 环节 | 限制 | 原因 |
|---|---|---|
| 权重加载 | `RAYON_NUM_THREADS=4` / `OMP_NUM_THREADS=4`（`config.py`） | 吞吐早被磁盘/PCIe 卡住，32 核全开只是更热 |
| 权重下载 | `max_workers=4` | 同上 |
| 视频解码 | `-threads 4`（`FLASHVSR_DEC_THREADS` 可覆盖） | 4 线程是拐点：wall 0.71 s / CPU 6.1 s，比 8/32 线程又快又省 |
| （不用硬件解码） | D3D11VA / NVDEC 实测更差 | 4K/50 帧：CPU 0.71 s vs d3d11va 1.06 s vs cuda 1.03 s；瓶颈是 YUV→rgb24 的 swscale，不是解码 |
| CPU 编码（x264/x265） | `-threads 4` | 编码耗时相对推理可忽略，没必要吃满 CPU |
| SVT-AV1 | `preset 4` + `lp=4` | 同上；且默认 32 核并行在 4K 下会 ENOMEM |

原则：**把 CPU 功耗省下来给 GPU**，因为整条链路（DiT + VAE 解码）是 GPU 瓶颈。

## 高精度解码与 10 位输出

模型内部是 bf16/fp8，输入帧却常常只有 8 bit——**量化发生在解码这一步**，和推理精度无关。两条独立的路子：

### 1. 浮点解码（面板「浮点解码」/ `--hi-precision`）

- 走 `libplacebo` 的 `gbrpf32le`：YUV→RGB 全程 float32，**不再经过 rgb24 的 8bit 四舍五入**，色度上采样也在核显（Vulkan）上做。
- 实测（544p，48 帧）：uint8 `0.25 s` → fp32 `1.09 s`；内存 4×（960×544×48：72 → 287 MiB）。
- 与 uint8 的差异：平均 `0.21/255`，p90 `0.66/255`；只有 **0.17% 的像素差 >8/255**，集中在高饱和色边——那是色度上采样核不同，不是错帧。
- 默认关。窗口占用超过 2 GB 会自动回退 8bit（按 `分段帧数 × 解码分辨率 × 3 × 4B` 估算）。
- 关键事实：`hwdownload,format=rgb24` **不支持**（Vulkan hwframe 不能直接下成 8bit RGB），所以 8bit 路径只能 `hwdownload,format=yuv420p` 再由 swscale 转 rgb24；fp32 反而可以 `gbrpf32le` 直接下载，这也是唯一能拿到 >8bit 精度的通道。
- 不能去掉 hw device：软件 libplacebo 慢 3.5 倍（544p 0.82 → 2.54 s），`-init_hw_device vulkan=pl -filter_hw_device pl` 必须保留。

### 2. 10 位编码（面板「位深」/ `--bitdepth 10`）

| 编码器 | 10 bit | 说明 |
|---|---|---|
| **libx265** | ✅ | `-pix_fmt yuv420p10le`，本机唯一验证可用的 |
| libx264 | ❌ | 本机 ffmpeg 的 x264 是 8 bit 构建 |
| libsvtav1 | ❌ | 4K 下 10 bit 缓冲翻倍 → `SvtMalloc: allocate memory failed` |
| h264_nvenc / av1_nvenc | ❌ | 需要 main10 profile，当前卡/驱动不支持 |
| h264_amf | ❌ | AMF 的 H.264 编码器只有 8 bit；且同码率画质比 x264/x265 差，只适合当“能出片就行”的兜底 |
| **hevc_qsv**（`qsv`） | ✅ | Intel 核显 main10（9 代+）；**本机无 Intel 核显（AMD 610M），代码就位但未实测** |
| h264_qsv（`qsv_h264`） | ❌ | H.264 硬件编码只有 8bit |

两者可以叠加，效果才完整：

```bat
REM 源精度一路保留到模型输入，再以 10 bit 落盘（低 2 位不被 8bit 编码丢掉）
python inference.py -s in.mp4 -o out.mp4 --hi-precision --bitdepth 10 --encoder x265 --crf 18
```

只开 `--bitdepth 10` 而不开 `--hi-precision` 意义有限：源在解码阶段就已经被量化到 8 bit 了。

## Intel QSV（核显编码）：通用可行方案

手头没有 Intel 机器实测，所以按"最可能一次跑通"的方式写，并把失败后果限制住：

```bat
REM qsv（HEVC 8bit）实际下发：
-c:v hevc_qsv -preset medium -global_quality 20 -profile:v main -vf format=nv12 -pix_fmt nv12
REM qsv_h264（H.264 8bit，兼容最老）：
-c:v h264_qsv -preset medium -global_quality 20 -profile:v high -vf format=nv12 -pix_fmt nv12
```

三个刻意的取舍：

| 决定 | 原因 |
|---|---|
| **系统内存输入 + `nv12`**，不用 `hwupload`/`hwframes` | 后者依赖 D3D11 纹理池，Windows 上老驱动常报 `Error creating a MFX session`；系统内存由驱动内部拷贝，多一次上传但兼容性最好 |
| **ICQ（`-global_quality`）**，不写死码率 | 4K 下固定码率要么糊要么爆；ICQ 与 CRF 语义接近（越小越好），`quality` 直接透传 |
| 拆成 `qsv` / `qsv_h264` 两个选项 | HEVC 压缩好但要求新；H.264 兼容性最高，老机器也能出片 |

硬件代际（厂商能力表，**不是本机实测**）：

| 能力 | 最低代际 |
|---|---|
| H.264 4K 编码 | 4 代 Haswell / 5 代 Broadwell |
| HEVC 8bit 4K 编码 | 6 代 Skylake |
| HEVC 10bit（`bitdepth=10`） | 9 代 Kaby Lake |
| AV1 编码 | 11 代 Tiger Lake（本项目未接 `av1_qsv`） |

### 怎么做到"不确定硬件也能跑"

二代到 14 代、乃至以后的核显，QSV 能力差异太大（编解码器、4K level、10bit、驱动新旧），
**没有一份静态表可以写死**。所以本项目不猜，改成三层兜底：

1. **按目标分辨率当场验证**（`probe_encoders(w, h)`）
   不是用 256×256 试，而是用**这次要输出的分辨率**（4K 就试 4K）各编 1 帧。
   老核显"支持 H.264 但撑不住 4K level"这种情况，只有这样才能提前发现。
   结果缓存在 `work/encoder_caps.json`（同一块 GPU + 同一尺寸，7 天内复用，二次启动 0 秒）。
2. **编码失败沿阶梯降级**（`FALLBACK_CHAIN`）
   `qsv → qsv_h264 → x265 → cpu`（NVENC/AMF/SVT-AV1 也都收敛到 `x265 → cpu`），
   第一段失败后就把 `backend` 换成能用的那个，后续段不再重试失败项。
3. **只降级，不中断**：每一步都往进度里写原因（如 `Error creating a MFX session: -9`），
   所以"跑不通"发生在第 1 段的前 2 秒，而不是跑完 87 段后才发现。

降级顺序可以改（不用动代码）：

```bat
set FLASHVSR_FALLBACK_CHAIN=x265>svtav1>cpu          REM 对所有后端生效
set FLASHVSR_FALLBACK_CHAIN=qsv=qsv_h264>x265>cpu    REM 只覆盖 qsv
```

默认链条里 AV1 排在 x265 **之后**，理由是「兜底要的是大概率成功 + 代价可控」：
AV1 在 4K + 10bit 下本机出现过 `SvtMalloc: allocate memory failed`，而兜底位放一个
已知会 OOM 的选项是自相矛盾的。AV1 本身是**一等选择**（`svtav1` / `av1_nvenc` 都在下拉框里），
只是不进兜底位。

### 编码器等画质实测（4K / 50 帧 / 4 线程 / lp=4）

**不同编码器的 CRF 与 preset 不是同一把尺子**，跨档比时间没有意义（下面这张表全部用
VMAF 对齐，参考系是上采样后的成片）：

| 配置 | 耗时 | 体积 | VMAF |
|---|---|---|---|
| av1 `p4` crf24 | 7.7 s | 3.53 MB | **96.90** |
| av1 `p6` crf25 | 5.4 s | 3.63 MB | 96.71 |
| av1 `p8` crf26 | **2.7 s** | 3.68 MB | 96.22 |
| av1 `p10` crf28 | 2.2 s | 3.34 MB | 95.70 |
| x265 `slow` crf20 | 6.0 s | **3.08 MB** | 96.33 |
| x265 `medium` crf20 | 3.2 s | 2.70 MB | 95.10 |
| x265 `fast` crf20 | 2.4 s | 2.59 MB | 95.30 |
| x264 `medium` crf18 | 8.9 s | 4.69 MB | 96.06 |
| x264 `slow` crf18 | 13.4 s | 4.48 MB | 96.05 |

结论（按"同 VMAF 看代价"）：

- **同等时间（≈2.5–3 s）**：AV1 `p8` 96.22 明显高于 x265 `medium` 95.10 → **AV1 每秒画质更好**；
- **同等体积（≈3.1–3.5 MB）**：x265 `slow` 96.33 略高于 AV1 `p10` 95.70 → **x265 每字节更好**；
- 两者各有胜场，不存在"AV1 全面更差"；此前用 `x265 medium` 对 `svtav1 p4` 比时间得出的结论是错的，已撤回。

据此定的策略：

- 用户主动选 `svtav1` → 仍是 **p4**（质量档；`FLASHVSR_SVT_PRESET` 可调，p6 约为 p4 的 70% 时间、少 0.2 dB）；
- `svtav1` 出现在**兜底链**里时 → 自动降为 **p8**（4K 实测 2.7 s，不值得为 0.7 dB 多等 3 倍）；
- 兜底默认仍是 `x265 → cpu`（AV1-in-mp4 在部分播放器需要系统扩展，兜底产物要保证能打开）；
  想要"每秒画质优先"可以改序：
  ```bat
  set FLASHVSR_FALLBACK_CHAIN=qsv=qsv_h264>svtav1>x265>cpu
  ```

`auto` 的实际选择顺序是 **cpu → x265 → nvenc → qsv → qsv_h264 → amf**：
GPU 编码会和推理抢显存与功耗预算（实测 NVENC/AMF 都不划算），只当 CPU 不可用时兜底。

**失败即回退**：任何硬件编码器（`qsv`/`qsv_h264`/`nvenc`/`av1_nvenc`/`amf`）初始化失败，
都会自动改用 `x265` 继续编码，并在进度里写明原因（例如 `Error creating a MFX session: -9`）。
所以即使 QSV 在你机器上跑不通，也只是慢一点，不会让跑了几小时的任务白费。

想换速度档：`FLASHVSR_QSV_PRESET=fast`（默认 `medium`）。

## 编码器：NVENC 需要驱动 ≥ 610

本机 `ffmpeg 9.0.1` 按 **NVENC API 13.1** 编译，而驱动 `591.86` 只提供 13.0，于是 `h264_nvenc` / `hevc_nvenc` / `av1_nvenc` **三个全部**初始化失败：

```
Driver does not support the required nvenc API version. Required: 13.1 Found: 13.0
The minimum required Nvidia driver for nvenc is 610.00 or newer
```

所以这不是参数问题，是驱动版本不够，别在原地调 profile/cq。两个选择：

- 升级到 **610+** 驱动，NVENC 才能用；
- 继续用 CPU 编码：`x265`（唯一支持 10 bit）／`svtav1`（同画质体积最小）／`cpu`（x264）。

面板启动时会跑一次能力探测（`probe_encoders()`），不可用的编码器会在下拉框里标出原因；`auto` 的选择顺序是 **CPU → NVENC → AMF**，所以本机会稳定落到 `cpu`，不会因为驱动问题把任务卡死。

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
> 国内直连 HF 不通时：hf-mirror 只有 ~0.1 MB/s，走本地代理分段能到 ~5 MB/s；**并发 4 路就够**（再多吞吐不变，只是徒增功耗和温度）。
>
> 加载权重同理：safetensors 默认按逻辑核数并行读盘，本项目在 `config.py` 里把 `RAYON_NUM_THREADS`/`OMP_NUM_THREADS` 限到 **4**——32 核全开只会更热，不会更快。

#### 可选：自建模型镜像（国内下载更快）

原模型是 **Apache-2.0**，允许再分发。如果想让国内用户少折腾，可以在 ModelScope 建一份镜像：

```bash
pip install modelscope
# 拉原版
modelscope download --model JunhaoZhuang/FlashVSR-v1.1 --local_dir ./FlashVSR-v1.1
# 传到自己的空间（首次会提示登录）
modelscope upload <你的空间>/FlashVSR-v1.1 ./FlashVSR-v1.1
```

或者传到自己的 HuggingFace repo：

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("FlashVSR-v1.1", repo_type="model", exist_ok=True)
api.upload_folder(folder_path=r"D:\ComfyUI_windows_portable\ComfyUI\models\FlashVSR-v1.1",
                  repo_id="<你的账号>/FlashVSR-v1.1", repo_type="model")
```

之后把上面的 `JunhaoZhuang/FlashVSR-v1.1` 换成你的镜像 id 即可，其余步骤（目录名、校验、文件清单）完全一样。
**注意**：镜像请保留原仓库的 LICENSE 与出处说明。

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
3. 「输入预缩放」决定是否先 /2 再喂 4× 模型（见下节，自定义档同样生效）
4. 长视频分段**默认开启**（48 帧/段；填 0 = 按显存自动，太大时也会自动下调）。中断后重提会被问「继续上次 / 从头开始 / 忘记历史缓存」
5. 开始超分 → 进度含「已用 / 预计剩余」→ 结果在右侧播放，历史可单删/清空
6. 成品在 `flashvsr-web\out\`，中间分段在 `work\segments\<指纹>\`（成功后自动删除）

## 档位与参数

| 档位 | scale | tile | kv | 说明 |
|---|---|---|---|---|
| 画质优先 | 4 | **128** | 3.0 | 默认推荐（配合 MXFP8）；8 GB 上实测不 OOM |
| 平衡 | 4 | 160 | 2.0 | 显存更紧时 |
| 快速 | 2 | 256 | 3.0 | 只要 2× 时用 |

- **tile 192 不再是默认**：量化后它确实快（单段 49.7 s），但只剩不到 1 GB 余量，且高分辨率素材必崩。需要时勾「tile 提到 192」，失败会自动降档重试。
- `kv` 上限放到 **4**：kv 越大保留的历史 KV 越多，时序更稳、显存更贵；`tile 128 / kv3` 是 8 GB 上的稳妥点，`tile 96 / kv4` 是需要更稳时序时的备选（见 `bench/sweep_tile_kv.py` 的扫描结果）。

- `tile_size` 与 `kv_ratio` **互斥吃显存**：8 GB 上 `tile128+kv3` 是稳妥点，`160+kv3` / `192+kv2` 已在上限边缘（会 OOM）

实测显存地图（4× / MXFP8 / 100 帧 640×360，性能模式，耗时同样是实测值）：

| tile / kv | 峰值 | 余量 |
|---|---|---|
| 96 / 4 | 3999 MiB | 4152 |
| 128 / 3（默认） | 4747 MiB | 3404 |
| 128 / 4 | 4819 MiB | 3332 |
| 160 / 3 | 5959 MiB | 2192 |
| 192 / 2 | 7114 MiB | 1037 ⚠️ |
| 192 / 3 | 7229 MiB | 922 ⚠️ |

高分辨率（960×544）下 tile192 必须配小段（实测，80 秒短探针）：

| tile / 段长 | 峰值 | 余量 | 结果 |
|---|---|---|---|
| 192 / 40 帧 | 7450 | 700 | ❌ OOM |
| 192 / 28 帧 | 7342 | 809 | ⚠️ 太紧 |
| **192 / 20 帧** | 6672 | 1479 | ✅ |
| 128 / 40 帧 | 4817 | 3333 | ✅ 最稳 |

自动分段因此按 tile 取预算：`tile≥192 → 1.0 GB`、否则 `3.2 GB`。


## 分段预览

勾了「长视频分段」时，**每跑完一段就随机抽一帧**（缩到 640 宽）推到界面进度条下方，带「第 i/n 段」标签。
边跑边能判断这一段有没有问题，不用等整片拼完。

- 抽帧走 `ffmpeg -ss <随机帧>/fps -frames:v 1 -vf scale=640:-2`
- 图片落在 `work/preview/<job>/seg_XXXX.jpg`，通过 `/api/preview/<job>/<name>` 访问（`no-store`）
- 预览自带段号（`preview.cur/total`），所以即使消息晚到、后面又刷了几条 tile 进度，标签也不会错位
- 一周前的预览会自动清理（`_gc_preview`）

## 显存不够时：先看分辨率，不是先降 tile

真正吃显存的是**整段解码缓存的 LQ 视频**：`帧数 × 输出分辨率 × 3 × 2B`。

| 源 | 输出 | 100 帧的 LQ 缓冲 | 不分段 |
|---|---|---|---|
| 640×360 | 2560×1440 | 2.2 GB | ✅ |
| **960×544** | 3840×2176 | **5.0 GB** | ❌ OOM（tile128 也救不了） |

面板会**自动分段**：估算超过 3.2 GB 就按 `3.2GB ÷ 每帧字节` 选 chunk（960×544/100 帧 → 68 帧/段），并在进度里说明。实测同参数下自动分段的 960×544 任务正常产出 3840×2176/100 帧。

**分段时解码也按段进行**：每段单独 `ffmpeg -ss … -frames:v N`，内存只跟「段长 + 重叠」有关，不再随总帧数增长（每段多花 0.2–0.5 s 的 seek）。

`kv` 从 3 加到 4 只多 ~72 MiB；但实测对段边界时序**没有改善**（z 均值 5.23 → 5.23），所以不建议为"稳"去加 kv。


- 显存不够会自动降档重试，进度里会写「自动降档重试（第 N 次）：tile=… kv=…」
- `权重`：MXFP8（推荐）/ bf16（基准）/ NVFP4（4bit，未必更快）/ 混合（FFN 4bit，省 0.16 GiB、画质 44.65 dB）/ **W8A16**（通用 fp8 权重，30/40 系也能用）
- `编码器`：`cpu`（libx264）/ `nvenc`（H.264 硬件）/ `qsv`（Intel 核显 HEVC 8bit，**未在本机验证**：本机是 AMD 610M）/ `qsv_h264`（Intel 核显 H.264 8bit，兼容最老）/ `amf`（AMD，**仅 8 bit，同码率画质明显弱于 x264/x265**）/ `svtav1`（SVT-AV1）/ `av1_nvenc`（40/50 系硬件 AV1）/ `x265`（H.265）
- 体积对比（业界 RD 基准，非本机实测）：AV1 比 x264 省 **40–50%** 码率、比 x265 省 **20–30%**；代价是 CPU 编码慢
- `svtav1` 默认 **preset 4 + lp=4**：preset 越低画质越好、同 CRF 体积越小（p4 比 p10 小 ~30%），而编码耗时相对推理可忽略（4K/50 帧：p4≈7 s、p2≈13 s，推理要 30–90 分钟）。
  可用 `FLASHVSR_SVT_PRESET=2` 换最高质量档；`lp` 固定 4（默认按 32 逻辑核并行，4K 下会 ENOMEM）
- `质量`：x264/x265/AOM 是 CRF（10–28，越小越好）；NVENC 是 CQ；AMF 是 QVBR；SVT-AV1 的 CRF 范围是 0–63（我们默认 15，已是高质量）
- 输出统一 `yuv420p`，保证 Windows 播放器与剪辑软件兼容（AV1 需要系统装 AV1 扩展才能用「电影和电视」播放）

## 登录与访问控制

面板同时监听两个端口：

| 入口 | 协议 | 鉴权 |
|---|---|---|
| `http://127.0.0.1:8288` | 明文 | **免登录**（只绑回环，外部摸不到） |
| `https://<EasyTier IP>:8288` | TLS（复用 `Desktop\server\certs`） | 账号密码 或 IP 已放行 |
| `http://<EasyTier IP>:8289` | 明文 | 同上（给**装不了自签 CA 的手机**用；端口可用 `FLASHVSR_PLAIN_PORT` 改） |

规则：

1. 回环地址永远放行；
2. 其他 IP 访问时先跳 `/login`，登录成功后**把该 IP 记入放行名单**（`work/auth_ips.json`），下次同 IP 免密；
3. API 未授权直接返回 `401`（不是跳转，方便脚本判断）；
4. 会话 cookie 用 HMAC 签名，30 天过期，`HttpOnly` + `SameSite=Lax`（TLS 下带 `Secure`）。

首次启动会生成 `admin` 账号并**在控制台打印随机密码**，也可用环境变量指定：

```powershell
$env:FLASHVSR_USER="admin"; $env:FLASHVSR_PASS="你的密码"
python app.py
```

管理命令（仅本机可用）：

```powershell
curl http://127.0.0.1:8288/api/auth/ips                     # 查看放行名单
curl -X DELETE http://127.0.0.1:8288/api/auth/ips          # 清空放行名单
```

> **为什么面板自己终结 TLS**：`Desktop\server\https-proxy.py` 是字节级管道，后端看到的客户端 IP
> 永远是多出来的 `127.0.0.1`，按 IP 放行会变成"所有人都放行"。所以 8288 不再走那个代理。



## 输入预缩放（源 /2 + 4× 模型）

FlashVSR 的权重是按 **4× 退化**训练的（GT 下采样 4 倍当 LQ，再 bicubic 放大回目标分辨率）。所以想做 2× 有两条路：

| 模式 | 做法 | 特点 |
|---|---|---|
| **原生 scale=2** | 源 bicubic 放大 2× 直接喂模型 | 快、LQ 信息多；但 LQ 比训练分布清楚一倍 → 文字/细线容易"补歪" |
| **源 /2 + 4× 模型**（默认） | 源先用 libplacebo(spline36) 在核显上降到 1/2，再走 4× 模型 | 模型留在训练分布内；输出同样是 2×；代价是丢掉一半源高频 |

两者速度几乎相同（实测 25 帧 29.1 s vs 29.6 s）。

面板里是独立的「**输入预缩放**」下拉，**预设档和自定义档都生效**：

| 选项 | 行为 |
|---|---|
| `自动`（默认） | 目标是 2× 时先 /2；其它倍率不缩放 |
| `强制 /2 + 4× 模型` | 不管填几倍，源先 /2、模型走 4× → **输出恒为 2×** |
| `关闭` | 不预缩放（填 2 就是原生 2×） |

自定义档填 `scale=4` + 「强制 /2」时，实际参数是 `scale=4 / pre_scale=0.5`，`tile` 自动减半（输入分辨率减半，同样的画面块只需一半 tile）。任务参数行会显示 `源/2` 标记。

实现（`engine.py:decode_frames`，解码阶段做，不落中间文件）：

```python
tw = max(2, int(round(w * pre_scale / 2)) * 2)   # 偶数，yuv420p 要求
th = max(2, int(round(h * pre_scale / 2)) * 2)
vf = f"libplacebo=w={tw}:h={th}:downscaler=spline36,hwdownload,format=yuv420p"
```

- 用 libplacebo（核显 Vulkan）而不是 CPU `scale`：不抢独显，几乎不吃 CPU
- `hwdownload` 只能下成 `yuv420p`（`rgb24` 会报 *Invalid output format*），再让 swscale 转 rgb24
- 顺序是「整段解码时降采样 → 内存里分窗 → 逐窗 4×」；512 帧 960×544 降到 480×272 只占 ~200 MB 内存

## 续跑与「忘记历史缓存」

分段任务每写完一段就把进度写进 `work/segments/<源指纹>_<参数指纹>/manifest.json`。

**再次提交同一个文件（参数一致）时，如果检测到未完成的分段，面板先弹窗问你**：

| 按钮 | 行为 |
|---|---|
| 继续上次 | 只补剩下的段，已完成段直接复用（`resumed>0`） |
| 从头开始 | 删掉这个任务的分段重算 |
| 忘记历史缓存 | 清空 `work/segments` 下**所有**任务缓存，再从零开始本次 |
| 取消 | 不建任务，临时上传保留 24 小时后自动清理 |

成功跑完的任务会自动删掉自己的分段（结果已经在 `out/`），所以只有中断/取消/报错的才有缓存。运行中的按钮区还常驻一个「忘记历史缓存」，并显示当前缓存占了多少 MB。


## DiT 分块（tile / 重叠）

节点内部：`stride = tile_size - tile_overlap`，羽化带宽度 = `tile_overlap × scale`。

| tile / overlap | tile 数（640×360） | 与 24 的 PSNR | SSIM |
|---|---|---|---|
| **128 / 24（默认）** | 24 | —（参考） | — |
| 128 / 16 | 24 | 43.15 dB | 0.9989 |
| 128 / 12（省时间） | 18（−25%） | 41.64 dB | 0.9985 |
| 128 / 8 | 18 | 41.06 dB | 0.9982 |

同机耗时（**性能模式实测**）：65 s → **52 s（−20%）**。默认已换回 **overlap 24**（画质优先），代价是那 20% 中的大部分——要速度再手动降到 12。

要点：`stride` 取整会跳变，overlap 16 这种中间值**没有任何收益**（tile 数还是 24）；只有跨过 12 才少一整排 tile。**tile 越大，overlap 也要跟着涨**（羽化带建议 `tile/8` 起步：tile128→16–24、tile192→24–32、tile256→32–48），否则接缝会随 tile 变大而变明显。

**还看得见网格伪影时**，overlap 的实际上限是 `tile/2`（不是 `tile/4`），可以这样等价实现「小步长 + 大上下文」：

| 配置 | 有效步长 | 每块左右留的上下文 | 相对 tile128/ov12 的算力 |
|---|---|---|---|
| tile128 / ov24（自动下限） | 104 | 12 px | 1.1× |
| tile192 / ov48 | 144 | 24 px | 1.6× |
| tile192 / ov64 | 128 | 32 px | 2.3× |
| tile256 / ov96 | 160 | 48 px | 2.4× |

**默认已改成自动 `overlap = max(24, tile/4)`**（tile128→32、tile192→48、tile256→64），面板里填 0 即可。4K 实测（tile256，1080p→4K）：ov12 缝处清晰度/非缝 = **0.866**（掉 13%，肉眼可见），ov64 = **0.997**（消失），代价是 tile 数 40→60（+50% 时间）。

`tile192/ov64` 和「步长 128、每块四周各多看 32 px」是等价的——模型见到的上下文翻倍，边界两侧预测更接近，接缝最不明显，代价是算力翻倍。先试 `tile192/ov48`，不够再上 `ov64`。

### 矩形 tile：试过了，负收益

| tile | tile 数 | 峰值显存 | 同机耗时 | 接缝 PSNR |
|---|---|---|---|---|
| 128×128（ov12） | 18 | 4886 MiB | 53 s | — |
| 128×192 | 12 | 5870 MiB | 54 s | 43.53 dB |
| 128×256 | 9 | 6662 MiB | 59 s | 43.60 dB |

成本取决于**重复覆盖面积** `1 ÷ [(1-ov/th)(1-ov/tw)]`，不是 tile 个数：
`128×128/ov12 = 1.22×`、`128×192/ov12 = 1.18×`（只省 3%）、`ov24 = 1.51×`。
矩形只改形状、不改重复率，所以白涨显存。代码保留 `tile_h/tile_w`，默认不用。


### tile 对齐规则（很重要）

节点内部会把**每个 tile 的输出**居中裁到 128 的倍数（`compute_scaled_and_target_dims`）。
若 `tile × scale` 不是 128 的倍数，裁掉的部分会让拼贴**几何错位**（不是少一块，而是内容平移），羽化也救不回来。

因此 tile 必须是 `128 / gcd(128, scale)` 的倍数：

| scale | tile 必须 | 例 |
|---|---|---|
| 2 | 64 的倍数 | 128 / 192 / 256 ✅　160 ❌ |
| 3 | 128 的倍数 | 128 ✅　192 ❌ |
| 4 | 32 的倍数 | 128 / 160 / 192 / 256 ✅ |

面板提交时会自动把 tile 对齐到安全步长，并在任务参数里显示实际使用值。

## 命令行（CLI）

面板不是唯一入口，`inference.py` 直接调引擎，参数语义与面板完全一致（共用 `engine.resolve_params`）：

```powershell
# 单个文件
python inference.py -s in.mp4 -o out.mp4 --preset quality

# 快速档 + 强制先 /2（模型走 4×）
python inference.py -s in.mp4 -o out.mp4 --preset fast --prescale half --quant mxfp8

# 手调参数
python inference.py -s in.mp4 -o out.mp4 --scale 4 --tile 128 --kv 3 --chunk 0 \
                    --quant mxfp8 --encoder nvenc --crf 18

# 批量目录（跳过已存在的成品，可挂机）
python inference.py -s D:\素材 -o D:\成品 --batch --preset quality
```

参数速查：`--quant none|nvfp4|mxfp8|mixed|w8`、`--prescale auto|half|off`、
`--encoder auto|cpu|nvenc|amf`、`--chunk 0=不分段`、`--overlap`（tile 羽化带）。

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
