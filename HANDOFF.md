# HANDOFF —— 给接手的 AI / 人

> 这份文件是"从零接手"的唯一入口。先读完它，再动代码。

## 1. 这是什么

**FlashVSR 的专用 WebUI + 引擎**：把视频超分（FlashVSR-v1.1，Wan2.1-1.3B DiT + TCDecoder）
包成一个单页面板，跑在 8 GB 显存的笔记本上（RTX 5060 Laptop）。
不依赖 ComfyUI 前端，但**复用**它的节点实现（`ComfyUI-FlashVSR_Ultra_Fast`）当运行时。

- 仓库：`https://github.com/ADCa-D/FlashVSR-Webui-Win`
- 本机路径：**`D:\flashvsr-web`**（工作目录）
- 分支：
  - `main` / `master` —— **产品**：面板 + 引擎（Python）
  - `cpp-port` —— **研究**：脱离 ComfyUI 的 C++ 推理（DiT / TCDecoder / NVRTC JIT），不进产品

## 2. 环境（本机已就绪，路径写死在这些位置）

| 项 | 路径 / 值 |
|---|---|
| Python | `D:\ComfyUI_windows_portable\python_embeded\python.exe`（3.13，torch 2.13+cu130） |
| ComfyUI 节点 | `D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast` |
| 模型 | `D:\ComfyUI_windows_portable\ComfyUI\models\FlashVSR-v1.1`（DiT 5.4 GB + TCDecoder 180 MB + VAE 484 MB + LQ_proj 549 MB） |
| ffmpeg | `D:\ffmpeg-9.0.1\ffmpeg.exe`（含 libplacebo / libsvtav1 / libvmaf，**无 heif muxer**） |
| GPU | RTX 5060 Laptop 8 GB（sm_120，驱动 591.86 → NVENC 不可用） |
| CPU | Ryzen 9 8940HX（Zen 4，AVX-512 **被 VBS/HVCI 屏蔽**，只能跑 AVX2） |

## 3. 怎么跑 / 怎么验

```bat
REM 自检（模型 / triton / 头文件 / ffmpeg / GPU / 端口）
D:\ComfyUI_windows_portable\python_embeded\python.exe check_env.py

REM 起面板（TLS 8288 / 明文 8289；127.0.0.1 免登录）
D:\ComfyUI_windows_portable\python_embeded\python.exe app.py

REM 测试（没有 pytest，直接跑，全部应 PASS）
for %f in (tests\test_*.py) do D:\ComfyUI_windows_portable\python_embeded\python.exe %f
```

⚠️ **同一时间只能有一个加载模型的进程**（并发必崩 `OSError 1455` / `0xC0000005`）。
跑任何 `bench/` 探针前先确认没有别的 python 在占显存。

## 4. 当前实测水平（544p→1088p，tile128，mxfp8）

| 指标 | 值 |
|---|---|
| 单窗（2 帧推进） | 5.5–5.6 s |
| tile192（输入 ≤960×600） | 4.44 s（−21%），峰值 3952 MiB |
| 1080p→2160p 峰值 | 6198 MiB（tile128）/ 6636 MiB（tile192） |
| 面板端到端（33 帧合成素材） | 复用关 40.3 s → 整窗复用 34.4 s（−14.6%） |
| 量化 | mxfp8 与 bf16 差 0.5 dB 内；**w8 更省显存但慢 1.8 dB** |

**开销分布（关键认知）：TCDecoder 卷积 ~53%，DiT 只占 1–2%，编解码 ~10–15%。**
所以"优化 DiT / 换语言"是没有收益的，只有"少干活"（复用）和"别浪费显存"（自管内存）有效。

## 5. 面板上的开关（都有实测支撑）

| 开关 | 效果 | 说明 |
|---|---|---|
| 量化：`mxfp8`（默认）/ `none` / `w8` | mxfp8 是速度与画质平衡点 | NVFP4 / mixed 已停用（更慢更占） |
| 复用：关闭 / 静止块 / 整窗 | **−4.7% / 54.9 dB**、**−8.9% / 50.6 dB** | 数据驱动，运动镜头自动不命中 |
| 编码器：auto / x265 / x264 / svtav1 / qsv | AV1 默认 CRF 23；切编码器自动带默认值 | `-threads` 对 libx265 无效，用 `pools=N` |
| 预缩放缩放器 | 默认 `ewa_lanczos4sharpest`（+0.5 dB） | 可选 spline36 等 |
| 8bit 抖动 `FLASHVSR_DITHER` | `ordered` / `tpdf` | 加在唯一量化点 `_quant8()` |

环境变量级（不进面板）：`FLASHVSR_REUSE_THR`、`FLASHVSR_WIN_THR`、`FLASHVSR_WIN_REUSE_MAX_RUN`、
`FLASHVSR_FRAME_DROP`（**实验、默认关、画质 33.6 dB 不可用**）、`FLASHVSR_AUTO_TILE`、
`FLASHVSR_FALLBACK_CHAIN`、`FLASHVSR_DEC_THREADS`。

## 6. 结构（按重要性）

| 文件 | 作用 |
|---|---|
| `engine.py` | 引擎主体：`resolve_params`（参数解析/自动分段/自动 tile）、`_param_key`（分段缓存指纹）、`_decode_cmd`（ffmpeg 解码 + libplacebo）、`flashvsr_gpu_canvas` / `flashvsr_aligned`（tile 拼接两种策略）、`upscale_video`（分段/续跑/编码/复用开关）、`_quant8`（**唯一 8bit 量化点**）、`install_stream_loader` |
| `app.py` | aiohttp 面板：鉴权、`/api/run`、`/api/capabilities`、`/api/segments`、SSE 进度 |
| `ui.html` | 单页 UI（预设 / 开发模式 / 试跑 / 复用 / 位深 / 调优） |
| `inference.py` | CLI 入口（`--frames` 试跑、`--tune`、`--scaler`、`--deband`、`--dither`、`--seam-fix`） |
| `worker.py` | 子进程隔离跑推理（避免面板污染显存） |
| `bench/` | 全部探针（见下） |
| `tests/` | 4 个直跑的测试，必须保持 PASS |

## 7. 踩过的坑（别重犯）

1. **分段缓存键漂移**：任何影响画面的参数都必须进 `_param_key`，且面板侧/引擎侧要**归一化**（`None` vs 默认值会造成两套指纹 → 续跑混用旧分段）。
2. **`chunk` 只切推理窗口不解码** → 峰值由总帧数决定，症状伪装成"参数无效"。
3. **并发加载模型必崩**（1455 / 访问冲突）。
4. **bf16 30 层 DiT 是混沌的**：加 1 ULP，输出 maxdiff 2.34 → **不要用 maxdiff 当验收线**，用成片 PSNR/VMAF 或"2× 混沌基线"。
5. **非连续张量喂裸指针 kernel（C++）** 会读出错误数据，症状是"量级对、数值错"。
6. **layernorm 的 `E[x²]−E[x]²`** 在近常数输入上会灾难性抵消，必须两趟或 Welford。
7. **Windows 把 GPU 分配算进进程 commit** → 量内存要看 working set，不看 commit。
8. **libplacebo 的 `preset=high_quality` 必须写成 `extra_opts=preset=high_quality`**（字典选项），且它会覆盖缩放器（VMAF −3.8）。
9. **PSNR 不能用来判缩放器/后处理优劣**（本例里它给出与 VMAF 相反的排序）。
10. 完整清单在知识库：`~/.config/opencode/reflection/`、`tricks/`、`have-solved/`（**先读这里，很多结论已经做过**）。
11. **任务管理器里的"专用 GPU 内存 8.0/8.0 GB"是假象**：这台机器 ReBAR 开着
    （`nvidia-smi -q` 里 BAR1 Total = 8192 MiB），任务管理器把 BAR1 映射窗口算成占用。
    真实占用看 `nvidia-smi --query-gpu=memory.used`（空闲时是 0 MiB）。
12. **PCIe 链路空载会降频到 Gen1**，负载时才升到 Gen4 x8（实测 10.3 GB/s ≈ 饱和）。
    别拿空载时的 `pcie.link.gen.current=1` 当"链路有问题"的证据。
13. **没有"每 tile 重新上传权重"这回事**：`TCDecoder` 只在第一次 pipe 调用时 CPU→GPU，
    之后常驻；`torch.cuda.memory_allocated()` 稳定在 ~2725 MiB 不抖动。任务管理器里
    Copy 引擎 ~3%、3D ~64%，说明是**算力跑满**，不是 PCIe 瓶颈。
    推论：mmap+pin 只对"一次性加载"有意义（能省 2–4 s）；ReBAR 直写（CPU write-combining
    写 VRAM）约 1–3 GB/s，**比 DMA 的 13 GB/s 更慢**，不要走。
14. **不要为了省搬运把 `uint8/255` 挪到 GPU 上做**：CUDA 的 float32 除法与 CPU 差 1 ULP
    （实测 1735424/3525120 个元素差 5.96e-8），进 bf16 后偶发跨舍入边界，再被 MXFP8 的
    block scale 放大 → 整段输出只有 36.9 dB、1% 像素差 >16。修法：只对**当前 tile 的切片**
    在 CPU 上归一化，整窗保持 uint8（省 199 MB/窗 驻留 + 逐位一致）。细节见
    `reflection/2026-11-17-gpu-uint8-division-1ulp.md`。
15. **`_SegEncoder` 这类模块级线程类不能按名字引用调用方闭包**：`encode_win` 定义在
    `upscale_video` 里，后台线程里直接调用会 `NameError: name 'encode_win' is not defined`，
    而且要到下一个 `_drain()` 才冒出来（日志看着一切正常）。必须把闭包当参数传：
    `_SegEncoder(encode_win)`。见 `reflection/2026-11-17-nested-fn-scope-in-thread.md`。
16. **自己的输出别直接拿去算 PSNR 帧数**：concat 出来的文件 `avg_frame_rate` 会是怪值
    （例：9 帧/0.36 s → `225/11`），`decode_frames_range` 带 `-r 25` 会把 9 帧解成 11 帧
    （补帧）。判帧数用 `ffprobe -count_frames` 的 `nb_read_frames`，别用解码出来的数组长度。
17. 量化权重有落盘缓存（`work/qcache/`，键 = checkpoint 的 size+mtime，1.34 GB）：
    重复加载 22.8 s → 11.5 s。`FLASHVSR_QCACHE=0` 关闭。注意 commit 峰值（13 GB→10.7 GB）
    和这个缓存关系不大——那段来自 5.4 GB f32 的读取/转换，不是量化。

## 8. bench/ 里能直接用的探针

| 探针 | 用途 |
|---|---|
| `probe_gpu_timeline.py` | 判"GPU 是否满载"（时间轴 vs 墙钟）——满载就别想换语言提速 |
| `probe_peak_tile.py` | tile 组合的耗时 + 峰值显存 |
| `probe_reuse_potential.py` | 任意素材的复用上限（重复帧、I/P/B、tile 静止率、一拍二周期检测） |
| `check_reuse_e2e.py` | 面板复用参数的端到端验证 |
| `probe_load_peak.py` | 加载模型时的 host 峰值（当前 5290 MiB） |
| `ab_tile_reuse.py` / `ab_frame_drop.py` | 复用/抽帧的 A/B（时间 + PSNR） |
| `dump_dit_ref.py` / `dump_tcdec_spec2.py` | 从真实流水线抓参考张量（跨语言移植的基准） |
| `smoke_opt.py` | 同进程跑 A(GPU prepare+GPU 画布) / B(节点 prepare+GPU 画布) / C(旧路径) 三段，比耗时、峰值显存与逐位差异 |
| `smoke_e2e.py` | 端到端分段冒烟：async 写盘 vs 同步写盘，逐帧比逐位一致 + 帧数校验 |
| `check_prepare_equiv.py` | 不加载模型，直接比 `_prepare_input_gpu` 与生产版 `prepare_input_tensor` 是否逐位一致（改 prepare 先跑它） |
| `diag_prepare_diff.py` | 输出差异的行/列剖面，判断误差是"接缝带"还是"纹理带" |

## 9. 下一步候选（按性价比）

先读这一节的**实测更正**，别照抄旧结论（2026-11-17 实测，素材 `work/src_1080p.mp4` = 1920×1080 24 帧，探针 `bench/probe_peak_tile.py`，`FLASHVSR_PROBE_PRE=1.0`）：

- **旧§9-1 的两条断言都不成立**：① 1080p 真实张量峰值是 **4552 MiB（tile128）/ 3188 MiB（tile192）**，reserved−alloc 间隙 1.9 GiB / 3.3 GiB；"3.4 GiB 分配器开销"是 tile192 的间隙被安到了默认配置上。② **tile192/ov48 在 1080p 上 93.2 s，比 tile128/ov32 的 89.4 s 慢 4.3%** —— 544p 上 tile192 −21% 是真的，1080p 上不成立。所以"静态池解锁大 tile 再快 8–15%"是推测，不要当路线图。
- **overlap 的成本模型**：成本 ∝ 覆盖倍数 `(tile/(tile−ov))²`。tile128、1080p 实测 ov32=89.4 s(1.78×) → ov24=82.2 s(−11%,1.51×) → ov16=67.2 s(−27%,1.31×)。ov 不能随便降的原因是：羽化只给权重不含信息，边界像素在 3D causal VAE(8× 下采样) + DiT(topk 稀疏注意力) 里缺上下文，**预测本身残缺**，残缺宽度由感受野定、与 tile 无关（`engine.py:1413-1415` 注释记 4K 实测：tile256 用 ov12 时缝处清晰度掉 13%，提到 64 后缝处/非缝比值 0.866→0.997）。验收请用这个"缝处/非缝清晰度比"口径，**别用全局 PSNR**（ov16 与 ov24 的 PSNR 几乎相同 34.7/34.2 dB，分辨不出接缝）。
- **CPU↔GPU 搬运量级已算清**：1080p tile128/ov32 每 tile 约 8.9 MB、220 tile ≈ 1.95 GB/窗 ≈ 0.2 s / 89 s = 0.2%。拼带宽无意义；真实开销在同步点（每 tile 两次 `clean_vram`、编码与推理串行）。

按性价比排序：

1. **TCDecoder 分块 + 静止块跳过**：解码器占 53%、静止块 40% → 理论上限 −30% 总时间；代价是块间接缝。
2. **编码流水线化（已落地，`FLASHVSR_ASYNC_ENCODE`）**：原来 `encode_win` 阻塞推理，编码占 10–15% 时 GPU 纯空转。现已改成单线程 `_SegEncoder` + 有界队列（depth 2），校验/预览/manifest 在 `_finish_seg` 里做，join 点是 `concat_segments` 之前。
3. **逐 tile 搬运清理（已落地，已验证）**：GPU 版 `prepare_input_tensor`（`_prepare_input_gpu`，不再 D2H→CPU stack→H2D）、画布 mask 缓存 + NHWC 预转换、每 10 tile 才 `empty_cache()` 一次（替掉每 tile 的 gc+empty_cache，后者实测慢 7%）、`decode_frames_range` 预分配 bytearray、`encode_frames` 零拷贝写 stdin。开关：`FLASHVSR_GPU_PREPARE=0` / `FLASHVSR_GPU_CANVAS=0` 可逐项回退。
   实测（`bench/smoke_opt.py 9 128`，544p 9 帧 272×480 uint8，tile128/ov32，mxfp8，同 seed）：
   A 新路径 5.28 s / 3336 MiB → B 只换画布 5.62 s → C 旧路径 6.01 s，**A 比 C 快 12.1%**；
   A vs B 逐位一致（prepare 不改变结果），A vs C 最大差 1 个 uint8 LSB（0.07% 像素，GPU vs CPU 浮点累加顺序）。
   端到端（`bench/smoke_e2e.py`，chunk=4 / overlap=1 / 9 帧 → 3 段）：async 写盘与同步写盘输出**逐帧逐位一致**，
   帧数 9/9；计时 32.6 s vs 29.1 s（小样三段的边界情况，重叠 1 帧 + 模型只加载一次，不代表稳态收益）。
4. **动画模式的时域插帧（VFI）**：唯一能吃到"抽帧 −48%"的路（现在抽帧只有 33.6 dB），但要多一个模型依赖。
5. `cpp-port` 分支：接 LQ_proj_in + 编解码，凑齐 C++ 全链路（价值在摆脱依赖、不在速度）。

**NVFP4 不要进默认**（旧§5 的"更慢更占"措辞也别照抄）：知识库 `tricks/comfyui-flashvsr-8gb.md:644-671` 更正为 GEMM 上 fp4 其实是 fp8 的 2×，"慢"来自每层 amax + 8 个 kernel 的 host 开销；真实短板是画质 44.65 dB（差约 6 dB）与 reserved 3.40 vs 2.05 GiB。而且本机 DiT 只占 1–2%（TCDecoder 卷积 53%），NVFP4 是 GEMM 路径、对卷积零作用 → 天花板就是那 1–2%，等 QAT/QAD。

## 10. 交接须知

- **别信"参数看起来没生效"**：先确认 ffmpeg/命令退出码，再确认产物帧数。
- **改动后必须跑 4 个测试**，尤其 `test_param_key`（它会抓指纹漂移）。
- 大改之前先看 `bench/` 里有没有对应的 A/B 探针，有就用它量，别凭感觉。
- 知识库（全局，跨会话）：`C:\Users\MECHREVO\.config\opencode\agent.md` 是状态入口。
