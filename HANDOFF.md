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
| 量化：`mxfp8`（默认）/ `none` / `w8` | mxfp8 是速度与画质平衡点 | NVFP4 / mixed 已停用并删除实现（更慢更占），旧参数自动降到 mxfp8 |
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

1. ~~**TCDecoder 分块 + 静止块跳过**（理论上限 −30%）~~ —— **2026-11-17 两条都实测否决/未转正**：
   - **空间分块否决**（`bench/probe_tcdec_chunk.py`，tile256 输出、T=6）：baseline `parallel=False` **117 ms / 253 MiB**；
     `parallel=True` 1659 ms（14×）/5602 MiB；time-chunk k=1..4 = 115–119 ms 且**逐位一致（99 dB）**；
     strips=2 halo=0 → 1010 ms / 2896 MiB / 37.9 dB（撕裂）；strips=2 halo=4 → 1235 ms / 3582 MiB / 61.8 dB；
     strips=4 halo=4 → 1024 ms / 2209 MiB / 61.1 dB。**空间分块慢 9–10× 且峰值反涨**（条带间必须 `clean_mem()` 丢时间轴状态 + 小张量占用率崩）。
     解码器峰值只有 253 MiB → **显存从来不是解码器的瓶颈**，分块无收益。
   - **静止块跳过未转正**（`bench/reuse_run.py` 33 帧 544p tile128 mxfp8，各配置同 seed）：
     off 37.19 s / pos@0.002 37.28 s（跳过 0 次）/ 更早的 CPU 缓存版 pos@0.002 38.88 s（跳过 19 次但净慢 0.8 s）。
     ① 关键事实：`decode_video` 是**按时轴 chunk** 调的（135 次 ≈ 15 tile × 9），不是每 tile 一次 → 位置键必须带 chunk 序号，
     否则不同 chunk 撞键（第一版"跳过 19 次"是撞键误判）。② 每 chunk 一次 `.item()` 强制同步就吃掉全部收益（−0.8 s）。
     ③ 改成 GPU 侧缓存 + 每 tile 只判定一次后，**thr=0.002 时命中 0 次** → cond（LQ，−1..1）下静止块的 `mean|Δ|` > 0.002，
     旧的"静止块跨窗差 0.26–0.41/255"是 RGB 尺度，换算到 cond 正好卡在阈值附近。要吃到收益必须抬阈值，而抬阈值就是拿画质赌。
     ④ 更根本的限制：同窗口内 tile 之间**共享 `TCDecoder.mem`**（`flashvsr_tiny_long.py:472` 只在窗口末尾 `clean_mem()`），
     跳过一格 decode = mem 少推进一次 → 污染下一格（实测 tile3 掉到 37.5 dB，tile4 恢复）；加每 tile `clean_mem()` 虽免费（+0.1%）
     但**本身改变输出**（ref_b vs ref_a 从 tile1 起 18.1 dB）。
   → 结论：**别在这条路上继续投入**，见 `reflection/2026-11-17-tcdecoder-chunk-and-static-skip.md`。
2. **编码流水线化（已落地，`FLASHVSR_ASYNC_ENCODE`）**：原来 `encode_win` 阻塞推理，编码占 10–15% 时 GPU 纯空转。现已改成单线程 `_SegEncoder` + 有界队列（depth 1，默认；depth 2 时最多同时驻留 3 段输出帧，更容易 OOM），校验/预览/manifest 在 `_finish_seg` 里做，join 点是 `concat_segments` 之前。
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

## 11. 2026-11-17 稳定性/画质修复清单（13 条，已落地）

外部 checklist 全部落在 engine.py 一处，逐条对应（行号会漂，按函数名找）：

| # | 问题 | 处理 |
|---|------|------|
| P0-1 | 流式加载把 mxfp8 的 skip 集合写死进闭包 → 切 none/w8 时跳过权重、填未初始化的 torch.empty → 花屏 | 模块级 _STREAM_SKIP_CFG + set_stream_skip(quant)；patched() 每次按当前 checkpoint 判断是不是本 DiT；install_stream_loader(quant) 在 guard 之前重算 |
| P0-2 | 矩形 tile 传元组给 canvas → int(tile) TypeError | 解包 th_t/tw_t，两个维度各自 %128 校验，建议步长 128//gcd(128,scale) |
| P0-3 | ffmpeg stderr=PIPE 不排空 → 管道写满死锁 | _start_stderr_drain() 后台线程 + deque 环形缓冲（解码/编码三处都接），失败时把尾部 stderr 一起抛出 |
| P0-4 | _param_key 漏 encoder/quality/bitdepth → 改参数后复用旧分段 | 三项进指纹；ENCODER_ARGS_REV 2→3（BT.709 也改变编码语义） |
| P0-5 | _SegEncoder 异常不收线程 / nonlocal backend,bitdepth 跨线程竞态 / 队列 2 段易 OOM | 段循环 try/finally 收线程；enc_lock + enc_state + _cur_enc() 取代 nonlocal；默认 max_pending=1 |
| P1-6 | rgb24→yuv420p 默认 BT.601 → 1080p 偏绿/发灰 | _inject_bt709_vf()（FLASHVSR_ENC_BT709=0 可关）：scale=in_range=full:out_color_matrix=bt709:out_range=tv,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv；qsv/vulkan 已有 -vf 时拼在它前面 |
| P1-7 | 非 fp32 解码经 yuv420p 中转丢精度 | 改 libplacebo …:format=bgra + hwdownload,format=bgra（FLASHVSR_DEC_DL 可退回 yuv420p/yuv444p/rgba） |
| P1-8 | 全段续跑时 out_wh 为 None | 拼接前用第一段 probe() 补 |
| P1-9 | _win_wrapped 闭包绑旧字典 | 模块级 _WIN_STATE，ensure_loaded 原地 update() |
| P1-10 | _drop_stride 全局平均误杀暗场微动 | 加 8×8 分块平均 MAE（FLASHVSR_DROP_BLK_THR，默认 2.0），全局或局部超阈值都算非重复 |
| P2-13 | TRT prewarm 只按方形 tile | 按实际 (tile_h, tile_w) × scale // 8 预热，支持传 [(h,w), …] |
| P2-11 | 连续解码替代每段 -ss | 已做（2026-11-17 追加）：_WindowStream 常驻一条 ffmpeg，跨窗口复用重叠帧 + 主线程预取线程，见第 12 节 |
| P2-12 | 真 Pinned Memory 异步 H2D | 已做（2026-11-17 追加）：_stage_uint8_to_gpu 双缓冲 pinned 暂存 + Event 轮换，见第 12 节 |

**实测否定**：清单 P1-7 原方案 hwdownload,format=rgb24 在本机 ffmpeg 9.0.1 + Vulkan 上不可用
（报 Failed to configure output pad on Parsed_libplacebo_0，vulkan filter 没有 24 位格式），改走 bgra 下载。
另外只写 -colorspace/-color_primaries/-color_trc 不够：实测 primaries/transfer 仍是 unknown（libx264 的 VUI 取帧侧数据），
必须再加 setparams=color_primaries=…:color_trc=…:colorspace=…:range=tv 才会写出完整 colr。

**验证**：tests/test_param_key.py、test_canvas_stitch.py、test_encoders.py、test_encoder_ladder.py、test_tune_encode.py 全绿；
encode_frames 编出的小样 ffprobe 均为 color_space/primaries/transfer=bt709、range=tv（cpu / x265 10bit / svtav1）；
decode_frames 在 8bit、10bit 源上 bgra 与 yuv420p 两条下载路径都正常出帧（同源 mean 一致）。
FLASHVSR_DEBUG_CHECK=1 打开 _debug_check_tensor，在 canvas 拼完与 _quant8 之前打印 shape/dtype/device/min/max/mean/NaN/Inf。


## 12. 2026-11-17 追加：常驻解码 + pinned 双缓冲（P2-11 / P2-12 落地）

三项改动都在 engine.py（行号会漂，按函数名找）：

**1. 解码补帧（_decode_cmd / decode_frames_range）**
各段帧数不等时，concat -c copy 出来的 mp4 里个别帧 duration 会被写成 0.12s（如 4-3-2 段的第 7 帧），
ffmpeg 默认按 CFR 补帧，把 9 帧成片解成 11 帧 → bench 的 load() 报"帧数不符"。
_decode_cmd 新增 fps_mode 参数，在 -f rawvideo 之前插 -fps_mode（默认 FLASHVSR_DEC_FPS=passthrough）；
decode_frames_range 先按 passthrough 解，只有"请求帧数没解够、且请求区间在探测到的总帧数之内"才用 cfr 重解一次兜底（VFR 源保持旧行为）。

**2. 常驻解码 _WindowStream（替代每段重启 ffmpeg）**
- 一个进程从头读到尾（count=None），read(start, count) 在锁内工作：上一窗缓冲覆盖 start 就切片复用（重叠帧零解码），
  否则拷 overlap 前缀 + 向后 readinto 补齐；需要回退时 _skip() 丢帧，退太远才重开进程。
- 缓冲池 2 块 pinned uint8（"_get_block(count, avoid)"，避免覆盖主线程还在算的那块）。
- upscale_video 里用 ThreadPoolExecutor(max_workers=1) 预取：GPU 跑本窗时解码线程已在读下一窗。
- 开关 FLASHVSR_STREAM_DEC=0 退回逐段 decode_frames_range；fp32（hi_precision）路径仍走逐段。
- finally 顺序：enc.close() → dec_pool.shutdown(wait=True) → stream.close()。
- 坑：_close_proc() 必须先 join stderr 排空线程再关 stdout/stderr，否则排空线程里抛 ValueError。

**3. pinned 双缓冲暂存 _stage_uint8_to_gpu（P2-12）**
- uint8 → float32 的 /255 仍在 CPU 上做（HANDOFF 坑14：GPU 除法差 1 ULP，fp8 放大后 PSNR 掉 ~37 dB），
  写进 pinned 暂存后 non_blocking 拷到 GPU；每键两块缓冲 + 两个 cuda.Event 轮换（复用前 event.synchronize()）。
- 键是 (shape, str(dev))；device 可能是 "cuda:0" 字符串，必须先归一成 torch.device（第一版忘了，e2e 报 AttributeError: 'str' object has no attribute 'type'）。
- 开关 FLASHVSR_PIN_STAGE=0 退回 image_tensor.float().div_(255.0).to(device, non_blocking=True)。

**实测（RTX 5060 Laptop / 544p→1088p / tile128 / mxfp8 / chunk8 overlap2 / 48 帧 6 段）**
| 配置 | 耗时 |
|---|---|
| stream=1 pin=1 | 41.96 s / 46.81 s（复测，笔记本噪音 ±12%） |
| stream=0 pin=1 | 48.75 s（常驻解码省 ~8–14%） |
| stream=1 pin=0 | 41.97 s（pinned 暂存在这个尺寸下无可测差异） |

纯解码基准（33 窗 0..100 帧，chunk4 overlap1）：常驻 308 ms / 逐段 6653 ms（21×，只开 1 次进程）；
流式与逐段解码逐位相同（np.array_equal），回退到较早帧才重开进程。
pinned 路径与旧实现逐位一致（0..255 全值、随机 shape、非连续视图各验一遍）。
bench/smoke_e2e.py 现在 VERDICT PASS（之前因为 11 帧误判 FAIL），async/sync max|Δ|=0、PSNR 99.00。
结论：常驻解码值得（省 8–14%，纯解码 21×）；pinned 暂存在 544p/128tile 下无感知收益，但对 4K/大 tile 才有意义，保留且零风险。

## 13. 2026-11-17 追加：稳定性加固（rotation / 小视频 / 原子分段 / 计划可测）

先稳后快（用户要求）：这一轮只做"不崩、不错、坏了能查"，性能优化留到下一轮。

**1. 手机竖拍的 rotation（_decode_cmd / probe）**
- ffprobe 只写 `stream=side_data_list` 拿不到旋转信息，必须 `-show_entries stream_side_data=rotation`。
- **ffmpeg 对 rawvideo 输出也会 autorotate**（实测：rotation=90 的 raw 帧与 `-noautorotate` + `transpose=2` 逐位相同；
  反过来只比每帧字节数分辨不出来，因为 90° 旋转前后 W×H 不变 —— 早前判断"rawvideo 不转"就是被这个坑骗了）。
- 所以：probe() 把 w/h 换成**显示朝向**并返回 rot；_decode_cmd **只加 `-autorotate`，绝不能再加 transpose**，
  否则二次旋转成 180°（帧尺寸对、内容错，肉眼才发现）。注意 `-autorotate` 是布尔开关，写 `-autorotate 1` 会把
  `1` 当输出文件名 → "cannot be applied to output url 1"。
- 回归：tests/test_rotation.py（remux 加 `-display_rotation 90`，断言 probe 宽高互换、解密帧 = 未旋转源的 ±90° 且不是 180°）。

**2. 小窗口画布（flashvsr_gpu_canvas）**
两个只在小视频上出现的坑，都会静默出黑：
- 窗口比 overlap 还窄时 `calculate_tile_coords` 返回**空表**（num_rows=0）→ 画布全黑（不报错，只有 [canvas] 警告）。
- 窗口比 tile 小时只有一块 tile，节点把 s 居中裁到 `max(128, floor(s/128)*128)`：s<128 时它不补边却仍把 128 当
  height/width 传给 pipe；s≥128 但没对齐时被裁掉的边距没有邻居 tile 来补 → 右/下黑边。
- 处理：只对"窗口 < tile"的那条轴用边缘复制补到 `128/gcd(128,scale)` 的整数倍，拼完再裁回原尺寸；
  另外空 coords 兜底成整帧一块。窗口 ≥ tile 的常规路径一点不动（544p 的耗时/结果不变）。
- 回归：tests/test_canvas_stitch.py 新增第 5 条（24×32 与 12×12 窗口，要求铺满无黑边）。

**3. 分段原子写 + 友好报错 + 磁盘预检**
- 先 encode 到 `seg_XXXX.mp4.tmp.mp4`（_tmp_name），校验帧数通过后才 os.replace 成正式名；启动时清理残留 tmp。
  这样进程被杀/断电不会留下"能 probe 但内容截断"的段，续跑校验更可靠。整段路径同样先写 tmp 再改名。
- probe()：ffprobe 找不到 / 解不开 → 中文 RuntimeError（原来直接抛 CalledProcessError 栈）。
- os.makedirs(seg_dir) 失败 → "磁盘满或没有写权限？"；输出盘剩余空间 < 估计值 ×1.2 时在 on_stage 文案里提示（不阻断）。

**4. 分段计划可测化（_seg_plans）**
把 `while start < total` 的窗口切分抽成纯函数，overlap 夹到 `chunk-1`（以前 overlap >= chunk 会死循环）。
tests/test_seg_plan.py 覆盖 total<chunk / ==chunk / 9-4-1 / 8-4-0 / chunk=1 / overlap>=chunk / 负数 / 0。

**验证**：6 个既有测试 + 2 个新测试全 PASS；bench/smoke_e2e.py VERDICT PASS（async vs sync max|Δ|=0）；
模型级边缘冒烟（rotation 90 / 单帧 / 320×240 / 64×48，均 chunk=4 分段 + pre_scale=0.5）四例全 PASS，
出图 mean 58–123、四边非黑；旋转 30 帧片源走常驻流式解码 + 3 段拼接，out_wh=[96,128] 正确。


## 14. 2026-11-17 追加：性能（tile 面积模型 / 自动重叠 / 2× 路径块大小 / 单块上限）

结论先行：GPU 在 pipe 内已 99.9% 饱和（16.43 s GPU 时间 / 16.45 s 墙钟，45 次 pipe），所以这一轮不是
「算得更快」而是「少算」：**总耗时 ≈ 处理的 LQ 面积 / 吞吐**。三条改动合计在 4× 默认路径上 −23…−30%，
在 2× 路径上 −22…−29%，并顺手挡掉两个会静默出坏图的 tile 配置。

**0. 预测模型（很好用，先算再跑）**
```
stride  = tile - overlap
cols    = ceil((W_lq - tile) / stride) + 1      # W_lq <= tile 时为 1
rows    = ceil((H_lq - tile) / stride) + 1
area    = cols * rows * tile * tile             # 需要模型处理的 LQ 像素总量
面积倍数 = area / (W_lq * H_lq)
```
实测吻合（544p 源，pre_scale=0.5 → LQ 480×272，16 帧，分段 chunk=8，warm）：
tile128/ov24 1.88×→15.9 s、tile192/ov24 1.69×→13.9 s、tile256/ov32 2.01×→16.7 s、
tile256/ov64 3.0×→24.5 s、矩形 r192×512 1.51×→12.9 s。面积差 5% 时耗时差 5% 量级 —— 调参前先按这个式子
算 tile 数，比试跑快得多。

**1. 自动 tile_overlap 对 tile ≤ 160 改用 24（resolve_params，engine.py:2192）**
- 老规则 `max(24, tile // 4)` 在 tile 128 上给 32（=tile/4），但那条「4K 实测要 tile/4」的结论是在
  **tile256/4K** 上量的。tile 128 多出来这一圈重叠直接把网格撑密：544p 4× 下 45 块 → 60 块。
- 实测（9 帧单窗，输出 3840×2176）：ov32 **82.98 s** / ov24 **58.47 s** / 新默认 63.55 s（±8% 噪声）。
- 画质等价性：两份输出相位相关 shift=(0,0)、NCC **0.9956**、mean|d| 0.88/255（第 0/4/8 帧一致）→
  重叠 24 LQ px（4× 即 96 输出 px 的羽化带）与 32 在肉眼和数值上都无差别。
- 新规则：tile ≥ 192 → `max(24, tile//4)`；否则 `min(tile//4, 24)`（96/128/160 都是 24，tile64 退到 16）。

**2. 2× 路径（prescale=half）不再把 tile 折半（engine.py:2145）**
- 老代码 `tile = max(64, tile // 2)`（「同一块画面」），实际效果是把 LQ 块缩小 4 倍：面板 tile128 → LQ 64，
  处理面积 344k px（1.88× → 2.64×），逐帧 5.2 s。
- 实测（544p → 输出 1920×1088，16 帧 chunk8）：LQ tile64 **82.77 s** / 96 61.20 s / 128 **58.87–64.43 s**。
- 新规则：`tile = max(64, min(tile, 128))` —— 4× 下 512 px 输出块正是「画质优先」的默认块（面板 256 →
  128 与老的折半结果相同），面板 128 也不会再被压到 64。峰值 3076 → 4505 MiB（8 GB 仍有余量）。

**3. 单个 tile 的输出上限（新夹子，engine.py:2154）**
不只是慢：单块输出太大模型会**静默崩坏**（几何错位 + 画面胡编），PSNR/肉眼在暗场里都不一定看得出来。
判定用相位相关求平移 + 对齐后 NCC（work/perf_align.py 的思路；只看 PSNR 会误判成「细节不同」）：

| 配置（LQ tile，scale） | LQ px | 输出块 | shift | NCC | 结论 |
|---|---|---|---|---|---|
| 128 @2×（参考） | 16k | 256² | (0,0) | 1.0000 | 正常 |
| 192 @2× | 37k | 384² | (0,0) | 0.9526 | 正常 |
| 256 @2× | 65k | 512² | (0,0) | 0.9524 | 正常 |
| r320×320 @2× | 102k | 640² | (0,0) | 0.9534 | 正常（边界） |
| r256×512 @2× | 131k | 512×1024 | (0,+32) | 0.3113 | **坏** |
| 384 @2× | 147k | 768² | (+16,0) | 0.1617 | **坏** |
| 512 @2×（整帧一块） | 262k | 1024² | (+16,+32) | 0.1133 | **坏**（那个 10.85 s 的「加速」是假象） |

所以 `tile * scale` 夹在 640 px（≈409k px）：4× → tile ≤ 160（画质优先 128 / 平衡 160 都在线内），
2× → ≤ 320。超了会降档并在 auto_note 里说明。顺带挡掉「面板 tile 512 + 2×」—— 老代码折半成 256@4×
= 1024² 单块，实测直接 CUDA out of memory。

**4. 验证**
- py_compile OK；tests/test_param_key、test_canvas_stitch、test_rotation、test_seg_plan、test_encoders、
  test_encoder_ladder、test_tune_encode 全 PASS；新增 tests/test_resolve_perf.py 钉住上面三条规则
  （自动 overlap 的四档、预设块大小、2× 不折半、单块上限与降档文案）。
- bench/smoke_e2e.py **VERDICT PASS**：async vs sync max|Δ|=0、PSNR 99.00 dB。
- 复测提示：本机 GPU 有别的负载时同一配置会漂 ±8%（58.47 vs 63.55 s 就是同一配置两次），比性能请连跑两遍看最小值。

**没做（评估后放弃）**：换 kernel/编译（DiT 只占 1–2%，TCDecoder 的卷积已吃满 GPU）；
空间分块 TCDecoder（早前实测慢 9–10×）；把 tile 继续加大（撞上面那张表的崩坏线）。

---

## 15. 2026-11-17 外部 review（回应.txt）8 条 bug 的核查与修复

另一模型对我们的 engine.py / trt_decoder.py / fp8_conv.py 做了 code review（`C:/Users/MECHREVO/Downloads/回应.txt`）。
逐条对源码核实：**8 条全部为真**（其中 1 条的机制说错了、2 条只在非默认配置下触发），4 条改进建议里做了 2 条。

**1. `_WindowStream` 缓冲池 `is` 比较失效（最严重）**
`read()` 返回 `arr[:n]` 是**基本切片**：新 ndarray 对象、同一片内存 → `ent[2] is not avoid` 恒为 True，
池子会把主线程正在读的那块交给预取线程覆盖，双缓冲实际从来没生效过（只是时序上撞不到）。
改：新增 `_shares(a, b)`（`np.shares_memory` + 身份兜底），`_get_block` 与池裁剪都改用它。

**2. `seam_fix` 被 GPU 画布整层旁路**
`_run_window` 里 `if use_gpu_canvas:` 在 `elif seam_fix == "align":` 之前，而 GPU 画布默认开启 →
面板选 align / off 完全没效果。改：`flashvsr_gpu_canvas(..., seam_fix=)` 新增参数，
`off` → 不羽化（硬边），`smooth` → 原行为，`align` → 累加前用重叠区估每通道 DC 偏差并整块平移
（只在 `mask<0.999 且已有内容` 的像素上估，样本 >512 才动手）。`flashvsr_aligned` 顺手补上
`tile_cpu[:canvas.shape[0]]` 尾帧截断与 `_PI_CROP_KEY` 偏移补偿（以前显存不足回退到它时每块偏一截、右下有黑边）。

**3. hi_precision（fp32 [0,1]）帧被 int16 截断**
`_drop_stride` 与 `budget_for` 都 `astype("int16")` → 全 0 → 帧差恒 0（`FLASHVSR_FRAME_DROP=1` 会误删一半帧）
且预算永远最低档（sparse 1.5 / local 9 / kv 2.0）。改：按 dtype 分流，浮点先 `*255` 再算。

**4. 流式跳过 + 量化缓存读失败 = 永久花屏**
权重既然被跳过读盘（`torch.empty` 未初始化），缓存又读不出来，旧代码会把垃圾量化后**写回缓存**。
改：`quantize_dit_mxfp8` 里比对 `_STREAM_SKIP_CFG["keys"]` 与缓存实际层数，缺层就删缓存并 raise，
让下次从原模型重新量化（不再静默污染）。

**5. 复用缓存被调用方原地改写**
`flashvsr_tiny_long.py` 外层是 `decode_video(...).transpose(1, 2).mul_(2).sub_(1)` —— transpose 是**视图**，
`mul_` 改的就是 `_decode_cl` 返回张量的存储，而缓存存的是同一个张量。改：存 `out.detach().clone()`、
返回 `.clone()`。另外：reuse 开启且未显式设 `FLASHVSR_REUSE_KEY` 时默认 `key="pos"`（`seq` 比的是扫描序相邻的不同 tile，
永远不可能命中）；每个任务开始清 `pos`/`prev`/`out`（以前跨视频残留，每项还持有两份 GPU 张量）。
`FLASHVSR_DEC_BACKEND=trt|compile` 的包装改到 `_decode_cl` **之前**，否则加速路径把 channels_last 与复用整层盖掉。

**6. `CompiledDecoder` 把 dict 键当形状**
`_get` 存的是 `{layer_idx: shape}`，`decode_video` 直接 `for s in shapes` → `torch.zeros(3)` 1D 张量 → 首帧必崩 →
`off=True` 永久回退。改：存 `[shapes[L] for L in layers]`（按层序的形状列表）。默认后端是 `torch`，只有显式开启才走到。

**7. fp8_conv Triton 核缺 CIN 掩码**
`CIN=784` 不能被 `BLOCK_K=64` 整除（784=12×64+16）→ 最后一轮读越界/读到下一个像素。改：两处 `tl.load` 都加 `ks < CIN`；
`inv_sa` 标定不再被首帧黑场锁死（只标定一次，黑场帧不落锁）；`sw`/`inv_sa` 缓存成 python float 去掉每次 `float(tensor)` 同步。
注意 fp8_conv 只被 `bench/probe_fp8conv_layout.py`、`bench/probe_vae_fp8*.py` 用，engine 不 import。

**8. `_SegEncoder` 背压不计在编段**
worker 一 pop 队列就空 → 可再提交一段 → 在编 1 + 排队 1 + 主线程 1 = 3 段输出帧驻留。改：
`while (len(q) + int(bool(_inflight))) >= _max`，默认 `max_pending=2`（= 在编 1 + 排队 1，与旧行为等价但与不变量一致；设 1 即串行）。

**另外做了的建议**：`_WindowStream` 跨段大跳（>32 帧）改用输入侧 `-ss` 重开，不再逐帧软解丢弃。
**没做**：量化权重改 `register_buffer`（要重写 `quantize_dit_mxfp8` 的闭包结构，且只影响 TRT 建引擎时的显存腾挪，
当前默认后端用不到）；`TCDecoder/trt_decoder` 的 `list.pop(0)`→`deque` 与 `trim_flag` 改按下标（每次只 pop 几个元素，收益在噪声内）。

**验证**：py_compile OK；8 个 tests 全 PASS（`test_canvas_stitch` 新增 seam_fix 三模式 + align 拉平逐 tile 偏置的断言）；
`bench/smoke_e2e.py` VERDICT PASS（async vs sync max|Δ|=0、PSNR 99.00 dB）；另跑脚本确认 `_shares` 对视图为 True、
对拷贝为 False，`_drop_stride` 在 u8 与 float32 上结果一致（1 / 1，静止序列 2 / 2），量化缓存守卫在缺层时抛错并删缓存。


## 16. 2026-11-17 第二份外部 review（千问）的核查与修复

千问读了 `engine_(1).py`（3278 行）/ `trt_decoder.py` / `fp8_conv.py`，给了 6 条 perf + 6 条 bug + 3 条收尾。
逐条核查：**9 条属实已修**、2 条实测不成立/已不复现、1 条（P2-6）属实但权衡后不改。

| # | 主张 | 结论 | 处理 |
|---|---|---|---|
| P0-1 | TRT 每帧重建 io_binding + 13 次 `torch.empty` | 真（且有两套 TRT 实现） | past 输出走 `_PastPool` A/B 双缓冲；frames 仍每帧分配（要 stack） |
| P0-2 | `cudnn.benchmark=True` 遇多变 shape 可能净负收益 | 实测无差 | 加 `FLASHVSR_CUDNN_BENCH`（默认 1） |
| P0-3 | fp8 核 CIN 小时反而慢、`patch_module` 无下限 | 真 | `MIN_CIN=32` + `FLASHVSR_FP8_MIN_CIN` |
| P1-4 | `_FEATHER_CACHE/_PI_STAGE` 全量 `clear()` 抖动 | 真 | 改逐条淘汰（羽化 24 条退 8、pinned 16 条退 4，pop 前 `synchronize` 事件） |
| P1-5 | `probe()` 重复起 ffprobe | 真 | `_PROBE_CACHE`，键 `(realpath, size, mtime_ns)`，32 条，返回副本 |
| P2-6 | `decode_frames_range` 预分配按 count | 真但影响小 | **不改**：管道带 `-frames:v count`，只有源提前结束时多分配一段；改按块增长会多一次全量 memcpy（4K 100+ MB/窗） |
| B0-1 | fp8 scale 首帧标定锁死 | 上一轮已修 | 黑场不再落锁（`amax0 > 1e-5` 才锁） |
| B0-2 | canvas 用 fp16 累加 | 真（数值上） | **默认改 fp32**；`FLASHVSR_CANVAS_FP32=0` 退回 fp16 |
| B1-3 | `_SegEncoder` 被 close 两次、close 后 submit 静默丢段 | 半真 | close 改幂等（`_done`）；「队列里的段被丢」不成立（worker 只在队列空时退出），补 close 后 submit 的告警 |
| B1-4 | bt709 注入与 qsv 10bit `format=p010le` 冲突 | **不成立** | 实测合并链 rc=0、pix_fmt/color 全对 |
| B1-5 | `_PI_CROP_KEY` 等靠模块属性传状态 | 描述准确、当前无竞态 | 不改结构（canvas 与 `_decode_cl` 都在主线程；预取线程只跑 ffmpeg） |
| B2-6 | TRT 异常后 `self.mem` 未复位 | 真 | 两处 except 补 `self.mem = None` |

**A/B 实测**（544p 源、4× 单窗 9 帧、kv=2.0、每个变体独立进程）：

| 变体 | 耗时 | 说明 |
|---|---|---|
| bench=1（默认） | 73.54 / 72.74 s | 两个进程 |
| bench=0 | 72.31 / 72.56 s | 差 <1%，属噪声 |
| canvas fp32 | 73.99 s | 比 fp16 慢 <1% |

数值影响：同配置两次运行**逐位一致**（`max|Δ|=0`），所以差异确实来自开关本身——bench on/off 差 `mean|d|=0.31/255`
（14.2% 像素，边缘最大 106），fp16/fp32 累加差 `mean|d|=0.18/255`（9.0% 像素，最大 58）。
两者都属「换 kernel / 换累加精度的正常抖动」，不是画质回归；fp32 因为重叠区最多叠 4 块、`hi_precision` 更敏感，所以做默认。

**验证**：py_compile OK；9 个测试全 PASS（新增 `tests/test_infra.py`：probe 缓存计数、羽化缓存淘汰、`_drop_stride` 与旧
逐块实现逐例对拍、`_SegEncoder` 幂等 close、`_PastPool` 交替复用）；`bench/smoke_e2e.py` VERDICT PASS。

**注意**：本机**没装 torch_tensorrt / tensorrt**（`find_spec` 均为 None），默认后端是 torch，所以 P0-1 的 `_PastPool`
只做了分配器单测 + 静态接线，**没有端到端跑过**；B2-6 同理。`FLASHVSR_DEC_BACKEND=trt/stage` 在这台机器上不可达。

**没做的收尾建议**：fp8 warm 钩子（`fp8_conv` 不被 engine import，只有 bench 探针用）；w8a16 预存 dequant scale
（只省一次 cast，真正的成本是权重 fp8→bf16 整表 cast，预存就失去省显存的意义）。`_drop_stride` 分块判据已向量化
（`np.add.reduceat` 取代 8×8 Python 循环）。