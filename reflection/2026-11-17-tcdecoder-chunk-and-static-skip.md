# TCDecoder 分块 / 静止块跳过：两条路都实测否决

日期 2026-11-17 · 素材 544p 33 帧（`work` 侧，同 seed）· GPU RTX 5060 Laptop 8 GB · torch 2.13.0+cu130

## 一、空间分块（否决）

探针 `bench/probe_tcdec_chunk.py`，tile256 输出（latent 32²、T=6、21 输出帧）：

| 配置 | 耗时 | 峰值 alloc | PSNR vs baseline |
|---|---|---|---|
| baseline `parallel=False` | **117 ms** | **253 MiB** | 参考 |
| `parallel=True` | 1659 ms | 5602 MiB | 61.8 dB |
| time-chunk k=1..4 | 115–119 ms | 281–300 MiB | **99 dB（max abs diff 0，逐位一致）** |
| strips=2 halo=0 | 1010 ms | 2896 MiB | 37.9 dB（撕裂） |
| strips=2 halo=4 | 1235 ms | 3582 MiB | 61.8 dB |
| strips=4 halo=4 | 1024 ms | 2209 MiB | 61.1 dB |

tile128：baseline 46 ms / 131 MiB，`parallel=True` 654 ms / 1471 MiB。

根因：条带之间必须 `clean_mem()` 丢掉时间轴状态才能拼，否则撕裂；一旦 clean，条带内小张量占用率崩 → **慢 9–10×**，峰值还反涨（每步重新分配）。
**解码器峰值只有 253 MiB**，显存从来不是它的瓶颈，所以"分块省显存换速度"的前提不成立。

踩坑：`torch.cat(parts, 2)` 写错维度（parts 是 (B,T,3,H,W)，拼接轴是 dim=3，写成 dim=2 会得到 6 通道且尺寸减半）。

## 二、静止块跳过（未转正）

探针 `bench/probe_static_skip.py` + 端到端 `bench/reuse_run.py`（33 帧 544p，tile128，mxfp8，`params["reuse"]="tile"`）。

### 源码事实（必须先知道）
- `src/pipelines/flashvsr_tiny_long.py:443` 的 `decode_video` 是**按时轴 chunk** 调用（不是一个 tile 一次）：135 次 ≈ 15 tile × 9。
- `src/models/TCDecoder.py:263` `x, self.mem = apply_model_with_memblocks(...)` 会改写 `self.mem`，而 `flashvsr_tiny_long.py:472` 只在**窗口末尾** `clean_mem()` → **同窗口内 tile 之间共享 mem**。
- 所以"跳过某格 decode" = 这一格的 mem 不推进 → 污染下一格。

### 实测污染与成本
- 单 chunk decode ≈ 117 ms（tile256）/ 46 ms（tile128）。
- 不隔离跳 tile2：tile3 掉到 **37.5 dB（mean 1.5/255）**、tile4 73 dB、tile5 逐位一致 → 污染只传一格且快速衰减。
- 每 tile `clean_mem()`：跳过零污染，耗时 **+0.1%（703 vs 702 ms，免费）**；但**隔离本身改变输出**（ref_b vs ref_a 从 tile1 起 18.1 dB / mean 25/255）→ 不能默认加。

### 端到端 A/B（同 seed，upscale 段计时）
| 配置 | 跳过 | 耗时 |
|---|---|---|
| off | 0/135 | 37.19 s / 38.09 s / 38.27 s（三次） |
| pos@0.002，CPU 缓存、每 chunk 判定 | 19/135 (14.1%) | **38.88 s（净慢 +2.1%）** |
| pos@0.002，GPU 缓存、每 tile 判定 | 0/135 | 37.28 s |
| pos@0.006 | 未跑完（用户叫停） | — |

### 三个致命细节
1. **位置键必须带 chunk 序号**。第一版键是 `(pos, cond.shape, x.shape)`，不同 chunk 形状相同会撞键 → 那 19 次"跳过"是拿错 chunk 的输出，属误判。
2. **每 chunk 一次 `.item()` 强制同步**就把收益吃掉：跳过 14% 的 decode 本该省 2 s 量级，结果净慢 0.8 s。同步点会排空 GPU 队列，破坏 CPU 提交 / GPU 执行的流水。
3. **阈值失真**：改成 GPU 缓存 + 每 tile 只判一次后，thr=0.002 命中 **0 次** —— cond 是 LQ 在 −1..1，静止块的 `mean|Δ|` 就卡在 0.002 之上。旧的"静止块跨窗差 0.26–0.41/255"是 **RGB 0..255 尺度**，换算到 cond 大致 0.002–0.003，正好落在阈值边缘。要吃收益必须抬阈值 = 拿画质赌。

## 三、结论

- 分块：**删掉**。HANDOFF §9-1 的"分块"这半边是错的（见 HANDOFF 已改）。
- 静止块跳过：**不进默认**，实验到此为止。真正卡死它的是 ①mem 跨 tile 共享（跳过会污染下一格，隔离会改变输出）②同步开销 ③阈值与画质耦合。除非把 statics 判定做进 GPU 内核（无需 host 回读），否则没有正收益空间。

## 四、可复用的教训

- 给 `decode_video` 打补丁做复用前，先确认它是 per-chunk 调用还是 per-tile 调用（`grep -n decode_video src/pipelines/*.py`）。
- 任何"跳过计算"的优化，先在**同一进程内**量出"同步一次"的成本；`.item()`/`.cpu()` 在流式 pipeline 里的代价远大于朴素估算。
- 阈值类实验必须先量出**判据量纲**（RGB 0..255 还是 −1..1），否则结论会差一个数量级。
