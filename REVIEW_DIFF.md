# 待审查改动包（FlashVSR / Wan-DiT 侧）

生成时间：2026-11-19
目的：把这一轮所有改动、验证数据、我自认的风险点，原样交给第三方审查。
配套文件：
- `work/diff_wan_video_dit.patch` —— `wan_video_dit.py.orig`（改动前）vs `wan_video_dit.py`（现在）的完整 diff，10 个 hunk
- 备份：`src/models/wan_video_dit.py.orig`（改动前原样）、`wan_video_dit.py.bak`（上游自带，与 .orig 内容相同）

---

## 0. 一句话摘要

改动分三类，**权限/风险完全不同**，请分开审：

| 类 | 内容 | 默认是否生效 | 风险 |
|---|---|---|---|
| A. 等价重构 | mask 展开缓存、`rearrange`→`reshape`、`cu_seqlens` 移入分支、freqs 表缓存在 GPU | **是** | 低（已验证逐位一致） |
| B. 行为修复 | `is_stream` 3 元组解包、`kv_len=None` 比较、freqs 越界扩表、mask 键改参数化 | **是** | 低（只在原代码会崩的路径上生效） |
| C. 新功能 | DiT 整步 CUDA Graph（`_StepGraph` / `_DiTGraphRunner`） | **否**（`FLASHVSR_DIT_GRAPH=1` 才开） | 中高（裸指针、显存） |
| D. 新增开关 | `FLASHVSR_MASK_TOPK=1` 用 topk indices 代替阈值比较 | **否** | 中（会改数值） |

---

## 1. 文件与改动清单

### 1.1 `src/models/wan_video_dit.py`（主文件，见 patch）

| # | 位置 | 改动 | 依据 |
|---|---|---|---|
| 1 | `flash_attention` | `cu_seqlens_q/k`、`head_mask_type` 从“无条件创建”移入 `if USE_BLOCK_ATTN and BLOCK_ATTN_AVAILABLE:` 分支 | 这两个张量只有 `block_sparse_attn_func` 用；sage 路径每层白做 3 次 H2D。也是 CUDA Graph 可捕获的前提 |
| 2 | 新增 `_local_mask_additive/_mask_sig/_select_topk_mask`、`_LOCAL_MASK_CACHE/_MASK_FP` | mask 展开按**参数**缓存（sig 由 `SelfAttention` 传入），返回 `[1,L,S]` 让 head 维走广播；非 sage / sage 用 `kv_double` 区分是否 `repeat_interleave(2)` | 原实现每层重建一张 mask，30 层内容相同 |
| 3 | `SelfAttention.forward` | `reorder_q/k/v` 与反变换由 `rearrange` 换成 `reshape`；新增 `mask_sig=(h//8, w//8, local_range, True)` 传给两个 mask 函数 | 模式是纯 view，等价零拷贝 |
| 4 | `SelfAttention.forward` | `if cache_num > kv_len:` → `if kv_len is not None and cache_num > kv_len:` | 非流式路径 `kv_len=None` 会 `TypeError` |
| 5 | `DiTBlock.forward` | 3 元组解包改为按 `is_stream` 分支接收 | 非流式只返回单张量 |
| 6 | 新增 `build_step_freqs` | 3D RoPE 表在 GPU 上按 `(f,h,w,idx,表形状,device)` 缓存；`off+f > 表长` 时按需扩 f 轴 | 原来每步在 CPU cat 三个 complex128 再 H2D；也是 capture 的前提 |
| 7 | 两个 `generate_draft_block_mask*` | 选块逻辑抽到 `_select_topk_mask`，默认仍是 `flat > 第(k+1)大值`；`FLASHVSR_MASK_TOPK=1` 时用 `topk.indices + scatter_` | 默认路径逐位不变 |
| 8 | `rope_apply` | **未改**（保留 complex128），只加了注释说明为什么实测不该改 | 实数 fp32 版实测慢 2.7 ms，且引入 1-ulp 抖动 |

### 1.2 `src/pipelines/flashvsr_{full,tiny,tiny_long}.py`

每份各两处（三份内容相同）：
- `from ..models.wan_video_dit import ... , build_step_freqs`
- `model_fn_wan_video` 里的 `if cur_process_idx == 0: freqs = torch.cat([...]) else: ...` → `freqs = build_step_freqs(dit, f, h, w, cur_process_idx, x.device)`

### 1.3 `engine.py`

- 新增 457–725 行：`_StepGraph`、`_DiTGraphRunner`、`install_dit_graphs`（整步 CUDA Graph，默认关）
- `Engine.__init__` 增加 `self.dit_graph = None`
- `ensure_loaded` 在 `init_pipeline` 之后装 runner 并 `reset()`（换模型/量化后旧图里钉的裸指针失效）

---

## 2. 我自认值得重点质疑的地方（请优先审这些）

1. **`_local_mask_additive` 的 `kv_double` 分支是否真的与两条旧路径等价**
   我做了 6 组 (bh,bw,repeat_len,repeat_num,repeat_head)×(sage/非sage) 的逐位对比，全 True。
   但**非 sage 路径在本机是死代码**（`BLOCK_ATTN_AVAILABLE=False`），只做了构造对比、没跑过真实前向。
2. **`_mask_sig` 作为缓存键是否足够唯一**
   键 = `(h//8, w//8, local_range, True)` + repeat 参数 + scores 形状 + dtype + device。
   风险：若将来引入 `include_self=False` 或不同窗型（`win`），sig 不含这些 → 会静默命中旧掩码。
3. **`build_step_freqs` 的扩表是否正确**
   只抽查了 `idx=520` 时 frame 1049 的第 0 个复数（与手算 cos/sin 一致）。整表未逐元素验证。
4. **CUDA Graph 的裸指针安全**
   `reset()` 在 `ensure_loaded` 里调用，但**如果有人在同一段内 `unload_dit` 把 DiT 搬回 CPU 再搬回来**，图里的指针就悬空了；我只在 `run()` 里比了 `device`，没比 data_ptr。
5. **`_StepGraph` 的 copy-in 覆盖是否完整**
   我拷贝了 `x / lq / 每层 pre_cache_k/v / t_mod / t / freqs`。
   `timestep`、`context` 在 capture 时直接传 `None`（原函数里 `timestep` 只给 TeaCache 用，`context` 只喂 cross-attn 且 cross-attn 用预缓存）——**这是基于阅读代码的推断，不是实测**。若将来 pipeline 改成用 `context`，图里会拿到 None。
6. **`mask=None` 分支（sageattn）是否会破坏跨块依赖**
   我把它当可选实验，没有合入默认路径。

---

## 3. 已验证的数据（含方法，便于复核）

| 结论 | 方法 | 结果 |
|---|---|---|
| 输出逐位不变 | 同 seed/同输入，改前改后各跑 1 步，比 `y` | `maxdiff 0.0000`，eager 与 graph 两条路径都是 |
| mask 展开等价 | 新旧实现各算一遍，`torch.equal` | 6 组形状全 True |
| `is_stream=False` 修复 | 直接调 `DiTBlock.forward(is_stream=False)` | 返回 `(1,128,256)`，修复前 `ValueError` |
| freqs 扩表 | `idx=520`，比手算 cos/sin | `0.957686-0.287817j` 一致 |
| 端到端确定性 | 同配置跑两遍，算 PSNR | **99.00 dB**（完全可复现） |
| mask 选块改动的影响 | 只切 `_select_topk_mask`，两段内容各跑一遍 | 40.3 / 41.3 dB，逐帧 \|diff\| 0.66–0.80 灰度级，锐度 +0.03%/+0.15% |
| 掩码饱和 | 真实潜变量统计 | 覆盖率 95.8%(tile128) / 99.0%(tile256)，`apply_topk` 恒 = 列数-1 |
| 空 mask 行为 | 传全 0 `mask_id` | `nan=0 inf=0`，输出整行为 **0**（不是 NaN） |
| attention 后端对比 | 真实形状 q/k/v `[1,12,1536,128]` | 现状 0.366 ms / SDPA 1.135 / 3×GEMM 1.576 / cuDNN SDPA 0.457 / sageattn 0.328 |
| attention 换 sage 的整步收益 | 只把 mask 函数换成返回 None | 174.7 → 174.6 ms（**0.0%**） |
| 单步时间构成 | torch.profiler | GEMM 71%，self-attn 6.3%，SDPA 7%，elementwise ~10% |

脚本都在 `bench/`：
`bench_dit_step.py`、`bench_dit_stream.py`、`ab_mask_select.py`、`probe_topk_saturation.py`、
`bench_attn_backends.py`、`bench_sageattn.py`、`ab_sage_dense.py`、`verify_dit_fixes.py`、
`bench_dit_graph`（`probe_dit_step_graph.py`）、`ab_dit_graph_e2e.py`

---

## 4. 我已知会误导人的说法（已更正，但可能还有）

1. 一开始说 “DiT 整步 CUDA Graph 1.63×” —— 那是 `topk_ratio=2.0` 的小网格配置。真实配置（22.5）下只有 1.06×，端到端甚至更慢。
2. 说 “attention 占单步 17%” —— 那是 384 token 的小网格；真实 1536 token 下是 6.3%。
3. 说 “空 mask 会 NaN” —— 实测是全 0，不是 NaN。
4. 说 “rope 改实数 fp32 更快” —— 方向反了，实测慢 2.7 ms。

---

## 5. 回滚方式

```powershell
# 单文件回滚（唯一有完整备份的）
Copy-Item "src\models\wan_video_dit.py.orig" "src\models\wan_video_dit.py" -Force

# pipeline 三份：把 freqs 那行换回原来的 if/else 块（patch 里没有这三份，改动仅 2 处/份）
# engine.py：删掉 457–725 行 与 ensure_loaded 里的 try 块，以及 __init__ 的 self.dit_graph = None
```

依赖新增：`pip install sageattention`（当前已装 1.0.6，只在我手写的 bench 里用，主路径不 import）。
