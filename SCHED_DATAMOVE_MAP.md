# 调度器 / 数据搬运 源码地图（给审查）

目的：把「谁决定搬什么」与「每步实际在搬什么」的代码位置列清楚，附 profile 证据，
供第三方审查判断哪些搬运是必要的、哪些是重复。

环境：RTX 5060 (sm120)、驱动 591.86、torch 2.13.0+cu130、triton-windows 3.9.0、
comfy_kitchen 0.2.35（第三方融合 kernel 包）、引擎路径 `D:\flashvsr-web\engine.py`。
节点包：`D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast`

---

## 0. 现状（mxfp8 / 1536 token / 30 block / 单步）

| 项目 | ms/步 | 占比 | 备注 |
|---|---|---|---|
| `comfy_kitchen::scaled_mm_mxfp8` | 36.0 | 40% | 720 次 / 3 步 = 240 次/步，fp8 cutlass sm120 |
| `aten::copy_` | 14.6 | 16% | **602 次/步** |
| `_attn_fwd`（sparse_sage Triton） | 11.2 | 12.5% | 60 次/步 |
| `aten::mul` | 9.8 | 11% | 361 次/步 |
| `aten::to` / `_to_copy` | 9.0 | 10% | **390 次/步** |
| `aten::clone` | 5.5 | 6% | **182 次/步（6/block）** |
| `quantize_mxfp8`（激活量化） | 4.5 CUDA + **19 CPU** | 5% | 720 次 / 3 步 |
| `aten::contiguous` | 4.3 | 5% | 121 次/步（≈4/block） |
| 合计 CUDA | 89.6 | | 墙钟 124 ms（其余是 launch 间隙） |

对照 bf16：单步 173.3 ms。**GEMM 已经是 fp8，剩余大头是搬运。**

---

## 1. 调度器（决定"搬什么、何时搬"）

| 文件 | 行数 | 关键位置 | 职责 / 与 profile 的关系 |
|---|---|---|---|
| `engine.py` | 3435 | `upscale_video` 2924；`_run_window` 2846；`budget_for` 3371；`run_win` 3388 | 外层调度：分段 / 分窗 / tile / kv / sparse 预算；决定每步提交几次 DiT |
| `engine.py` | | `_tile_housekeeping` 493 | 每 tile 的显存清理策略（注释里记录了 gc/empty_cache 的实测代价） |
| `engine.py` | | `install_stream_loader` 835 | 权重流式加载：`mxfp8` 命中量化缓存时跳过读盘（启动期，不在热路径） |
| `engine.py` | | `quantize_dit_mxfp8` 1281 / `_qcache_*` 1178-1279 | 权重量化 + 缓存（启动期）；注意**激活量化在热路径**，见 §3.1 |
| `pipelines/flashvsr_tiny_long.py` | 593 | `model_fn_wan_video` 535；主循环 363-436 | 每个 denoise step 组 batch：LQ_proj_in.stream_forward、latent 切片、cache 列表传递 |
| `pipelines/base.py` | 130 | `load_models_to_device` 94 | 模块级 CPU↔GPU 调度 |
| `vram_management/layers.py` | 95 | `AutoWrappedModule` 11-38；`AutoWrappedLinear` 41-72 | ComfyUI 路径的按层 onload/offload；**forward 里有 deepcopy，见 §3.5** |

---

## 2. 数据搬运（每步真正发生拷贝的位置）

| 文件:行 | 代码 | 频率/步 | profile 对应 |
|---|---|---|---|
| `wan_video_dit.py:433-441` `rope_apply` | `x.to(torch.float64)` → complex mul → `.to(dtype)`，Q/K 各一次 | 60 次 cast + 60 次 complex mul | `aten::to` 390/步、`mul` 361/步 的主要来源之一 |
| `wan_video_dit.py:100-119` `WindowPartition3D.partition/reverse` | `permute(...).contiguous()`，q/k/v 三次 + 反向一次 | 4/block = 120/步 | `contiguous` 121/步（对齐） |
| `wan_video_dit.py:512-516` | `k_w = torch.cat([pre_cache_k, k_w])`、v 同 | 2/block = 60/步 | 分配 + 全量拷贝 |
| `wan_video_dit.py:534-537` | `attention_mask.to(torch.int8)` 传给 kernel | 1/block = 30/步 | `_to_copy` 的一部分 |
| `wan_video_dit.py:123-160` `_local_mask_additive` | 首次展开后缓存；热路径只 `.to(dev)` 命中 | ~0（缓存后） | — |
| `wan_video_dit.py:400-430` `build_step_freqs` | 首次构建后缓存（GPU 常驻） | ~0（缓存后） | — |
| `engine.py:561-620` `_prepare_input_gpu` / `prepare_input_tensor` | 每 tile：上采样 + dtype/布局转换，**全程留在 GPU**（原版会 D2H→CPU stack→H2D） | 每 tile 一次 | 不在 DiT 步内 |
| `engine.py:623-830` `flashvsr_gpu_canvas` | 逐 tile 写入画布 + 羽化权重累加（NHWC） | 每 tile | 不在 DiT 步内，但决定总时长 |
| `wan_video_vae.py:33-53` `CausalConv3d` | 空间 padding 也走 `F.pad` → 每层 Conv3d 前多一次全量 5D 拷贝 | 每层 | VAE 解码段 |
| `wan_video_vae.py:74-80` `Upsample` | `super().forward(x.float()).type_as(x)`：强制 fp32 上采样再转回 | 最后两级 | 峰值显存 ×2、带宽 ×3 的嫌疑点 |
| `wan_video_vae.py:542 / 575` `encode/decode` | 循环内 `out = torch.cat([out, out_], 2)`（O(T²) 累积搬运） | 每帧 | VAE 段 |
| `TCDecoder.py:41-52` `MemBlock` | `torch.cat([x, past], 1)` 每层每帧 | 每层每帧 | 已用 TRT 解码时另计 |
| `sparse_sage/sparse_int8_attn.py` / `quant_per_block.py` | 注意力内部 INT8 量化 + 布局转换 | 每层 | 已计入 `_attn_fwd` 11 ms |
| `comfy_kitchen/backends/cuda/__init__.py:1729` `quantize_mxfp8` | 每次 GEMM 前对激活量化（720 次/3 步） | 240/步 | 4.5 ms CUDA + **19 ms CPU** |

---

## 3. 最可疑的 6 处（按预期收益）

### 3.1 激活量化每层每 GEMM 一次，且 CPU 开销 19 ms/步
`quantize_mxfp8` 720 次 / 3 步 = 240 次/步（30 block × 8 个 Linear）。
其中 **Q、K、V 三个 Linear 共享同一个 `input_x`**，却各量化一次 → 至少 1/3 的量化是冗余的（60 次/步）。
FFN 的两个 Linear 输入不同，不可合并。
验证：按 `x.data_ptr()` 统计量化次数；或把 QKV 合成一个 `Linear(dim, 3*dim)` 后比 ms。

### 3.2 `aten::copy_` 602 次/步 —— 来源未归因
候选：`rope_apply` 的 `.to()` 落地、`cat` 的写出、`partition/reverse` 的 contiguous、
`mask.to(int8)`、`modulation.to(dtype)`（30 次）、`freqs` 的 `.to(dev)`（缓存后应只 1 次）。
验证：带 `with_stack=True` 的 profiler 按 Python 行聚合（本机 torch 的 `key_averages(group_by_stack_n)` 会 RecursionError，需手工遍历 `prof.events()` 或用 chrome trace 解析）。

### 3.3 `aten::clone` 182 次/步（6/block）—— 不是上面的任何一处显式调用
节点代码里热路径没有 `.clone()`；`quantize_mxfp8` 的 Python 包装（comfy_kitchen 1729-1764）
可能对激活/scale 做了 clone。这是唯一还没归因的 5.5 ms。
验证：把 `quantize_mxfp8` 的 python 包装用 `torch.profiler` 单独抓一次，看 clone 是否在它下面。

### 3.4 VAE：`Upsample` 强转 fp32 + `CausalConv3d` 空间 padding 走 F.pad
`wan_video_vae.py:80` 的 `.float()` 是 torch 1.x 时代补丁；`33-53` 把空间 padding 也塞进 `F.pad`，
而 `nn.Conv3d` 本身支持空间零填充（cuDNN 内做，免费）。解码最后两级分辨率最高，这两处都在最贵的位置。
预期：显存 −1.5~2 GB、解码时间 −10~20%（需实测）。
注意：这属于 VAE 段，不在 §0 的 DiT 步 profile 里。

### 3.5 `AutoWrappedModule.forward` 里的 `copy.deepcopy`
`vram_management/layers.py:33-38`：只有 `onload_dtype == computation_dtype` 且 device 相等才跳过 deepcopy。
ComfyUI 路径会走这里（`nodes.py:188` 调 `pipe.enable_vram_management`），**引擎路径不走**。
所以：两套入口的性能可能差很多。需要确认 pipeline 传入的三组 dtype/device 是否相等。

### 3.6 VAE/TCDecoder 的循环内 `torch.cat`
`wan_video_vae.py:542/575` 每帧 `cat` 历史；`TCDecoder.py:52` 每层 `cat([x, past])`。
前者改成 `list + 一次 cat` 是纯结构改动；后者是 MemBlock 的算法要求（时序递归），只能靠预分配 ring buffer。

---

## 4. 已排除（有数据）

| 项 | 依据 |
|---|---|
| fp8 GEMM 本身 | 36 ms/步，cutlass sm120 block-scaled，已是第三方最优 kernel |
| self-attn 换后端 | 现状 Triton 0.366 ms/层；SDPA 1.135；cuDNN SDPA 0.457；sageattn 0.328（整步 −0.1%） |
| mask / rope 微优化 | 前向本身非确定（同输入连跑 rel 0.2），单次 A/B 无法分辨 |
| CUDA Graph 整步 | 真实配置 1.06×，端到端更慢；已删除（存档 `work/dit_graph_engine_block.py.txt`） |
| tile/kv 扫描 | 上游 README 已做（tile160/192 × kv2） |

---

## 5. 必须先知道的实验前提

**这个模型不是逐位可复现的**（已实测）：
- 当前代码连跑 3 次：相对差 0.19–0.23
- 全部退回 `.orig`：0.21（不是我改的）
- `use_deterministic_algorithms(True)`：仍 0.23
- self-attn 输出置零后仍 0.0042 → 基础抖动在 GEMM（≈1 bf16 ulp）
- 关闭 attention 后从 0.20 → 0.004，说明 **topk 选块把 1 ulp 放大 50 倍**

所以：任何像素级 A/B 必须 N≥5、报 mean±std 与噪声底；性能对比用 CUDA event 多次取中位数。

---

## 6. 外部依赖

| 包 | 版本 / 路径 | 相关点 |
|---|---|---|
| comfy_kitchen | 0.2.35，`python_embeded\Lib\site-packages\comfy_kitchen` | `backends/cuda/__init__.py:1729 quantize_mxfp8`、`scaled_mm_mxfp8`（4392 行 python 包装） |
| triton-windows | 3.9.0.post30 | `sparse_sage/*` 三个 Triton kernel |
| torch | 2.13.0+cu130 | SDPA 无 flash kernel；cuDNN attention 默认 runtime disabled |
