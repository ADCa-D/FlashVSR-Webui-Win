# 源码审查包（调度 / 数据搬运）

这是**原样摘录的源码**，不是问题清单——目的是让审查者看到真实实现再判断。

生成方式：`bench/make_review_bundle.py`（ast 按函数/类边界切片，行号为文件内真实行号）。
重新生成：`python bench/make_review_bundle.py`

| 文件 | 内容 | 代码量 |
|---|---|---|
| `01_dit_hotpath.py.md` | DiT 每步 30 个 block 的实际算子：RoPE、窗口划分、稀疏选块、Triton 注意力入口、RMSNorm、SelfAttention、DiTBlock | 424 行 / 22 KB |
| `02a_engine_schedule.py.md` | 外层调度：`upscale_video`（510 行，含分段/分窗/复用/续跑）、`_run_window`、tile 坐标、显存清理 | 638 行 / 35 KB |
| `02b_engine_dataflow.py.md` | tile 预处理（GPU 版 vs 节点原版）、画布羽化混合、权重流式加载、MXFP8 权重量化与缓存 | 418 行 / 23 KB |
| `03_vae_tcdecoder.py.md` | VAE：`CausalConv3d` / `Upsample` / `Resample` / `VideoVAE_` / `WanVideoVAE`（含 tiled_decode）；TCDecoder：`MemBlock` / `TAEHV` | 612 行 / 26 KB |
| `04_pipeline_vram_quant.py.md` | 管线主循环 `__call__`、`model_fn_wan_video`、`load_models_to_device`、VRAM 管理层、comfy_kitchen 的 `quantize_mxfp8` 及对齐/克隆辅助 | 407 行 / 19 KB |

---

## 实测基线（必须先知道，否则无法判断"值不值得改"）

环境：RTX 5060 (sm120)、驱动 591.86、torch 2.13.0+cu130、triton-windows 3.9.0、comfy_kitchen 0.2.35。
配置：`tiny-long` / `sparse` / `mxfp8` / tile 128（DiT 网格 16×16，1536 token）/ kv 2 / 30 block。

单步（torch.profiler，3 步平均）：

| 项 | ms/步 | 占比 | 次数/步 |
|---|---|---|---|
| `comfy_kitchen::scaled_mm_mxfp8` | 36.0 | 40% | 240 |
| `aten::copy_` | 14.6 | 16% | **602** |
| `_attn_fwd`（sparse_sage Triton） | 11.2 | 12.5% | 60 |
| `aten::mul` | 9.8 | 11% | 361 |
| `aten::to` / `_to_copy` | 9.0 | 10% | **390** |
| `aten::clone` | 5.5 | 6% | **182** |
| `quantize_mxfp8` | 4.5 CUDA + **19 CPU** | 5% | 240 |
| `aten::contiguous` | 4.3 | 5% | 121 |
| 合计 | 89.6 CUDA | | 墙钟 124 ms |

对照：同配置 bf16 单步 173.3 ms。**GEMM 已经 fp8，剩下的是搬运。**

---

## 希望审查者回答的具体问题（按文件）

### `01_dit_hotpath.py.md`

1. `SelfAttention.forward`（468–555）里每 block 有 4 次 `contiguous()`（`WindowPartition3D.partition` ×3 + `reverse` ×1，见 100–119）。
   能否把窗口划分与 `b n s d` 布局转换合并成一次 permute+contiguous？
2. `rope_apply`（433–438）：`x.to(float64)` → complex 乘 → `.to(dtype)`。
   实测过"改成实数 fp32"反而慢 2.7 ms（多了 5–6 个 kernel），所以现在保留 fp64。
   在 fp8 GEMM 已经把总时间压到 124 ms 后，这里是否值得用单个 Triton kernel 融合？
3. `generate_draft_block_mask_sage`（234–279）：`flat > 第(k+1)大值` 在并列时会少选块
   （实测 1.3% 的行少于 k，最差 24/95）。改成 `topk.indices + scatter_` 的代价与风险？
4. 稀疏度：实测 `apply_topk` 恒 = 列数−1（95/96，覆盖率 99%），即这套"稀疏注意力"实际是 dense 减 1。
   如果改成真正稀疏（比如 k = 0.5×列数），画质代价怎么看？

### `02a_engine_schedule.py.md`

5. `upscale_video`（2924–3434）里的分段/续跑/复用逻辑：每个 tile 之后调用 `_tile_housekeeping`，
   以及"静止块复用"。哪些路径会在热循环里做隐式同步或全量拷贝？
6. `_run_window`（2846–2922）→ 管线 `__call__` 的参数传递是否有可以合并的重复转换？

### `02b_engine_dataflow.py.md`

7. `_prepare_input_gpu`（561–592，替换节点 `prepare_input_tensor`）与节点原版（`nodes.py:75`）都摘了，
   逐行对比：GPU 版是否真的消掉了 D2H→CPU→H2D？
8. `flashvsr_gpu_canvas`（623–746）：逐 tile 写入 + 羽化权重累加。有没有多余的整画布读写？
9. `quantize_dit_mxfp8`（1281–1352）里的 `_call`（1288–1293）：
   `x.reshape(-1, D).contiguous()` + `quantize_mxfp8` + `scaled_mm_mxfp8`，每个 Linear 各来一次。
   **Q/K/V 共享同一个输入**，能否只量化一次？`x.reshape().contiguous()` 是否必然产生拷贝？

### `03_vae_tcdecoder.py.md`

10. `CausalConv3d`（33–53）：空间 padding 也走 `F.pad`（每层全量 5D 拷贝）。
    `nn.Conv3d` 自带空间零填充，能否把空间 padding 交回去、只对时间维 pad？
11. `Upsample`（74–80）：`super().forward(x.float()).type_as(x)`。PyTorch 2.13 下 bf16 nearest-exact
    是否已无必要？去掉后峰值显存/带宽预计变化多少？
12. `VideoVAE_.decode/stream_decode`（553–625）：循环内 `torch.cat([out, out_], 2)` 的 O(T²) 累积，
    改 `list + 一次 cat` 是否等价（注意因果关系/tiling 边界）？

### `04_pipeline_vram_quant.py.md`

13. `AutoWrappedModule.forward`（11–38）：当 `onload_dtype != computation_dtype` 时**每次前向 deepcopy**
    整个模块。ComfyUI 路径会走这里（`nodes.py:188` 调用 `enable_vram_management`），引擎路径不走。
    请判断 pipeline 传入的参数是否可能触发这条分支。
14. `comfy_kitchen.quantize_mxfp8`（1729–1762）：`torch.empty` + `torch.zeros` + `_wrap_for_dlpack`。
    每步 240 次调用对应 240 次 `zeros`（memset）与 240 次包装；有无批量/预分配方案？
15. profile 里 `aten::clone` 182 次/步（6/block），但节点热路径源码里没有显式 `.clone()`
    （唯一的克隆点在 `models/utils.py` 的 VAE cache 与 `wan_video_vae.py:131/167`）。
    请从摘录代码里找可能的来源。

---

## 读这份代码前必须知道的两件事

1. **模型不是逐位可复现的**（已实测）：
   - 当前代码连跑 3 次相对差 0.19–0.23；全部退回上游 `.orig` 仍 0.21
   - `torch.use_deterministic_algorithms(True)` 无效
   - 把 self-attn 输出置零后仍 0.0042（≈1 bf16 ulp），说明基础抖动在 GEMM
   - 从 0.004 → 0.20 的放大来自 **topk 选块的离散性**
   → 任何像素级 A/B 必须 N≥5 取 mean±std 并报噪声底；性能对比用 CUDA event 取中位数。

2. **`sparse` 参数在当前配置下是死的**：`topk_ratio = sparse×768×1280/(tile·scale)²`
   乘上 `square_num` 后恒为可用列数的 ~10×sparse 倍，默认 1.5 → 15 倍 → 永远被
   `min(列数−1, topk)` 截断。要它生效需 < 0.1。
