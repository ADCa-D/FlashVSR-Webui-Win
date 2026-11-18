# DiT/VAE 性能改动交接（2026-11-19）

同配置基准：RTX 5060（sm120）/ `tiny-long` / `mxfp8` / DiT 网格 16×16（1536 token）/ 30 block。

## 1. 结果

| 项 | 数值 |
|---|---|
| DiT 单步 | **124.3 → 83.0 ms（−33%）** |
| 端到端（100 帧 480×272→960×544） | **34.2 → 28.6 s（−16.6%，1.20×）**，4 次运行均 100 帧 |
| VAE 解码（full 模式，9×32×32 latent） | −7.1% 时间、−20% 峰值显存 |
| 峰值显存（DiT 步） | 不变（4.10–4.29 GiB） |

## 2. 开关（默认已是最优组合）

| 环境变量 | 默认 | 实测收益 | 说明 |
|---|---|---|---|
| `FLASHVSR_FUSE_QKV` | 1 | **−8.7%** | q/k/v 权重拼成 `[3D,D]`，一次量化 + 一次 GEMM |
| `FLASHVSR_SKIP_MASK` | 1 | **−12.7%** | 掩码饱和（`k ≥ 列数−1` 且局部窗全 True）时跳过构造 |
| `FLASHVSR_FUSED_NORM` | 1 | **−13.3%** | `F.rms_norm` 单 kernel |
| `FLASHVSR_FUSED_MODULATE` | 1 | −1.5% | `torch.addcmul` |
| `FLASHVSR_QUANT_REUSE` | **0** | −0.6% | 激活量化缓冲复用；**不值得**，见 §5 |
| `FLASHVSR_MASK_TOPK` | **0** | ~0 | 用 topk indices 代替阈值比较；改数值，仅作选项 |

三项主改动近似可加（−13.3 −12.7 ≈ −24%，实测 −21.9%）。

## 3. 热点函数（消融法，基线 106.6 ms/步）

| 消融项 | Δ | 结论 |
|---|---|---|
| `quantize_mxfp8` | −15.0 | 主要是“读热内存”的假收益，真实可省只有 0.5 |
| `RMSNorm.forward` | −14.4 | ✅ 已融合 |
| `mask` 构造 | −11.6 | ✅ 已跳过 |
| `modulate` | −3.8 | ✅ 已压成 2 遍读写（实得 1.5） |
| `rope_apply` | −2.0 | 不做（Gemini 估 4–12，实际 2） |
| self-attn sparse → SDPA | **+13.3** | 自定义 Triton kernel 明显更快，别换 |

完整的表与脚本见 `review/HOT_FUNCS.md`、`bench/hot_ablate.py`。

## 4. 方法：怎么测的（重要）

这个模型**不可逐位复现**（同输入连跑相对差 0.19–0.23），所以：

- 单元级：固定输入、同进程、同 kernel 才谈“相同”，且**必须用非退化输入**
  （踩过：全 True 掩码掩盖了 permute 错位）
- 步级：CUDA event、N≥5、交替进程、取中位（`bench/ab_*.py` 都是这个形状）
- 端到端：N≥2 交替、报中位 + 帧数
- 完整协议：`VERIFY_PROTOCOL.md`

三个失败的工具路径，别再试：

| 方法 | 现象 |
|---|---|
| `profiler(with_stack=True)` + `key_averages(group_by_stack_n)` | 函数名被截断，看不到归因 |
| 同上 + `sys.setrecursionlimit(200000)` | profiler 递归建树，**卡死**（吃 2.7 GB 显存无进展） |
| 逐调用 CUDA event | 嵌套 + 队列反压 → 区间和 139.9 ms > 总时长 101.9 |
| NVTX range | 这个 torch 构建不开 CUDA 用户注解、trace 里也没有 |

## 5. 明确不要再做的（已证伪）

| 想做的事 | 实测 |
|---|---|
| 激活量化缓冲复用（109 行实现） | **−0.6%**：单步已 GPU-bound；消融的 −15 是 L2 热命中假象 |
| RoPE 改实数 fp32 | 更慢 2.7 ms（kernel 数变多），且引入 1 ulp 抖动 |
| 换 SageAttention / SDPA | 整步 **0.0%**（sage 1.12× on kernel，但 attention 只占 6%） |
| 整步 CUDA Graph | 真实配置 1.06×，端到端 **更慢**；代码已删除（存档 `work/dit_graph_engine_block.py.txt`） |
| 真·稀疏（0.5×列数） | 收益 <3%，但可能引入跨帧闪烁；当前配置本来就是 99% dense |
| VAE 的 `.float()` / `F.pad` / 循环 cat | 只对 **ComfyUI `full` 模式**有效，engine 的 tiny/tiny-long 不经过 VAE |

## 6. 已知风险 / 边界

- `SKIP_MASK` / `FUSED_NORM` / `FUSED_MODULATE` / `FUSE_QKV` 都会改变输出
  （1 个 bf16 ulp 或每行少 1 块）；在 0.2 的噪声底以下，**无法用 diff 证明画质不变**，
  只能按结构等价 + 消融收益接受。不接受就单独关掉对应开关。
- VAE 的 `CausalConv3d` 换 padding 方式后，TF32 开启时输出与旧版差 ~8e-4
  （cuDNN 换了算法，不是逻辑错；关 TF32 时逐位相同）。
- `sparse`（`topk_ratio`）在当前 tile/kv 下是**死参数**：预算恒为可用列数的 ~10×sparse 倍，
  默认 1.5 → 15 倍 → 永远被 `min(列数−1, topk)` 截断。要它生效得 < 0.1。
- 节点改动以补丁形式保存在 `node_patches/`（含回滚方式）。

## 7. 复现

```powershell
# 步级 A/B（三项主改动）
python bench/ab_mask_norm.py
python bench/ab_fused_qkv.py
python bench/ab_modulate.py

# 端到端
python bench/ab_e2e_today.py

# 热点（消融）
python bench/hot_ablate.py

# 噪声底
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"; python bench/make_deterministic.py
```
