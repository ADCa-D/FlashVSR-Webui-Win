# 节点侧补丁（ComfyUI-FlashVSR_Ultra_Fast）

这两个补丁是 DiT/VAE 的改动。节点的源码在
`ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast\src\models\`，**不在本仓库里**
（ComfyUI 把 custom_nodes 当未跟踪目录），所以以补丁形式保存。

| 补丁 | 内容 |
|---|---|
| `wan_video_dit.py.patch` | Fused QKV 分支、mask 饱和跳过、RMSNorm 融合、modulate 融合、mask 缓存修正（去掉错误的 permute）、`is_stream=False` 解包、`kv_len=None`、freqs 扩表 |
| `wan_video_vae.py.patch` | CausalConv3d 空间 padding 交回 nn.Conv3d、Upsample 去 `.float()`、循环 cat→list、encode 尾帧补齐、encode 的 tile_size 循环内累乘、feat_idx 可变默认参数 |

## 打补丁

```powershell
cd "D:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-FlashVSR_Ultra_Fast\src\models"
git apply -p1 D:\flashvsr-web\node_patches\wan_video_dit.py.patch
git apply -p1 D:\flashvsr-web\node_patches\wan_video_vae.py.patch
```

补丁只针对 .orig 基线；如果节点被上游更新过，先 `git apply --check` 确认。

## 回滚

改动前的原文件就在同一目录：`wan_video_dit.py.orig`、`wan_video_vae.py.orig`：

```powershell
Copy-Item wan_video_dit.py.orig wan_video_dit.py -Force
Copy-Item wan_video_vae.py.orig wan_video_vae.py -Force
```

## 注意

- 这些改动**会改变输出**（mask 少选 1 块 / RMSNorm 1 ulp / modulate 1 ulp），
  且本模型本身不可复现（同输入连跑差异 0.19–0.23），所以**无法用逐位对比验证**。
  验收见仓库根目录 `VERIFY_PROTOCOL.md`。
- 收益与开关见 `HANDOFF_DIT_PERF.md`。
