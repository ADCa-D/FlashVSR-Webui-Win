# GPU 上做 `uint8/255`：1 ULP 差异被 fp8 量化放大成 37 dB

## 现象
把「整窗 float32 预转换」改成「整窗保持 uint8、每个 tile 上卡再 /255」之后，同 seed 同输入的输出与旧路径对比：

```
A vs C（新路径 vs 旧路径）: max|Δ|=235.0 mean=0.75 PSNR=36.89 dB  >4:4.16%  >16:1.021%
A vs A（同路径重跑）:        max|Δ|=0.0   mean=0.00 PSNR=168.13 dB   （流水线本身完全确定）
```
说明不是随机性，是输入的确定性偏差被放大。

## 根因
逐层定位（不加载模型，直接比 prepare 的输出）：
1. 喂**完全相同的 float32 [0,1]** 时，GPU 版 prepare 与节点版 `prepare_input_tensor` **逐位一致**（diff=0/1769472）。所以插值、bf16 cast、`*2-1` 的顺序都不是问题。
2. 差异只出现在 uint8 直通路径。实测：
```
gpu 先 float 再 div_(255.0)   diff=1735424/3525120  max=5.960e-08   ← 一半元素差 1 个 float32 ULP
gpu div(255) / mul_(1/255)   同上
cpu float().div_(255.0)      基准
```
即 **CUDA 的 float32 除法与 CPU 的结果差 1 ULP**（GPU 走倒数乘）。1 个 float32 ULP 在 bf16 之后偶尔跨过舍入边界（`max|Δ|=7.812e-03` = bf16 的 2^-7），再经 MXFP8 逐层 block scale 放大，整段输出掉到 37 dB。

## 修法
`engine._prepare_input_gpu`：uint8 输入**在 CPU 上**做 `/255`，只对当前 tile 的切片（512 KB 级），随后 H2D：
```python
src = image_tensor.float().div_(255.0).to(device=device, non_blocking=True)
```
整窗仍然是 uint8（省掉 199 MB/窗 的 float32 驻留），float 转换摊到每个 tile，代价可忽略。改完 prepare 与节点版 diff=0。

## 教训
- **"数学上等价"不等于"逐位等价"**：设备间 1 ULP 差异在量化流水线里会被放大两个数量级，验收必须比逐位，不能只看肉眼或 PSNR 的粗口径。
- 定位顺序：先做"同路径重跑"（证明是否确定性）→ 再做"只改一处的 A/B" → 最后把改动**下沉到最小单元**单独比（这里是 prepare，不需要加载模型，秒级）。
- `bench/check_prepare_equiv.py` 就是这个下沉对比脚本，以后改 prepare/canvas 先跑它。
