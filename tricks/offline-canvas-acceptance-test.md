# 用假 pipe + ast 抽纯函数，给 GPU 画布写零依赖离线验收

**场景**：FlashVSR 的分块拼接（`engine.flashvsr_gpu_canvas`）只能在 GPU 上跑，动辄要加载模型、几十秒一窗；但其中 90% 的 bug（偏移、黑边、权重为 0、tile 不对齐）都是纯几何/算术问题，CPU 就能测。

## 做法
1. 不要 `import` 那个 ComfyUI 节点包（会连带拉 src/models、huggingface_hub、可能初始化 CUDA）。
   用 `ast.parse` 从 `nodes.py` 里摘出纯函数定义，塞进一个 namespace 执行：
   ```python
   tree = ast.parse(src)
   keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in _PURE]
   exec(compile(ast.Module(body=keep, type_ignores=[]), "nodes_pure", "exec"), ns)
   ```
   项目里现成可用：`engine.load_node_module()`（engine.py:1198）已自动装 folder_paths/comfy 桩，测试也可直接复用。
2. 假 pipe：`device=cpu`、`torch_dtype=float32`、`__call__(**kw)` 直接 `return kw["LQ_video"]` —— 模型"完美复现输入"，于是拼接口有**解析解**（常数进 → 常数出、无黑边）。
3. 假 mod：挂纯函数 + `tensor2video` 手写等价（注意 `permute(1,2,3,0)`，不是 (1,0,2,3)）+ `clean_vram()` 空实现。
4. 断言挑能证明语义的，不挑数字好看：
   - `_prepare_input_gpu` 与节点 `prepare_input_tensor` **逐位等价**（含 scale 1/2/4 与非 128 倍数尺寸）；
   - 常数输入 → 常数输出，`max|Δ|` 为 0（这一条直接抓出"最外圈 1 像素黑边"）；
   - `wsum.min() > 0`、tile64 vs tile128 差异 < 阈值。

## 收益实证
这套测试一次就抓出两个真 bug：linspace ramp 首尾为 0 导致画布边缘纯黑；`tile*scale` 非 128 倍数时节点居中裁剪而画布未补偿偏移。
见 `D:\flashvsr-web\tests\test_canvas_stitch.py`。
