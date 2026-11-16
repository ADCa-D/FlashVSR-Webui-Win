# tricks 索引（正面经验，可直接抄）

| 主题 | 一句话 | 文件 |
|---|---|---|
| 离线验收 GPU 画布 | 用假 pipe + ast 抽 nodes.py 纯函数，CPU 上跑拼接语义断言，抓出黑边/错位 | [offline-canvas-acceptance-test.md](offline-canvas-acceptance-test.md) |
| 逐位等价验收 | 改推理前处理时，用"同路径重跑 → 单点 A/B → 下沉到最小单元"三步定位，脚本 `bench/check_prepare_equiv.py` | [bit-exact-equivalence-check.md](bit-exact-equivalence-check.md) |
