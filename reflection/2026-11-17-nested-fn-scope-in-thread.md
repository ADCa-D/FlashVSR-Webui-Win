# 把 `upscale_video` 里的闭包挪到后台线程：`NameError: name 'encode_win' is not defined`

## 报错原文
```
File "D:\flashvsr-web\engine.py", line 2531, in upscale_video
    _drain()
File "D:\flashvsr-web\engine.py", line 2527, in _drain
    raise RuntimeError(f"第 {idx + 1} 段编码失败：{err}")
RuntimeError: 第 1 段编码失败：name 'encode_win' is not defined
```

## 根因
`_SegEncoder`（D:\flashvsr-web\engine.py:1852）是**模块级类**，它的 `_run()` 在 `encode_win(frames, path, None)` 里直接按名字找全局变量；
而 `encode_win` 是 `Engine.upscale_video`（D:\flashvsr-web\engine.py:2345）内部的**嵌套函数**，只在那个调用的栈帧里存在，模块级根本看不见。

而且这个错误是"延迟爆炸"的：GPU 推理正常跑完、日志全是正常进度，**直到下一个 `_drain()` 才抛**——表面上像"写盘到一半失败"。

## 修法
把闭包显式传进线程对象，不要依赖名字查找：
```python
class _SegEncoder(object):
    def __init__(self, fn, max_pending=2):
        self._fn = fn          # encode_win 定义在 upscale_video 内部，必须显式传入
        ...
        self._fn(frames, path, None)

enc = _SegEncoder(encode_win) if async_encode else None
```

## 教训
- 任何"模块级类 + 调用方内部闭包"的组合，闭包**必须作为参数传**；别指望作用域能串起来。
- 后台线程里的失败要在**提交点**就冒泡（drain 每段开头都有），否则错误会滞后一整段，日志看起来"一切正常"。
- 异步化的正确性验收必须跑**分段**路径（`chunk < total`）；不分段时 `enc` 根本不接管，同一个 bug 测不出来。
