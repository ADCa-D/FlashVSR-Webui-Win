# edit 只替换子串，不保留缩进语义 → SyntaxError / IndentationError

## 报错原文
```
SyntaxError: no binding for nonlocal 'backend' found
  File "D:\flashvsr-web\engine.py", line 2342
IndentationError: unindent does not match any outer indentation level
  File "D:\flashvsr-web\engine.py", line 2371
    def budget_for(...):
```

## 根因
`edit` 的 `old_string` 是纯子串替换。
- 我写 `old_string="def encode_win(frames_in, path, audio_src=None):"`（**不带行首 8 空格**），替换文本里塞了新类 → 原来那 8 个空格留在原地，新类被"顶"成缩进 8（嵌在 `upscale_video` 体内），`encode_win` 反而掉到 0 缩进。
- 于是 `encode_win` 从嵌套函数变成模块级函数，它里面的 `nonlocal backend, bitdepth` 失去绑定 → SyntaxError；后续手工 dedent 又切断了外层缩进链 → IndentationError。

## 修法
一次性脚本 `work\_fix_indent.py`：python 读全文行、按行号切片，把 62 行类块整体搬到模块级（插在第一个 `^class ` 行之前，即原第 1848 行前），并恢复 `        def encode_win` 的 8 空格，`io.open(..., newline="\n")` 写回 → `SYNTAX_OK`。

## 教训
- **`edit` 的 `old_string` 必须以行首缩进为锚**，且带足够多的上下文保证唯一，否则会切断 Python 的缩进结构。
- 大块移动（几十行）用一次性的行号脚本，比连续 edit 反复试探更稳。
- 每次结构性编辑后立刻用只读编译验语法（见 `2026-11-17-ffmpeg-write-denied-by-sandbox.md` 末尾）。
