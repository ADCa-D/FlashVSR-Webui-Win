# ffmpeg 在本机沙箱里写盘被拒（partial enforcement）

## 报错原文
```
Error opening output D:\flashvsr-web\work\_pv_probe\t.avif: Permission denied
FFMPEG_EXIT=-13
```
```
PermissionError: [Errno 13] Permission denied: 'D:\flashvsr-web\__pycache__\engine.cpython-313.pyc.…'
FFMPEG_EXIT=-13 时 encode 探测静默回落到 jpg → tests/test_encoders.py 两个用例 FAIL
tests/test_tune_encode.py 13 项全是 BrokenPipeError: [Errno 32] Broken pipe
```

## 根因
1. 本机 DSH 沙箱 `enforcement: partial`：python 进程自己写工作区没问题，但 python 拉起的**子进程**（ffmpeg.exe）写同一目录会被拒，rc=4294967283 = `0xFFFFFFF3`。
2. 另有 ACL 层面问题：`D:\flashvsr-web\work` 及其 273 个子项**关掉了 ACL 继承**，只有 `D:\` 的通用 ACE（Administrators/SYSTEM/Authenticated Users/Users），既没有 `MECHREVO` 的 FullControl，也没有沙箱授权 SID `S-1-4-975549821-1049304264`（根目录有）。表现为 python 建子目录 `WinError 5`、`Set-Content` 被拒。

## 修法
ACL（必须 `danger-full-access`，沙箱内 icacls 自己也被拒）：
```
icacls work /inheritance:e
icacls work /grant "MECHREVO:(OI)(CI)F" /T
```
ffmpeg 落盘：**凡测试/编码/生成素材需要 ffmpeg 写文件，一律带 `sandbox_permissions: danger-full-access`**。不需要落盘时（解码走 stdout 管道）confined 即可。

## 连带经验
- `python -m py_compile` 写不了 `__pycache__` → 用只读方式验语法：
  `python -c "import io;src=io.open(r'D:\flashvsr-web\engine.py',encoding='utf-8').read();compile(src,'engine.py','exec');print('SYNTAX_OK')"`
- 不要给 Triton 指 `work/` 下的 cache（`PermissionError: [WinError 5] ... triton_cache`）。
- `pwsh` 重定向 `*> work\x.log` 会被拒（python 能写，pwsh 不能）；看输出直接用 job_output。
