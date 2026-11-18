# run_code 硬上限 120 s：长任务必须走后台 job

## 报错原文
```
Error: code run failed (timeout): execution timed out during execution (120000ms)
File sandbox: danger-full-access.
```
即使给 `pwsh` 传了 `timeoutMs: 400000`，外层 `run_code` 仍在 120 s 处掐断整个程序，pwsh 子进程被连带杀掉（`Start-Process` 起的 `python app.py` 也随之消失，`app_err.log` 为空、无 traceback，很像"服务莫名停止"）。

## 根因
`run_code` 的执行预算 ≠ 内部 `pwsh` 的 `timeoutMs`。外层到点即终止 Node 进程，内部任何前台等待（轮询 GPU 任务、`Start-Sleep` 累计、`Invoke-WebRequest` 循环）都会被一起砍掉；被掐断的 python 子进程不会留日志，容易误判成代码 bug。

## 修法（本次采用，已成功）
1. 任何预期 > ~90 s 的动作（加载模型、跑窗口、端到端 WebUI 任务）一律 `run_in_background: true`，拿 `jobId`。
2. 后台脚本自己把结果写文件（`work\ui_smoke.log`），因为 `job_output` 只给尾部、且被 tqdm 淹掉过。
3. 用 `job_output({ wait: true, timeout_ms: 90000 })` 分多次收，别一次等满。
4. 判活看产物而不是进程：`out\<name>_<jobid>_4x.mp4` 出现 + `ffprobe` 可读 = 真跑完；`app_alive=0` 可能只是脚本自己 `Stop-Process` 了。

## 复用结论
- 前台 `pwsh` 只用于 < 60 s 的检查（`ffprobe`、`Get-Content`、`nvidia-smi`）。
- 端到端验证脚本模板：`Start-Process app.py` → `Start-Sleep 12` → `curl.exe -F file=@... /api/run` → 轮询 `/api/result/<jid>` → 全部追加写 log → `Stop-Process`。
