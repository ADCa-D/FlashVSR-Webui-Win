@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 默认用 ComfyUI 便携包自带的 Python；换机器时改这一行或用环境变量
set PY=%FLASHVSR_PY%
if "%PY%"=="" set PY=D:\ComfyUI_windows_portable\python_embeded\python.exe

if not exist "%PY%" (
  echo [错误] 找不到 Python: %PY%
  echo 请设置环境变量 FLASHVSR_PY 指向 ComfyUI 便携包里的 python_embeded\python.exe
  pause
  exit /b 1
)

echo 启动 FlashVSR 面板… 浏览器会自动打开 http://127.0.0.1:8288
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8288"
"%PY%" app.py
pause
