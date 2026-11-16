@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem ---- 找 Python：FLASHVSR_PY > 便携包默认位置 > PATH
set "PY=%FLASHVSR_PY%"
if not defined PY if exist "D:\ComfyUI_windows_portable\python_embeded\python.exe" set "PY=D:\ComfyUI_windows_portable\python_embeded\python.exe"
if not defined PY if exist "C:\ComfyUI_windows_portable\python_embeded\python.exe" set "PY=C:\ComfyUI_windows_portable\python_embeded\python.exe"
if not defined PY if exist "%USERPROFILE%\ComfyUI_windows_portable\python_embeded\python.exe" set "PY=%USERPROFILE%\ComfyUI_windows_portable\python_embeded\python.exe"
if not defined PY for %%I in (python.exe) do if not "%%~$PATH:I"=="" set "PY=%%~$PATH:I"

if not defined PY (
  echo [错误] 找不到 Python。请设置 FLASHVSR_PY 指向 ComfyUI 便携包里的 python_embeded\python.exe
  echo 例：set FLASHVSR_PY^=D:\ComfyUI_windows_portable\python_embeded\python.exe
  pause
  exit /b 1
)

rem ---- 部署自检：缺模型/triton/头文件时直接说清楚，不要等到跑一半才崩
"%PY%" check_env.py
if errorlevel 1 (
  echo.
  echo [停止] 上面标 [FAIL] 的项补齐后再运行本脚本。
  pause
  exit /b 1
)

echo.
echo 启动 FlashVSR 面板，浏览器会自动打开 http://127.0.0.1:8288
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8288"
"%PY%" app.py
pause
