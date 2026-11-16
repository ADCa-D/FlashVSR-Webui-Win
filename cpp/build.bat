@echo off
rem Build a C++ app against the libtorch bundled with ComfyUI (no extra download).
rem Usage: build.bat [name]   -> compiles cpp\<name>.cpp into cpp\<name>.exe
rem CUDA headers come from pip packages (cuda_runtime + cuda_nvcc crt headers).
setlocal
set SRC=%1
if "%SRC%"=="" set SRC=spike
set VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
set TORCH=D:\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch
set CUDART=C:\Users\MECHREVO\AppData\Local\Temp\opencode\cuda_hdr12\nvidia\cuda_runtime
set CUDACRT=C:\Users\MECHREVO\AppData\Local\Temp\opencode\cuda_nvcc12\nvidia\cuda_nvcc
if not exist "%TORCH%\include\torch\extension.h" (
  echo [ERROR] libtorch headers not found: %TORCH%\include
  exit /b 1
)
pushd cpp
cl /nologo /std:c++20 /utf-8 /EHsc /MD /O2 /arch:AVX2 ^
   /I"%TORCH%\include" /I"%TORCH%\include\torch\csrc\api\include" ^
   /I"%CUDART%\include" /I"%CUDACRT%\include" ^
   %SRC%.cpp /Fe:%SRC%.exe ^
   /link /LIBPATH:"%TORCH%\lib" /LIBPATH:"." ^
   torch.lib torch_cuda.lib torch_cpu.lib c10.lib c10_cuda.lib cudart64_13.lib
del /q *.obj >nul 2>nul
del /q *.exp >nul 2>nul
del /q %SRC%.lib >nul 2>nul
popd
