@echo off
rem Build the C++ spike against the libtorch bundled with ComfyUI (no extra download).
rem CUDA headers come from pip packages (cuda_runtime + cuda_nvcc crt headers).
setlocal
set VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul
set TORCH=D:\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch
set CUDART=C:\Users\MECHREVO\AppData\Local\Temp\opencode\cuda_hdr12\nvidia\cuda_runtime
set CUDACRT=C:\Users\MECHREVO\AppData\Local\Temp\opencode\cuda_nvcc12\nvidia\cuda_nvcc
if not exist "%TORCH%\include\torch\extension.h" (
  echo [ERROR] libtorch headers not found: %TORCH%\include
  exit /b 1
)
cl /nologo /std:c++20 /utf-8 /EHsc /MD /O2 ^
   /I"%TORCH%\include" /I"%TORCH%\include\torch\csrc\api\include" ^
   /I"%CUDART%\include" /I"%CUDACRT%\include" ^
   cpp\spike.cpp /Fe:cpp\spike.exe ^
   /link /LIBPATH:"%TORCH%\lib" /LIBPATH:"cpp" ^
   torch.lib torch_cuda.lib torch_cpu.lib c10.lib c10_cuda.lib cudart64_13.lib
