@echo off
rem Build a C++ app against the libtorch bundled with ComfyUI (no extra download).
rem Usage: build.bat [name]   -> compiles cpp\<name>.cpp into cpp\<name>.exe
rem
rem The simd\ kernels are compiled one ISA level per translation unit (MSVC has no
rem per-function target attributes on x86), so the main program stays baseline and
rem the dispatcher picks the kernel at runtime.
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
if not exist build mkdir build

set CF=/nologo /c /std:c++20 /utf-8 /EHsc /MD /O2 /I"%TORCH%\include" /I"%TORCH%\include\torch\csrc\api\include" /I"%CUDART%\include" /I"%CUDACRT%\include"

rem main program: baseline x64 only, it must run on any CPU
cl %CF% %SRC%.cpp /Fo:build\main.obj || goto :fail
rem kernels: each with its own ISA
cl %CF% simd\bf16_base.cpp  /Fo:build\bf16_base.obj  || goto :fail
cl %CF% simd\dispatch.cpp   /Fo:build\dispatch.obj   || goto :fail
cl %CF% /arch:AVX2   simd\bf16_avx2.cpp   /Fo:build\bf16_avx2.obj   || goto :fail
cl %CF% /arch:AVX512 simd\bf16_avx512.cpp /Fo:build\bf16_avx512.obj || goto :fail

link /nologo /LIBPATH:"%TORCH%\lib" /LIBPATH:"." /OUT:%SRC%.exe ^
   build\main.obj build\bf16_base.obj build\bf16_avx2.obj build\bf16_avx512.obj build\dispatch.obj ^
   torch.lib torch_cuda.lib torch_cpu.lib c10.lib c10_cuda.lib cudart64_13.lib || goto :fail
popd
exit /b 0

:fail
popd
exit /b 1
