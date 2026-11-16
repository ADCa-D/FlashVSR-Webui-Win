@echo off
rem Build a C++ app against a libtorch install (defaults to ComfyUI's portable
rem python, override with env vars for other setups).
rem
rem   set TORCH_DIR=D:\path\to\site-packages\torch
rem   set CUDA_HDR_INC=...\nvidia\cuda_runtime\include     (cuda_runtime.h)
rem   set CUDA_NVCC_INC=...\nvidia\cuda_nvcc\include        (crt/host_config.h)
rem
rem Usage: build.bat [name]   -> compiles cpp\<name>.cpp into cpp\<name>.exe
rem
rem The simd\ kernels are compiled one ISA level per translation unit (MSVC has
rem no per-function target attributes on x86), so the main program stays
rem baseline and the dispatcher picks the kernel at runtime.
setlocal
set SRC=%1
if "%SRC%"=="" set SRC=spike

if "%TORCH_DIR%"=="" set TORCH_DIR=D:\ComfyUI_windows_portable\python_embeded\Lib\site-packages\torch
if "%CUDA_HDR_INC%"=="" set CUDA_HDR_INC=C:\Users\%USERNAME%\AppData\Local\Temp\opencode\cuda_hdr12\nvidia\cuda_runtime\include
if "%CUDA_NVCC_INC%"=="" set CUDA_NVCC_INC=C:\Users\%USERNAME%\AppData\Local\Temp\opencode\cuda_nvcc12\nvidia\cuda_nvcc\include

set VS=%VSPATH%
if "%VS%"=="" set VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
if not exist "%VS%\VC\Auxiliary\Build\vcvars64.bat" (
  echo [ERROR] vcvars64.bat not found. Set VSPATH to your VS/BuildTools root.
  exit /b 1
)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul

if not exist "%TORCH_DIR%\include\torch\extension.h" (
  echo [ERROR] libtorch headers not found: %TORCH_DIR%\include
  echo         set TORCH_DIR to ...\site-packages\torch
  exit /b 1
)
pushd cpp
if not exist build mkdir build

set CF=/nologo /c /std:c++20 /utf-8 /EHsc /MD /O2 /I"%TORCH_DIR%\include" /I"%TORCH_DIR%\include\torch\csrc\api\include" /I"%CUDA_HDR_INC%" /I"%CUDA_NVCC_INC%"

rem main program: baseline x64 only, it must run on any CPU
cl %CF% %SRC%.cpp /Fo:build\main.obj || goto :fail
rem kernels: each with its own ISA
cl %CF% simd\bf16_base.cpp  /Fo:build\bf16_base.obj  || goto :fail
cl %CF% simd\dispatch.cpp   /Fo:build\dispatch.obj   || goto :fail
cl %CF% /arch:AVX2   simd\bf16_avx2.cpp   /Fo:build\bf16_avx2.obj   || goto :fail
cl %CF% /arch:AVX512 simd\bf16_avx512.cpp /Fo:build\bf16_avx512.obj || goto :fail

rem cudart64_13.lib is generated locally from the torch-shipped DLL (see README):
rem   dumpbin /exports "%TORCH_DIR%\lib\cudart64_13.dll"  -> cudart64_13.def
rem   lib /def:cudart64_13.def /machine:x64 /out:cudart64_13.lib
link /nologo /LIBPATH:"%TORCH_DIR%\lib" /LIBPATH:"." /OUT:%SRC%.exe ^
   build\main.obj build\bf16_base.obj build\bf16_avx2.obj build\bf16_avx512.obj build\dispatch.obj ^
   torch.lib torch_cuda.lib torch_cpu.lib c10.lib c10_cuda.lib cudart64_13.lib || goto :fail
popd
exit /b 0

:fail
popd
exit /b 1
