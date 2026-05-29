@echo off
REM Build llama.cpp with CUDA as MSVC shared DLLs (llama.dll + ggml*.dll incl
REM ggml-cuda.dll). CUDA/MSVC stays sealed in the DLLs; the MinGW cgo host links
REM only llama.dll's C API. Requires VS2022 + CUDA toolkit (nvcc) + CCCL headers.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
set "PATH=%USERPROFILE%\scoop\shims;%PATH%"
set "CUDA=%USERPROFILE%\scoop\apps\cuda\current"
cd /d "%~dp0.." || exit /b 1
rmdir /s /q build-cuda 2>nul
cmake -S llama.cpp -B build-cuda -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 ^
  -DCMAKE_CUDA_COMPILER="%CUDA%\bin\nvcc.exe" ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DGGML_NATIVE=ON || exit /b 1
cmake --build build-cuda --config Release --target llama || exit /b 1
echo === CUDA DLLs ===
dir /s /b build-cuda\*.dll
