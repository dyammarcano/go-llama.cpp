#!/usr/bin/env bash
# Compile binding.cpp -> libbinding.a (binding.o only; llama libs are linked by
# cgo). Shared by the CPU/Vulkan (static) and CUDA (DLL) builds.
set -euo pipefail
export PATH="$HOME/scoop/shims:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WINVER="-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00 -DNTDDI_VERSION=0x0A000007 -DGGML_NO_THREAD_POWER_THROTTLING"
g++ -std=c++17 -O3 $WINVER -I llama.cpp/include -I llama.cpp/common -I llama.cpp/ggml/include \
  -c binding.cpp -o binding.o
rm -f libbinding.a
ar rcs libbinding.a binding.o
echo "built libbinding.a"
