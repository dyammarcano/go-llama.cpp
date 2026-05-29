#!/usr/bin/env bash
# Build llama.cpp static libs with MinGW for the CPU or Vulkan backend.
# Usage: llamacpp.sh [cpu|vulkan]   (default: cpu)
# Vulkan requires the Vulkan SDK (glslc + headers) on PATH / VULKAN_SDK set.
set -euo pipefail
export PATH="$HOME/scoop/shims:$PATH"

BACKEND="${1:-cpu}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SRC=llama.cpp
BUILD="$SRC/build-$BACKEND"

# MinGW headers here lack THREAD_POWER_THROTTLING_STATE; target Win10 for
# cpp-httplib and compile out that optional ggml-cpu block.
WINVER="-D_WIN32_WINNT=0x0A00 -DWINVER=0x0A00 -DNTDDI_VERSION=0x0A000007 -DGGML_NO_THREAD_POWER_THROTTLING"
CPUC="$SRC/ggml/src/ggml-cpu/ggml-cpu.c"
if ! grep -q "GGML_NO_THREAD_POWER_THROTTLING" "$CPUC"; then
  sed -i 's/#if _WIN32_WINNT >= 0x0602$/#if _WIN32_WINNT >= 0x0602 \&\& !defined(GGML_NO_THREAD_POWER_THROTTLING)/' "$CPUC"
fi

EXTRA=""
[ "$BACKEND" = "vulkan" ] && EXTRA="-DGGML_VULKAN=ON"

rm -rf "$BUILD"
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF -DGGML_NATIVE=ON $EXTRA \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_FLAGS="$WINVER" -DCMAKE_CXX_FLAGS="$WINVER"
cmake --build "$BUILD" -j --target llama llama-common
echo "=== $BACKEND static libs ==="
find "$BUILD" -name "*.a" | sort
