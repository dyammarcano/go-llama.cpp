//go:build windows && cuda

package llama

// CUDA build (-tags cuda): links the MSVC-built llama.dll (pulls
// ggml.dll -> ggml-cuda.dll at runtime) via the C API only. Build with
// .scripts/build-cuda.bat. llama.dll/ggml*.dll + cudart64_13.dll/cublas64_13.dll
// must be on PATH (or beside the exe) at runtime.

/*
#cgo LDFLAGS: ${SRCDIR}/build-cuda/bin/llama.dll -lstdc++
*/
import "C"
