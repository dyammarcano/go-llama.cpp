//go:build windows && vulkan && !cuda

package llama

// Vulkan build (-tags vulkan): MinGW static libs from `task build:vulkan`
// (GGML_VULKAN=ON), linked against the system Vulkan loader (vulkan-1.dll in
// System32). Run `task build:vulkan` first to produce llama.cpp/build-vulkan.
// NOTE: the link paths mirror the CPU layout + ggml-vulkan.a; verify against the
// actual build-vulkan output if the lib names differ.

/*
#cgo LDFLAGS: -Wl,--start-group ${SRCDIR}/llama.cpp/build-vulkan/common/libllama-common.a ${SRCDIR}/llama.cpp/build-vulkan/common/libllama-common-base.a ${SRCDIR}/llama.cpp/build-vulkan/src/libllama.a ${SRCDIR}/llama.cpp/build-vulkan/ggml/src/ggml-cpu.a ${SRCDIR}/llama.cpp/build-vulkan/ggml/src/ggml.a ${SRCDIR}/llama.cpp/build-vulkan/ggml/src/ggml-base.a ${SRCDIR}/llama.cpp/build-vulkan/ggml/src/ggml-vulkan.a ${SRCDIR}/llama.cpp/build-vulkan/vendor/cpp-httplib/libcpp-httplib.a -Wl,--end-group -fopenmp -lws2_32 -lbcrypt -lstdc++ -lm C:/Windows/System32/vulkan-1.dll
*/
import "C"
