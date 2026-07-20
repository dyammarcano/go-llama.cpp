//go:build windows && !cuda && !vulkan

package llama

// CPU build (default): links the MinGW static llama.cpp libs produced by
// scripts/llamacpp.sh (task build:cpu), which writes to llama.cpp/build-cpu.
// ggml*.a lack a lib prefix so are given by path; --start-group resolves
// circular refs.

/*
#cgo LDFLAGS: -Wl,--start-group ${SRCDIR}/llama.cpp/build-cpu/common/libllama-common.a ${SRCDIR}/llama.cpp/build-cpu/common/libllama-common-base.a ${SRCDIR}/llama.cpp/build-cpu/src/libllama.a ${SRCDIR}/llama.cpp/build-cpu/ggml/src/ggml-cpu.a ${SRCDIR}/llama.cpp/build-cpu/ggml/src/ggml.a ${SRCDIR}/llama.cpp/build-cpu/ggml/src/ggml-base.a ${SRCDIR}/llama.cpp/build-cpu/vendor/cpp-httplib/libcpp-httplib.a -Wl,--end-group -fopenmp -lws2_32 -lbcrypt -lstdc++ -lm
*/
import "C"
