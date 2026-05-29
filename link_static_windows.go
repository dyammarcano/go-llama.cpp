//go:build windows && !cuda && !vulkan

package llama

// CPU build (default): links the MinGW static llama.cpp libs from
// .scripts/build-llamacpp.sh. ggml*.a lack a lib prefix so are given by path;
// --start-group resolves circular refs.

/*
#cgo LDFLAGS: -Wl,--start-group ${SRCDIR}/llama.cpp/build/common/libllama-common.a ${SRCDIR}/llama.cpp/build/common/libllama-common-base.a ${SRCDIR}/llama.cpp/build/src/libllama.a ${SRCDIR}/llama.cpp/build/ggml/src/ggml-cpu.a ${SRCDIR}/llama.cpp/build/ggml/src/ggml.a ${SRCDIR}/llama.cpp/build/ggml/src/ggml-base.a ${SRCDIR}/llama.cpp/build/vendor/cpp-httplib/libcpp-httplib.a -Wl,--end-group -fopenmp -lws2_32 -lbcrypt -lstdc++ -lm
*/
import "C"
