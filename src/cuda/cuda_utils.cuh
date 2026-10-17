#pragma once
// Internal CUDA helpers shared by the .cu translation units.

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

// Every CUDA runtime call goes through CUDA_CHECK; failures become exceptions so
// the benchmark driver can record the error in the CSV instead of crashing.
#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            std::ostringstream msg_;                                                  \
            msg_ << #call << " failed at " << __FILE__ << ":" << __LINE__ << ": "     \
                 << cudaGetErrorName(err_) << " (" << cudaGetErrorString(err_) << ")"; \
            throw std::runtime_error(msg_.str());                                     \
        }                                                                             \
    } while (0)

// Kernel launchers. Each checks the launch with cudaGetLastError(); execution
// errors surface at the next synchronizing CUDA call.
void launch_matmul_naive(const float* dA, const float* dB, float* dC, int N,
                         cudaStream_t stream);
void launch_matmul_tiled(const float* dA, const float* dB, float* dC, int N, int tile,
                         cudaStream_t stream);
