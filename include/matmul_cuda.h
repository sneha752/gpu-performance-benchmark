#pragma once
// Host-side interface to the CUDA implementations. Deliberately free of CUDA
// headers so it can be included from plain C++ translation units.

#include <string>
#include <vector>

enum class CudaKernel { Naive, Tiled };

struct GpuInfo {
    std::string name;
    int cc_major = 0;
    int cc_minor = 0;
    int runtime_version = 0;  // e.g. 12040 for CUDA 12.4
    int driver_version = 0;
    double total_mem_gib = 0.0;
};

// Per-iteration timings in milliseconds, measured with CUDA events.
struct GpuTimings {
    std::vector<float> h2d_ms;     // copy A and B host -> device
    std::vector<float> kernel_ms;  // kernel only
    std::vector<float> d2h_ms;     // copy C device -> host
    std::vector<float> e2e_ms;     // H2D + kernel + D2H
};

// Returns false (and a reason) if no usable CUDA device is present.
bool cuda_device_available(std::string* reason);
GpuInfo cuda_device_info();

// Thread-block edge of the naive kernel (block is kNaiveBlock x kNaiveBlock).
constexpr int kNaiveBlock = 16;

// Runs `warmup` untimed and `iters` timed H2D -> kernel -> D2H sequences.
// The result of the last iteration is written to hC. Throws std::runtime_error
// on any CUDA error. `tile` (8, 16 or 32) only applies to CudaKernel::Tiled.
GpuTimings run_cuda_matmul(CudaKernel kernel, const float* hA, const float* hB, float* hC,
                           int N, int tile, int warmup, int iters);
