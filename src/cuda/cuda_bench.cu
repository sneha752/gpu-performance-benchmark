// CUDA timing harness: device management, pinned staging buffers, and
// CUDA-event timing of H2D copy, kernel, and D2H copy.

#include "cuda_utils.cuh"
#include "matmul_cuda.h"

#include <cstring>

namespace {

// Minimal RAII owners so buffers are released even when CUDA_CHECK throws.
// Destructors ignore errors: they may run while a sticky error is unwinding.
struct DeviceBuffer {
    float* ptr = nullptr;
    explicit DeviceBuffer(size_t bytes) { CUDA_CHECK(cudaMalloc(&ptr, bytes)); }
    ~DeviceBuffer() { cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct PinnedBuffer {
    float* ptr = nullptr;
    explicit PinnedBuffer(size_t bytes) { CUDA_CHECK(cudaMallocHost(&ptr, bytes)); }
    ~PinnedBuffer() { cudaFreeHost(ptr); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

struct Event {
    cudaEvent_t ev = nullptr;
    Event() { CUDA_CHECK(cudaEventCreate(&ev)); }
    ~Event() { cudaEventDestroy(ev); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

float elapsed_ms(const Event& start, const Event& stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.ev, stop.ev));
    return ms;
}

}  // namespace

bool cuda_device_available(std::string* reason) {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        if (reason) *reason = std::string("cudaGetDeviceCount: ") + cudaGetErrorString(err);
        cudaGetLastError();  // clear the error so it does not leak into later calls
        return false;
    }
    if (count == 0) {
        if (reason) *reason = "no CUDA-capable device found";
        return false;
    }
    return true;
}

GpuInfo cuda_device_info() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    GpuInfo info;
    info.name = prop.name;
    info.cc_major = prop.major;
    info.cc_minor = prop.minor;
    info.total_mem_gib = static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);
    CUDA_CHECK(cudaRuntimeGetVersion(&info.runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&info.driver_version));
    return info;
}

GpuTimings run_cuda_matmul(CudaKernel kernel, const float* hA, const float* hB, float* hC,
                           int N, int tile, int warmup, int iters) {
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    // One-time setup, excluded from all timings: pinned staging buffers (so the
    // copies measure DMA bandwidth, not an extra pageable bounce), device
    // buffers, and events.
    PinnedBuffer pA(bytes), pB(bytes), pC(bytes);
    std::memcpy(pA.ptr, hA, bytes);
    std::memcpy(pB.ptr, hB, bytes);
    DeviceBuffer dA(bytes), dB(bytes), dC(bytes);
    Event e_start, e_h2d, e_kernel, e_d2h;
    const cudaStream_t stream = 0;  // legacy default stream: operations run in order

    auto run_once = [&]() {
        CUDA_CHECK(cudaEventRecord(e_start.ev, stream));
        CUDA_CHECK(cudaMemcpyAsync(dA.ptr, pA.ptr, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(dB.ptr, pB.ptr, bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(e_h2d.ev, stream));
        if (kernel == CudaKernel::Naive)
            launch_matmul_naive(dA.ptr, dB.ptr, dC.ptr, N, stream);
        else
            launch_matmul_tiled(dA.ptr, dB.ptr, dC.ptr, N, tile, stream);
        CUDA_CHECK(cudaEventRecord(e_kernel.ev, stream));
        CUDA_CHECK(cudaMemcpyAsync(pC.ptr, dC.ptr, bytes, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(e_d2h.ev, stream));
        // Waits for the whole sequence; also surfaces asynchronous kernel faults.
        CUDA_CHECK(cudaEventSynchronize(e_d2h.ev));
    };

    // Warm-up absorbs context creation, module loading and clock ramp-up.
    for (int i = 0; i < warmup; ++i) run_once();

    GpuTimings t;
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemset(dC.ptr, 0, bytes));  // stale results cannot pass validation
        run_once();
        t.h2d_ms.push_back(elapsed_ms(e_start, e_h2d));
        t.kernel_ms.push_back(elapsed_ms(e_h2d, e_kernel));
        t.d2h_ms.push_back(elapsed_ms(e_kernel, e_d2h));
        t.e2e_ms.push_back(elapsed_ms(e_start, e_d2h));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    std::memcpy(hC, pC.ptr, bytes);
    return t;
}
