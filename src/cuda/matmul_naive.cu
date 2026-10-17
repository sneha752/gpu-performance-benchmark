// Naive CUDA matmul: one thread computes one element of C, reading A and B
// directly from global memory.

#include "cuda_utils.cuh"
#include "matmul_cuda.h"

__global__ void matmul_naive_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int N) {
    // threadIdx.x walks along a row of C, so consecutive threads in a warp read
    // consecutive B[k][col] and write consecutive C[row][col] (coalesced), while
    // A[row][k] is the same address for the whole warp (broadcast).
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    const size_t n = static_cast<size_t>(N);
    float acc = 0.0f;
    for (int k = 0; k < N; ++k) {
        acc += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = acc;
}

void launch_matmul_naive(const float* dA, const float* dB, float* dC, int N,
                         cudaStream_t stream) {
    const dim3 block(kNaiveBlock, kNaiveBlock);
    const dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    matmul_naive_kernel<<<grid, block, 0, stream>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
}
