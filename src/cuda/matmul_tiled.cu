// Shared-memory tiled CUDA matmul. Each TILE x TILE thread block computes a
// TILE x TILE tile of C. The block marches along the K dimension, cooperatively
// staging one tile of A and one tile of B in shared memory per step, so each
// global element is loaded once per block instead of once per thread.

#include "cuda_utils.cuh"
#include "matmul_cuda.h"

#include <string>

template <int TILE>
__global__ void matmul_tiled_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                    float* __restrict__ C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;
    const size_t n = static_cast<size_t>(N);

    float acc = 0.0f;
    const int num_tiles = (N + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        // Out-of-range elements are padded with zeros so the inner loop needs no
        // bounds checks. No thread returns early: every thread must reach both
        // __syncthreads() calls.
        const int a_col = t * TILE + tx;
        const int b_row = t * TILE + ty;
        As[ty][tx] = (row < N && a_col < N) ? A[row * n + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < N && col < N) ? B[b_row * n + col] : 0.0f;
        __syncthreads();  // tile fully loaded before anyone reads it

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            // As[ty][k] is a broadcast within a warp row; Bs[k][tx] hits
            // consecutive banks, so neither access has bank conflicts.
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();  // everyone done reading before the next overwrite
    }

    if (row < N && col < N) C[row * n + col] = acc;
}

template <int TILE>
static void launch_tiled(const float* dA, const float* dB, float* dC, int N,
                         cudaStream_t stream) {
    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmul_tiled_kernel<TILE><<<grid, block, 0, stream>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
}

void launch_matmul_tiled(const float* dA, const float* dB, float* dC, int N, int tile,
                         cudaStream_t stream) {
    switch (tile) {
        case 8: launch_tiled<8>(dA, dB, dC, N, stream); break;
        case 16: launch_tiled<16>(dA, dB, dC, N, stream); break;
        case 32: launch_tiled<32>(dA, dB, dC, N, stream); break;
        default:
            throw std::invalid_argument("unsupported tile size " + std::to_string(tile) +
                                        " (supported: 8, 16, 32)");
    }
}
