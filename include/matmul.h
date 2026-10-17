#pragma once
// Common declarations for the square float32 matrix-multiplication benchmark.
// All matrices are N x N, row-major, stored contiguously.

#include <cstddef>
#include <vector>

// C = A * B using the straightforward i-j-k triple loop (CPU baseline).
void matmul_cpu(const float* A, const float* B, float* C, int N);

// Fill m with uniform values in [-1, 1) from a fixed-seed std::mt19937.
void fill_random(std::vector<float>& m, unsigned seed);

struct ErrorStats {
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    long long mismatches = 0;  // elements violating |x - ref| <= atol + rtol * |ref|
    long long checked = 0;     // elements compared
    bool pass() const { return checked > 0 && mismatches == 0; }
};

// Element-wise comparison of `test` against `ref` (count elements).
ErrorStats compare_with_tolerance(const float* test, const float* ref, std::size_t count,
                                  double atol, double rtol);

// Independent correctness check: recompute entries of A*B with double-precision
// accumulation and compare them against C. Every entry is checked when
// N*N <= max_samples, otherwise max_samples entries chosen with a fixed seed.
ErrorStats check_against_fp64(const float* A, const float* B, const float* C, int N,
                              double atol, double rtol, long long max_samples, unsigned seed);
