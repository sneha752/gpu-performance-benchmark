#include "matmul.h"

#include <algorithm>
#include <cmath>
#include <random>

void matmul_cpu(const float* A, const float* B, float* C, int N) {
    const std::size_t n = static_cast<std::size_t>(N);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < n; ++k) {
                acc += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = acc;
        }
    }
}

void fill_random(std::vector<float>& m, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : m) x = dist(rng);
}

namespace {

void accumulate_error(ErrorStats& s, double value, double ref, double atol, double rtol) {
    const double abs_err = std::fabs(value - ref);
    const double rel_err = abs_err / std::max(std::fabs(ref), 1e-30);
    if (abs_err > s.max_abs_err) s.max_abs_err = abs_err;
    if (rel_err > s.max_rel_err) s.max_rel_err = rel_err;
    // NaN fails this comparison and is therefore counted as a mismatch.
    if (!(abs_err <= atol + rtol * std::fabs(ref))) ++s.mismatches;
    ++s.checked;
}

}  // namespace

ErrorStats compare_with_tolerance(const float* test, const float* ref, std::size_t count,
                                  double atol, double rtol) {
    ErrorStats s;
    for (std::size_t i = 0; i < count; ++i) accumulate_error(s, test[i], ref[i], atol, rtol);
    return s;
}

ErrorStats check_against_fp64(const float* A, const float* B, const float* C, int N,
                              double atol, double rtol, long long max_samples, unsigned seed) {
    const std::size_t n = static_cast<std::size_t>(N);
    auto ref_entry = [&](std::size_t i, std::size_t j) {
        double acc = 0.0;
        for (std::size_t k = 0; k < n; ++k)
            acc += static_cast<double>(A[i * n + k]) * static_cast<double>(B[k * n + j]);
        return acc;
    };

    ErrorStats s;
    if (static_cast<long long>(n * n) <= max_samples) {
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                accumulate_error(s, C[i * n + j], ref_entry(i, j), atol, rtol);
    } else {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<std::size_t> idx(0, n - 1);
        for (long long t = 0; t < max_samples; ++t) {
            const std::size_t i = idx(rng), j = idx(rng);
            accumulate_error(s, C[i * n + j], ref_entry(i, j), atol, rtol);
        }
    }
    return s;
}
