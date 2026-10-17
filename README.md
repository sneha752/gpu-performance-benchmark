# GPU Performance Benchmarking Suite

Square float32 matrix multiplication (`C = A × B`, N×N, row-major) implemented three ways
and benchmarked side by side:

| Implementation | File | Idea |
|---|---|---|
| CPU baseline | `src/cpu/matmul_cpu.cpp` | Straightforward i-j-k triple loop |
| Naive CUDA | `src/cuda/matmul_naive.cu` | One thread per output element, operands read from global memory |
| Tiled CUDA | `src/cuda/matmul_tiled.cu` | Shared-memory tiling (tile 8/16/32, default 16×16) with zero-padded boundary tiles |

Every result is validated, every measurement is written to a CSV with metadata, and the
plots are generated only from that CSV.

## Repository layout

```
gpu-performance-benchmark/
├── CMakeLists.txt            # CUDA is optional: CPU-only build when nvcc is absent
├── include/
│   ├── matmul.h              # CPU kernel, RNG init, validation helpers
│   └── matmul_cuda.h         # host-side CUDA API (no CUDA headers needed)
├── src/
│   ├── main.cpp              # benchmark driver + CSV writer
│   ├── cpu/matmul_cpu.cpp    # triple loop, fp64 reference check, tolerance compare
│   └── cuda/
│       ├── cuda_utils.cuh    # CUDA_CHECK macro, launcher declarations
│       ├── matmul_naive.cu   # naive kernel + launcher
│       ├── matmul_tiled.cu   # tiled kernel template + launcher
│       └── cuda_bench.cu     # pinned buffers, CUDA-event timing harness
├── scripts/
│   ├── run_benchmarks.py     # correctness gate → benchmark → run_metadata.json
│   └── plot_results.py       # pandas/matplotlib plots + plots/summary.md
├── results/                  # benchmark_results.csv, run_metadata.json
└── plots/                    # *.png, summary.md
```

## Setup and reproduction

Requirements: CMake ≥ 3.18, a C++17 compiler, Python 3 with pandas and matplotlib. For GPU
results, an NVIDIA GPU with driver and CUDA Toolkit (nvcc). Without nvcc the project builds a
CPU-only binary and marks every CUDA row `not_run`.

### Linux / WSL (Ubuntu)

```bash
sudo apt update && sudo apt install -y build-essential cmake python3-pandas python3-matplotlib
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure      # correctness tests
python3 scripts/run_benchmarks.py               # correctness gate + benchmark → results/
python3 scripts/plot_results.py                 # plots/ + plots/summary.md
```

### Free cloud GPU (Google Colab / Kaggle, NVIDIA T4)

Colab images ship with nvcc, CMake, pandas and matplotlib. In a notebook with a GPU runtime:

```bash
!nvidia-smi
!git clone <your-repo-url> && cd gpu-performance-benchmark && \
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && \
  ctest --test-dir build --output-on-failure && \
  python3 scripts/run_benchmarks.py && python3 scripts/plot_results.py
```

Then download `results/` and `plots/` (or commit them from the notebook).

### Native Windows (MSVC + CUDA)

Install Visual Studio 2022 Build Tools (C++ workload), CMake, and the CUDA Toolkit, then from a
*Developer PowerShell*:

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
python scripts\run_benchmarks.py
python scripts\plot_results.py
```

### Options

`scripts/run_benchmarks.py --help` lists all options: `--sizes 128,256,512,1024,2048`,
`--iters`, `--warmup`, `--cpu-iters`, `--cpu-iters-large`, `--cpu-max`, `--tile {8,16,32}`,
`--seed`. The binary accepts the same flags directly (`build/matmul_bench --help`). The
default tile size can also be set at configure time with `-DMATMUL_TILE_SIZE=32`.

## Algorithms

**CPU baseline.** `C[i][j] = Σ_k A[i][k]·B[k][j]` with an i-j-k loop and a float accumulator.
The inner loop walks B down a column (stride N), so it is cache-unfriendly on purpose. It is
the textbook baseline, not an optimized BLAS.

**Naive CUDA.** A 2-D grid of 16×16 blocks, with one thread per element of C. `threadIdx.x` maps
to the column, so a warp reads 32 consecutive `B[k][col]` values (one coalesced transaction)
and all threads read the same `A[row][k]` (a broadcast). Every thread still reads 2N floats from
global memory to perform 2N FLOPs, so the kernel is memory-bound.

**Tiled CUDA.** Each TILE×TILE block computes one TILE×TILE tile of C. For each of ⌈N/TILE⌉
steps along K, the block cooperatively loads one tile of A and one tile of B into shared memory
(each thread loads one element of each, coalesced). Then `__syncthreads()`, a TILE-long inner
product out of shared memory, and a second `__syncthreads()` before the tiles are overwritten.
Global traffic drops by a factor of TILE. Boundary handling: loads outside the matrix
write 0 to shared memory, and only in-range threads store to C. No thread exits early, so
every thread reaches both barriers. This is required for correctness when N is not a
multiple of TILE.

## Benchmark methodology

| Quantity | How it is measured |
|---|---|
| CPU time | `std::chrono::steady_clock` around one `matmul_cpu` call. One untimed warm-up for N ≤ 512, then `--cpu-iters` (default 3) runs for N < 1024 and `--cpu-iters-large` (default 1) for N ≥ 1024, because a 2048³ triple loop takes many seconds. |
| H2D | CUDA events around `cudaMemcpyAsync` of A and B (2·N²·4 bytes) from **pinned** host memory |
| Kernel | CUDA events around the kernel launch only |
| D2H | CUDA events around `cudaMemcpyAsync` of C (N²·4 bytes) to pinned host memory |
| End-to-end | First to last event: H2D + kernel + D2H |
| Statistic | **Median** over `--iters` (default 10) timed iterations. Min and max are also recorded. |
| GFLOPS | `2·N³ / time`, using kernel time (`gflops`) or end-to-end time (`gflops_e2e`) |
| Speedup | `cpu_median / kernel_median` (`speedup_vs_cpu`) and `cpu_median / e2e_median` (`speedup_e2e_vs_cpu`) |

Excluded from all timings: matrix initialization, `cudaMalloc`, `cudaMallocHost`, copying
inputs into pinned staging buffers, event creation, and CUDA context creation. `--warmup`
(default 3) full H2D → kernel → D2H sequences run untimed first, to absorb lazy module
loading and GPU clock ramp-up. `cudaEventSynchronize` on the last event waits for the whole
sequence and surfaces asynchronous kernel faults. `C` is zeroed on the device before each timed
iteration so stale results cannot pass validation. Pinned memory means the transfer times show
achievable PCIe bandwidth. Pageable `std::vector` transfers would be slower.

**Correctness.** Inputs are uniform in [-1, 1) from `std::mt19937` (seed `--seed`, offset per
size), so every run is reproducible.
- The CPU result is checked against an independent fp64-accumulated reference: every element
  for N ≤ 512, and 262,144 seeded random elements above that.
- Each GPU result is compared element-wise against the full CPU result.
- The pass rule for every element is `|x − ref| ≤ atol + rtol·|ref|`, with `atol = rtol = 1e-3`.
- `max_abs_err`, `max_rel_err`, `mismatches` and `checked` are recorded.
- `ctest` and `run_benchmarks.py` first run a validation pass on awkward sizes (1, 7, 17, 33,
  100, 129, 257 …). Benchmarks don't run if it fails.

**CSV** (`results/benchmark_results.csv`) has one row per (N, implementation). The
`status` column holds `ok`, `failed_validation`, `error` or `not_run`, and a `notes` column
gives the reason. Other columns hold timings, errors, the device name, compute capability,
CUDA runtime/driver versions, compiler, build type, seed, tile size and run id.
`results/run_metadata.json` adds host, OS, CPU model, git commit and the exact command.
Plots use only rows with `status == ok` and `correct == true`, and say on the figure
which rows were left out and why.

## Results

<!-- RESULTS:START -->
### Run 20261016T122113Z: development laptop, CPU only

**Machine:** Intel Core i5-1334U (13th gen), 16 GB RAM, WSL2 Ubuntu 26.04, gcc 15.2.0, CMake
4.2.3, Release (`-O3`). **No NVIDIA GPU** (Intel Iris Xe only), so the build has no CUDA and
every CUDA row in the CSV is `not_run`. **No GPU numbers have been measured yet.** The plots
that need GPU data say so on the figure.

| N | CPU median (ms) | Timed runs | GFLOPS | max abs error vs fp64 | Validation |
|---:|---:|---:|---:|---:|---|
| 128 | 0.948 | 3 | 4.42 | 6.2e-06 | pass (all 16,384 elements) |
| 256 | 7.96 | 3 | 4.22 | 1.3e-05 | pass (all 65,536 elements) |
| 512 | 102.7 | 3 | 2.61 | 3.2e-05 | pass (all 262,144 elements) |
| 1024 | 2,511 | 1 | 0.86 | 5.9e-05 | pass (262,144 sampled elements) |
| 2048 | 52,923 | 1 | 0.32 | 1.1e-04 | pass (262,144 sampled elements) |

Source: `results/benchmark_results.csv`, `results/run_metadata.json`, `plots/summary.md`.

**Observations.**
- Throughput falls about 14× from N=256 to N=2048. The inner loop reads B down a
  column, touching a new cache line for every multiply-add. Once N rows of B no longer fit in
  cache, each access goes to DRAM. Power-of-two strides make it worse by mapping a column's
  lines onto the same cache sets.
- The float32 error against the fp64 reference grows roughly with N, as expected for
  sequential summation, and stays more than 10× inside the `1e-3` tolerance.
- N ≥ 1024 has a single timed run (by design, to keep the 2048 run near one minute), so those
  medians carry more noise than the small sizes.

To add GPU results, run the Colab/Kaggle steps above. The CSV and plots will then include
the naive and tiled kernels, transfer times, speedups and GFLOPS for that GPU.
<!-- RESULTS:END -->

## Technical summary

- **Memory coalescing.** A warp's 32 global loads are combined into as few 32-byte sectors as
  possible when the threads touch consecutive addresses. Mapping `threadIdx.x` to the column
  makes reads of `B[k][col]` and writes of `C[row][col]` coalesced. Mapping it to the row
  instead would make every thread touch a different cache line, multiplying memory traffic.
- **Shared-memory tiling.** The naive kernel loads each element of A and B N times from
  global memory. Tiling stages TILE×TILE blocks in on-chip shared memory and reuses each loaded
  value TILE times, so global traffic falls by a factor of TILE. This raises arithmetic
  intensity from about 0.25 FLOP/byte to about TILE/4 FLOP/byte, so the kernel moves toward
  being compute-bound. Barriers make sure a tile is fully loaded before use and fully consumed
  before it is overwritten.
- **Data-transfer overhead.** Copies move 3·N²·4 bytes over PCIe (O(N²)), while compute is
  O(N³). For small N the copies dominate end-to-end time, and the GPU can lose to the CPU. As N
  grows, compute dominates and transfers are amortized. In real applications, keeping data
  resident on the GPU across many operations avoids paying for transfers each time.
- **Speedup.** Speedup is the ratio of CPU time to GPU time. Kernel-only speedup shows the raw
  device advantage. End-to-end speedup is what a caller that starts and ends on the host
  actually sees. Both are reported against the same unoptimized CPU baseline, so they overstate
  the advantage over a tuned multithreaded BLAS.
- **GFLOPS.** One N×N matmul does N³ multiply-adds, which is 2N³ floating-point operations, so
  GFLOPS = 2N³ / (seconds · 10⁹). Comparing it with the device's peak FP32 rate shows how far a
  kernel is from the hardware limit. A simple tiled kernel typically reaches a modest fraction
  of peak. cuBLAS gets much closer with register blocking, vectorized loads and tensor cores.

## Limitations

- The CPU baseline is single-threaded and deliberately naive. It is not a fair "best CPU" comparison.
- Only square matrices are supported, and only one GPU and the default stream are used.
  There is no copy/compute overlap.
- Neither kernel is close to cuBLAS. Next steps would be register blocking (several outputs
  per thread), vectorized `float4` loads, double buffering, and a cuBLAS comparison row.
- Timings depend on GPU clocks, thermals and other load. Medians reduce but don't remove noise.
- The development laptop (Intel Iris Xe) has no NVIDIA GPU, so CUDA results must come
  from another machine. See the Results section for what was actually measured, and where.
