// Benchmark driver: runs the CPU baseline and (when built with CUDA and a device
// is present) the naive and tiled CUDA kernels for each matrix size, validates
// every result, and writes one CSV row per (size, implementation).

#include "matmul.h"
#ifdef HAVE_CUDA
#include "matmul_cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef MATMUL_DEFAULT_TILE
#define MATMUL_DEFAULT_TILE 16
#endif
#ifndef BENCH_BUILD_TYPE
#define BENCH_BUILD_TYPE "unknown"
#endif

namespace {

struct Options {
    std::vector<int> sizes{128, 256, 512, 1024, 2048};
    int warmup = 3;           // untimed GPU runs per size and kernel
    int iters = 10;           // timed GPU runs per size and kernel
    int cpu_iters = 3;        // timed CPU runs for N < 1024
    int cpu_iters_large = 1;  // timed CPU runs for N >= 1024
    int cpu_max = 4096;       // skip the CPU baseline above this size
    int tile = MATMUL_DEFAULT_TILE;
    unsigned seed = 42;
    double atol = 1e-3;
    double rtol = 1e-3;
    long long fp64_samples = 512LL * 512;  // full fp64 check up to N = 512
    std::string csv = "results/benchmark_results.csv";
    std::string run_id;
    bool validate_only = false;
};

void usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "  --sizes LIST          comma-separated N values (default 128,256,512,1024,2048)\n"
        << "  --warmup K            untimed GPU iterations (default 3)\n"
        << "  --iters K             timed GPU iterations (default 10)\n"
        << "  --cpu-iters K         timed CPU iterations for N < 1024 (default 3)\n"
        << "  --cpu-iters-large K   timed CPU iterations for N >= 1024 (default 1)\n"
        << "  --cpu-max N           skip CPU baseline for sizes above N (default 4096)\n"
        << "  --tile T              tiled-kernel tile size: 8, 16 or 32 (default "
        << MATMUL_DEFAULT_TILE << ")\n"
        << "  --seed S              RNG seed for input matrices (default 42)\n"
        << "  --atol X --rtol X     validation tolerances (default 1e-3, 1e-3)\n"
        << "  --csv PATH            output CSV (default results/benchmark_results.csv)\n"
        << "  --run-id ID           identifier stored in every CSV row\n"
        << "  --validate-only       one run per implementation, no CSV; exit 1 on failure\n";
}

std::vector<int> parse_sizes(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const int n = std::stoi(item);
        if (n <= 0) throw std::invalid_argument("matrix size must be positive: " + item);
        out.push_back(n);
    }
    if (out.empty()) throw std::invalid_argument("empty --sizes list");
    return out;
}

Options parse_args(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
            return argv[++i];
        };
        if (a == "--sizes") o.sizes = parse_sizes(next());
        else if (a == "--warmup") o.warmup = std::stoi(next());
        else if (a == "--iters") o.iters = std::stoi(next());
        else if (a == "--cpu-iters") o.cpu_iters = std::stoi(next());
        else if (a == "--cpu-iters-large") o.cpu_iters_large = std::stoi(next());
        else if (a == "--cpu-max") o.cpu_max = std::stoi(next());
        else if (a == "--tile") o.tile = std::stoi(next());
        else if (a == "--seed") o.seed = static_cast<unsigned>(std::stoul(next()));
        else if (a == "--atol") o.atol = std::stod(next());
        else if (a == "--rtol") o.rtol = std::stod(next());
        else if (a == "--csv") o.csv = next();
        else if (a == "--run-id") o.run_id = next();
        else if (a == "--validate-only") o.validate_only = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); std::exit(0); }
        else throw std::invalid_argument("unknown option " + a);
    }
    if (o.iters < 1 || o.cpu_iters < 1 || o.cpu_iters_large < 1 || o.warmup < 0)
        throw std::invalid_argument("iteration counts must be >= 1 (warmup >= 0)");
    if (o.tile != 8 && o.tile != 16 && o.tile != 32)
        throw std::invalid_argument("--tile must be 8, 16 or 32");
    if (o.validate_only) {
        o.warmup = 0;
        o.iters = o.cpu_iters = o.cpu_iters_large = 1;
    }
    return o;
}

struct Stats {
    double median = NAN, min = NAN, max = NAN;
};

template <typename T>
Stats summarize(std::vector<T> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    s.median = (n % 2) ? v[n / 2] : 0.5 * (static_cast<double>(v[n / 2 - 1]) + v[n / 2]);
    s.min = v.front();
    s.max = v.back();
    return s;
}

double gflops(int N, double ms) {
    return 2.0 * N * static_cast<double>(N) * N / (ms * 1e-3) / 1e9;
}

std::string num(double x) {
    if (std::isnan(x)) return "";
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.6g", x);
    return buf;
}

std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
    return buf;
}

std::string compiler_string() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

const std::vector<std::string> kColumns = {
    "run_id", "timestamp_utc", "impl", "N", "status", "correct", "reference",
    "max_abs_err", "max_rel_err", "mismatches", "checked", "atol", "rtol",
    "warmup", "iters", "time_ms_median", "time_ms_min", "time_ms_max",
    "h2d_ms_median", "d2h_ms_median", "e2e_ms_median", "e2e_ms_min", "e2e_ms_max",
    "gflops", "gflops_e2e", "speedup_vs_cpu", "speedup_e2e_vs_cpu", "block",
    "tile_size", "seed", "device", "compute_capability", "cuda_runtime",
    "cuda_driver", "compiler", "build_type", "notes"};

using Row = std::map<std::string, std::string>;

std::string csv_escape(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) out += (c == '"') ? std::string("\"\"") : std::string(1, c);
    return out + "\"";
}

void write_csv(const std::string& path, const std::vector<Row>& rows) {
    const std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path + " for writing");
    for (size_t i = 0; i < kColumns.size(); ++i) f << (i ? "," : "") << kColumns[i];
    f << "\n";
    for (const Row& r : rows) {
        for (size_t i = 0; i < kColumns.size(); ++i) {
            auto it = r.find(kColumns[i]);
            f << (i ? "," : "") << csv_escape(it == r.end() ? "" : it->second);
        }
        f << "\n";
    }
}

void set_errors(Row& r, const ErrorStats& e) {
    r["correct"] = e.pass() ? "true" : "false";
    r["status"] = e.pass() ? "ok" : "failed_validation";
    r["max_abs_err"] = num(e.max_abs_err);
    r["max_rel_err"] = num(e.max_rel_err);
    r["mismatches"] = std::to_string(e.mismatches);
    r["checked"] = std::to_string(e.checked);
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    try {
        opt = parse_args(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        usage(argv[0]);
        return 2;
    }

    Row base;
    base["run_id"] = opt.run_id;
    base["timestamp_utc"] = utc_timestamp();
    base["seed"] = std::to_string(opt.seed);
    base["atol"] = num(opt.atol);
    base["rtol"] = num(opt.rtol);
    base["compiler"] = compiler_string();
    base["build_type"] = BENCH_BUILD_TYPE;

    bool cuda_ok = false;
    std::string cuda_note;
#ifdef HAVE_CUDA
    try {
        cuda_ok = cuda_device_available(&cuda_note);
        if (cuda_ok) {
            const GpuInfo gi = cuda_device_info();
            base["device"] = gi.name;
            base["compute_capability"] = std::to_string(gi.cc_major) + "." + std::to_string(gi.cc_minor);
            base["cuda_runtime"] = std::to_string(gi.runtime_version);
            base["cuda_driver"] = std::to_string(gi.driver_version);
            std::cout << "GPU: " << gi.name << " (sm_" << gi.cc_major << gi.cc_minor << ", "
                      << num(gi.total_mem_gib) << " GiB)\n";
        }
    } catch (const std::exception& ex) {
        cuda_ok = false;
        cuda_note = ex.what();
    }
#else
    cuda_note = "built without CUDA (no CUDA compiler found at configure time)";
#endif
    if (!cuda_ok) std::cout << "CUDA not run: " << cuda_note << "\n";

    std::vector<Row> rows;
    int failures = 0;

    for (const int N : opt.sizes) {
        const size_t count = static_cast<size_t>(N) * N;
        std::vector<float> A(count), B(count), C_cpu, C_gpu;
        fill_random(A, opt.seed + 2u * static_cast<unsigned>(N));
        fill_random(B, opt.seed + 2u * static_cast<unsigned>(N) + 1u);

        // ---- CPU baseline -------------------------------------------------
        Row cpu = base;
        cpu["impl"] = "cpu";
        cpu["N"] = std::to_string(N);
        const bool have_cpu = N <= opt.cpu_max;
        double cpu_ms = NAN;
#ifndef HAVE_CUDA
        (void)cpu_ms;  // only consumed by the CUDA speedup columns
#endif
        if (have_cpu) {
            C_cpu.assign(count, 0.0f);
            const int reps = (N >= 1024) ? opt.cpu_iters_large : opt.cpu_iters;
            const int warm = (!opt.validate_only && N <= 512) ? 1 : 0;
            for (int i = 0; i < warm; ++i) matmul_cpu(A.data(), B.data(), C_cpu.data(), N);
            std::vector<double> t;
            for (int i = 0; i < reps; ++i) {
                const auto t0 = std::chrono::steady_clock::now();
                matmul_cpu(A.data(), B.data(), C_cpu.data(), N);
                const auto t1 = std::chrono::steady_clock::now();
                t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            const Stats s = summarize(t);
            cpu_ms = s.median;
            const ErrorStats e = check_against_fp64(A.data(), B.data(), C_cpu.data(), N, opt.atol,
                                                    opt.rtol, opt.fp64_samples, opt.seed);
            set_errors(cpu, e);
            cpu["reference"] = (e.checked == static_cast<long long>(count)) ? "fp64_full" : "fp64_sampled";
            cpu["warmup"] = std::to_string(warm);
            cpu["iters"] = std::to_string(reps);
            cpu["time_ms_median"] = num(s.median);
            cpu["time_ms_min"] = num(s.min);
            cpu["time_ms_max"] = num(s.max);
            cpu["gflops"] = num(gflops(N, s.median));
            cpu["speedup_vs_cpu"] = "1";
            if (!e.pass()) ++failures;
            std::printf("N=%5d %-11s %-17s median %10.3f ms  %8.2f GFLOPS  max_abs_err %.2e\n", N,
                        "cpu", cpu["status"].c_str(), s.median, gflops(N, s.median), e.max_abs_err);
        } else {
            cpu["status"] = "not_run";
            cpu["notes"] = "N > --cpu-max";
        }
        rows.push_back(cpu);

        // ---- CUDA kernels -------------------------------------------------
        for (const char* name : {"cuda_naive", "cuda_tiled"}) {
            const bool tiled = std::string(name) == "cuda_tiled";
            Row r = base;
            r["impl"] = name;
            r["N"] = std::to_string(N);
            r["block"] = tiled ? std::to_string(opt.tile) + "x" + std::to_string(opt.tile) : "16x16";
            if (tiled) r["tile_size"] = std::to_string(opt.tile);
            if (!cuda_ok) {
                r["status"] = "not_run";
                r["notes"] = cuda_note;
                rows.push_back(r);
                continue;
            }
#ifdef HAVE_CUDA
            try {
                C_gpu.assign(count, NAN);
                const GpuTimings gt =
                    run_cuda_matmul(tiled ? CudaKernel::Tiled : CudaKernel::Naive, A.data(),
                                    B.data(), C_gpu.data(), N, opt.tile, opt.warmup, opt.iters);
                ErrorStats e;
                if (have_cpu) {
                    e = compare_with_tolerance(C_gpu.data(), C_cpu.data(), count, opt.atol, opt.rtol);
                    r["reference"] = "cpu_full";
                } else {
                    e = check_against_fp64(A.data(), B.data(), C_gpu.data(), N, opt.atol, opt.rtol,
                                           opt.fp64_samples, opt.seed);
                    r["reference"] = (e.checked == static_cast<long long>(count)) ? "fp64_full" : "fp64_sampled";
                }
                set_errors(r, e);
                const Stats k = summarize(gt.kernel_ms), h2d = summarize(gt.h2d_ms),
                            d2h = summarize(gt.d2h_ms), e2e = summarize(gt.e2e_ms);
                r["warmup"] = std::to_string(opt.warmup);
                r["iters"] = std::to_string(opt.iters);
                r["time_ms_median"] = num(k.median);
                r["time_ms_min"] = num(k.min);
                r["time_ms_max"] = num(k.max);
                r["h2d_ms_median"] = num(h2d.median);
                r["d2h_ms_median"] = num(d2h.median);
                r["e2e_ms_median"] = num(e2e.median);
                r["e2e_ms_min"] = num(e2e.min);
                r["e2e_ms_max"] = num(e2e.max);
                r["gflops"] = num(gflops(N, k.median));
                r["gflops_e2e"] = num(gflops(N, e2e.median));
                r["speedup_vs_cpu"] = num(cpu_ms / k.median);
                r["speedup_e2e_vs_cpu"] = num(cpu_ms / e2e.median);
                if (!e.pass()) ++failures;
                std::printf("N=%5d %-11s %-17s kernel %9.3f ms  %8.2f GFLOPS  e2e %9.3f ms  max_abs_err %.2e\n",
                            N, name, r["status"].c_str(), k.median, gflops(N, k.median), e2e.median,
                            e.max_abs_err);
            } catch (const std::exception& ex) {
                r["status"] = "error";
                r["correct"] = "false";
                r["notes"] = ex.what();
                ++failures;
                std::printf("N=%5d %-11s error: %s\n", N, name, ex.what());
            }
#endif
            rows.push_back(r);
        }
    }

    if (!opt.validate_only) {
        try {
            write_csv(opt.csv, rows);
            std::cout << "Wrote " << rows.size() << " rows to " << opt.csv << "\n";
        } catch (const std::exception& ex) {
            std::cerr << "error: " << ex.what() << "\n";
            return 2;
        }
    }
    if (failures) std::cout << failures << " implementation(s) FAILED validation or errored\n";
    else std::cout << "All executed implementations passed validation\n";
    return failures ? 1 : 0;
}
