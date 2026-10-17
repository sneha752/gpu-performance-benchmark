#!/usr/bin/env python3
"""Run correctness tests, then the benchmark, and record run metadata.

Standard library only. Steps:
  1. Correctness gate: `matmul_bench --validate-only` on awkward sizes
     (non-multiples of the tile size). Benchmarks are not run if it fails.
  2. Benchmark: `matmul_bench` over the requested sizes, writing the CSV.
  3. Metadata: results/run_metadata.json (host, CPU, GPU, git commit, command).
"""

import argparse
import csv
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALIDATION_SIZES = "1,2,7,15,16,17,31,32,33,64,100,127,128,129,255,256,257"


def find_binary(build_dir: Path) -> Path:
    names = ["matmul_bench", "matmul_bench.exe"]
    for sub in ["", "Release", "RelWithDebInfo", "Debug"]:
        for name in names:
            p = build_dir / sub / name
            if p.is_file():
                return p
    sys.exit(f"error: matmul_bench not found under {build_dir}; build it first (see README).")


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def command_output(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=ROOT).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run(cmd):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=ROOT).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default="build", help="CMake build directory (default: build)")
    ap.add_argument("--sizes", default="128,256,512,1024,2048", help="comma-separated matrix sizes")
    ap.add_argument("--warmup", type=int, default=3, help="untimed GPU iterations")
    ap.add_argument("--iters", type=int, default=10, help="timed GPU iterations")
    ap.add_argument("--cpu-iters", type=int, default=3, help="timed CPU iterations for N < 1024")
    ap.add_argument("--cpu-iters-large", type=int, default=1, help="timed CPU iterations for N >= 1024")
    ap.add_argument("--cpu-max", type=int, default=4096, help="skip the CPU baseline above this N")
    ap.add_argument("--tile", type=int, default=16, choices=[8, 16, 32], help="tiled-kernel tile size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="results/benchmark_results.csv")
    ap.add_argument("--skip-validation", action="store_true", help="skip the correctness gate")
    args = ap.parse_args()

    binary = find_binary(ROOT / args.build_dir)
    output = ROOT / args.output
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if not args.skip_validation:
        print("== Step 1/2: correctness tests ==")
        rc = run([binary, "--validate-only", "--tile", args.tile, "--seed", args.seed,
                  "--sizes", VALIDATION_SIZES])
        if rc != 0:
            sys.exit(f"error: correctness tests failed (exit code {rc}); benchmark not run.")

    print("== Step 2/2: benchmark ==")
    bench_cmd = [binary, "--sizes", args.sizes, "--warmup", args.warmup, "--iters", args.iters,
                 "--cpu-iters", args.cpu_iters, "--cpu-iters-large", args.cpu_iters_large,
                 "--cpu-max", args.cpu_max, "--tile", args.tile, "--seed", args.seed,
                 "--csv", output, "--run-id", run_id]
    rc = run(bench_cmd)

    metadata = {
        "run_id": run_id,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "exit_code": rc,
        "command": [str(c) for c in bench_cmd],
        "host": platform.node(),
        "os": platform.platform(),
        "cpu_model": cpu_model(),
        "python": platform.python_version(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(command_output(["git", "status", "--porcelain"])),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                      "--format=csv,noheader"]) if shutil.which("nvidia-smi") else None,
        "nvcc": command_output(["nvcc", "--version"]) if shutil.which("nvcc") else None,
    }
    meta_path = output.parent / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {meta_path}")

    if output.is_file():
        print("\nimpl         N      status             median_ms   GFLOPS   speedup")
        with output.open(newline="") as f:
            for r in csv.DictReader(f):
                print(f"{r['impl']:<12} {r['N']:<6} {r['status']:<18} {r['time_ms_median'] or '-':>10} "
                      f"{r['gflops'] or '-':>8} {r['speedup_vs_cpu'] or '-':>9}")
    if rc != 0:
        sys.exit(f"error: benchmark reported failures (exit code {rc}); see CSV status/notes columns.")


if __name__ == "__main__":
    main()
