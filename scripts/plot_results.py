#!/usr/bin/env python3
"""Generate plots from results/benchmark_results.csv.

Only rows with status == "ok" and correct == true are plotted. Rows that were
not run, failed validation, or errored are listed on the affected plot and in
plots/summary.md, so a missing line is never silent.
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Fixed identity per implementation (validated categorical palette); markers
# and line styles carry identity too, so color is never the only cue.
STYLE = {
    "cpu":        dict(color="#2a78d6", marker="o", linestyle="-",  label="CPU (triple loop)"),
    "cuda_naive": dict(color="#eb6834", marker="s", linestyle="-",  label="CUDA naive (kernel)"),
    "cuda_tiled": dict(color="#1baf7a", marker="^", linestyle="-",  label="CUDA tiled (kernel)"),
    "e2e_naive":  dict(color="#eb6834", marker="s", linestyle="--", label="CUDA naive (end-to-end)"),
    "e2e_tiled":  dict(color="#1baf7a", marker="^", linestyle="--", label="CUDA tiled (end-to-end)"),
}
SEGMENTS = [("h2d_ms_median", "Host→device copy", "#2a78d6"),
            ("time_ms_median", "Kernel", "#1baf7a"),
            ("d2h_ms_median", "Device→host copy", "#eda100")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2, "lines.markersize": 7, "font.size": 10, "legend.frameon": False,
})


def shown(path: Path) -> str:
    """Path relative to the repo when inside it, else absolute."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load(path: Path):
    df = pd.read_csv(path)
    df["correct"] = df["correct"].astype(str).str.lower().eq("true")
    df["valid"] = df["status"].eq("ok") & df["correct"]
    return df


def series(df, impl, col):
    s = df[(df["impl"] == impl) & df["valid"]].dropna(subset=[col]).sort_values("N")
    return s["N"], s[col]


def note_missing(ax, df, impls, width=95):
    """Write below the axes which rows are absent from this plot and why."""
    bad = df[df["impl"].isin(impls) & ~df["valid"]]
    if bad.empty:
        return
    lines = []
    for (impl, status), g in bad.groupby(["impl", "status"]):
        reason = g["notes"].dropna().astype(str)
        why = f": {reason.iloc[0][:70]}" if not reason.empty else ""
        lines.append(textwrap.fill(f"{impl} {status} for N={','.join(map(str, sorted(g['N'])))}{why}",
                                   width, subsequent_indent="    "))
    ax.text(0.0, -0.16, "Not plotted:\n" + "\n".join(lines), transform=ax.transAxes,
            fontsize=7.5, color=MUTED, va="top", ha="left")


def finish(fig, ax, title, ylabel, out, logy=True):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlabel("Matrix size N (N×N, float32)")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
        # Plain-number ticks at 1-2-5 steps read better than bare powers of ten.
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {shown(out)}")


def plot_lines(df, out, title, ylabel, entries, logy=True):
    """entries: list of (style_key, impl, column)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0
    for key, impl, col in entries:
        x, y = series(df, impl, col)
        y = y[y > 0] if logy else y
        if len(y):
            ax.plot(x[y.index], y, **STYLE[key])
            plotted += 1
    sizes = sorted(df["N"].unique())
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(n) for n in sizes])
    ax.set_xlim(sizes[0] / 1.25, sizes[-1] * 1.25)
    ax.minorticks_off()
    if not plotted:
        # A log axis cannot be drawn without data; say so instead of crashing.
        ax.text(0.5, 0.55, "No valid measurements for this plot", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color=MUTED)
        ax.set_yticks([])
        logy = False
    note_missing(ax, df, {impl for _, impl, _ in entries})
    finish(fig, ax, title, ylabel, out, logy)


def plot_transfer_overhead(df, out):
    impls = ["cuda_naive", "cuda_tiled"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
    for ax, impl in zip(axes, impls):
        g = df[(df["impl"] == impl) & df["valid"]].sort_values("N")
        if g.empty:
            ax.text(0.5, 0.5, f"No valid {impl} measurements", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            labels = g["N"].astype(str)
            totals = g[[c for c, _, _ in SEGMENTS]].sum(axis=1)
            bottom = pd.Series(0.0, index=g.index)
            for col, name, color in SEGMENTS:
                share = 100 * g[col] / totals
                ax.bar(labels, share, bottom=bottom, color=color, label=name,
                       edgecolor="#fcfcfb", linewidth=1.5, width=0.6)
                bottom += share
            for i, tot in enumerate(totals):
                ax.text(i, 101, f"{tot:.3g} ms", ha="center", va="bottom", fontsize=8, color=MUTED)
            ax.set_ylim(0, 110)
        note_missing(ax, df, {impl}, width=60)
        ax.set_title(f"{STYLE[impl]['label'].split(' (')[0]}: share of end-to-end time",
                     loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Matrix size N")
        ax.set_ylabel("Share of H2D + kernel + D2H (%)")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Transfer overhead vs computation (labels: total median time)", x=0.01,
                 ha="left", fontsize=12, fontweight="bold")
    handles, labels = next((a.get_legend_handles_labels() for a in axes
                            if a.get_legend_handles_labels()[0]), ([], []))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if handles:
        fig.legend(handles, labels, loc="upper right", ncol=3, fontsize=9)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {shown(out)}")


def write_summary(df, out):
    cols = ["impl", "N", "status", "correct", "time_ms_median", "h2d_ms_median", "d2h_ms_median",
            "e2e_ms_median", "gflops", "speedup_vs_cpu", "speedup_e2e_vs_cpu", "max_abs_err"]
    t = df[cols].sort_values(["N", "impl"])
    t = t.astype(str).mask(t.isna(), "")
    # "correct" is only meaningful for implementations that actually ran.
    t.loc[t["status"].isin(["not_run", "error"]), "correct"] = ""
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in t.itertuples(index=False)]
    meta = df.iloc[0]
    header = (f"# Benchmark summary\n\nrun_id `{meta.get('run_id', '')}`, device "
              f"`{meta.get('device', '') if pd.notna(meta.get('device')) else 'none'}`, "
              f"compiler `{meta.get('compiler', '')}`. Times are medians in ms.\n\n")
    out.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {shown(out)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="results/benchmark_results.csv")
    ap.add_argument("--outdir", default="plots")
    args = ap.parse_args()

    path = ROOT / args.input
    if not path.is_file():
        raise SystemExit(f"error: {path} not found; run scripts/run_benchmarks.py first.")
    df = load(path)
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    plot_lines(df, outdir / "cpu_vs_gpu_time.png", "CPU vs GPU execution time", "Median time (ms, log scale)",
               [("cpu", "cpu", "time_ms_median"), ("cuda_naive", "cuda_naive", "time_ms_median"),
                ("cuda_tiled", "cuda_tiled", "time_ms_median"), ("e2e_tiled", "cuda_tiled", "e2e_ms_median")])
    plot_lines(df, outdir / "naive_vs_tiled_kernel.png", "Naive vs tiled CUDA kernel time",
               "Median kernel time (ms, log scale)",
               [("cuda_naive", "cuda_naive", "time_ms_median"), ("cuda_tiled", "cuda_tiled", "time_ms_median")])
    plot_lines(df, outdir / "speedup.png", "Speedup over the CPU baseline", "Speedup (× vs CPU, log scale)",
               [("cuda_naive", "cuda_naive", "speedup_vs_cpu"), ("cuda_tiled", "cuda_tiled", "speedup_vs_cpu"),
                ("e2e_naive", "cuda_naive", "speedup_e2e_vs_cpu"), ("e2e_tiled", "cuda_tiled", "speedup_e2e_vs_cpu")])
    plot_lines(df, outdir / "gflops.png", "Achieved throughput", "GFLOPS (2N³ / time, log scale)",
               [("cpu", "cpu", "gflops"), ("cuda_naive", "cuda_naive", "gflops"),
                ("cuda_tiled", "cuda_tiled", "gflops")])
    plot_transfer_overhead(df, outdir / "transfer_overhead.png")
    write_summary(df, outdir / "summary.md")


if __name__ == "__main__":
    main()
