#!/usr/bin/env python3
"""
Benchmark visualization script for CUDA kernel performance analysis.

Parses CSV output from the C++ Benchmarker class and generates
performance comparison charts and summary statistics.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_benchmark_data(csv_path: Path) -> pd.DataFrame:
    """Load benchmark CSV and return a DataFrame with normalized column names."""
    df = pd.read_csv(csv_path)

    # Normalize column names to expected format
    column_mapping = {
        "Algorithm": "tag",
        "N": "data_size",
        "MinTime_ms": "min_time_ms",
        "MaxTime_ms": "max_time_ms",
        "MeanTime_ms": "mean_time_ms",
        "MedianTime_ms": "median_time_ms",
        "StdDev_ms": "stddev_ms",
        "Bandwidth_GBs": "bandwidth_gbps",
        "PeakBandwidth_GBs": "peak_bandwidth_gbps",
    }
    df = df.rename(columns=column_mapping)

    # Extract base kernel name (remove _N_XXX suffix) for grouping
    df["kernel_type"] = df["tag"].str.replace(r"_\d+(x\d+)*$", "", regex=True)

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics to console."""
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY")
    print("=" * 90)

    # Group by kernel type (base name without size suffix)
    grouped = df.groupby("kernel_type")

    print(f"\n{'Kernel Type':<20} {'Mean (ms)':<12} {'Min (ms)':<12} {'StdDev (ms)':<12} {'BW (GB/s)':<12} {'Speedup':<10}")
    print("-" * 90)

    # Find baseline: slowest kernel (highest mean time)
    kernel_times = df.groupby("kernel_type")["mean_time_ms"].mean()
    baseline_type = kernel_times.idxmax()
    baseline_time = kernel_times.max()

    print(f"Using '{baseline_type}' as baseline (slowest kernel)\n")

    for kernel_type, group in grouped:
        mean_time = group["mean_time_ms"].mean()
        min_time = group["min_time_ms"].mean()
        stddev = group["stddev_ms"].mean()
        avg_bandwidth = group["bandwidth_gbps"].mean()
        speedup = baseline_time / mean_time if mean_time > 0 else 0
        print(f"{kernel_type:<20} {mean_time:<12.4f} {min_time:<12.4f} {stddev:<12.6f} {avg_bandwidth:<12.2f} {speedup:<10.2f}x")

    print("\n" + "=" * 90)

    # Data size info
    print(f"\nData sizes tested: {sorted(df['data_size'].unique())}")
    print(f"Total benchmark runs: {len(df)}")


def plot_kernel_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar chart comparing kernel execution times by kernel type."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique kernel types and their statistics
    kernel_stats = df.groupby("kernel_type").agg({
        "mean_time_ms": "mean",
        "min_time_ms": "mean",
        "max_time_ms": "mean",
        "stddev_ms": "mean"
    }).sort_values("mean_time_ms")

    x = range(len(kernel_stats))
    bars = ax.bar(x, kernel_stats["mean_time_ms"], color="steelblue", label="Mean", alpha=0.7)

    # Add error bars showing min/max range
    yerr_low = kernel_stats["mean_time_ms"] - kernel_stats["min_time_ms"]
    yerr_high = kernel_stats["max_time_ms"] - kernel_stats["mean_time_ms"]
    ax.errorbar(x, kernel_stats["mean_time_ms"],
                yerr=[yerr_low, yerr_high],
                fmt='none', ecolor='black', capsize=5, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(kernel_stats.index, rotation=45, ha='right')
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title("Kernel Performance Comparison (averaged across all data sizes)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "kernel_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_bandwidth_vs_size(df: pd.DataFrame, output_dir: Path) -> None:
    """Create line chart of bandwidth vs data size for each kernel type."""
    fig, ax = plt.subplots(figsize=(12, 6))

    kernel_types = df["kernel_type"].unique()
    colors = plt.cm.tab10.colors

    for i, kernel_type in enumerate(kernel_types):
        kernel_data = df[df["kernel_type"] == kernel_type].sort_values("data_size")
        # Plot mean bandwidth
        ax.plot(
            kernel_data["data_size"],
            kernel_data["bandwidth_gbps"],
            marker="o",
            label=f"{kernel_type} (mean)",
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=6,
        )
        # Plot peak bandwidth (lighter)
        ax.plot(
            kernel_data["data_size"],
            kernel_data["peak_bandwidth_gbps"],
            marker="^",
            label=f"{kernel_type} (peak)",
            color=colors[i % len(colors)],
            linewidth=1,
            linestyle="--",
            markersize=4,
            alpha=0.5,
        )

    ax.set_xlabel("Data Size (elements)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Memory Bandwidth vs Data Size")
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "bandwidth_vs_size.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_speedup_chart(df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar chart showing speedup relative to baseline kernel."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Find baseline: slowest kernel (highest mean time)
    kernel_times = df.groupby("kernel_type")["mean_time_ms"].mean()
    baseline_type = kernel_times.idxmax()
    baseline_time = kernel_times.max()

    avg_times = df.groupby("kernel_type")["mean_time_ms"].mean()
    speedups = baseline_time / avg_times

    bars = ax.barh(range(len(speedups)), speedups.values, color="forestgreen")
    ax.set_yticks(range(len(speedups)))
    ax.set_yticklabels(speedups.index)
    ax.set_xlabel(f"Speedup vs {baseline_type}")
    ax.set_title("Kernel Speedup Comparison")
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="Baseline")

    # Add value labels
    for bar, val in zip(bars, speedups.values):
        ax.text(val + 0.02 * max(speedups.values), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}x", va="center", fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "speedup_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_variance_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Create chart showing timing variance (coefficient of variation) by kernel type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate coefficient of variation (CV = stddev / mean)
    kernel_stats = df.groupby("kernel_type").agg({
        "mean_time_ms": "mean",
        "stddev_ms": "mean"
    })
    kernel_stats["cv_percent"] = (kernel_stats["stddev_ms"] / kernel_stats["mean_time_ms"]) * 100
    kernel_stats = kernel_stats.sort_values("cv_percent")

    bars = ax.barh(range(len(kernel_stats)), kernel_stats["cv_percent"], color="coral")
    ax.set_yticks(range(len(kernel_stats)))
    ax.set_yticklabels(kernel_stats.index)
    ax.set_xlabel("Coefficient of Variation (%)")
    ax.set_title("Timing Variance by Kernel (lower is more consistent)")

    # Add value labels
    for bar, val in zip(bars, kernel_stats["cv_percent"].values):
        ax.text(val + 0.02 * max(kernel_stats["cv_percent"].values),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}%", va="center", fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "variance_analysis.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CUDA benchmark results and generate visualizations."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to benchmark CSV file (e.g., ../dot_product_benchmark.csv)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same as CSV file)",
    )

    args = parser.parse_args()

    if not args.csv_path.exists():
        print(f"Error: CSV file not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or args.csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading benchmark data from: {args.csv_path}")
    df = load_benchmark_data(args.csv_path)

    print_summary(df)

    print("\nGenerating plots...")
    plot_kernel_comparison(df, output_dir)
    plot_bandwidth_vs_size(df, output_dir)
    plot_speedup_chart(df, output_dir)
    plot_variance_analysis(df, output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
