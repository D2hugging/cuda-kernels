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
        "AvgTime_ms": "avg_time_ms",
        "Bandwidth_GBs": "bandwidth_gbps",
    }
    df = df.rename(columns=column_mapping)

    # Extract base kernel name (remove _N_XXX suffix) for grouping
    df["kernel_type"] = df["tag"].str.replace(r"_N_\d+K?$", "", regex=True)

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print summary statistics to console."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Group by kernel type (base name without size suffix)
    grouped = df.groupby("kernel_type")

    print(f"\n{'Kernel Type':<20} {'Avg Time (ms)':<15} {'Bandwidth (GB/s)':<18} {'Speedup':<10}")
    print("-" * 70)

    # Find baseline (Stage1Naive) for speedup calculation
    baseline_data = df[df["kernel_type"] == "Stage1Naive"]
    baseline_time = baseline_data["avg_time_ms"].mean() if not baseline_data.empty else df["avg_time_ms"].mean()

    for kernel_type, group in grouped:
        avg_time = group["avg_time_ms"].mean()
        avg_bandwidth = group["bandwidth_gbps"].mean()
        speedup = baseline_time / avg_time if avg_time > 0 else 0
        print(f"{kernel_type:<20} {avg_time:<15.4f} {avg_bandwidth:<18.2f} {speedup:<10.2f}x")

    print("\n" + "=" * 70)

    # Data size info
    print(f"\nData sizes tested: {sorted(df['data_size'].unique())}")
    print(f"Total benchmark runs: {len(df)}")


def plot_kernel_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar chart comparing kernel execution times by kernel type."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Get unique kernel types and their average times
    avg_times = df.groupby("kernel_type")["avg_time_ms"].mean().sort_values()

    bars = ax.barh(range(len(avg_times)), avg_times.values, color="steelblue")
    ax.set_yticks(range(len(avg_times)))
    ax.set_yticklabels(avg_times.index)
    ax.set_xlabel("Average Execution Time (ms)")
    ax.set_title("Kernel Performance Comparison (averaged across all data sizes)")

    # Add value labels on bars
    for bar, val in zip(bars, avg_times.values):
        ax.text(val + 0.01 * max(avg_times.values), bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "kernel_comparison.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_bandwidth_vs_size(df: pd.DataFrame, output_dir: Path) -> None:
    """Create line chart of bandwidth vs data size for each kernel type."""
    fig, ax = plt.subplots(figsize=(10, 6))

    kernel_types = df["kernel_type"].unique()
    colors = plt.cm.tab10.colors

    for i, kernel_type in enumerate(kernel_types):
        kernel_data = df[df["kernel_type"] == kernel_type].sort_values("data_size")
        ax.plot(
            kernel_data["data_size"],
            kernel_data["bandwidth_gbps"],
            marker="o",
            label=kernel_type,
            color=colors[i % len(colors)],
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Data Size (elements)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Memory Bandwidth vs Data Size")
    ax.set_xscale("log")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "bandwidth_vs_size.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_speedup_chart(df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar chart showing speedup relative to baseline kernel."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Use Stage1Naive as baseline
    baseline_data = df[df["kernel_type"] == "Stage1Naive"]
    if baseline_data.empty:
        baseline_type = df["kernel_type"].iloc[0]
        baseline_data = df[df["kernel_type"] == baseline_type]
    else:
        baseline_type = "Stage1Naive"
    baseline_time = baseline_data["avg_time_ms"].mean()

    avg_times = df.groupby("kernel_type")["avg_time_ms"].mean()
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

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
