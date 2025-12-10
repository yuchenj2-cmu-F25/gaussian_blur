import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_benchmarks(results_dir: Path):
    sizes = []
    gaussian_ref = []
    gaussian_opt = []
    sobel_ref = []
    sobel_opt = []
    pipeline_ref = []
    pipeline_opt = []

    # Expect files like benchmark_128x128.csv, benchmark_256x256.csv, ...
    for size in [128, 256, 512, 1024]:
        csv_path = results_dir / f"benchmark_{size}x{size}.csv"
        if not csv_path.is_file():
            continue

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = {row["Benchmark"]: float(row["FLOPS_per_Cycle"]) for row in reader}

        sizes.append(size)
        gaussian_ref.append(rows["gaussian_ref"])
        gaussian_opt.append(rows["gaussian_4x96"])
        sobel_ref.append(rows["sobel_ref"])
        sobel_opt.append(rows["sobel_4x96"])
        pipeline_ref.append(rows["pipeline_ref"])
        pipeline_opt.append(rows["pipeline_4x96"])

    return (
        sizes,
        gaussian_ref,
        gaussian_opt,
        sobel_ref,
        sobel_opt,
        pipeline_ref,
        pipeline_opt,
    )


def main():
    results_dir = Path("benchmark_results")
    (
        sizes,
        gaussian_ref,
        gaussian_opt,
        sobel_ref,
        sobel_opt,
        pipeline_ref,
        pipeline_opt,
    ) = load_benchmarks(results_dir)

    if not sizes:
        print(f"No benchmark_*.csv files found in {results_dir}")
        return

    x_labels = [f"{s}x{s}" for s in sizes]

    # Gaussian figure
    fig_g, ax_g = plt.subplots(figsize=(6, 4))
    ax_g.plot(sizes, gaussian_ref, marker="o", label="gaussian_ref")
    ax_g.plot(sizes, gaussian_opt, marker="o", label="gaussian_4x96")
    ax_g.axhline(32.0, color="gray", linestyle="--", label="theoretical peak (32)")
    ax_g.set_xticks(sizes)
    ax_g.set_xticklabels(x_labels)
    ax_g.set_xlabel("Image size (pixels)")
    ax_g.set_ylabel("FLOPS_per_Cycle")
    ax_g.set_title("Gaussian Blur: reference vs 4x96")
    ax_g.legend()
    ax_g.grid(True, linestyle=":", alpha=0.5)
    fig_g.tight_layout()
    fig_g.savefig(results_dir / "gaussian_benchmark.png", dpi=150)

    # Sobel figure
    fig_s, ax_s = plt.subplots(figsize=(6, 4))
    ax_s.plot(sizes, sobel_ref, marker="o", label="sobel_ref")
    ax_s.plot(sizes, sobel_opt, marker="o", label="sobel_4x96")
    ax_s.axhline(32.0, color="gray", linestyle="--", label="theoretical peak (32)")
    ax_s.set_xticks(sizes)
    ax_s.set_xticklabels(x_labels)
    ax_s.set_xlabel("Image size (pixels)")
    ax_s.set_ylabel("FLOPS_per_Cycle")
    ax_s.set_title("Sobel: reference vs 4x96")
    ax_s.legend()
    ax_s.grid(True, linestyle=":", alpha=0.5)
    fig_s.tight_layout()
    fig_s.savefig(results_dir / "sobel_benchmark.png", dpi=150)

    # Pipeline figure
    fig_p, ax_p = plt.subplots(figsize=(6, 4))
    ax_p.plot(sizes, pipeline_ref, marker="o", label="pipeline_ref")
    ax_p.plot(sizes, pipeline_opt, marker="o", label="pipeline_4x96")
    ax_p.axhline(32.0, color="gray", linestyle="--", label="theoretical peak (32)")
    ax_p.set_xticks(sizes)
    ax_p.set_xticklabels(x_labels)
    ax_p.set_xlabel("Image size (pixels)")
    ax_p.set_ylabel("FLOPS_per_Cycle")
    ax_p.set_title("Full pipeline: reference vs 4x96")
    ax_p.legend()
    ax_p.grid(True, linestyle=":", alpha=0.5)
    fig_p.tight_layout()
    fig_p.savefig(results_dir / "pipeline_benchmark.png", dpi=150)


if __name__ == "__main__":
    main()
