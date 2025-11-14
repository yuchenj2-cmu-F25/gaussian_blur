# Automated Benchmarking System

This document explains the automated benchmarking system for the Gaussian blur implementations.

## Overview

The automated benchmark system runs all blur implementations across different image dimensions, executes multiple iterations for statistical reliability, and outputs results in CSV format.

## Files

- **run_benchmarks.sh** - Main automation script
- **main.c** - Modified to support CSV output mode via `--csv` flag
- **benchmark_results/** - Directory where CSV results are stored

## Quick Start

```bash
# Make the script executable
chmod +x run_benchmarks.sh

# Run the full benchmark suite
./run_benchmarks.sh
```

## What It Does

The script performs the following steps:

1. **For each image dimension configuration** (600×800 and 300×400):
   - Compiles the code with dimension-specific macros (`-DHEIGHT=X -DWIDTH=Y`)
   - Runs the complete benchmark suite 10 times (outer loop)
   - Each run executes all 5 implementations with 100 internal iterations (RUNS=100)

2. **Collects FLOPS/cycle** for each implementation:
   - `reference` - Scalar baseline (no auto-vectorization)
   - `5x16` - Standard 5×16 SIMD implementation
   - `5x16_lowload` - Low memory load variant
   - `10x8` - 10×8 block size
   - `4x80` - Unrolled 4×80 implementation

3. **Calculates statistics**:
   - Averages FLOPS/cycle across the 10 outer runs
   - Provides stable, representative performance metrics

4. **Generates CSV files**:
   - `benchmark_results/benchmark_600x800.csv`
   - `benchmark_results/benchmark_300x400.csv`

## Configuration

Edit `run_benchmarks.sh` to customize:

```bash
# Number of outer iterations (default: 10)
OUTER_RUNS=10

# Image dimensions to test
declare -a CONFIGS=(
    "600:800"   # HEIGHT:WIDTH
    "300:400"   # HEIGHT:WIDTH
)
```

## CSV Output Format

Each CSV file contains:

```csv
Benchmark,FLOPS_per_Cycle
reference,1.234567
5x16,4.567890
5x16_lowload,4.678901
10x8,4.789012
4x80,4.890123
```

## How --csv Mode Works

When `main.c` is run with the `--csv` flag:

```bash
./blur_benchmark --csv
```

It:
- Suppresses normal printf output
- Runs all benchmarks silently
- Outputs a single CSV line: `ref,5x16,5x16_low,10x8,4x80`

This makes parsing simple and reliable.

## Compilation Details

The script compiles with:

```bash
gcc -O3 -march=native -Wall -Wextra \
    -DHEIGHT=${height} -DWIDTH=${width} -DRUNS=100 \
    -o blur_benchmark \
    main.c blur.c utils.c kernels_vert.c kernels_horiz.c \
    kernels_vert_10x8.c kernels_vert_4x80.c
```

The `-D` flags override the macros in `config.h`, allowing dimension changes without modifying source files.

## Performance Metric

**FLOPS/cycle** is calculated as:

```c
FLOPS_per_cycle = (12.0 * HEIGHT * WIDTH) / average_cycles
```

Where:
- 12 operations per pixel = 6 FLOPs (3 horizontal + 3 vertical) × 2 (multiply-add counts as 2)
- `average_cycles` is averaged over RUNS=100 internal iterations

## Interpreting Results

Higher FLOPS/cycle is better. Typical results:

- **reference**: ~1-2 FLOPS/cycle (scalar baseline)
- **SIMD variants**: ~4-8 FLOPS/cycle (depending on block size and unrolling)
- **Best performers**: Usually `4x80` or `5x16_lowload`

## Adding New Dimensions

To test additional image sizes:

```bash
declare -a CONFIGS=(
    "600:800"
    "300:400"
    "1024:1024"  # Add new dimension
    "480:640"    # Add another
)
```

**Important**: Ensure dimensions are divisible by block sizes:
- WIDTH should be divisible by 80 (for 4×80), 16 (for 5×16, 10×8)
- HEIGHT should be divisible by 4 (for 4×80), 5 (for 5×16), 10 (for 10×8)

## Troubleshooting

### Compilation Fails
- Check that all source files are present
- Ensure GCC supports `-march=native` and AVX2
- Verify dimensions are compatible with block sizes

### Inconsistent Results
- Increase `OUTER_RUNS` for more stable averages
- Check system load (close other applications)
- Pin to specific CPU cores if needed

### CSV Parsing Errors
- Ensure `--csv` mode outputs exactly one line
- Check for stderr output polluting stdout
- Verify `bc` is installed for averaging calculations

## Example Output

```
========================================
Gaussian Blur Automated Benchmark
========================================

========================================
Benchmarking: HEIGHT=600, WIDTH=800
========================================
Compiling with HEIGHT=600, WIDTH=800...
Compilation successful!
Outer iteration 1/10
  Running benchmark iteration...
    Results: ref=1.234567, 5x16=4.567890, 5x16_low=4.678901, 10x8=4.789012, 4x80=4.890123
...
Outer iteration 10/10
  Running benchmark iteration...
    Results: ref=1.234567, 5x16=4.567890, 5x16_low=4.678901, 10x8=4.789012, 4x80=4.890123

Calculating averages across 10 runs...
Writing results to benchmark_results/benchmark_600x800.csv

Results for 600x800:
Benchmark,FLOPS_per_Cycle
reference,1.234567
5x16,4.567890
5x16_lowload,4.678901
10x8,4.789012
4x80,4.890123

...

Benchmark complete!
Results saved in: benchmark_results/
```

## Integration with Analysis Tools

The CSV format is designed for easy import into:

- **Python/Pandas**: `pd.read_csv('benchmark_results/benchmark_600x800.csv')`
- **Excel/LibreOffice**: Direct import
- **R**: `read.csv('benchmark_results/benchmark_600x800.csv')`
- **gnuplot**: Plotting performance comparisons

## Notes

- Each full benchmark run takes several minutes
- Results are appended if `benchmark_results/` already exists
- The script uses `bc` for floating-point arithmetic
- Color output requires ANSI terminal support

---

**Created**: 2025-11-13
**Author**: Automated benchmark system for Gaussian blur optimization study
