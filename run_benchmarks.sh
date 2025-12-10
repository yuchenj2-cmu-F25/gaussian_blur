#!/bin/bash

# Automated Canny/Sobel Benchmark Script
# Runs benchmarks with different image dimensions and outputs CSV results

set -e  # Exit on error

# Configuration
OUTER_RUNS=10  # Number of times to run the entire benchmark suite
OUTPUT_DIR="benchmark_results"

# OpenMP configuration
export OMP_NUM_THREADS=6
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export OMP_SCHEDULE=static

# Image dimension configurations (square images)
declare -a CONFIGS=(
    "128:128"   # HEIGHT:WIDTH
    "256:256"
    "512:512"
    "1024:1024"
)

# Benchmark names (must match order in output)
BENCHMARK_NAMES=("gaussian_ref" "sobel_ref" "pipeline_ref" "gaussian_4x96" "sobel_4x96" "pipeline_4x96")

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Canny/Sobel Automated Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to compile with specific dimensions
compile_with_dimensions() {
    local height=$1
    local width=$2

    echo -e "${YELLOW}Compiling with HEIGHT=${height}, WIDTH=${width}...${NC}"

    # Clean previous build
    make clean > /dev/null 2>&1 || true

    # Compile with dimension macros
    # Override HEIGHT, WIDTH, and RUNS from config.h
    gcc -O3 -march=native -Wall -Wextra -fopenmp -Ikernel1 \
        -DHEIGHT=${height} -DWIDTH=${width} -DRUNS=100 \
        -o blur_benchmark \
        main.c \
        kernel1/blur_main.c kernel1/utils.c \
        kernel1/kernels_vert_4x96_macro.c kernel1/kernels_horiz_4x96_macro.c \
        kernel1/kernels_sobel_4x96_macro.c

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Compilation successful!${NC}"
        return 0
    else
        echo "Compilation failed!"
        return 1
    fi
}

# Function to run single benchmark iteration
run_single_benchmark() {
    local height=$1
    local width=$2

    # Run the benchmark in CSV mode and capture output
    # Output format: gaussian_ref,sobel_ref,pipeline_ref,gaussian_4x96,sobel_4x96,pipeline_4x96
    local output=$(./blur_benchmark --csv 2>&1)

    # Parse CSV output
    echo "$output"
}

# Function to calculate average from array
calculate_average() {
    local sum=0
    local count=$#

    for val in "$@"; do
        sum=$(echo "$sum + $val" | bc -l)
    done

    echo "scale=6; $sum / $count" | bc -l
}

# Main benchmark loop
for config in "${CONFIGS[@]}"; do
    IFS=':' read -r HEIGHT WIDTH <<< "$config"

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Benchmarking: HEIGHT=${HEIGHT}, WIDTH=${WIDTH}${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Compile with these dimensions
    compile_with_dimensions "$HEIGHT" "$WIDTH"

    # Arrays to store results from each outer run
    declare -a gaussian_ref_results=()
    declare -a sobel_ref_results=()
    declare -a pipeline_ref_results=()
    declare -a gaussian_4x96_results=()
    declare -a sobel_4x96_results=()
    declare -a pipeline_4x96_results=()

    # Run benchmark OUTER_RUNS times
    for i in $(seq 1 $OUTER_RUNS); do
        echo -e "${YELLOW}Outer iteration $i/$OUTER_RUNS${NC}"

        # Run single benchmark and parse CSV results
        csv_line=$(run_single_benchmark "$HEIGHT" "$WIDTH")

        # Parse CSV: gaussian_ref,sobel_ref,pipeline_ref,gaussian_4x96,sobel_4x96,pipeline_4x96
        IFS=',' read -r g_ref s_ref p_ref g_4x96 s_4x96 p_4x96 <<< "$csv_line"

        # Store results
        gaussian_ref_results+=("$g_ref")
        sobel_ref_results+=("$s_ref")
        pipeline_ref_results+=("$p_ref")
        gaussian_4x96_results+=("$g_4x96")
        sobel_4x96_results+=("$s_4x96")
        pipeline_4x96_results+=("$p_4x96")

        echo "    Ref: g=$g_ref, s=$s_ref, p=$p_ref | 4x96: g=$g_4x96, s=$s_4x96, p=$p_4x96"
    done

    # Calculate averages
    echo ""
    echo -e "${GREEN}Calculating averages across $OUTER_RUNS runs...${NC}"

    avg_gaussian_ref=$(calculate_average "${gaussian_ref_results[@]}")
    avg_sobel_ref=$(calculate_average "${sobel_ref_results[@]}")
    avg_pipeline_ref=$(calculate_average "${pipeline_ref_results[@]}")
    avg_gaussian_4x96=$(calculate_average "${gaussian_4x96_results[@]}")
    avg_sobel_4x96=$(calculate_average "${sobel_4x96_results[@]}")
    avg_pipeline_4x96=$(calculate_average "${pipeline_4x96_results[@]}")

    # Generate CSV filename
    csv_file="${OUTPUT_DIR}/benchmark_${HEIGHT}x${WIDTH}.csv"

    # Write CSV
    echo -e "${GREEN}Writing results to $csv_file${NC}"

    cat > "$csv_file" << EOF
Benchmark,FLOPS_per_Cycle
gaussian_ref,$avg_gaussian_ref
sobel_ref,$avg_sobel_ref
pipeline_ref,$avg_pipeline_ref
gaussian_4x96,$avg_gaussian_4x96
sobel_4x96,$avg_sobel_4x96
pipeline_4x96,$avg_pipeline_4x96
EOF

    echo ""
    echo -e "${GREEN}Results for ${HEIGHT}x${WIDTH}:${NC}"
    cat "$csv_file"

    # Clean up arrays for next iteration
    unset gaussian_ref_results
    unset sobel_ref_results
    unset pipeline_ref_results
    unset gaussian_4x96_results
    unset sobel_4x96_results
    unset pipeline_4x96_results
done

# Clean up
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Benchmark complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results saved in: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR"/*.csv

# Optional: combine results into summary
echo ""
echo -e "${YELLOW}Summary of all benchmarks:${NC}"
for csv in "$OUTPUT_DIR"/*.csv; do
    echo ""
    echo "=== $(basename $csv) ==="
    cat "$csv"
done
