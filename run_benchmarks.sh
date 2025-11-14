#!/bin/bash

# Automated Gaussian Blur Benchmark Script
# Runs benchmarks with different image dimensions and outputs CSV results

set -e  # Exit on error

# Configuration
OUTER_RUNS=10  # Number of times to run the entire benchmark suite
OUTPUT_DIR="benchmark_results"

# Image dimension configurations
declare -a CONFIGS=(
    "600:800"   # HEIGHT:WIDTH
    "300:400"   # HEIGHT:WIDTH
)

# Benchmark names (must match order in output)
BENCHMARK_NAMES=("reference" "5x16" "5x16_lowload" "10x8" "4x80")

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Gaussian Blur Automated Benchmark${NC}"
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
    gcc -O3 -march=native -Wall -Wextra \
        -DHEIGHT=${height} -DWIDTH=${width} -DRUNS=100 \
        -o blur_benchmark \
        main.c blur.c utils.c kernels_vert.c kernels_horiz.c \
        kernels_vert_10x8.c kernels_vert_4x80.c

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

    echo "  Running benchmark iteration..."

    # Run the benchmark in CSV mode and capture output
    # Output format: reference,5x16,5x16_lowload,10x8,4x80
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
    declare -a ref_results=()
    declare -a b5x16_results=()
    declare -a b5x16_lowload_results=()
    declare -a b10x8_results=()
    declare -a b4x80_results=()

    # Run benchmark OUTER_RUNS times
    for i in $(seq 1 $OUTER_RUNS); do
        echo -e "${YELLOW}Outer iteration $i/$OUTER_RUNS${NC}"

        # Run single benchmark and parse CSV results
        csv_line=$(run_single_benchmark "$HEIGHT" "$WIDTH")

        # Parse CSV: reference,5x16,5x16_lowload,10x8,4x80
        IFS=',' read -r ref b5x16 b5x16_low b10x8 b4x80 <<< "$csv_line"

        # Store results
        ref_results+=("$ref")
        b5x16_results+=("$b5x16")
        b5x16_lowload_results+=("$b5x16_low")
        b10x8_results+=("$b10x8")
        b4x80_results+=("$b4x80")

        echo "    Results: ref=$ref, 5x16=$b5x16, 5x16_low=$b5x16_low, 10x8=$b10x8, 4x80=$b4x80"
    done

    # Calculate averages
    echo ""
    echo -e "${GREEN}Calculating averages across $OUTER_RUNS runs...${NC}"

    avg_ref=$(calculate_average "${ref_results[@]}")
    avg_5x16=$(calculate_average "${b5x16_results[@]}")
    avg_5x16_lowload=$(calculate_average "${b5x16_lowload_results[@]}")
    avg_10x8=$(calculate_average "${b10x8_results[@]}")
    avg_4x80=$(calculate_average "${b4x80_results[@]}")

    # Generate CSV filename
    csv_file="${OUTPUT_DIR}/benchmark_${HEIGHT}x${WIDTH}.csv"

    # Write CSV
    echo -e "${GREEN}Writing results to $csv_file${NC}"

    cat > "$csv_file" << EOF
Benchmark,FLOPS_per_Cycle
reference,$avg_ref
5x16,$avg_5x16
5x16_lowload,$avg_5x16_lowload
10x8,$avg_10x8
4x80,$avg_4x80
EOF

    echo ""
    echo -e "${GREEN}Results for ${HEIGHT}x${WIDTH}:${NC}"
    cat "$csv_file"

    # Clean up arrays for next iteration
    unset ref_results
    unset b5x16_results
    unset b5x16_lowload_results
    unset b10x8_results
    unset b4x80_results
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
