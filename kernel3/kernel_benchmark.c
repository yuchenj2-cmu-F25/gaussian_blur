#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <sys/types.h>
#include <unistd.h>

#include "direction_kernels.h"

/* Configuration */
#define IMAGE_WIDTH   2048
#define IMAGE_HEIGHT  2048
#define NUM_PIXELS    (IMAGE_WIDTH * IMAGE_HEIGHT)
#define NUM_RUNS      100
#define WARMUP_RUNS   10

/* Benchmark result structure */
typedef struct {
    const char *name;
    double cycles;
    double flops;
    double flops_per_cycle;
    double throughput_mpixels;
    int correctness;
} BenchmarkResult;

/* RDTSC for cycle counting (Broadwell-specific) */
static inline unsigned long long rdtsc(void) {
    unsigned int lo, hi;
    asm volatile("rdtsc" : "=a" (lo), "=d" (hi));
    return ((unsigned long long)hi << 32) | lo;
}

/* Serialize execution for accurate timing */
static inline void serialize(void) {
    asm volatile("cpuid" : : "a" (0) : "rbx", "rcx", "rdx");
}

/* Initialize test data */
void init_test_data(float *g_x, float *g_y, int count) {
    srand48(12345);
    for (int i = 0; i < count; i++) {
        // Use uniform distribution [-1, 1] for gradients
        g_x[i] = (float)(drand48() * 2.0 - 1.0);
        g_y[i] = (float)(drand48() * 2.0 - 1.0);
    }
}

/* Reference implementation for correctness checking */
void reference_implementation(const float *g_x, const float *g_y,
                             float *direction, int count) {
    for (int i = 0; i < count; i++) {
        float gx = g_x[i];
        float gy = g_y[i];
        
        float abs_gx = fabsf(gx);
        float abs_gy = fabsf(gy);
        
        float dir;
        if (abs_gx > abs_gy) {
            dir = (gx * gy > 0.0f) ? 1.0f : 0.0f;
        } else {
            dir = (gx * gy < 0.0f) ? 3.0f : 2.0f;
        }
        
        direction[i] = dir;
    }
}

/* Check correctness within floating-point tolerance */
int check_correctness(const float *result, const float *reference, int count, 
                     float tolerance) {
    int errors = 0;
    for (int i = 0; i < count; i++) {
        if (fabsf(result[i] - reference[i]) > tolerance) {
            errors++;
            if (errors <= 5) {
                fprintf(stderr, 
                    "Mismatch at pixel %d: got %.1f, expected %.1f\n",
                    i, result[i], reference[i]);
            }
        }
    }
    return errors == 0;
}

/* Run benchmark with cycle counting */
BenchmarkResult benchmark_kernel(const char *name,
                                void (*kernel_func)(const float*, const float*, float*, int),
                                float *g_x, float *g_y, float *direction,
                                float *reference, int count) {
    BenchmarkResult result;
    result.name = name;
    
    // Warmup runs
    for (int r = 0; r < WARMUP_RUNS; r++) {
        kernel_func(g_x, g_y, direction, count);
    }
    
    // Measure with cycle counting
    serialize();
    unsigned long long t0 = rdtsc();
    
    for (int r = 0; r < NUM_RUNS; r++) {
        kernel_func(g_x, g_y, direction, count);
    }
    
    unsigned long long t1 = rdtsc();
    serialize();
    
    result.cycles = (double)(t1 - t0) / NUM_RUNS;
    
    // Calculate FLOPS
    // Per professor feedback: kernel should mainly do bitwise ops (sign checking)
    // not actual floating-point multiply for sign detection
    // However, current implementation still uses g_x * g_y multiply
    // 
    // Realistic counting:
    // - Multiply g_x * g_y: 1 FLOP (current implementation)
    // - All other ops (AND, CMP, BLEND): 0 FLOPS (bitwise/logical)
    // Total: 1 FLOP per pixel (but could be optimized to 0 FLOPS with sign-bit ops)
    //
    // For comparison with kernel1 (7-8 FLOPS/cycle):
    // If we're getting 34.90 FLOPS/cycle, that suggests either:
    // 1. Different FLOP counting methodology
    // 2. The multiply isn't the bottleneck
    // 3. We may be measuring something other than raw FLOPS
    
    result.flops = (double)count * 1.0 * NUM_RUNS;
    result.flops_per_cycle = result.flops / result.cycles;
    result.throughput_mpixels = ((double)count * NUM_RUNS) / result.cycles / 1e6 * 2.4e9; // Base freq
    
    // Check correctness
    result.correctness = check_correctness(direction, reference, count, 0.01f);
    
    return result;
}

int main(void) {
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Kernel 3: Direction Classification Benchmark\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("Image Size: %d × %d = %d pixels\n", IMAGE_WIDTH, IMAGE_HEIGHT, NUM_PIXELS);
    printf("Benchmark Runs: %d (after %d warmup runs)\n", NUM_RUNS, WARMUP_RUNS);
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n\n");
    
    // Allocate aligned memory
    float *g_x = (float*)aligned_alloc(32, NUM_PIXELS * sizeof(float));
    float *g_y = (float*)aligned_alloc(32, NUM_PIXELS * sizeof(float));
    float *direction = (float*)aligned_alloc(32, NUM_PIXELS * sizeof(float));
    float *reference = (float*)aligned_alloc(32, NUM_PIXELS * sizeof(float));
    
    if (!g_x || !g_y || !direction || !reference) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    // Initialize test data
    init_test_data(g_x, g_y, NUM_PIXELS);
    reference_implementation(g_x, g_y, reference, NUM_PIXELS);
    
    // Benchmark array
    BenchmarkResult results[4];
    int result_count = 0;
    
    printf("Running benchmarks...\n\n");
    
    // Benchmark 1: Naive Scalar
    printf("Kernel 3.1: Naive Scalar Implementation...\n");
    results[result_count++] = benchmark_kernel(
        "3.1 Naive Scalar",
        kernel_direction_naive,
        g_x, g_y, direction, reference, NUM_PIXELS
    );
    
    // Benchmark 2: Basic AVX2
    printf("Kernel 3.2: Basic AVX2...\n");
    results[result_count++] = benchmark_kernel(
        "3.2 Basic AVX2",
        kernel_direction_basic_avx2,
        g_x, g_y, direction, reference, NUM_PIXELS
    );
    
    // Benchmark 3: Integer Optimized
    printf("Kernel 3.3: Integer Optimized...\n");
    results[result_count++] = benchmark_kernel(
        "3.3 Integer Opt",
        kernel_direction_integer_opt,
        g_x, g_y, direction, reference, NUM_PIXELS
    );
    
    // Benchmark 4: Reduction Tree
    printf("Kernel 3.4: Reduction Tree Optimized...\n");
    results[result_count++] = benchmark_kernel(
        "3.4 Reduction Tree",
        kernel_direction_reduction_tree,
        g_x, g_y, direction, reference, NUM_PIXELS
    );
    
    printf("\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("BENCHMARK RESULTS\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n\n");
    
    printf("%-20s | %12s | %12s | %10s | %8s\n",
           "Kernel", "Cycles", "FLOPS/cycle", "Correct", "Speedup");
    printf("-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "-" "\n");
    
    for (int i = 0; i < result_count; i++) {
        double speedup = (i == 0) ? 1.0 : results[0].cycles / results[i].cycles;
        printf("%-20s | %12.0f | %12.2f | %8s | %7.2f×\n",
               results[i].name,
               results[i].cycles,
               results[i].flops_per_cycle,
               results[i].correctness ? "PASS" : "FAIL",
               speedup);
    }
    
    printf("\n" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n");
    printf("DETAILED ANALYSIS\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "\n\n");
    
    for (int i = 0; i < result_count; i++) {
        printf("%s:\n", results[i].name);
        printf("  Cycles per iteration: %.0f\n", results[i].cycles);
        printf("  FLOPS per cycle: %.2f\n", results[i].flops_per_cycle);
        printf("  Throughput (MP/s): %.1f\n", results[i].throughput_mpixels);
        printf("  Correctness: %s\n", results[i].correctness ? "PASS" : "FAIL");
        printf("\n");
    }
    
    printf("Theoretical Peak Performance (Broadwell Base 2.4 GHz):\n");
    printf("  Actual FLOPs per pixel: 1 (only g_x * g_y multiplication)\n");
    printf("  Best achievable FLOPS/cycle: ~40 (with perfect pipeline)\n");
    printf("  Bottleneck: Multiply instruction throughput (3-4 cycle latency)\n\n");
    
    // Cleanup
    free(g_x);
    free(g_y);
    free(direction);
    free(reference);
    
    return 0;
}
