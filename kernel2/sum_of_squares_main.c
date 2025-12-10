#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "config.h"
#include "sum_of_squares.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    // Check for CSV output mode
    int csv_mode = 0;
    if (argc > 1 && strcmp(argv[1], "--csv") == 0) {
        csv_mode = 1;
    }

    int size = WIDTH * HEIGHT;
    static float gx[WIDTH * HEIGHT];
    static float gy[WIDTH * HEIGHT];
    static float mag2[WIDTH * HEIGHT];
    static float mag2_ref[WIDTH * HEIGHT];
    static float mag2_omp[WIDTH * HEIGHT];

    srand((unsigned)time(NULL));
    fill_random_array(gx, size);
    fill_random_array(gy, size);

    // Measure implementations
    unsigned long long t0, t1;
    double cycles;
    double sums;

    // Store FLOPS/cycle results for CSV output
    double flops_optimized = 0.0;
    double flops_omp = 0.0;
    double flops_reference = 0.0;

    // ========== BENCHMARK OPTIMIZED ==========

    // Benchmark optimized implementation
    if (!csv_mode) {
        t0 = rdtsc();
        kernel2_sum_of_squares(gx, gy, mag2, size);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Sum of squares optimized took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (2.0*size)/cycles);
    }

    sums = 0.0;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        kernel2_sum_of_squares(gx, gy, mag2, size);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_optimized = (2.0*size)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Sum of squares optimized average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_optimized);
    }

    // ========== BENCHMARK OMP ==========

    // Benchmark OpenMP implementation
    if (!csv_mode) {
        t0 = rdtsc();
        kernel2_sum_of_squares_omp(gx, gy, mag2_omp, size);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Sum of squares OMP took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (2.0*size)/cycles);
    }

    sums = 0.0;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        kernel2_sum_of_squares_omp(gx, gy, mag2_omp, size);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_omp = (2.0*size)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Sum of squares OMP average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_omp);
    }

    // ========== BENCHMARK REFERENCE ==========

    // Benchmark reference implementation
    if (!csv_mode) {
        t0 = rdtsc();
        kernel2_sum_of_squares_reference(gx, gy, mag2_ref, size);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Sum of squares reference took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (2.0*size)/cycles);
    }

    sums = 0.0;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        kernel2_sum_of_squares_reference(gx, gy, mag2_ref, size);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_reference = (2.0*size)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Sum of squares reference average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_reference);
    }

    // Check correctness
    if (!csv_mode) {
        float diff_optimized = compare_arrays(mag2_ref, mag2, size);
        float diff_omp = compare_arrays(mag2_ref, mag2_omp, size);
        printf("Average difference optimized from reference: %.6f\n", diff_optimized);
        printf("Average difference OMP from reference: %.6f\n", diff_omp);
    }

    // CSV output mode: print single line with all FLOPS/cycle values
    // Format: reference,optimized,omp
    if (csv_mode) {
        printf("%.6f,%.6f,%.6f\n",
               flops_reference, flops_optimized, flops_omp);
    }

    return 0;
}

