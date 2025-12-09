#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "config.h"
#include "blur.h"
#include "utils.h"

int main(int argc, char *argv[]) {
    // Check for CSV output mode
    int csv_mode = 0;
    if (argc > 1 && strcmp(argv[1], "--csv") == 0) {
        csv_mode = 1;
    }
    static float input[HEIGHT][WIDTH];
    static float output[HEIGHT][WIDTH];
    static float reference[HEIGHT][WIDTH];

    srand((unsigned)time(NULL));
    fill_random_image(input);

    // Measure custom implementation
    unsigned long long t0, t1;
    double cycles;
    double sums;

    // Store FLOPS/cycle results for CSV output
    double flops_5x16 = 0.0;
    double flops_5x16_lowload = 0.0;
    double flops_10x8 = 0.0;
    double flops_4x80 = 0.0;
    double flops_24x80_v2 = 0.0;
    double flops_2d_24x80 = 0.0;
    double flops_separable_24x80 = 0.0;
    double flops_combined_24x80 = 0.0;
    double flops_reference = 0.0;



    // Benchmark 5x16
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_5x16(input, output);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Gaussian blur 5x16 took %.0f cycles\n", cycles);
        printf(" %lf\n", (12.0*HEIGHT*WIDTH)/((double)(cycles/(1.0))));
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_5x16(input, output);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_5x16 = (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Gaussian blur 5x16 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %lf\n", flops_5x16);
    }
    sums = .0f;



    // Benchmark 10x8
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_10x8(input, output);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Gaussian blur 10x8 took %.0f cycles\n", cycles);
        printf(" %lf\n", (12.0*HEIGHT*WIDTH)/((double)(cycles/(1.0))));
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_10x8(input, output);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_10x8 = (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Gaussian blur 10x8 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %lf\n", flops_10x8);
    }
    sums = .0f;




    // Benchmark 5x16_lowload
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_5x16_lowload(input, output);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Gaussian blur 5x16_lowload took %.0f cycles\n", cycles);
        printf(" %lf\n", (12.0*HEIGHT*WIDTH)/((double)(cycles/(1.0))));
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_5x16_lowload(input, output);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_5x16_lowload = (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Gaussian blur 5x16_lowload average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %lf\n", flops_5x16_lowload);
    }
    sums = .0f;








    // Benchmark 4x80
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_4x80(input, output);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Gaussian blur 4x80 took %.0f cycles\n", cycles);
        printf(" %lf\n", (12.0*HEIGHT*WIDTH)/((double)(cycles/(1.0))));
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_4x80(input, output);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_4x80 = (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Gaussian blur 4x80 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %lf\n", flops_4x80);
    }
    sums = .0f;


    // Benchmark 24x80_v2
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_24x80_v2(input, output);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Gaussian blur 24x80_v2 took %.0f cycles\n", cycles);
        printf(" %lf\n", (12.0*HEIGHT*WIDTH)/((double)(cycles/(1.0))));
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_24x80_v2(input, output);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_24x80_v2 = (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Gaussian blur 24x80_v2 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %lf\n", flops_24x80_v2);
    }
    sums = .0f;


    // Benchmark 2D 24x80
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_2d_24x80(input, output);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Gaussian blur 2d_24x80 took %.0f cycles\n", cycles);
        printf(" %lf\n", (12.0*HEIGHT*WIDTH)/((double)(cycles/(1.0))));
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_2d_24x80(input, output);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_2d_24x80 = (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Gaussian blur 2d_24x80 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %lf\n", flops_2d_24x80);
    }
    sums = .0f;


    // Benchmark separable 24x80
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_separable_24x80(input, output);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Gaussian blur separable_24x80 took %.0f cycles\n", cycles);
        printf(" %lf\n", (12.0*HEIGHT*WIDTH)/((double)(cycles/(1.0))));
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_separable_24x80(input, output);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_separable_24x80 = (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Gaussian blur separable_24x80 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %lf\n", flops_separable_24x80);
    }
    sums = .0f;


    // Benchmark combined 24x80
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_combined_24x80(input, output);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Gaussian blur combined_24x80 took %.0f cycles\n", cycles);
        printf(" %lf\n", (12.0*HEIGHT*WIDTH)/((double)(cycles/(1.0))));
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_combined_24x80(input, output);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_combined_24x80 = (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Gaussian blur combined_24x80 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %lf\n", flops_combined_24x80);
    }
    sums = .0f;


    // Benchmark reference (for correctness and as scalar baseline)
    gaussian_blur_reference(input, reference);

    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_faster(input, reference);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Gaussian blur reference took %.0f cycles\n", cycles);
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_faster(input, reference);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_reference = (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Gaussian blur reference average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %lf\n", flops_reference);
    }

    // Check correctness
    if (!csv_mode) {
        float diff = compare_images(reference, output);
        printf("Average difference from reference: %.6f\n", diff);
    }

    // CSV output mode: print single line with all FLOPS/cycle values
    if (csv_mode) {
        printf("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               flops_reference, flops_5x16, flops_5x16_lowload, flops_10x8, flops_4x80, flops_24x80_v2, flops_2d_24x80, flops_separable_24x80, flops_combined_24x80);
    }

    return 0;
}
