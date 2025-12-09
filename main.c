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

    srand((unsigned)time(NULL));
    fill_random_image(input);

    // Measure custom implementation
    unsigned long long t0, t1;
    double cycles;
    double sums;

    // Store FLOPS/cycle results for CSV output
    double flops_gaussian_4x80 = 0.0;
    double flops_sobel_4x80 = 0.0;
    double flops_pipeline_4x80 = 0.0;
    double flops_gaussian_ref = 0.0;
    double flops_sobel_ref = 0.0;
    double flops_pipeline_ref = 0.0;

    // Gradient maps for Sobel
    static float grad_x[HEIGHT][WIDTH];
    static float grad_y[HEIGHT][WIDTH];
    static float grad_x_ref[HEIGHT][WIDTH];
    static float grad_y_ref[HEIGHT][WIDTH];

    // Blurred images
    static float blurred[HEIGHT][WIDTH];
    static float blurred_ref[HEIGHT][WIDTH];


    // ========== BENCHMARK OPTIMIZED 4x80 ==========

    // Benchmark Gaussian blur 4x80
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_4x80(input, blurred);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Gaussian blur 4x80 took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (12.0*HEIGHT*WIDTH)/cycles);
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_4x80(input, blurred);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_gaussian_4x80 = (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Gaussian blur 4x80 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_gaussian_4x80);
    }

    // Benchmark Sobel 4x80 (operates on blurred image)
    if (!csv_mode) {
        t0 = rdtsc();
        sobel_4x80(blurred, grad_x, grad_y);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Sobel 4x80 took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (20.0*HEIGHT*WIDTH)/cycles);
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        sobel_4x80(blurred, grad_x, grad_y);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_sobel_4x80 = (20.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Sobel 4x80 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_sobel_4x80);
    }

    // Benchmark full pipeline 4x80
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_4x80(input, blurred);
        sobel_4x80(blurred, grad_x, grad_y);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Canny/Sobel pipeline 4x80 took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (32.0*HEIGHT*WIDTH)/cycles);
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_4x80(input, blurred);
        sobel_4x80(blurred, grad_x, grad_y);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_pipeline_4x80 = (32.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Canny/Sobel pipeline 4x80 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_pipeline_4x80);
    }


    // ========== BENCHMARK REFERENCE ==========

    // Benchmark Gaussian blur reference
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_reference(input, blurred_ref);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Gaussian blur reference took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (12.0*HEIGHT*WIDTH)/cycles);
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_reference(input, blurred_ref);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_gaussian_ref = (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Gaussian blur reference average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_gaussian_ref);
    }

    // Benchmark Sobel reference (operates on blurred image)
    if (!csv_mode) {
        t0 = rdtsc();
        sobel_reference(blurred_ref, grad_x_ref, grad_y_ref);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Sobel reference took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (20.0*HEIGHT*WIDTH)/cycles);
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        sobel_reference(blurred_ref, grad_x_ref, grad_y_ref);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_sobel_ref = (20.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Sobel reference average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_sobel_ref);
    }

    // Benchmark full pipeline reference
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_reference(input, blurred_ref);
        sobel_reference(blurred_ref, grad_x_ref, grad_y_ref);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Canny/Sobel pipeline reference took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (32.0*HEIGHT*WIDTH)/cycles);
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_reference(input, blurred_ref);
        sobel_reference(blurred_ref, grad_x_ref, grad_y_ref);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_pipeline_ref = (32.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Canny/Sobel pipeline reference average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_pipeline_ref);
    }

    // Check correctness
    if (!csv_mode) {
        float diff_x = compare_images(grad_x_ref, grad_x);
        float diff_y = compare_images(grad_y_ref, grad_y);
        printf("Average difference grad_x from reference: %.6f\n", diff_x);
        printf("Average difference grad_y from reference: %.6f\n", diff_y);
    }

    // CSV output mode: print single line with all FLOPS/cycle values
    // Format: gaussian_ref,sobel_ref,pipeline_ref,gaussian_4x80,sobel_4x80,pipeline_4x80
    if (csv_mode) {
        printf("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               flops_gaussian_ref, flops_sobel_ref, flops_pipeline_ref,
               flops_gaussian_4x80, flops_sobel_4x80, flops_pipeline_4x80);
    }

    return 0;
}
