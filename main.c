#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "config.h"
#include "blur.h"
#include "utils.h"
#include "kernel2/sum_of_squares.h"

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
    double flops_gaussian_4x96 = 0.0;
    double flops_sobel_4x96 = 0.0;
    double flops_sum_squares = 0.0;
    double flops_pipeline_4x96 = 0.0;
    double flops_gaussian_ref = 0.0;
    double flops_sobel_ref = 0.0;
    double flops_sum_squares_ref = 0.0;
    double flops_pipeline_ref = 0.0;

    // Gradient maps for Sobel
    static float grad_x[HEIGHT][WIDTH];
    static float grad_y[HEIGHT][WIDTH];
    static float grad_x_ref[HEIGHT][WIDTH];
    static float grad_y_ref[HEIGHT][WIDTH];

    // Magnitude squared maps for Kernel 2
    static float mag2[HEIGHT * WIDTH];
    static float mag2_ref[HEIGHT * WIDTH];

    // Blurred images
    static float blurred[HEIGHT][WIDTH];
    static float blurred_ref[HEIGHT][WIDTH];


    // ========== BENCHMARK OPTIMIZED 4x96 ==========

    // Benchmark Gaussian blur 4x96
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_4x96(input, blurred);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Gaussian blur 4x96 took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (12.0*HEIGHT*WIDTH)/cycles);
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_4x96(input, blurred);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_gaussian_4x96 = (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Gaussian blur 4x96 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_gaussian_4x96);
    }

    // Benchmark Sobel 4x96 (operates on blurred image)
    if (!csv_mode) {
        t0 = rdtsc();
        sobel_4x96(blurred, grad_x, grad_y);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Sobel 4x96 took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (20.0*HEIGHT*WIDTH)/cycles);
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        sobel_4x96(blurred, grad_x, grad_y);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_sobel_4x96 = (20.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Sobel 4x96 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_sobel_4x96);
    }

    // Benchmark Sum of Squares (Kernel 2)
    // Convert 2D arrays to 1D for kernel2
    static float gx_1d[HEIGHT * WIDTH];
    static float gy_1d[HEIGHT * WIDTH];
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            gx_1d[r * WIDTH + c] = grad_x[r][c];
            gy_1d[r * WIDTH + c] = grad_y[r][c];
        }
    }

    if (!csv_mode) {
        t0 = rdtsc();
        kernel2_sum_of_squares(gx_1d, gy_1d, mag2, HEIGHT * WIDTH);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Sum of squares took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (2.0*HEIGHT*WIDTH)/cycles);
    }

    sums = 0.0;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        kernel2_sum_of_squares(gx_1d, gy_1d, mag2, HEIGHT * WIDTH);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_sum_squares = (2.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Sum of squares average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_sum_squares);
    }

    // Benchmark full pipeline 4x96 (including kernel2)
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_4x96(input, blurred);
        sobel_4x96(blurred, grad_x, grad_y);
        // Convert to 1D and compute sum of squares
        for (int r = 0; r < HEIGHT; r++) {
            for (int c = 0; c < WIDTH; c++) {
                gx_1d[r * WIDTH + c] = grad_x[r][c];
                gy_1d[r * WIDTH + c] = grad_y[r][c];
            }
        }
        kernel2_sum_of_squares(gx_1d, gy_1d, mag2, HEIGHT * WIDTH);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Full pipeline (blur+sobel+sum_squares) 4x96 took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (34.0*HEIGHT*WIDTH)/cycles);
    }

    sums = 0.0;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_4x96(input, blurred);
        sobel_4x96(blurred, grad_x, grad_y);
        // Convert to 1D and compute sum of squares
        for (int r = 0; r < HEIGHT; r++) {
            for (int c = 0; c < WIDTH; c++) {
                gx_1d[r * WIDTH + c] = grad_x[r][c];
                gy_1d[r * WIDTH + c] = grad_y[r][c];
            }
        }
        kernel2_sum_of_squares(gx_1d, gy_1d, mag2, HEIGHT * WIDTH);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_pipeline_4x96 = (34.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Full pipeline (blur+sobel+sum_squares) 4x96 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_pipeline_4x96);
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

    // Benchmark Sum of Squares reference (Kernel 2)
    // Convert 2D arrays to 1D for kernel2
    static float gx_ref_1d[HEIGHT * WIDTH];
    static float gy_ref_1d[HEIGHT * WIDTH];
    for (int r = 0; r < HEIGHT; r++) {
        for (int c = 0; c < WIDTH; c++) {
            gx_ref_1d[r * WIDTH + c] = grad_x_ref[r][c];
            gy_ref_1d[r * WIDTH + c] = grad_y_ref[r][c];
        }
    }

    if (!csv_mode) {
        t0 = rdtsc();
        kernel2_sum_of_squares_reference(gx_ref_1d, gy_ref_1d, mag2_ref, HEIGHT * WIDTH);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Sum of squares reference took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (2.0*HEIGHT*WIDTH)/cycles);
    }

    sums = 0.0;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        kernel2_sum_of_squares_reference(gx_ref_1d, gy_ref_1d, mag2_ref, HEIGHT * WIDTH);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_sum_squares_ref = (2.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Sum of squares reference average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_sum_squares_ref);
    }

    // Benchmark full pipeline reference (including kernel2)
    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_reference(input, blurred_ref);
        sobel_reference(blurred_ref, grad_x_ref, grad_y_ref);
        // Convert to 1D and compute sum of squares
        for (int r = 0; r < HEIGHT; r++) {
            for (int c = 0; c < WIDTH; c++) {
                gx_ref_1d[r * WIDTH + c] = grad_x_ref[r][c];
                gy_ref_1d[r * WIDTH + c] = grad_y_ref[r][c];
            }
        }
        kernel2_sum_of_squares_reference(gx_ref_1d, gy_ref_1d, mag2_ref, HEIGHT * WIDTH);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Full pipeline (blur+sobel+sum_squares) reference took %.0f cycles\n", cycles);
        printf(" %.6lf FLOPS/cycle\n", (34.0*HEIGHT*WIDTH)/cycles);
    }

    sums = 0.0;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_reference(input, blurred_ref);
        sobel_reference(blurred_ref, grad_x_ref, grad_y_ref);
        // Convert to 1D and compute sum of squares
        for (int r = 0; r < HEIGHT; r++) {
            for (int c = 0; c < WIDTH; c++) {
                gx_ref_1d[r * WIDTH + c] = grad_x_ref[r][c];
                gy_ref_1d[r * WIDTH + c] = grad_y_ref[r][c];
            }
        }
        kernel2_sum_of_squares_reference(gx_ref_1d, gy_ref_1d, mag2_ref, HEIGHT * WIDTH);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_pipeline_ref = (34.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Full pipeline (blur+sobel+sum_squares) reference average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %.6lf FLOPS/cycle\n", flops_pipeline_ref);
    }

    // Check correctness
    if (!csv_mode) {
        float diff_x = compare_images(grad_x_ref, grad_x);
        float diff_y = compare_images(grad_y_ref, grad_y);
        float diff_mag2 = compare_arrays(mag2_ref, mag2, HEIGHT * WIDTH);
        printf("Average difference grad_x from reference: %.6f\n", diff_x);
        printf("Average difference grad_y from reference: %.6f\n", diff_y);
        printf("Average difference mag2 from reference: %.6f\n", diff_mag2);
    }

    // CSV output mode: print single line with all FLOPS/cycle values
    // Format: gaussian_ref,sobel_ref,sum_squares_ref,pipeline_ref,gaussian_4x96,sobel_4x96,sum_squares,pipeline_4x96
    if (csv_mode) {
        printf("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
               flops_gaussian_ref, flops_sobel_ref, flops_sum_squares_ref, flops_pipeline_ref,
               flops_gaussian_4x96, flops_sobel_4x96, flops_sum_squares, flops_pipeline_4x96);
    }

    return 0;
}
