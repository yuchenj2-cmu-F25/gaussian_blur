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

    double flops_canny_sobel_4x80 = 0.0;
    double flops_reference = 0.0;

    // Gradient maps for Canny/Sobel
    static float grad_x[HEIGHT][WIDTH];
    static float grad_y[HEIGHT][WIDTH];



    // Benchmark Canny/Sobel 4x80 (Gaussian blur + Sobel X + Sobel Y)
    if (!csv_mode) {
        t0 = rdtsc();
        canny_sobel_4x80(input, grad_x, grad_y);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Canny/Sobel 4x80 took %.0f cycles\n", cycles);
        printf(" %lf\n", (32.0*HEIGHT*WIDTH)/((double)(cycles/(1.0))));
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        canny_sobel_4x80(input, grad_x, grad_y);
        t1 = rdtsc();
        sums += (double)(t1 - t0);
    }
    flops_canny_sobel_4x80 = (32.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS)));
    if (!csv_mode) {
        printf("Canny/Sobel 4x80 average took %.0f cycles\n", sums/(1.0*RUNS));
        printf(" %lf\n", flops_canny_sobel_4x80);
    }
    sums = .0f;


    // Benchmark reference (for correctness and as scalar baseline)
    gaussian_blur_reference(input, reference);

    if (!csv_mode) {
        t0 = rdtsc();
        gaussian_blur_reference(input, reference);
        t1 = rdtsc();
        cycles = (double)(t1 - t0);
        printf("Gaussian blur reference took %.0f cycles\n", cycles);
    }

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        gaussian_blur_reference(input, reference);
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
        printf("%.6f,%.6f\n",
               flops_reference, flops_canny_sobel_4x80);
    }

    return 0;
}
