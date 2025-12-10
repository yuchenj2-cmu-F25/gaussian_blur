#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "blur.h"
#include "kernel2/sum_of_squares.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s input_image.bin output_prefix\n", argv[0]);
        fprintf(stderr, "  Expects input_image.bin to contain WIDTH*HEIGHT float32 pixels\n");
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_prefix = argv[2];

    static float input[HEIGHT][WIDTH];
    static float blurred[HEIGHT][WIDTH];
    static float grad_x[HEIGHT][WIDTH];
    static float grad_y[HEIGHT][WIDTH];
    static float mag2[HEIGHT * WIDTH];

    FILE *f = fopen(input_path, "rb");
    if (!f) {
        perror("fopen input");
        return 1;
    }

    size_t expected = (size_t)WIDTH * (size_t)HEIGHT;
    size_t read = fread(input, sizeof(float), expected, f);
    fclose(f);

    if (read != expected) {
        fprintf(stderr,
                "Error: expected %zu float32 values, but read %zu from %s\n",
                expected, read, input_path);
        return 1;
    }

    gaussian_blur_4x96(input, blurred);
    sobel_4x96(blurred, grad_x, grad_y);

    // Convert 2D arrays to 1D for kernel2
    static float gx_1d[HEIGHT * WIDTH];
    static float gy_1d[HEIGHT * WIDTH];
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            gx_1d[i * WIDTH + j] = grad_x[i][j];
            gy_1d[i * WIDTH + j] = grad_y[i][j];
        }
    }

    // Compute sum of squares (kernel2)
    kernel2_sum_of_squares(gx_1d, gy_1d, mag2, HEIGHT * WIDTH);

    char path_gx[1024];
    char path_gy[1024];
    char path_mag2[1024];

    snprintf(path_gx, sizeof(path_gx), "%s_grad_x.bin", output_prefix);
    snprintf(path_gy, sizeof(path_gy), "%s_grad_y.bin", output_prefix);
    snprintf(path_mag2, sizeof(path_mag2), "%s_mag2.bin", output_prefix);

    FILE *fgx = fopen(path_gx, "wb");
    if (!fgx) {
        perror("fopen grad_x");
        return 1;
    }
    size_t written_x = fwrite(grad_x, sizeof(float), expected, fgx);
    fclose(fgx);

    if (written_x != expected) {
        fprintf(stderr,
                "Error: wrote only %zu of %zu float32 values to %s\n",
                written_x, expected, path_gx);
        return 1;
    }

    FILE *fgy = fopen(path_gy, "wb");
    if (!fgy) {
        perror("fopen grad_y");
        return 1;
    }
    size_t written_y = fwrite(grad_y, sizeof(float), expected, fgy);
    fclose(fgy);

    if (written_y != expected) {
        fprintf(stderr,
                "Error: wrote only %zu of %zu float32 values to %s\n",
                written_y, expected, path_gy);
        return 1;
    }

    FILE *fmag2 = fopen(path_mag2, "wb");
    if (!fmag2) {
        perror("fopen mag2");
        return 1;
    }
    size_t written_mag2 = fwrite(mag2, sizeof(float), expected, fmag2);
    fclose(fmag2);

    if (written_mag2 != expected) {
        fprintf(stderr,
                "Error: wrote only %zu of %zu float32 values to %s\n",
                written_mag2, expected, path_mag2);
        return 1;
    }

    printf("Wrote gradients and magnitude squared to:\n");
    printf("  %s\n", path_gx);
    printf("  %s\n", path_gy);
    printf("  %s\n", path_mag2);

    return 0;
}

