#include "blur.h"
#include "kernels.h"
#include "config.h"

void gaussian_blur_10x8(float input[HEIGHT][WIDTH],
                        float output[HEIGHT][WIDTH])
{
    const int src_stride = WIDTH;  // in floats
    const int dst_stride = WIDTH;  // in floats
    const int block_h    = 10;
    const int block_w    = 8;
    const int num_blocks = HEIGHT / block_h;
    float tmp[HEIGHT][WIDTH];

    // First block - use upper variant
    int row0 = 0 * block_h;
    for (int c = 0; c < WIDTH; c += block_w) {
        const float *src_block = &input[row0][c];
        float       *dst_block = &tmp[row0][c];

        kernel_conv3_vert_10x8_upper(
            src_block,
            src_stride,
            dst_block,
            dst_stride
        );
    }

    // Middle blocks - use regular variant
    for (int b = 1; b < num_blocks - 1; ++b) {
        row0 = b * block_h;

        for (int c = 0; c < WIDTH; c += block_w) {
            const float *src_block = &input[row0][c];
            float       *dst_block = &tmp[row0][c];

            kernel_conv3_vert_10x8(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }
    }

    // Last block - use lower variant
    row0 = (num_blocks - 1) * block_h;
    for (int c = 0; c < WIDTH; c += block_w) {
        const float *src_block = &input[row0][c];
        float       *dst_block = &tmp[row0][c];

        kernel_conv3_vert_10x8_lower(
            src_block,
            src_stride,
            dst_block,
            dst_stride
        );
    }

    // Horizontal pass using 5x16 kernels
    const int horiz_block_h = 5;
    const int horiz_block_w = 16;
    const int horiz_num_blocks = HEIGHT / horiz_block_h;

    for (int b = 0; b < horiz_num_blocks; ++b) {
        row0 = b * horiz_block_h;

        // Left edge
        for (int c = 0; c < horiz_block_w; c += horiz_block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16_left(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Middle
        for (int c = horiz_block_w; c < WIDTH - horiz_block_w; c += horiz_block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Right edge
        for (int c = WIDTH - horiz_block_w; c < WIDTH; c += horiz_block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16_right(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }
    }
}

void gaussian_blur_5x16_lowload(float input[HEIGHT][WIDTH],
                                      float output[HEIGHT][WIDTH])
{
    const int src_stride = WIDTH;  // in floats
    const int dst_stride = WIDTH;  // in floats
    const int block_h    = 5;
    const int block_w    = 16;
    const int num_blocks = HEIGHT / block_h;
    float tmp[HEIGHT][WIDTH];

    int row0 = 0 * block_h;
    for (int c = 0; c < WIDTH; c += block_w) {
        const float *src_block = &input[row0][c];
        float       *dst_block = &tmp[row0][c];

        kernel_conv3_vert_5x16_lowload_upper(
            src_block,
            src_stride,
            dst_block,
            dst_stride
        );
    }

    // Loop over row blocks of height 13
    for (int b = 1; b < num_blocks - 1; ++b) {
        row0 = b * block_h;  // first output row of this 13-row block

        // Now sweep across columns in chunks of 8
        for (int c = 0; c < WIDTH; c += block_w) {
            const float *src_block = &input[row0][c];
            float       *dst_block = &tmp[row0][c];

            kernel_conv3_vert_5x16_lowload(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }
    }

    row0 = (num_blocks - 1) * block_h;
    for (int c = 0; c < WIDTH; c += block_w) {
        const float *src_block = &input[row0][c];
        float       *dst_block = &tmp[row0][c];

        kernel_conv3_vert_5x16_lowload_lower(
            src_block,
            src_stride,
            dst_block,
            dst_stride
        );
    }

    // Loop over row blocks of height 13
    for (int b = 0; b < num_blocks; ++b) {
        row0 = b * block_h;  // first output row of this 13-row block

        // Now sweep across columns in chunks of 8
        for (int c = 0; c < block_w; c += block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16_left(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Now sweep across columns in chunks of 8
        for (int c = block_w; c < WIDTH - block_w; c += block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Now sweep across columns in chunks of 8
        for (int c = WIDTH - block_w; c < WIDTH; c += block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16_right(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

    }
}

void gaussian_blur_5x16(float input[HEIGHT][WIDTH],
                                      float output[HEIGHT][WIDTH])
{
    const int src_stride = WIDTH;  // in floats
    const int dst_stride = WIDTH;  // in floats
    const int block_h    = 5;
    const int block_w    = 16;
    const int num_blocks = HEIGHT / block_h;
    float tmp[HEIGHT][WIDTH];

    int row0 = 0 * block_h;
    for (int c = 0; c < WIDTH; c += block_w) {
        const float *src_block = &input[row0][c];
        float       *dst_block = &tmp[row0][c];

        kernel_conv3_vert_5x16_upper(
            src_block,
            src_stride,
            dst_block,
            dst_stride
        );
    }

    // Loop over row blocks of height 13
    for (int b = 1; b < num_blocks - 1; ++b) {
        row0 = b * block_h;  // first output row of this 13-row block

        // Now sweep across columns in chunks of 8
        for (int c = 0; c < WIDTH; c += block_w) {
            const float *src_block = &input[row0][c];
            float       *dst_block = &tmp[row0][c];

            kernel_conv3_vert_5x16(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }
    }

    row0 = (num_blocks - 1) * block_h;
    for (int c = 0; c < WIDTH; c += block_w) {
        const float *src_block = &input[row0][c];
        float       *dst_block = &tmp[row0][c];

        kernel_conv3_vert_5x16_lower(
            src_block,
            src_stride,
            dst_block,
            dst_stride
        );
    }

    // Loop over row blocks of height 13
    for (int b = 0; b < num_blocks; ++b) {
        row0 = b * block_h;  // first output row of this 13-row block

        // Now sweep across columns in chunks of 8
        for (int c = 0; c < block_w; c += block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16_left(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Now sweep across columns in chunks of 8
        for (int c = block_w; c < WIDTH - block_w; c += block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Now sweep across columns in chunks of 8
        for (int c = WIDTH - block_w; c < WIDTH; c += block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16_right(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

    }
}

void gaussian_blur_4x80(float input[HEIGHT][WIDTH],
                        float output[HEIGHT][WIDTH])
{
    const int src_stride = WIDTH;  // in floats
    const int dst_stride = WIDTH;  // in floats
    const int block_h    = 4;      // process 4 rows at a time
    const int block_w    = 80;     // process 80 columns (full width for WIDTH=512)
    const int num_blocks = HEIGHT / block_h;  // 600 / 4 = 150 blocks
    float tmp[HEIGHT][WIDTH];

    // Vertical pass with 4x80 unrolled kernels
    // First block - use upper variant
    int row0 = 0 * block_h;
    for (int c = 0; c < WIDTH; c += block_w) {
        const float *src_block = &input[row0][c];
        float       *dst_block = &tmp[row0][c];

        kernel_conv3_vert_4x80_upper(
            src_block,
            src_stride,
            dst_block,
            dst_stride
        );
    }

    // Middle blocks - use regular variant
    for (int b = 1; b < num_blocks - 1; ++b) {
        row0 = b * block_h;

        for (int c = 0; c < WIDTH; c += block_w) {
            const float *src_block = &input[row0][c];
            float       *dst_block = &tmp[row0][c];

            kernel_conv3_vert_4x80(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }
    }

    // Last block - use lower variant
    row0 = (num_blocks - 1) * block_h;
    for (int c = 0; c < WIDTH; c += block_w) {
        const float *src_block = &input[row0][c];
        float       *dst_block = &tmp[row0][c];

        kernel_conv3_vert_4x80_lower(
            src_block,
            src_stride,
            dst_block,
            dst_stride
        );
    }

    // Horizontal pass using existing 5x16 kernels (same as other implementations)
    const int horiz_block_h = 5;
    const int horiz_block_w = 16;
    const int horiz_num_blocks = HEIGHT / horiz_block_h;

    for (int b = 0; b < horiz_num_blocks; ++b) {
        row0 = b * horiz_block_h;

        // Left edge
        for (int c = 0; c < horiz_block_w; c += horiz_block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16_left(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Middle
        for (int c = horiz_block_w; c < WIDTH - horiz_block_w; c += horiz_block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Right edge
        for (int c = WIDTH - horiz_block_w; c < WIDTH; c += horiz_block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16_right(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }
    }
}

// Disable auto-vectorization for reference implementation
// This ensures we have a true scalar baseline for comparison
#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC optimize ("no-tree-vectorize")
#endif

#ifdef __clang__
#pragma clang optimize off
#endif

void gaussian_blur_reference(float input[HEIGHT][WIDTH], float output[HEIGHT][WIDTH]) {
    const float k0f = 0.25f, k1f = 0.5f, k2f = 0.25f;
    static float tmp[HEIGHT][WIDTH];

    for (int r = 0; r < HEIGHT; ++r) {
        volatile const float *row = &input[r][0];
        volatile float *dst = &tmp[r][0];

        // Handle left edge
        dst[0] = k0f * row[0] + k1f * row[0] + k2f * row[1];

        // Inner loop - volatile pointers prevent auto-vectorization
        #ifdef __clang__
        #pragma clang loop vectorize(disable)
        #endif
        for (int c = 1; c < WIDTH - 1; ++c) {
            dst[c] = k0f * row[c-1] + k1f * row[c] + k2f * row[c+1];
        }

        // Handle right edge
        dst[WIDTH - 1] = k0f * row[WIDTH - 2] + k1f * row[WIDTH - 1] + k2f * row[WIDTH - 1];
    }

    for (int r = 0; r < HEIGHT; ++r) {
        int rm1 = (r == 0) ? 0 : (r - 1);
        int rp1 = (r == HEIGHT - 1) ? (HEIGHT - 1) : (r + 1);
        volatile const float *row_m1 = &tmp[rm1][0];
        volatile const float *row_0  = &tmp[r][0];
        volatile const float *row_p1 = &tmp[rp1][0];
        volatile float *dst = &output[r][0];

        #ifdef __clang__
        #pragma clang loop vectorize(disable)
        #endif
        for (int c = 0; c < WIDTH; ++c) {
            dst[c] = k0f * row_m1[c] + k1f * row_0[c] + k2f * row_p1[c];
        }
    }
}

#ifdef __clang__
#pragma clang optimize on
#endif

#ifdef __GNUC__
#pragma GCC pop_options
#endif

void gaussian_blur(float input[HEIGHT][WIDTH], float output[HEIGHT][WIDTH]) {
    const float kernel[3] = {0.25f, 0.5f, 0.25f}; // normalized 1D kernel

    static float temp[HEIGHT][WIDTH];

    // horiztal pass
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            float acc = 0.0f;
            for (int k = -1; k <= 1; k++) {
                int jj = j + k;
                if (jj < 0) jj = 0;
                if (jj >= WIDTH) jj = WIDTH - 1;
                acc += input[i][jj] * kernel[k + 1];
            }
            temp[i][j] = acc;
        }
    }

    // Vertical pass
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            float acc = 0.0f;
            for (int k = -1; k <= 1; k++) {
                int ii = i + k;
                if (ii < 0) ii = 0;
                if (ii >= HEIGHT) ii = HEIGHT - 1;
                acc += temp[ii][j] * kernel[k + 1];
            }
            output[i][j] = acc;
        }
    }
}
