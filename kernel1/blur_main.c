// Active implementations used in main.c
// This file contains only the implementations that are benchmarked

#include "blur.h"
#include "kernels.h"
#include "config.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void gaussian_blur_4x96(float input[HEIGHT][WIDTH],
                        float output[HEIGHT][WIDTH])
{
    const int src_stride = WIDTH;
    const int dst_stride = WIDTH;
    const int block_h = 4;
    const int block_w = 96;
    const int scalar_left = 8; // multiple of 8 for optimization

    static float tmp[HEIGHT][WIDTH];

    // 1D Gaussian kernel coefficients
    const float k0f = 0.25f;
    const float k1f = 0.5f;
    const float k2f = 0.25f;

    // ========== VERTICAL PASS ==========

    // Handle top row (row 0) with scalar code
    for (int c = 0; c < WIDTH; ++c) {
        const float *row = &input[0][0];
        tmp[0][c] = k0f * row[c] + k1f * row[c] + k2f * input[1][c];
    }

    // Handle left columns (0 to scalar_left-1) for rows 1 to HEIGHT-1 with scalar code
    for (int r = 1; r < HEIGHT; ++r) {
        int r_m1 = r - 1;
        int r_p1 = (r == HEIGHT - 1) ? (HEIGHT - 1) : (r + 1);
        const float *row_m1 = &input[r_m1][0];
        const float *row_0  = &input[r][0];
        const float *row_p1 = &input[r_p1][0];

        for (int c = 0; c < scalar_left; ++c) {
            tmp[r][c] = k0f * row_m1[c] + k1f * row_0[c] + k2f * row_p1[c];
        }
    }

    // Process bulk vertical with 4x96 kernel
    const int vert_start = 1;
    int vert_bottom_start;
    {
        int range = HEIGHT - vert_start;
        if (range <= 0) {
            vert_bottom_start = vert_start;
        } else {
            int remainder = range % block_h;
            vert_bottom_start = HEIGHT - remainder;
        }
    }

    #pragma omp parallel for
    for (int r = vert_start; r < vert_bottom_start; r += block_h) {
        int c;
        for (c = scalar_left; c + block_w <= WIDTH; c += block_w) {
            const float *src_block = &input[r][c];
            float *dst_block = &tmp[r][c];

            kernel_conv3_vert_4x96(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Handle remaining right columns for this row block with scalar
        for (int rr = r; rr < r + block_h && rr < HEIGHT; ++rr) {
            int rr_m1 = rr - 1;
            int rr_p1 = (rr == HEIGHT - 1) ? (HEIGHT - 1) : (rr + 1);
            const float *row_m1 = &input[rr_m1][0];
            const float *row_0  = &input[rr][0];
            const float *row_p1 = &input[rr_p1][0];

            for (int cc = c; cc < WIDTH; ++cc) {
                tmp[rr][cc] = k0f * row_m1[cc] + k1f * row_0[cc] + k2f * row_p1[cc];
            }
        }
    }

    // Handle remaining bottom rows with scalar
    for (int r = vert_bottom_start; r < HEIGHT; ++r) {
        int r_m1 = r - 1;
        int r_p1 = (r == HEIGHT - 1) ? (HEIGHT - 1) : (r + 1);
        const float *row_m1 = &input[r_m1][0];
        const float *row_0  = &input[r][0];
        const float *row_p1 = &input[r_p1][0];

        for (int c = scalar_left; c < WIDTH; ++c) {
            tmp[r][c] = k0f * row_m1[c] + k1f * row_0[c] + k2f * row_p1[c];
        }
    }

    // ========== HORIZONTAL PASS ==========

    // Handle all rows
    for (int r = 0; r < HEIGHT; ++r) {
        const float *row = &tmp[r][0];

        // Handle left edge
        output[r][0] = k0f * row[0] + k1f * row[0] + k2f * row[1];

        for (int c = 1; c < scalar_left; ++c) {
            output[r][c] = k0f * row[c-1] + k1f * row[c] + k2f * row[c+1];
        }
    }

    // Process bulk horizontal with 4x96 kernel
    const int horiz_start = 0;
    int horiz_bottom_start;
    {
        int range = HEIGHT - horiz_start;
        if (range <= 0) {
            horiz_bottom_start = horiz_start;
        } else {
            int remainder = range % block_h;
            horiz_bottom_start = HEIGHT - remainder;
        }
    }

    #pragma omp parallel for
    for (int r = horiz_start; r < horiz_bottom_start; r += block_h) {
        int c;
        for (c = scalar_left; c + block_w < WIDTH; c += block_w) {
            const float *src_block = &tmp[r][c];
            float *dst_block = &output[r][c];

            kernel_conv3_horiz_4x96(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Handle remaining right columns for this row block with scalar
        for (int rr = r; rr < r + block_h && rr < HEIGHT; ++rr) {
            const float *row = &tmp[rr][0];
            for (int cc = c; cc < WIDTH - 1; ++cc) {
                output[rr][cc] = k0f * row[cc-1] + k1f * row[cc] + k2f * row[cc+1];
            }
            // Handle right edge
            output[rr][WIDTH-1] = k0f * row[WIDTH-2] + k1f * row[WIDTH-1] + k2f * row[WIDTH-1];
        }
    }

    // Handle remaining bottom rows with scalar
    for (int r = horiz_bottom_start; r < HEIGHT; ++r) {
        const float *row = &tmp[r][0];
        for (int c = scalar_left; c < WIDTH - 1; ++c) {
            output[r][c] = k0f * row[c-1] + k1f * row[c] + k2f * row[c+1];
        }
        // Handle right edge
        output[r][WIDTH-1] = k0f * row[WIDTH-2] + k1f * row[WIDTH-1] + k2f * row[WIDTH-1];
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

// Sobel operator: computes X and Y gradients from blurred input
void sobel_4x96(float blurred[HEIGHT][WIDTH],
                float grad_x[HEIGHT][WIDTH],
                float grad_y[HEIGHT][WIDTH])
{
    const int src_stride = WIDTH;
    const int dst_stride = WIDTH;
    const int block_h = 4;
    const int block_w = 96;
    const int scalar_left = 8; // multiple of 8 for optimization

    static float tmp_x[HEIGHT][WIDTH];
    static float tmp_y[HEIGHT][WIDTH];

    // Sobel kernel coefficients
    const float k_smooth0 = 1.0f;  // smoothing kernel [1, 2, 1]
    const float k_smooth1 = 2.0f;
    const float k_deriv0 = -1.0f;  // derivative kernel [-1, 0, 1]
    const float k_deriv1 = 1.0f;

    // ========== SOBEL X (VERTICAL EDGES) ==========
    // Vertical smoothing pass [1, 2, 1]

    // Handle top row (row 0) with scalar code
    for (int c = 0; c < WIDTH; ++c) {
        const float *row = &blurred[0][0];
        tmp_x[0][c] = k_smooth0 * row[c] + k_smooth1 * row[c] + k_smooth0 * blurred[1][c];
    }

    // Handle left columns for rows 1+
    for (int r = 1; r < HEIGHT; ++r) {
        int r_m1 = r - 1;
        int r_p1 = (r == HEIGHT - 1) ? (HEIGHT - 1) : (r + 1);
        const float *row_m1 = &blurred[r_m1][0];
        const float *row_0  = &blurred[r][0];
        const float *row_p1 = &blurred[r_p1][0];

        for (int c = 0; c < scalar_left; ++c) {
            tmp_x[r][c] = k_smooth0 * row_m1[c] + k_smooth1 * row_0[c] + k_smooth0 * row_p1[c];
        }
    }

    // Process bulk vertical smoothing
    const int sobel_x_vert_start = 1;
    int sobel_x_vert_bottom_start;
    {
        int range = HEIGHT - sobel_x_vert_start;
        if (range <= 0) {
            sobel_x_vert_bottom_start = sobel_x_vert_start;
        } else {
            int remainder = range % block_h;
            sobel_x_vert_bottom_start = HEIGHT - remainder;
        }
    }

    #pragma omp parallel for
    for (int r = sobel_x_vert_start; r < sobel_x_vert_bottom_start; r += block_h) {
        int c;
        for (c = scalar_left; c + block_w <= WIDTH; c += block_w) {
            const float *src_block = &blurred[r][c];
            float *dst_block = &tmp_x[r][c];

            kernel_sobel_vert_smooth_4x96(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Handle remaining right columns with scalar
        for (int rr = r; rr < r + block_h && rr < HEIGHT; ++rr) {
            int rr_m1 = rr - 1;
            int rr_p1 = (rr == HEIGHT - 1) ? (HEIGHT - 1) : (rr + 1);
            const float *row_m1 = &blurred[rr_m1][0];
            const float *row_0  = &blurred[rr][0];
            const float *row_p1 = &blurred[rr_p1][0];

            for (int cc = c; cc < WIDTH; ++cc) {
                tmp_x[rr][cc] = k_smooth0 * row_m1[cc] + k_smooth1 * row_0[cc] + k_smooth0 * row_p1[cc];
            }
        }
    }

    // Handle remaining bottom rows with scalar
    for (int r = sobel_x_vert_bottom_start; r < HEIGHT; ++r) {
        int r_m1 = r - 1;
        int r_p1 = (r == HEIGHT - 1) ? (HEIGHT - 1) : (r + 1);
        const float *row_m1 = &blurred[r_m1][0];
        const float *row_0  = &blurred[r][0];
        const float *row_p1 = &blurred[r_p1][0];

        for (int c = scalar_left; c < WIDTH; ++c) {
            tmp_x[r][c] = k_smooth0 * row_m1[c] + k_smooth1 * row_0[c] + k_smooth0 * row_p1[c];
        }
    }

    // Horizontal derivative pass [-1, 0, 1]
    for (int r = 0; r < HEIGHT; ++r) {
        const float *row = &tmp_x[r][0];

        // Handle left edge (clamp to left boundary)
        grad_x[r][0] = k_deriv0 * row[0] + k_deriv1 * row[1];

        for (int c = 1; c < scalar_left; ++c) {
            grad_x[r][c] = k_deriv0 * row[c-1] + k_deriv1 * row[c+1];
        }
    }

    // Process bulk horizontal derivative
    const int sobel_x_horiz_start = 0;
    int sobel_x_horiz_bottom_start;
    {
        int range = HEIGHT - sobel_x_horiz_start;
        if (range <= 0) {
            sobel_x_horiz_bottom_start = sobel_x_horiz_start;
        } else {
            int remainder = range % block_h;
            sobel_x_horiz_bottom_start = HEIGHT - remainder;
        }
    }

    #pragma omp parallel for
    for (int r = sobel_x_horiz_start; r < sobel_x_horiz_bottom_start; r += block_h) {
        int c;
        for (c = scalar_left; c + block_w < WIDTH; c += block_w) {
            const float *src_block = &tmp_x[r][c];
            float *dst_block = &grad_x[r][c];

            kernel_sobel_horiz_deriv_4x96(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Handle remaining right columns with scalar
        for (int rr = r; rr < r + block_h && rr < HEIGHT; ++rr) {
            const float *row = &tmp_x[rr][0];
            for (int cc = c; cc < WIDTH - 1; ++cc) {
                grad_x[rr][cc] = k_deriv0 * row[cc-1] + k_deriv1 * row[cc+1];
            }
            // Handle right edge
            grad_x[rr][WIDTH-1] = k_deriv0 * row[WIDTH-2] + k_deriv1 * row[WIDTH-1];
        }
    }

    // Handle remaining bottom rows with scalar
    for (int r = sobel_x_horiz_bottom_start; r < HEIGHT; ++r) {
        const float *row = &tmp_x[r][0];
        for (int c = scalar_left; c < WIDTH - 1; ++c) {
            grad_x[r][c] = k_deriv0 * row[c-1] + k_deriv1 * row[c+1];
        }
        // Handle right edge
        grad_x[r][WIDTH-1] = k_deriv0 * row[WIDTH-2] + k_deriv1 * row[WIDTH-1];
    }

    // ========== SOBEL Y (HORIZONTAL EDGES) ==========
    // Vertical derivative pass [-1, 0, 1]

    // Handle top row (row 0) with scalar code (clamp to top boundary)
    for (int c = 0; c < WIDTH; ++c) {
        const float *row_0  = &blurred[0][0];
        const float *row_p1 = &blurred[1][0];
        tmp_y[0][c] = k_deriv0 * row_0[c] + k_deriv1 * row_p1[c];
    }

    // Handle left columns for rows 1+
    for (int r = 1; r < HEIGHT; ++r) {
        int r_m1 = r - 1;
        int r_p1 = (r == HEIGHT - 1) ? (HEIGHT - 1) : (r + 1);
        const float *row_m1 = &blurred[r_m1][0];
        const float *row_p1 = &blurred[r_p1][0];

        for (int c = 0; c < scalar_left; ++c) {
            tmp_y[r][c] = k_deriv0 * row_m1[c] + k_deriv1 * row_p1[c];
        }
    }

    // Process bulk vertical derivative
    const int sobel_y_vert_start = 1;
    int sobel_y_vert_bottom_start;
    {
        int range = HEIGHT - sobel_y_vert_start;
        if (range <= 0) {
            sobel_y_vert_bottom_start = sobel_y_vert_start;
        } else {
            int remainder = range % block_h;
            sobel_y_vert_bottom_start = HEIGHT - remainder;
        }
    }

    #pragma omp parallel for
    for (int r = sobel_y_vert_start; r < sobel_y_vert_bottom_start; r += block_h) {
        int c;
        for (c = scalar_left; c + block_w <= WIDTH; c += block_w) {
            const float *src_block = &blurred[r][c];
            float *dst_block = &tmp_y[r][c];

            kernel_sobel_vert_deriv_4x96(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Handle remaining right columns with scalar
        for (int rr = r; rr < r + block_h && rr < HEIGHT; ++rr) {
            int rr_m1 = rr - 1;
            int rr_p1 = (rr == HEIGHT - 1) ? (HEIGHT - 1) : (rr + 1);
            const float *row_m1 = &blurred[rr_m1][0];
            const float *row_p1 = &blurred[rr_p1][0];

            for (int cc = c; cc < WIDTH; ++cc) {
                tmp_y[rr][cc] = k_deriv0 * row_m1[cc] + k_deriv1 * row_p1[cc];
            }
        }
    }

    // Handle remaining bottom rows with scalar
    for (int r = sobel_y_vert_bottom_start; r < HEIGHT; ++r) {
        int r_m1 = r - 1;
        int r_p1 = (r == HEIGHT - 1) ? (HEIGHT - 1) : (r + 1);
        const float *row_m1 = &blurred[r_m1][0];
        const float *row_p1 = &blurred[r_p1][0];

        for (int c = scalar_left; c < WIDTH; ++c) {
            tmp_y[r][c] = k_deriv0 * row_m1[c] + k_deriv1 * row_p1[c];
        }
    }

    // Horizontal smoothing pass [1, 2, 1]
    for (int r = 0; r < HEIGHT; ++r) {
        const float *row = &tmp_y[r][0];

        // Handle left edge
        grad_y[r][0] = k_smooth0 * row[0] + k_smooth1 * row[0] + k_smooth0 * row[1];

        for (int c = 1; c < scalar_left; ++c) {
            grad_y[r][c] = k_smooth0 * row[c-1] + k_smooth1 * row[c] + k_smooth0 * row[c+1];
        }
    }

    // Process bulk horizontal smoothing
    const int sobel_y_horiz_start = 0;
    int sobel_y_horiz_bottom_start;
    {
        int range = HEIGHT - sobel_y_horiz_start;
        if (range <= 0) {
            sobel_y_horiz_bottom_start = sobel_y_horiz_start;
        } else {
            int remainder = range % block_h;
            sobel_y_horiz_bottom_start = HEIGHT - remainder;
        }
    }

    #pragma omp parallel for
    for (int r = sobel_y_horiz_start; r < sobel_y_horiz_bottom_start; r += block_h) {
        int c;
        for (c = scalar_left; c + block_w < WIDTH; c += block_w) {
            const float *src_block = &tmp_y[r][c];
            float *dst_block = &grad_y[r][c];

            kernel_sobel_horiz_smooth_4x96(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Handle remaining right columns with scalar
        for (int rr = r; rr < r + block_h && rr < HEIGHT; ++rr) {
            const float *row = &tmp_y[rr][0];
            for (int cc = c; cc < WIDTH - 1; ++cc) {
                grad_y[rr][cc] = k_smooth0 * row[cc-1] + k_smooth1 * row[cc] + k_smooth0 * row[cc+1];
            }
            // Handle right edge
            grad_y[rr][WIDTH-1] = k_smooth0 * row[WIDTH-2] + k_smooth1 * row[WIDTH-1] + k_smooth0 * row[WIDTH-1];
        }
    }

    // Handle remaining bottom rows with scalar
    for (int r = sobel_y_horiz_bottom_start; r < HEIGHT; ++r) {
        const float *row = &tmp_y[r][0];
        for (int c = scalar_left; c < WIDTH - 1; ++c) {
            grad_y[r][c] = k_smooth0 * row[c-1] + k_smooth1 * row[c] + k_smooth0 * row[c+1];
        }
        // Handle right edge
        grad_y[r][WIDTH-1] = k_smooth0 * row[WIDTH-2] + k_smooth1 * row[WIDTH-1] + k_smooth0 * row[WIDTH-1];
    }
}

// Scalar reference implementation for Sobel operator
// Disable auto-vectorization for true scalar baseline
#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC optimize ("no-tree-vectorize")
#endif

#ifdef __clang__
#pragma clang optimize off
#endif

void sobel_reference(float blurred[HEIGHT][WIDTH],
                     float grad_x[HEIGHT][WIDTH],
                     float grad_y[HEIGHT][WIDTH])
{
    // Sobel smoothing kernel: [1, 2, 1]
    const float s0 = 1.0f, s1 = 2.0f;

    // Sobel derivative kernel: [-1, 0, 1]
    const float d0 = -1.0f, d1 = 1.0f;

    static float tmp_x[HEIGHT][WIDTH];
    static float tmp_y[HEIGHT][WIDTH];

    // ========== SOBEL X (VERTICAL EDGES) ==========
    // Vertical smoothing pass [1, 2, 1]
    for (int r = 0; r < HEIGHT; ++r) {
        int rm1 = (r == 0) ? 0 : (r - 1);
        int rp1 = (r == HEIGHT - 1) ? (HEIGHT - 1) : (r + 1);
        volatile const float *row_m1 = &blurred[rm1][0];
        volatile const float *row_0  = &blurred[r][0];
        volatile const float *row_p1 = &blurred[rp1][0];
        volatile float *dst = &tmp_x[r][0];

        #ifdef __clang__
        #pragma clang loop vectorize(disable)
        #endif
        for (int c = 0; c < WIDTH; ++c) {
            dst[c] = s0 * row_m1[c] + s1 * row_0[c] + s0 * row_p1[c];
        }
    }

    // Horizontal derivative pass [-1, 0, 1]
    for (int r = 0; r < HEIGHT; ++r) {
        volatile const float *row = &tmp_x[r][0];
        volatile float *dst = &grad_x[r][0];

        // Handle left edge (clamp to left boundary)
        dst[0] = d0 * row[0] + d1 * row[1];

        #ifdef __clang__
        #pragma clang loop vectorize(disable)
        #endif
        for (int c = 1; c < WIDTH - 1; ++c) {
            dst[c] = d0 * row[c-1] + d1 * row[c+1];
        }

        // Handle right edge
        dst[WIDTH - 1] = d0 * row[WIDTH - 2] + d1 * row[WIDTH - 1];
    }

    // ========== SOBEL Y (HORIZONTAL EDGES) ==========
    // Vertical derivative pass [-1, 0, 1]
    for (int r = 0; r < HEIGHT; ++r) {
        int rm1 = (r == 0) ? 0 : (r - 1);
        int rp1 = (r == HEIGHT - 1) ? (HEIGHT - 1) : (r + 1);
        volatile const float *row_m1 = &blurred[rm1][0];
        volatile const float *row_p1 = &blurred[rp1][0];
        volatile float *dst = &tmp_y[r][0];

        #ifdef __clang__
        #pragma clang loop vectorize(disable)
        #endif
        for (int c = 0; c < WIDTH; ++c) {
            dst[c] = d0 * row_m1[c] + d1 * row_p1[c];
        }
    }

    // Horizontal smoothing pass [1, 2, 1]
    for (int r = 0; r < HEIGHT; ++r) {
        volatile const float *row = &tmp_y[r][0];
        volatile float *dst = &grad_y[r][0];

        // Handle left edge
        dst[0] = s0 * row[0] + s1 * row[0] + s0 * row[1];

        #ifdef __clang__
        #pragma clang loop vectorize(disable)
        #endif
        for (int c = 1; c < WIDTH - 1; ++c) {
            dst[c] = s0 * row[c-1] + s1 * row[c] + s0 * row[c+1];
        }

        // Handle right edge
        dst[WIDTH - 1] = s0 * row[WIDTH - 2] + s1 * row[WIDTH - 1] + s0 * row[WIDTH - 1];
    }
}

#ifdef __clang__
#pragma clang optimize on
#endif

#ifdef __GNUC__
#pragma GCC pop_options
#endif
