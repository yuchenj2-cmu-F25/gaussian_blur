#ifndef DIRECTION_KERNELS_H
#define DIRECTION_KERNELS_H

#include <stddef.h>

/* Direction classification kernels - all take 8 floats (1 AVX2 vector) at a time
 * Input: g_x and g_y gradient components
 * Output: direction classification (0, 1, 2, or 3)
 * 
 * Direction mapping:
 *   0: horizontal (|g_x| > |g_y|, no sign conditions)
 *   1: diagonal +45° (|g_x| > |g_y|, sign(g_x)*sign(g_y) > 0)
 *   2: vertical (|g_x| <= |g_y|, no sign conditions)
 *   3: diagonal -45° (|g_x| <= |g_y|, sign(g_x)*sign(g_y) < 0)
 */

/* Kernel 3.1: Naive scalar implementation - baseline reference */
void kernel_direction_naive(const float *restrict g_x, const float *restrict g_y,
                           float *restrict direction, int count);

/* Kernel 3.2: Movemask + Blend with 16-pixel kernel */
void kernel_direction_basic_avx2(const float *restrict g_x, const float *restrict g_y,
                                float *restrict direction, int count);

/* Kernel 3.3: Integer bit operations with 16-pixel kernel */
void kernel_direction_integer_opt(const float *restrict g_x, const float *restrict g_y,
                                 float *restrict direction, int count);

/* Kernel 3.4: Highly optimized with instruction pipelining and 16-pixel kernel */
void kernel_direction_reduction_tree(const float *restrict g_x, const float *restrict g_y,
                                    float *restrict direction, int count);

#endif
