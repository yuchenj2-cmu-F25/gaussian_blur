#include <immintrin.h>
#include <math.h>
#include "direction_kernels.h"

/*
 * Kernel 3.1: Naive Scalar Implementation
 * 
 * Strategy: Simple element-wise scalar operations
 * - Process one pixel at a time
 * - Use floating-point comparisons and branches
 * - No SIMD vectorization
 * - Baseline reference for performance comparison
 */

void kernel_direction_naive(const float *restrict g_x, const float *restrict g_y,
                           float *restrict direction, int count)
{
    for (int i = 0; i < count; i++) {
        float gx = g_x[i];
        float gy = g_y[i];
        
        float abs_gx = fabsf(gx);
        float abs_gy = fabsf(gy);
        float product = gx * gy;
        
        float dir;
        if (abs_gx > abs_gy) {
            // Horizontal dominant
            dir = (product > 0.0f) ? 1.0f : 0.0f;
        } else {
            // Vertical dominant
            dir = (product < 0.0f) ? 3.0f : 2.0f;
        }
        
        direction[i] = dir;
    }
}

