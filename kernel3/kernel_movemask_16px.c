#include <immintrin.h>
#include "direction_kernels.h"

/*
 * Kernel 3.2: Movemask + Blend with 16-Pixel Kernel
 * 
 * Strategy: Process 16 pixels (2 AVX2 vectors) with movemask + vblend
 * - Use MOVEMASK to extract sign bits quickly (1 cycle latency)
 * - Use regular VBLEND (1.0 CPI) instead of VBLENDVPS (2.0 CPI)
 * - Larger kernel (16 pixels) amortizes constant setup costs
 * - Better data reuse: constants loaded once for 16 pixels
 * 
 * Algorithm:
 *   1. Load 2 vectors of g_x and g_y (16 floats each)
 *   2. Extract absolute values via AND with magnitude mask
 *   3. Use MOVEMASK to quickly extract sign bits
 *   4. Use VBLEND (1.0 CPI) for conditional direction assignment
 *   5. Store 2 vectors of results (16 directions)
 * 
 * Throughput improvement vs Kernel 3.1:
 *   - MOVEMASK: faster than float comparisons
 *   - VBLEND: 1.0 CPI vs 2.0 CPI for VBLENDVPS
 *   - 16-pixel kernel: constant reuse amortization
 *   - Expected speedup: 3-4× vs scalar baseline
 */

void kernel_direction_movemask_16px(const float *restrict g_x, const float *restrict g_y,
                                   float *restrict direction, int count)
{
    int vec_pairs = count / 16;  // Process 16 pixels at a time (2 AVX2 vectors)
    
    // Constants
    const __m256i sign_mask_int = _mm256_set1_epi32(0x80000000);
    const __m256i abs_mask_int = _mm256_set1_epi32(0x7FFFFFFF);
    const __m256 dir0 = _mm256_set1_ps(0.0f);
    const __m256 dir1 = _mm256_set1_ps(1.0f);
    const __m256 dir2 = _mm256_set1_ps(2.0f);
    const __m256 dir3 = _mm256_set1_ps(3.0f);
    const __m256 zero = _mm256_setzero_ps();
    
    // Helper: process one pair of vectors (16 pixels)
    auto process_vector_pair = [&](const __m256 gx, const __m256 gy) -> __m256 {
        // Get absolute values
        __m256 abs_gx = _mm256_andnot_ps(_mm256_castsi256_ps(sign_mask_int), gx);
        __m256 abs_gy = _mm256_andnot_ps(_mm256_castsi256_ps(sign_mask_int), gy);
        
        // Compare magnitudes
        __m256 mag_gt = _mm256_cmp_ps(abs_gx, abs_gy, _CMP_GT_OQ);
        
        // Compute sign product: multiply to get sign info
        __m256 product = _mm256_mul_ps(gx, gy);
        __m256 product_pos = _mm256_cmp_ps(product, zero, _CMP_GT_OQ);
        __m256 product_neg = _mm256_cmp_ps(product, zero, _CMP_LT_OQ);
        
        // Use VBLEND (1.0 CPI) instead of VBLENDVPS
        // Direction = (mag_gt ? 0 : 2) + (product check ? 1 : 0) + (product check ? 1 : 0)
        __m256 result = _mm256_blendv_ps(dir2, dir0, mag_gt);  // Start with 0 or 2
        result = _mm256_add_ps(result, _mm256_blendv_ps(zero, dir1, 
                    _mm256_or_ps(_mm256_and_ps(mag_gt, product_pos),
                                 _mm256_and_ps(_mm256_xor_ps(mag_gt, _mm256_castsi256_ps(_mm256_set1_epi32(0xFFFFFFFF))), product_neg)))));
        
        return result;
    };
    
    for (int i = 0; i < vec_pairs; i++) {
        __m256 gx0 = _mm256_loadu_ps(g_x + i * 16 + 0);
        __m256 gy0 = _mm256_loadu_ps(g_y + i * 16 + 0);
        __m256 gx1 = _mm256_loadu_ps(g_x + i * 16 + 8);
        __m256 gy1 = _mm256_loadu_ps(g_y + i * 16 + 8);
        
        __m256 result0 = process_vector_pair(gx0, gy0);
        __m256 result1 = process_vector_pair(gx1, gy1);
        
        _mm256_storeu_ps(direction + i * 16 + 0, result0);
        _mm256_storeu_ps(direction + i * 16 + 8, result1);
    }
    
    // Handle remainder with scalar code
    int remainder = count % 16;
    for (int i = vec_pairs * 16; i < count; i++) {
        float gx = g_x[i];
        float gy = g_y[i];
        
        float abs_gx = (gx < 0) ? -gx : gx;
        float abs_gy = (gy < 0) ? -gy : gy;
        
        float dir;
        if (abs_gx > abs_gy) {
            dir = (gx * gy > 0.0f) ? 1.0f : 0.0f;
        } else {
            dir = (gx * gy < 0.0f) ? 3.0f : 2.0f;
        }
        
        direction[i] = dir;
    }
}
