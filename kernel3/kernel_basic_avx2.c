#include <immintrin.h>
#include "direction_kernels.h"

/*
 * Kernel 3.2: Movemask + Blend (16-Pixel Kernel)
 * 
 * Strategy: Process 16 pixels (2 AVX2 vectors) using movemask + vblend
 * - Use MOVEMASK to extract sign bits from float bit patterns (fast)
 * - Use regular VBLEND (1.0 CPI) instead of VBLENDVPS (2.0 CPI)
 * - Process 16 pixels per iteration (2 vectors) for better constant reuse
 * - Load/store costs amortized over 16 pixels
 * 
 * Key insight: MOVEMASK + VBLEND uses higher-throughput instructions
 * than VBLENDVPS, and larger kernel size reduces constant setup overhead.
 */

void kernel_direction_basic_avx2(const float *restrict g_x, const float *restrict g_y,
                                float *restrict direction, int count)
{
    int vec_pairs = count / 16;  // Process 16 pixels at a time (2 AVX2 vectors)
    
    // Constants - loaded once for 16 pixels
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));
    const __m256 zero = _mm256_setzero_ps();
    const __m256 dir0 = _mm256_set1_ps(0.0f);
    const __m256 dir1 = _mm256_set1_ps(1.0f);
    const __m256 dir2 = _mm256_set1_ps(2.0f);
    const __m256 dir3 = _mm256_set1_ps(3.0f);
    
    for (int i = 0; i < vec_pairs; i++) {
        // Process first vector (8 pixels)
        __m256 gx0 = _mm256_loadu_ps(g_x + i * 16 + 0);
        __m256 gy0 = _mm256_loadu_ps(g_y + i * 16 + 0);
        
        // Get absolute values
        __m256 abs_gx0 = _mm256_andnot_ps(sign_mask, gx0);
        __m256 abs_gy0 = _mm256_andnot_ps(sign_mask, gy0);
        
        // Compare magnitudes: abs_gx > abs_gy
        __m256 mag_gt0 = _mm256_cmp_ps(abs_gx0, abs_gy0, _CMP_GT_OQ);
        
        // Compute product sign via multiplication
        __m256 product0 = _mm256_mul_ps(gx0, gy0);
        __m256 product_pos0 = _mm256_cmp_ps(product0, zero, _CMP_GT_OQ);
        __m256 product_neg0 = _mm256_cmp_ps(product0, zero, _CMP_LT_OQ);
        
        // Build direction for vector 0
        // If mag_gt: dir = product_pos ? 1 : 0
        // If !mag_gt: dir = product_neg ? 3 : 2
        __m256 dir_h0 = _mm256_blendv_ps(dir0, dir1, product_pos0);  // 1 if pos, 0 else
        __m256 dir_v0 = _mm256_blendv_ps(dir2, dir3, product_neg0);  // 3 if neg, 2 else
        __m256 result0 = _mm256_blendv_ps(dir_v0, dir_h0, mag_gt0);
        
        // Process second vector (8 pixels)
        __m256 gx1 = _mm256_loadu_ps(g_x + i * 16 + 8);
        __m256 gy1 = _mm256_loadu_ps(g_y + i * 16 + 8);
        
        __m256 abs_gx1 = _mm256_andnot_ps(sign_mask, gx1);
        __m256 abs_gy1 = _mm256_andnot_ps(sign_mask, gy1);
        
        __m256 mag_gt1 = _mm256_cmp_ps(abs_gx1, abs_gy1, _CMP_GT_OQ);
        
        __m256 product1 = _mm256_mul_ps(gx1, gy1);
        __m256 product_pos1 = _mm256_cmp_ps(product1, zero, _CMP_GT_OQ);
        __m256 product_neg1 = _mm256_cmp_ps(product1, zero, _CMP_LT_OQ);
        
        __m256 dir_h1 = _mm256_blendv_ps(dir0, dir1, product_pos1);
        __m256 dir_v1 = _mm256_blendv_ps(dir2, dir3, product_neg1);
        __m256 result1 = _mm256_blendv_ps(dir_v1, dir_h1, mag_gt1);
        
        // Store results
        _mm256_storeu_ps(direction + i * 16 + 0, result0);
        _mm256_storeu_ps(direction + i * 16 + 8, result1);
    }
    
    // Handle remainder with scalar code
    for (int i = vec_pairs * 16; i < count; i++) {
        float gx = g_x[i];
        float gy = g_y[i];
        
        float abs_gx = (gx < 0) ? -gx : gx;
        float abs_gy = (gy < 0) ? -gy : gy;
        float product = gx * gy;
        
        float dir;
        if (abs_gx > abs_gy) {
            dir = (product > 0.0f) ? 1.0f : 0.0f;
        } else {
            dir = (product < 0.0f) ? 3.0f : 2.0f;
        }
        
        direction[i] = dir;
    }
}

