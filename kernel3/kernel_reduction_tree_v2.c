#include <immintrin.h>
#include "direction_kernels.h"

/*
 * Kernel 3.5: Professor-Optimized with Sign-Bit Checking (16-Pixel Kernel)
 * 
 * Based on professor feedback:
 * 1. Use AND_SI256 instead of AND_PS (higher throughput on Broadwell)
 * 2. Extract sign bits directly instead of computing multiply
 * 3. Better instruction pipelining with reduction tree approach
 * 4. Eliminate unnecessary floating-point operations
 * 
 * Key optimizations:
 *   - All operations on integer units (faster AND_SI256)
 *   - Sign bit extraction via XOR instead of multiply
 *   - Reduced dependency chain for blends
 *   - 16-pixel kernel for constant reuse
 * 
 * Expected improvement: ~2-3× vs Kernel 3.4 (fewer dependencies)
 */

void kernel_direction_reduction_tree(const float *restrict g_x, const float *restrict g_y,
                                    float *restrict direction, int count)
{
    int vec_pairs = count / 16;
    
    // Integer constants for bit operations
    const __m256i abs_mask_int = _mm256_set1_epi32(0x7FFFFFFF);
    const __m256i sign_mask_int = _mm256_set1_epi32(0x80000000);
    
    // Float constants for output
    const __m256 dir0 = _mm256_set1_ps(0.0f);
    const __m256 dir1 = _mm256_set1_ps(1.0f);
    const __m256 dir2 = _mm256_set1_ps(2.0f);
    const __m256 dir3 = _mm256_set1_ps(3.0f);
    const __m256 zero = _mm256_setzero_ps();
    
    for (int i = 0; i < vec_pairs; i++) {
        // ===== Stream 1: Process vector at offset +0 =====
        __m256 gx0 = _mm256_loadu_ps(g_x + i * 16 + 0);
        __m256 gy0 = _mm256_loadu_ps(g_y + i * 16 + 0);
        
        // Convert to integers for bit operations (faster on integer units)
        __m256i gx0_int = _mm256_castps_si256(gx0);
        __m256i gy0_int = _mm256_castps_si256(gy0);
        
        // Extract magnitudes using AND_SI256 (higher throughput than AND_PS)
        __m256i abs_gx0_int = _mm256_and_si256(gx0_int, abs_mask_int);
        __m256i abs_gy0_int = _mm256_and_si256(gy0_int, abs_mask_int);
        __m256 abs_gx0 = _mm256_castsi256_ps(abs_gx0_int);
        __m256 abs_gy0 = _mm256_castsi256_ps(abs_gy0_int);
        
        // Compare magnitudes: |g_x| > |g_y|
        __m256 mag_gt0 = _mm256_cmp_ps(abs_gx0, abs_gy0, _CMP_GT_OQ);
        
        // Extract sign bits (professor recommendation: no multiply needed!)
        // If sign bits are equal: g_x and g_y have same sign → product positive
        // If sign bits differ: g_x and g_y have different sign → product negative
        __m256i sign_gx0 = _mm256_and_si256(gx0_int, sign_mask_int);
        __m256i sign_gy0 = _mm256_and_si256(gy0_int, sign_mask_int);
        __m256i signs_xor0 = _mm256_xor_si256(sign_gx0, sign_gy0);
        
        // Convert XOR result to float for blending
        __m256 product_same_sign0 = _mm256_cmp_ps(_mm256_castsi256_ps(signs_xor0), zero, _CMP_EQ_OQ);
        __m256 product_diff_sign0 = _mm256_cmp_ps(_mm256_castsi256_ps(signs_xor0), zero, _CMP_NEQ_OQ);
        
        // Direction logic with reduced dependencies:
        // If mag_gt: dir = same_sign ? 1 : 0
        // If !mag_gt: dir = diff_sign ? 3 : 2
        __m256 dir_h0 = _mm256_blendv_ps(dir0, dir1, product_same_sign0);
        __m256 dir_v0 = _mm256_blendv_ps(dir2, dir3, product_diff_sign0);
        __m256 result0 = _mm256_blendv_ps(dir_v0, dir_h0, mag_gt0);
        
        // ===== Stream 2: Process vector at offset +8 (parallel) =====
        __m256 gx1 = _mm256_loadu_ps(g_x + i * 16 + 8);
        __m256 gy1 = _mm256_loadu_ps(g_y + i * 16 + 8);
        
        __m256i gx1_int = _mm256_castps_si256(gx1);
        __m256i gy1_int = _mm256_castps_si256(gy1);
        
        __m256i abs_gx1_int = _mm256_and_si256(gx1_int, abs_mask_int);
        __m256i abs_gy1_int = _mm256_and_si256(gy1_int, abs_mask_int);
        __m256 abs_gx1 = _mm256_castsi256_ps(abs_gx1_int);
        __m256 abs_gy1 = _mm256_castsi256_ps(abs_gy1_int);
        
        __m256 mag_gt1 = _mm256_cmp_ps(abs_gx1, abs_gy1, _CMP_GT_OQ);
        
        __m256i sign_gx1 = _mm256_and_si256(gx1_int, sign_mask_int);
        __m256i sign_gy1 = _mm256_and_si256(gy1_int, sign_mask_int);
        __m256i signs_xor1 = _mm256_xor_si256(sign_gx1, sign_gy1);
        
        __m256 product_same_sign1 = _mm256_cmp_ps(_mm256_castsi256_ps(signs_xor1), zero, _CMP_EQ_OQ);
        __m256 product_diff_sign1 = _mm256_cmp_ps(_mm256_castsi256_ps(signs_xor1), zero, _CMP_NEQ_OQ);
        
        __m256 dir_h1 = _mm256_blendv_ps(dir0, dir1, product_same_sign1);
        __m256 dir_v1 = _mm256_blendv_ps(dir2, dir3, product_diff_sign1);
        __m256 result1 = _mm256_blendv_ps(dir_v1, dir_h1, mag_gt1);
        
        // Store both results
        _mm256_storeu_ps(direction + i * 16 + 0, result0);
        _mm256_storeu_ps(direction + i * 16 + 8, result1);
    }
    
    // Handle remainder with scalar code
    for (int i = vec_pairs * 16; i < count; i++) {
        float gx = g_x[i];
        float gy = g_y[i];
        
        float abs_gx = (gx < 0) ? -gx : gx;
        float abs_gy = (gy < 0) ? -gy : gy;
        
        // Sign check: same sign = product positive
        int same_sign = ((*(int*)&gx ^ *(int*)&gy) & 0x80000000) == 0;
        int diff_sign = !same_sign;
        
        float dir;
        if (abs_gx > abs_gy) {
            dir = same_sign ? 1.0f : 0.0f;
        } else {
            dir = diff_sign ? 3.0f : 2.0f;
        }
        
        direction[i] = dir;
    }
}
