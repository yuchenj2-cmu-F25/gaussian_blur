#include <immintrin.h>
#include "direction_kernels.h"

/*
 * Kernel 3.4: Highly Optimized with Instruction Pipelining (16-Pixel Kernel)
 * 
 * Strategy: Maximum parallelism with multiple independent vector streams
 * - Process 2 vector pairs (16 pixels) with 3 levels of parallelism
 * - All operations independent except final blend
 * - Hide VBLEND latency through instruction-level parallelism
 * - Process in groups of 16 with multiple streams in flight
 * 
 * Key optimizations:
 *   - 2 independent accumulators process in parallel
 *   - No dependency between magnitude and sign checks
 *   - VBLEND latency hidden by parallel processing
 *   - Larger kernel: constant reuse
 *   - Cache-friendly: sequential loads/stores
 * 
 * Expected speedup:
 *   - vs Kernel 3.2: ~2.0× (better parallelism, larger kernel)
 *   - vs Kernel 3.3: ~1.3× (additional parallelism)
 */

void kernel_direction_reduction_tree(const float *restrict g_x, const float *restrict g_y,
                                    float *restrict direction, int count)
{
    int vec_pairs = count / 16;
    
    const __m256i abs_mask_int = _mm256_set1_epi32(0x7FFFFFFF);
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 three = _mm256_set1_ps(3.0f);
    const __m256 dir0 = _mm256_set1_ps(0.0f);
    const __m256 dir2 = _mm256_set1_ps(2.0f);
    
    for (int i = 0; i < vec_pairs; i++) {
        // ===== Stream 1: Process vector at offset +0 =====
        __m256 gx0 = _mm256_loadu_ps(g_x + i * 16 + 0);
        __m256 gy0 = _mm256_loadu_ps(g_y + i * 16 + 0);
        
        __m256i gx0_int = _mm256_castps_si256(gx0);
        __m256i gy0_int = _mm256_castps_si256(gy0);
        
        // Extract magnitudes
        __m256i abs_gx0_int = _mm256_and_si256(gx0_int, abs_mask_int);
        __m256i abs_gy0_int = _mm256_and_si256(gy0_int, abs_mask_int);
        __m256 abs_gx0 = _mm256_castsi256_ps(abs_gx0_int);
        __m256 abs_gy0 = _mm256_castsi256_ps(abs_gy0_int);
        __m256 mag_gt0 = _mm256_cmp_ps(abs_gx0, abs_gy0, _CMP_GT_OQ);
        
        // Compute sign product via multiplication
        __m256 product0 = _mm256_mul_ps(gx0, gy0);
        __m256 product_pos0 = _mm256_cmp_ps(product0, zero, _CMP_GT_OQ);
        __m256 product_neg0 = _mm256_cmp_ps(product0, zero, _CMP_LT_OQ);
        
        // Direction logic: mag_gt ? (product_pos ? 1 : 0) : (product_neg ? 3 : 2)
        __m256 dir_h0 = _mm256_blendv_ps(dir0, one, product_pos0);
        __m256 dir_v0 = _mm256_blendv_ps(dir2, three, product_neg0);
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
        
        __m256 product1 = _mm256_mul_ps(gx1, gy1);
        __m256 product_pos1 = _mm256_cmp_ps(product1, zero, _CMP_GT_OQ);
        __m256 product_neg1 = _mm256_cmp_ps(product1, zero, _CMP_LT_OQ);
        
        __m256 dir_h1 = _mm256_blendv_ps(dir0, one, product_pos1);
        __m256 dir_v1 = _mm256_blendv_ps(dir2, three, product_neg1);
        __m256 result1 = _mm256_blendv_ps(dir_v1, dir_h1, mag_gt1);
        
        // Store both results
        _mm256_storeu_ps(direction + i * 16 + 0, result0);
        _mm256_storeu_ps(direction + i * 16 + 8, result1);
    }
    
    // Handle remainder
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
