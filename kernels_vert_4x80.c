#include <immintrin.h>
#include <stddef.h>
#include "kernels.h"

// 4x80 kernel: processes 4 rows, 80 columns (10 accumulators × 8 floats each)
// Fully unrolled - each of the 4 output rows is computed explicitly

void kernel_conv3_vert_4x80(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9;
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    // Macro to load and initialize accumulators (first load, no accumulation)
    #define LOAD_AND_INIT(row, coeff) \
        ymm0=_mm256_loadu_ps(ROWCOL(row,0));  ymm0=_mm256_mul_ps(coeff,ymm0); \
        ymm1=_mm256_loadu_ps(ROWCOL(row,8));  ymm1=_mm256_mul_ps(coeff,ymm1); \
        ymm2=_mm256_loadu_ps(ROWCOL(row,16)); ymm2=_mm256_mul_ps(coeff,ymm2); \
        ymm3=_mm256_loadu_ps(ROWCOL(row,24)); ymm3=_mm256_mul_ps(coeff,ymm3); \
        ymm4=_mm256_loadu_ps(ROWCOL(row,32)); ymm4=_mm256_mul_ps(coeff,ymm4); \
        ymm5=_mm256_loadu_ps(ROWCOL(row,40)); ymm5=_mm256_mul_ps(coeff,ymm5); \
        ymm6=_mm256_loadu_ps(ROWCOL(row,48)); ymm6=_mm256_mul_ps(coeff,ymm6); \
        ymm7=_mm256_loadu_ps(ROWCOL(row,56)); ymm7=_mm256_mul_ps(coeff,ymm7); \
        ymm8=_mm256_loadu_ps(ROWCOL(row,64)); ymm8=_mm256_mul_ps(coeff,ymm8); \
        ymm9=_mm256_loadu_ps(ROWCOL(row,72)); ymm9=_mm256_mul_ps(coeff,ymm9);

    // Macro to load and accumulate with a coefficient
    #define LOAD_AND_ACCUMULATE(row, coeff) \
        ymm15=_mm256_loadu_ps(ROWCOL(row,0));  ymm0=_mm256_fmadd_ps(coeff,ymm15,ymm0); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,8));  ymm1=_mm256_fmadd_ps(coeff,ymm15,ymm1); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,16)); ymm2=_mm256_fmadd_ps(coeff,ymm15,ymm2); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,24)); ymm3=_mm256_fmadd_ps(coeff,ymm15,ymm3); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,32)); ymm4=_mm256_fmadd_ps(coeff,ymm15,ymm4); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,40)); ymm5=_mm256_fmadd_ps(coeff,ymm15,ymm5); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,48)); ymm6=_mm256_fmadd_ps(coeff,ymm15,ymm6); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,56)); ymm7=_mm256_fmadd_ps(coeff,ymm15,ymm7); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,64)); ymm8=_mm256_fmadd_ps(coeff,ymm15,ymm8); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,72)); ymm9=_mm256_fmadd_ps(coeff,ymm15,ymm9);

    // Macro to store all accumulators to an output row
    #define STORE_OUTPUT_ROW(row) \
        _mm256_storeu_ps(dst+(row)*dst_stride+0,  ymm0); \
        _mm256_storeu_ps(dst+(row)*dst_stride+8,  ymm1); \
        _mm256_storeu_ps(dst+(row)*dst_stride+16, ymm2); \
        _mm256_storeu_ps(dst+(row)*dst_stride+24, ymm3); \
        _mm256_storeu_ps(dst+(row)*dst_stride+32, ymm4); \
        _mm256_storeu_ps(dst+(row)*dst_stride+40, ymm5); \
        _mm256_storeu_ps(dst+(row)*dst_stride+48, ymm6); \
        _mm256_storeu_ps(dst+(row)*dst_stride+56, ymm7); \
        _mm256_storeu_ps(dst+(row)*dst_stride+64, ymm8); \
        _mm256_storeu_ps(dst+(row)*dst_stride+72, ymm9);

    // Macro to compute one complete output row with vertical convolution
    // output[row] = 0.25*input[row-1] + 0.5*input[row] + 0.25*input[row+1]
    #define COMPUTE_OUTPUT_ROW(row) \
        LOAD_AND_INIT((row)-1, ymm13) \
        LOAD_AND_ACCUMULATE((row),   ymm14) \
        LOAD_AND_ACCUMULATE((row)+1, ymm13) \
        STORE_OUTPUT_ROW(row)

    // Compute all 4 output rows
    COMPUTE_OUTPUT_ROW(0)
    COMPUTE_OUTPUT_ROW(1)
    COMPUTE_OUTPUT_ROW(2)
    COMPUTE_OUTPUT_ROW(3)
    // COMPUTE_OUTPUT_ROW(4)
    // COMPUTE_OUTPUT_ROW(5)
    // COMPUTE_OUTPUT_ROW(6)
    // COMPUTE_OUTPUT_ROW(7)

    // Clean up macros
    #undef COMPUTE_OUTPUT_ROW
    #undef STORE_OUTPUT_ROW
    #undef LOAD_AND_ACCUMULATE
    #undef LOAD_AND_INIT
    #undef ROWCOL
}

void kernel_conv3_vert_4x80_upper(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    // Upper boundary version: replicate row 0 for row -1
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9;
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    // ========== OUTPUT ROW 0 (using row 0 instead of row -1) ==========
    ymm0=_mm256_setzero_ps(); ymm1=_mm256_setzero_ps(); ymm2=_mm256_setzero_ps(); ymm3=_mm256_setzero_ps(); ymm4=_mm256_setzero_ps();
    ymm5=_mm256_setzero_ps(); ymm6=_mm256_setzero_ps(); ymm7=_mm256_setzero_ps(); ymm8=_mm256_setzero_ps(); ymm9=_mm256_setzero_ps();

    ymm15=_mm256_loadu_ps(ROWCOL(0,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(0,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(0,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(0,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(0,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(0,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(0,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(0,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,0));  ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8));  ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(0,16)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(0,24)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(0,32)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,40)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(0,48)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(0,56)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(0,64)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(0,72)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(1,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(1,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(1,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(1,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(1,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(1,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(1,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(1,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(1,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    _mm256_storeu_ps(dst+0*dst_stride+0,  ymm0); _mm256_storeu_ps(dst+0*dst_stride+8,  ymm1);
    _mm256_storeu_ps(dst+0*dst_stride+16, ymm2); _mm256_storeu_ps(dst+0*dst_stride+24, ymm3);
    _mm256_storeu_ps(dst+0*dst_stride+32, ymm4); _mm256_storeu_ps(dst+0*dst_stride+40, ymm5);
    _mm256_storeu_ps(dst+0*dst_stride+48, ymm6); _mm256_storeu_ps(dst+0*dst_stride+56, ymm7);
    _mm256_storeu_ps(dst+0*dst_stride+64, ymm8); _mm256_storeu_ps(dst+0*dst_stride+72, ymm9);

    // Rows 1-3 are normal (same as regular kernel)
    // Copy row 1 computation
    ymm0=_mm256_setzero_ps(); ymm1=_mm256_setzero_ps(); ymm2=_mm256_setzero_ps(); ymm3=_mm256_setzero_ps(); ymm4=_mm256_setzero_ps();
    ymm5=_mm256_setzero_ps(); ymm6=_mm256_setzero_ps(); ymm7=_mm256_setzero_ps(); ymm8=_mm256_setzero_ps(); ymm9=_mm256_setzero_ps();

    ymm15=_mm256_loadu_ps(ROWCOL(0,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(0,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(0,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(0,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(0,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(0,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(0,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(0,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(1,0));  ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8));  ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(1,16)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(1,24)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(1,32)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(1,40)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(1,48)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(1,56)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(1,64)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(1,72)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(2,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(2,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(2,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(2,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(2,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(2,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    _mm256_storeu_ps(dst+1*dst_stride+0,  ymm0); _mm256_storeu_ps(dst+1*dst_stride+8,  ymm1);
    _mm256_storeu_ps(dst+1*dst_stride+16, ymm2); _mm256_storeu_ps(dst+1*dst_stride+24, ymm3);
    _mm256_storeu_ps(dst+1*dst_stride+32, ymm4); _mm256_storeu_ps(dst+1*dst_stride+40, ymm5);
    _mm256_storeu_ps(dst+1*dst_stride+48, ymm6); _mm256_storeu_ps(dst+1*dst_stride+56, ymm7);
    _mm256_storeu_ps(dst+1*dst_stride+64, ymm8); _mm256_storeu_ps(dst+1*dst_stride+72, ymm9);

    // Row 2
    ymm0=_mm256_setzero_ps(); ymm1=_mm256_setzero_ps(); ymm2=_mm256_setzero_ps(); ymm3=_mm256_setzero_ps(); ymm4=_mm256_setzero_ps();
    ymm5=_mm256_setzero_ps(); ymm6=_mm256_setzero_ps(); ymm7=_mm256_setzero_ps(); ymm8=_mm256_setzero_ps(); ymm9=_mm256_setzero_ps();
    ymm15=_mm256_loadu_ps(ROWCOL(1,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(1,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(1,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(1,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(1,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(1,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(1,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(1,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(1,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0));  ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8));  ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,16)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(2,24)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(2,32)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(2,40)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(2,48)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,56)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(2,64)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(2,72)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(3,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(3,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(3,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(3,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(3,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(3,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(3,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    _mm256_storeu_ps(dst+2*dst_stride+0,  ymm0); _mm256_storeu_ps(dst+2*dst_stride+8,  ymm1);
    _mm256_storeu_ps(dst+2*dst_stride+16, ymm2); _mm256_storeu_ps(dst+2*dst_stride+24, ymm3);
    _mm256_storeu_ps(dst+2*dst_stride+32, ymm4); _mm256_storeu_ps(dst+2*dst_stride+40, ymm5);
    _mm256_storeu_ps(dst+2*dst_stride+48, ymm6); _mm256_storeu_ps(dst+2*dst_stride+56, ymm7);
    _mm256_storeu_ps(dst+2*dst_stride+64, ymm8); _mm256_storeu_ps(dst+2*dst_stride+72, ymm9);

    // Row 3
    ymm0=_mm256_setzero_ps(); ymm1=_mm256_setzero_ps(); ymm2=_mm256_setzero_ps(); ymm3=_mm256_setzero_ps(); ymm4=_mm256_setzero_ps();
    ymm5=_mm256_setzero_ps(); ymm6=_mm256_setzero_ps(); ymm7=_mm256_setzero_ps(); ymm8=_mm256_setzero_ps(); ymm9=_mm256_setzero_ps();
    ymm15=_mm256_loadu_ps(ROWCOL(2,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(2,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(2,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(2,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(2,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(2,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(2,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0));  ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8));  ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,16)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(3,24)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(3,32)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(3,40)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(3,48)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(3,56)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(3,64)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(3,72)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(4,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(4,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(4,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(4,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(4,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(4,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(4,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    _mm256_storeu_ps(dst+3*dst_stride+0,  ymm0); _mm256_storeu_ps(dst+3*dst_stride+8,  ymm1);
    _mm256_storeu_ps(dst+3*dst_stride+16, ymm2); _mm256_storeu_ps(dst+3*dst_stride+24, ymm3);
    _mm256_storeu_ps(dst+3*dst_stride+32, ymm4); _mm256_storeu_ps(dst+3*dst_stride+40, ymm5);
    _mm256_storeu_ps(dst+3*dst_stride+48, ymm6); _mm256_storeu_ps(dst+3*dst_stride+56, ymm7);
    _mm256_storeu_ps(dst+3*dst_stride+64, ymm8); _mm256_storeu_ps(dst+3*dst_stride+72, ymm9);

    #undef ROWCOL
}

void kernel_conv3_vert_4x80_lower(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    // Lower boundary version: replicate last row for out-of-bounds access
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9;
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    // Rows 0-2 are normal (same as regular kernel)
    // Row 0
    ymm0=_mm256_setzero_ps(); ymm1=_mm256_setzero_ps(); ymm2=_mm256_setzero_ps(); ymm3=_mm256_setzero_ps(); ymm4=_mm256_setzero_ps();
    ymm5=_mm256_setzero_ps(); ymm6=_mm256_setzero_ps(); ymm7=_mm256_setzero_ps(); ymm8=_mm256_setzero_ps(); ymm9=_mm256_setzero_ps();
    ymm15=_mm256_loadu_ps(ROWCOL(-1,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(-1,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(-1,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(-1,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(-1,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(-1,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(-1,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(-1,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(-1,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(-1,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(0,0));  ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8));  ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(0,16)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(0,24)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(0,32)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,40)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(0,48)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(0,56)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(0,64)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(0,72)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(1,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(1,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(1,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(1,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(1,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(1,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(1,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(1,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(1,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    _mm256_storeu_ps(dst+0*dst_stride+0,  ymm0); _mm256_storeu_ps(dst+0*dst_stride+8,  ymm1);
    _mm256_storeu_ps(dst+0*dst_stride+16, ymm2); _mm256_storeu_ps(dst+0*dst_stride+24, ymm3);
    _mm256_storeu_ps(dst+0*dst_stride+32, ymm4); _mm256_storeu_ps(dst+0*dst_stride+40, ymm5);
    _mm256_storeu_ps(dst+0*dst_stride+48, ymm6); _mm256_storeu_ps(dst+0*dst_stride+56, ymm7);
    _mm256_storeu_ps(dst+0*dst_stride+64, ymm8); _mm256_storeu_ps(dst+0*dst_stride+72, ymm9);

    // Row 1
    ymm0=_mm256_setzero_ps(); ymm1=_mm256_setzero_ps(); ymm2=_mm256_setzero_ps(); ymm3=_mm256_setzero_ps(); ymm4=_mm256_setzero_ps();
    ymm5=_mm256_setzero_ps(); ymm6=_mm256_setzero_ps(); ymm7=_mm256_setzero_ps(); ymm8=_mm256_setzero_ps(); ymm9=_mm256_setzero_ps();
    ymm15=_mm256_loadu_ps(ROWCOL(0,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(0,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(0,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(0,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(0,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(0,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(0,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(0,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(1,0));  ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8));  ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(1,16)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(1,24)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(1,32)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(1,40)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(1,48)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(1,56)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(1,64)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(1,72)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(2,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(2,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(2,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(2,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(2,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(2,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    _mm256_storeu_ps(dst+1*dst_stride+0,  ymm0); _mm256_storeu_ps(dst+1*dst_stride+8,  ymm1);
    _mm256_storeu_ps(dst+1*dst_stride+16, ymm2); _mm256_storeu_ps(dst+1*dst_stride+24, ymm3);
    _mm256_storeu_ps(dst+1*dst_stride+32, ymm4); _mm256_storeu_ps(dst+1*dst_stride+40, ymm5);
    _mm256_storeu_ps(dst+1*dst_stride+48, ymm6); _mm256_storeu_ps(dst+1*dst_stride+56, ymm7);
    _mm256_storeu_ps(dst+1*dst_stride+64, ymm8); _mm256_storeu_ps(dst+1*dst_stride+72, ymm9);

    // Row 2
    ymm0=_mm256_setzero_ps(); ymm1=_mm256_setzero_ps(); ymm2=_mm256_setzero_ps(); ymm3=_mm256_setzero_ps(); ymm4=_mm256_setzero_ps();
    ymm5=_mm256_setzero_ps(); ymm6=_mm256_setzero_ps(); ymm7=_mm256_setzero_ps(); ymm8=_mm256_setzero_ps(); ymm9=_mm256_setzero_ps();
    ymm15=_mm256_loadu_ps(ROWCOL(1,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(1,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(1,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(1,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(1,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(1,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(1,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(1,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(1,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0));  ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8));  ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,16)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(2,24)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(2,32)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(2,40)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(2,48)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,56)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(2,64)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(2,72)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(3,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(3,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(3,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(3,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(3,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(3,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(3,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    _mm256_storeu_ps(dst+2*dst_stride+0,  ymm0); _mm256_storeu_ps(dst+2*dst_stride+8,  ymm1);
    _mm256_storeu_ps(dst+2*dst_stride+16, ymm2); _mm256_storeu_ps(dst+2*dst_stride+24, ymm3);
    _mm256_storeu_ps(dst+2*dst_stride+32, ymm4); _mm256_storeu_ps(dst+2*dst_stride+40, ymm5);
    _mm256_storeu_ps(dst+2*dst_stride+48, ymm6); _mm256_storeu_ps(dst+2*dst_stride+56, ymm7);
    _mm256_storeu_ps(dst+2*dst_stride+64, ymm8); _mm256_storeu_ps(dst+2*dst_stride+72, ymm9);

    // Row 3 (last row - use row 3 instead of row 4)
    ymm0=_mm256_setzero_ps(); ymm1=_mm256_setzero_ps(); ymm2=_mm256_setzero_ps(); ymm3=_mm256_setzero_ps(); ymm4=_mm256_setzero_ps();
    ymm5=_mm256_setzero_ps(); ymm6=_mm256_setzero_ps(); ymm7=_mm256_setzero_ps(); ymm8=_mm256_setzero_ps(); ymm9=_mm256_setzero_ps();
    ymm15=_mm256_loadu_ps(ROWCOL(2,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(2,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(2,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(2,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(2,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(2,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(2,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0));  ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8));  ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,16)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(3,24)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(3,32)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(3,40)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(3,48)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(3,56)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(3,64)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(3,72)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,16)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(3,24)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(3,32)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(3,40)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(3,48)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(3,56)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(3,64)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(3,72)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);
    _mm256_storeu_ps(dst+3*dst_stride+0,  ymm0); _mm256_storeu_ps(dst+3*dst_stride+8,  ymm1);
    _mm256_storeu_ps(dst+3*dst_stride+16, ymm2); _mm256_storeu_ps(dst+3*dst_stride+24, ymm3);
    _mm256_storeu_ps(dst+3*dst_stride+32, ymm4); _mm256_storeu_ps(dst+3*dst_stride+40, ymm5);
    _mm256_storeu_ps(dst+3*dst_stride+48, ymm6); _mm256_storeu_ps(dst+3*dst_stride+56, ymm7);
    _mm256_storeu_ps(dst+3*dst_stride+64, ymm8); _mm256_storeu_ps(dst+3*dst_stride+72, ymm9);

    #undef ROWCOL
}
