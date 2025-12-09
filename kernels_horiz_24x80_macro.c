#include <immintrin.h>
#include <stddef.h>
#include "kernels.h"

// 24x80 horizontal kernel: processes 24 rows, 80 columns (10 accumulators × 8 floats each)
// Macro-based version to eliminate code repetition

void kernel_conv3_horiz_24x80(const float *restrict src, int src_stride,
                               float *restrict dst, int dst_stride)
{
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9;
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    // Macro to zero all accumulators
    #define ZERO_ACCUMULATORS() \
        ymm0=_mm256_setzero_ps(); ymm1=_mm256_setzero_ps(); ymm2=_mm256_setzero_ps(); \
        ymm3=_mm256_setzero_ps(); ymm4=_mm256_setzero_ps(); ymm5=_mm256_setzero_ps(); \
        ymm6=_mm256_setzero_ps(); ymm7=_mm256_setzero_ps(); ymm8=_mm256_setzero_ps(); \
        ymm9=_mm256_setzero_ps();

    // Macro to load and accumulate with horizontal offset and coefficient
    #define LOAD_AND_ACCUMULATE(row, col_offset, coeff) \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+0));  ymm0=_mm256_fmadd_ps(coeff,ymm15,ymm0); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+8));  ymm1=_mm256_fmadd_ps(coeff,ymm15,ymm1); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+16)); ymm2=_mm256_fmadd_ps(coeff,ymm15,ymm2); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+24)); ymm3=_mm256_fmadd_ps(coeff,ymm15,ymm3); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+32)); ymm4=_mm256_fmadd_ps(coeff,ymm15,ymm4); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+40)); ymm5=_mm256_fmadd_ps(coeff,ymm15,ymm5); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+48)); ymm6=_mm256_fmadd_ps(coeff,ymm15,ymm6); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+56)); ymm7=_mm256_fmadd_ps(coeff,ymm15,ymm7); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+64)); ymm8=_mm256_fmadd_ps(coeff,ymm15,ymm8); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+72)); ymm9=_mm256_fmadd_ps(coeff,ymm15,ymm9);

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

    // Macro to compute one complete output row with horizontal convolution
    // Each output row uses 3 column offsets: (col-1, col, col+1) with coefficients (0.25, 0.5, 0.25)
    #define COMPUTE_OUTPUT_ROW(row) \
        ZERO_ACCUMULATORS() \
        LOAD_AND_ACCUMULATE((row), -1, ymm13) \
        LOAD_AND_ACCUMULATE((row),  0, ymm14) \
        LOAD_AND_ACCUMULATE((row),  1, ymm13) \
        STORE_OUTPUT_ROW(row)

    // Compute all 24 output rows
    COMPUTE_OUTPUT_ROW(0)
    COMPUTE_OUTPUT_ROW(1)
    COMPUTE_OUTPUT_ROW(2)
    COMPUTE_OUTPUT_ROW(3)
    COMPUTE_OUTPUT_ROW(4)
    COMPUTE_OUTPUT_ROW(5)
    COMPUTE_OUTPUT_ROW(6)
    COMPUTE_OUTPUT_ROW(7)
    COMPUTE_OUTPUT_ROW(8)
    COMPUTE_OUTPUT_ROW(9)
    COMPUTE_OUTPUT_ROW(10)
    COMPUTE_OUTPUT_ROW(11)
    COMPUTE_OUTPUT_ROW(12)
    COMPUTE_OUTPUT_ROW(13)
    COMPUTE_OUTPUT_ROW(14)
    COMPUTE_OUTPUT_ROW(15)
    COMPUTE_OUTPUT_ROW(16)
    COMPUTE_OUTPUT_ROW(17)
    COMPUTE_OUTPUT_ROW(18)
    COMPUTE_OUTPUT_ROW(19)
    COMPUTE_OUTPUT_ROW(20)
    COMPUTE_OUTPUT_ROW(21)
    COMPUTE_OUTPUT_ROW(22)
    COMPUTE_OUTPUT_ROW(23)

    // Clean up macros
    #undef COMPUTE_OUTPUT_ROW
    #undef STORE_OUTPUT_ROW
    #undef LOAD_AND_ACCUMULATE
    #undef ZERO_ACCUMULATORS
    #undef ROWCOL
}
