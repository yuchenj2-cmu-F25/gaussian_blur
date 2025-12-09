#include <immintrin.h>
#include <stddef.h>
#include "kernels.h"

// Combined 24x80 kernel: horizontal pass followed by vertical pass
// Processes 24 output rows, 80 columns
// Uses internal tmp buffer passed as parameter

void kernel_conv3_combined_24x80(const float *restrict src, int src_stride,
                                  float *restrict dst, int dst_stride,
                                  float *restrict tmp, int tmp_stride)
{
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9;
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)
    #define TMPROWCOL(s,t) (tmp + (s)*(ptrdiff_t)tmp_stride + (t)*(ptrdiff_t)1)

    // ========== HORIZONTAL PASS: src -> tmp (26 rows needed for 24 output rows) ==========

    // Macro to load and initialize accumulators for horizontal pass
    #define LOAD_AND_INIT_HORIZ(row, col_offset, coeff) \
        ymm0=_mm256_loadu_ps(ROWCOL(row,(col_offset)+0));  ymm0=_mm256_mul_ps(coeff,ymm0); \
        ymm1=_mm256_loadu_ps(ROWCOL(row,(col_offset)+8));  ymm1=_mm256_mul_ps(coeff,ymm1); \
        ymm2=_mm256_loadu_ps(ROWCOL(row,(col_offset)+16)); ymm2=_mm256_mul_ps(coeff,ymm2); \
        ymm3=_mm256_loadu_ps(ROWCOL(row,(col_offset)+24)); ymm3=_mm256_mul_ps(coeff,ymm3); \
        ymm4=_mm256_loadu_ps(ROWCOL(row,(col_offset)+32)); ymm4=_mm256_mul_ps(coeff,ymm4); \
        ymm5=_mm256_loadu_ps(ROWCOL(row,(col_offset)+40)); ymm5=_mm256_mul_ps(coeff,ymm5); \
        ymm6=_mm256_loadu_ps(ROWCOL(row,(col_offset)+48)); ymm6=_mm256_mul_ps(coeff,ymm6); \
        ymm7=_mm256_loadu_ps(ROWCOL(row,(col_offset)+56)); ymm7=_mm256_mul_ps(coeff,ymm7); \
        ymm8=_mm256_loadu_ps(ROWCOL(row,(col_offset)+64)); ymm8=_mm256_mul_ps(coeff,ymm8); \
        ymm9=_mm256_loadu_ps(ROWCOL(row,(col_offset)+72)); ymm9=_mm256_mul_ps(coeff,ymm9);

    #define LOAD_AND_ACCUMULATE_HORIZ(row, col_offset, coeff) \
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

    #define STORE_TMP_ROW(row) \
        _mm256_storeu_ps(TMPROWCOL(row,0),  ymm0); \
        _mm256_storeu_ps(TMPROWCOL(row,8),  ymm1); \
        _mm256_storeu_ps(TMPROWCOL(row,16), ymm2); \
        _mm256_storeu_ps(TMPROWCOL(row,24), ymm3); \
        _mm256_storeu_ps(TMPROWCOL(row,32), ymm4); \
        _mm256_storeu_ps(TMPROWCOL(row,40), ymm5); \
        _mm256_storeu_ps(TMPROWCOL(row,48), ymm6); \
        _mm256_storeu_ps(TMPROWCOL(row,56), ymm7); \
        _mm256_storeu_ps(TMPROWCOL(row,64), ymm8); \
        _mm256_storeu_ps(TMPROWCOL(row,72), ymm9);

    #define COMPUTE_HORIZ_ROW(row) \
        LOAD_AND_INIT_HORIZ((row), -1, ymm13) \
        LOAD_AND_ACCUMULATE_HORIZ((row),  0, ymm14) \
        LOAD_AND_ACCUMULATE_HORIZ((row),  1, ymm13) \
        STORE_TMP_ROW(row)

    // Horizontal pass: process rows -1 to 24 (26 rows total)
    COMPUTE_HORIZ_ROW(-1)
    COMPUTE_HORIZ_ROW(0)
    COMPUTE_HORIZ_ROW(1)
    COMPUTE_HORIZ_ROW(2)
    COMPUTE_HORIZ_ROW(3)
    COMPUTE_HORIZ_ROW(4)
    COMPUTE_HORIZ_ROW(5)
    COMPUTE_HORIZ_ROW(6)
    COMPUTE_HORIZ_ROW(7)
    COMPUTE_HORIZ_ROW(8)
    COMPUTE_HORIZ_ROW(9)
    COMPUTE_HORIZ_ROW(10)
    COMPUTE_HORIZ_ROW(11)
    COMPUTE_HORIZ_ROW(12)
    COMPUTE_HORIZ_ROW(13)
    COMPUTE_HORIZ_ROW(14)
    COMPUTE_HORIZ_ROW(15)
    COMPUTE_HORIZ_ROW(16)
    COMPUTE_HORIZ_ROW(17)
    COMPUTE_HORIZ_ROW(18)
    COMPUTE_HORIZ_ROW(19)
    COMPUTE_HORIZ_ROW(20)
    COMPUTE_HORIZ_ROW(21)
    COMPUTE_HORIZ_ROW(22)
    COMPUTE_HORIZ_ROW(23)
    COMPUTE_HORIZ_ROW(24)

    #undef COMPUTE_HORIZ_ROW
    #undef STORE_TMP_ROW
    #undef LOAD_AND_ACCUMULATE_HORIZ
    #undef LOAD_AND_INIT_HORIZ

    // ========== VERTICAL PASS: tmp -> dst (24 rows) ==========

    // Macro to load and initialize accumulators for vertical pass
    #define LOAD_AND_INIT_VERT(row, coeff) \
        ymm0=_mm256_loadu_ps(TMPROWCOL(row,0));  ymm0=_mm256_mul_ps(coeff,ymm0); \
        ymm1=_mm256_loadu_ps(TMPROWCOL(row,8));  ymm1=_mm256_mul_ps(coeff,ymm1); \
        ymm2=_mm256_loadu_ps(TMPROWCOL(row,16)); ymm2=_mm256_mul_ps(coeff,ymm2); \
        ymm3=_mm256_loadu_ps(TMPROWCOL(row,24)); ymm3=_mm256_mul_ps(coeff,ymm3); \
        ymm4=_mm256_loadu_ps(TMPROWCOL(row,32)); ymm4=_mm256_mul_ps(coeff,ymm4); \
        ymm5=_mm256_loadu_ps(TMPROWCOL(row,40)); ymm5=_mm256_mul_ps(coeff,ymm5); \
        ymm6=_mm256_loadu_ps(TMPROWCOL(row,48)); ymm6=_mm256_mul_ps(coeff,ymm6); \
        ymm7=_mm256_loadu_ps(TMPROWCOL(row,56)); ymm7=_mm256_mul_ps(coeff,ymm7); \
        ymm8=_mm256_loadu_ps(TMPROWCOL(row,64)); ymm8=_mm256_mul_ps(coeff,ymm8); \
        ymm9=_mm256_loadu_ps(TMPROWCOL(row,72)); ymm9=_mm256_mul_ps(coeff,ymm9);

    #define LOAD_AND_ACCUMULATE_VERT(row, coeff) \
        ymm15=_mm256_loadu_ps(TMPROWCOL(row,0));  ymm0=_mm256_fmadd_ps(coeff,ymm15,ymm0); \
        ymm15=_mm256_loadu_ps(TMPROWCOL(row,8));  ymm1=_mm256_fmadd_ps(coeff,ymm15,ymm1); \
        ymm15=_mm256_loadu_ps(TMPROWCOL(row,16)); ymm2=_mm256_fmadd_ps(coeff,ymm15,ymm2); \
        ymm15=_mm256_loadu_ps(TMPROWCOL(row,24)); ymm3=_mm256_fmadd_ps(coeff,ymm15,ymm3); \
        ymm15=_mm256_loadu_ps(TMPROWCOL(row,32)); ymm4=_mm256_fmadd_ps(coeff,ymm15,ymm4); \
        ymm15=_mm256_loadu_ps(TMPROWCOL(row,40)); ymm5=_mm256_fmadd_ps(coeff,ymm15,ymm5); \
        ymm15=_mm256_loadu_ps(TMPROWCOL(row,48)); ymm6=_mm256_fmadd_ps(coeff,ymm15,ymm6); \
        ymm15=_mm256_loadu_ps(TMPROWCOL(row,56)); ymm7=_mm256_fmadd_ps(coeff,ymm15,ymm7); \
        ymm15=_mm256_loadu_ps(TMPROWCOL(row,64)); ymm8=_mm256_fmadd_ps(coeff,ymm15,ymm8); \
        ymm15=_mm256_loadu_ps(TMPROWCOL(row,72)); ymm9=_mm256_fmadd_ps(coeff,ymm15,ymm9);

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

    #define COMPUTE_VERT_ROW(row) \
        LOAD_AND_INIT_VERT((row)-1, ymm13) \
        LOAD_AND_ACCUMULATE_VERT((row),   ymm14) \
        LOAD_AND_ACCUMULATE_VERT((row)+1, ymm13) \
        STORE_OUTPUT_ROW(row)

    // Vertical pass: process 24 output rows
    COMPUTE_VERT_ROW(0)
    COMPUTE_VERT_ROW(1)
    COMPUTE_VERT_ROW(2)
    COMPUTE_VERT_ROW(3)
    COMPUTE_VERT_ROW(4)
    COMPUTE_VERT_ROW(5)
    COMPUTE_VERT_ROW(6)
    COMPUTE_VERT_ROW(7)
    COMPUTE_VERT_ROW(8)
    COMPUTE_VERT_ROW(9)
    COMPUTE_VERT_ROW(10)
    COMPUTE_VERT_ROW(11)
    COMPUTE_VERT_ROW(12)
    COMPUTE_VERT_ROW(13)
    COMPUTE_VERT_ROW(14)
    COMPUTE_VERT_ROW(15)
    COMPUTE_VERT_ROW(16)
    COMPUTE_VERT_ROW(17)
    COMPUTE_VERT_ROW(18)
    COMPUTE_VERT_ROW(19)
    COMPUTE_VERT_ROW(20)
    COMPUTE_VERT_ROW(21)
    COMPUTE_VERT_ROW(22)
    COMPUTE_VERT_ROW(23)

    #undef COMPUTE_VERT_ROW
    #undef STORE_OUTPUT_ROW
    #undef LOAD_AND_ACCUMULATE_VERT
    #undef LOAD_AND_INIT_VERT
    #undef TMPROWCOL
    #undef ROWCOL
}
