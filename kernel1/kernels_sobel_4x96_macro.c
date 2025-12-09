#include <immintrin.h>
#include <stddef.h>
#include "kernels.h"

// Sobel vertical smoothing kernel [1, 2, 1]: processes 4 rows, 80 columns
void kernel_sobel_vert_smooth_4x96(const float *restrict src, int src_stride,
                                    float *restrict dst, int dst_stride)
{
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
    const __m256 ymm13=_mm256_set1_ps(1.0f), ymm14=_mm256_set1_ps(2.0f);
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
        ymm9=_mm256_loadu_ps(ROWCOL(row,72)); ymm9=_mm256_mul_ps(coeff,ymm9); \
        ymm10=_mm256_loadu_ps(ROWCOL(row,80)); ymm10=_mm256_mul_ps(coeff,ymm10); \
        ymm11=_mm256_loadu_ps(ROWCOL(row,88)); ymm11=_mm256_mul_ps(coeff,ymm11);

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
        ymm15=_mm256_loadu_ps(ROWCOL(row,72)); ymm9=_mm256_fmadd_ps(coeff,ymm15,ymm9); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,80)); ymm10=_mm256_fmadd_ps(coeff,ymm15,ymm10); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,88)); ymm11=_mm256_fmadd_ps(coeff,ymm15,ymm11);

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
        _mm256_storeu_ps(dst+(row)*dst_stride+72, ymm9); \
        _mm256_storeu_ps(dst+(row)*dst_stride+80, ymm10); \
        _mm256_storeu_ps(dst+(row)*dst_stride+88, ymm11);

    #define COMPUTE_OUTPUT_ROW(row) \
        LOAD_AND_INIT((row)-1, ymm13) \
        LOAD_AND_ACCUMULATE((row),   ymm14) \
        LOAD_AND_ACCUMULATE((row)+1, ymm13) \
        STORE_OUTPUT_ROW(row)

    COMPUTE_OUTPUT_ROW(0)
    COMPUTE_OUTPUT_ROW(1)
    COMPUTE_OUTPUT_ROW(2)
    COMPUTE_OUTPUT_ROW(3)

    #undef COMPUTE_OUTPUT_ROW
    #undef STORE_OUTPUT_ROW
    #undef LOAD_AND_ACCUMULATE
    #undef LOAD_AND_INIT
    #undef ROWCOL
}

// Sobel vertical derivative kernel [-1, 0, 1]: processes 4 rows, 80 columns
void kernel_sobel_vert_deriv_4x96(const float *restrict src, int src_stride,
                                   float *restrict dst, int dst_stride)
{
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
    const __m256 ymm13=_mm256_set1_ps(-1.0f), ymm15_one=_mm256_set1_ps(1.0f);
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
        ymm9=_mm256_loadu_ps(ROWCOL(row,72)); ymm9=_mm256_mul_ps(coeff,ymm9); \
        ymm10=_mm256_loadu_ps(ROWCOL(row,80)); ymm10=_mm256_mul_ps(coeff,ymm10); \
        ymm11=_mm256_loadu_ps(ROWCOL(row,88)); ymm11=_mm256_mul_ps(coeff,ymm11);

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
        ymm15=_mm256_loadu_ps(ROWCOL(row,72)); ymm9=_mm256_fmadd_ps(coeff,ymm15,ymm9); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,80)); ymm10=_mm256_fmadd_ps(coeff,ymm15,ymm10); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,88)); ymm11=_mm256_fmadd_ps(coeff,ymm15,ymm11);

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
        _mm256_storeu_ps(dst+(row)*dst_stride+72, ymm9); \
        _mm256_storeu_ps(dst+(row)*dst_stride+80, ymm10); \
        _mm256_storeu_ps(dst+(row)*dst_stride+88, ymm11);

    // Derivative kernel: [-1, 0, 1] - middle term is 0, so we skip it
    #define COMPUTE_OUTPUT_ROW(row) \
        LOAD_AND_INIT((row)-1, ymm13) \
        LOAD_AND_ACCUMULATE((row)+1, ymm15_one) \
        STORE_OUTPUT_ROW(row)

    COMPUTE_OUTPUT_ROW(0)
    COMPUTE_OUTPUT_ROW(1)
    COMPUTE_OUTPUT_ROW(2)
    COMPUTE_OUTPUT_ROW(3)

    #undef COMPUTE_OUTPUT_ROW
    #undef STORE_OUTPUT_ROW
    #undef LOAD_AND_ACCUMULATE
    #undef LOAD_AND_INIT
    #undef ROWCOL
}

// Sobel horizontal smoothing kernel [1, 2, 1]: processes 4 rows, 80 columns
void kernel_sobel_horiz_smooth_4x96(const float *restrict src, int src_stride,
                                     float *restrict dst, int dst_stride)
{
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
    const __m256 ymm13=_mm256_set1_ps(1.0f), ymm14=_mm256_set1_ps(2.0f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    // Macro to load and initialize accumulators (first load, no accumulation)
    #define LOAD_AND_INIT(row, col_offset, coeff) \
        ymm0=_mm256_loadu_ps(ROWCOL(row,(col_offset)+0));  ymm0=_mm256_mul_ps(coeff,ymm0); \
        ymm1=_mm256_loadu_ps(ROWCOL(row,(col_offset)+8));  ymm1=_mm256_mul_ps(coeff,ymm1); \
        ymm2=_mm256_loadu_ps(ROWCOL(row,(col_offset)+16)); ymm2=_mm256_mul_ps(coeff,ymm2); \
        ymm3=_mm256_loadu_ps(ROWCOL(row,(col_offset)+24)); ymm3=_mm256_mul_ps(coeff,ymm3); \
        ymm4=_mm256_loadu_ps(ROWCOL(row,(col_offset)+32)); ymm4=_mm256_mul_ps(coeff,ymm4); \
        ymm5=_mm256_loadu_ps(ROWCOL(row,(col_offset)+40)); ymm5=_mm256_mul_ps(coeff,ymm5); \
        ymm6=_mm256_loadu_ps(ROWCOL(row,(col_offset)+48)); ymm6=_mm256_mul_ps(coeff,ymm6); \
        ymm7=_mm256_loadu_ps(ROWCOL(row,(col_offset)+56)); ymm7=_mm256_mul_ps(coeff,ymm7); \
        ymm8=_mm256_loadu_ps(ROWCOL(row,(col_offset)+64)); ymm8=_mm256_mul_ps(coeff,ymm8); \
        ymm9=_mm256_loadu_ps(ROWCOL(row,(col_offset)+72)); ymm9=_mm256_mul_ps(coeff,ymm9); \
        ymm10=_mm256_loadu_ps(ROWCOL(row,(col_offset)+80)); ymm10=_mm256_mul_ps(coeff,ymm10); \
        ymm11=_mm256_loadu_ps(ROWCOL(row,(col_offset)+88)); ymm11=_mm256_mul_ps(coeff,ymm11);

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
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+72)); ymm9=_mm256_fmadd_ps(coeff,ymm15,ymm9); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+80)); ymm10=_mm256_fmadd_ps(coeff,ymm15,ymm10); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+88)); ymm11=_mm256_fmadd_ps(coeff,ymm15,ymm11);

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
        _mm256_storeu_ps(dst+(row)*dst_stride+72, ymm9); \
        _mm256_storeu_ps(dst+(row)*dst_stride+80, ymm10); \
        _mm256_storeu_ps(dst+(row)*dst_stride+88, ymm11);

    #define COMPUTE_OUTPUT_ROW(row) \
        LOAD_AND_INIT((row), -1, ymm13) \
        LOAD_AND_ACCUMULATE((row),  0, ymm14) \
        LOAD_AND_ACCUMULATE((row),  1, ymm13) \
        STORE_OUTPUT_ROW(row)

    COMPUTE_OUTPUT_ROW(0)
    COMPUTE_OUTPUT_ROW(1)
    COMPUTE_OUTPUT_ROW(2)
    COMPUTE_OUTPUT_ROW(3)

    #undef COMPUTE_OUTPUT_ROW
    #undef STORE_OUTPUT_ROW
    #undef LOAD_AND_ACCUMULATE
    #undef LOAD_AND_INIT
    #undef ROWCOL
}

// Sobel horizontal derivative kernel [-1, 0, 1]: processes 4 rows, 80 columns
void kernel_sobel_horiz_deriv_4x96(const float *restrict src, int src_stride,
                                    float *restrict dst, int dst_stride)
{
    __m256 ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11;
    const __m256 ymm13=_mm256_set1_ps(-1.0f), ymm15_one=_mm256_set1_ps(1.0f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    // Macro to load and initialize accumulators (first load, no accumulation)
    #define LOAD_AND_INIT(row, col_offset, coeff) \
        ymm0=_mm256_loadu_ps(ROWCOL(row,(col_offset)+0));  ymm0=_mm256_mul_ps(coeff,ymm0); \
        ymm1=_mm256_loadu_ps(ROWCOL(row,(col_offset)+8));  ymm1=_mm256_mul_ps(coeff,ymm1); \
        ymm2=_mm256_loadu_ps(ROWCOL(row,(col_offset)+16)); ymm2=_mm256_mul_ps(coeff,ymm2); \
        ymm3=_mm256_loadu_ps(ROWCOL(row,(col_offset)+24)); ymm3=_mm256_mul_ps(coeff,ymm3); \
        ymm4=_mm256_loadu_ps(ROWCOL(row,(col_offset)+32)); ymm4=_mm256_mul_ps(coeff,ymm4); \
        ymm5=_mm256_loadu_ps(ROWCOL(row,(col_offset)+40)); ymm5=_mm256_mul_ps(coeff,ymm5); \
        ymm6=_mm256_loadu_ps(ROWCOL(row,(col_offset)+48)); ymm6=_mm256_mul_ps(coeff,ymm6); \
        ymm7=_mm256_loadu_ps(ROWCOL(row,(col_offset)+56)); ymm7=_mm256_mul_ps(coeff,ymm7); \
        ymm8=_mm256_loadu_ps(ROWCOL(row,(col_offset)+64)); ymm8=_mm256_mul_ps(coeff,ymm8); \
        ymm9=_mm256_loadu_ps(ROWCOL(row,(col_offset)+72)); ymm9=_mm256_mul_ps(coeff,ymm9); \
        ymm10=_mm256_loadu_ps(ROWCOL(row,(col_offset)+80)); ymm10=_mm256_mul_ps(coeff,ymm10); \
        ymm11=_mm256_loadu_ps(ROWCOL(row,(col_offset)+88)); ymm11=_mm256_mul_ps(coeff,ymm11);

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
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+72)); ymm9=_mm256_fmadd_ps(coeff,ymm15,ymm9); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+80)); ymm10=_mm256_fmadd_ps(coeff,ymm15,ymm10); \
        ymm15=_mm256_loadu_ps(ROWCOL(row,(col_offset)+88)); ymm11=_mm256_fmadd_ps(coeff,ymm15,ymm11);

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
        _mm256_storeu_ps(dst+(row)*dst_stride+72, ymm9); \
        _mm256_storeu_ps(dst+(row)*dst_stride+80, ymm10); \
        _mm256_storeu_ps(dst+(row)*dst_stride+88, ymm11);

    // Derivative kernel: [-1, 0, 1] - middle term is 0, so we skip it
    #define COMPUTE_OUTPUT_ROW(row) \
        LOAD_AND_INIT((row), -1, ymm13) \
        LOAD_AND_ACCUMULATE((row),  1, ymm15_one) \
        STORE_OUTPUT_ROW(row)

    COMPUTE_OUTPUT_ROW(0)
    COMPUTE_OUTPUT_ROW(1)
    COMPUTE_OUTPUT_ROW(2)
    COMPUTE_OUTPUT_ROW(3)

    #undef COMPUTE_OUTPUT_ROW
    #undef STORE_OUTPUT_ROW
    #undef LOAD_AND_ACCUMULATE
    #undef LOAD_AND_INIT
    #undef ROWCOL
}
