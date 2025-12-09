#include <immintrin.h>
#include <stddef.h>
#include "kernels.h"

void kernel_conv3_vert_10x8(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    __m256 ymm0=_mm256_setzero_ps(), ymm1=_mm256_setzero_ps(), ymm2=_mm256_setzero_ps(), ymm3=_mm256_setzero_ps(), ymm4=_mm256_setzero_ps(),
           ymm5=_mm256_setzero_ps(), ymm6=_mm256_setzero_ps(), ymm7=_mm256_setzero_ps(), ymm8=_mm256_setzero_ps(), ymm9=_mm256_setzero_ps();
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    ymm15=_mm256_loadu_ps(ROWCOL(-1,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4); ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(5,0)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6); ymm15=_mm256_loadu_ps(ROWCOL(6,0)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(7,0)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8); ymm15=_mm256_loadu_ps(ROWCOL(8,0)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4); ymm15=_mm256_loadu_ps(ROWCOL(5,0)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(6,0)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6); ymm15=_mm256_loadu_ps(ROWCOL(7,0)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(8,0)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8); ymm15=_mm256_loadu_ps(ROWCOL(9,0)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(5,0)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4); ymm15=_mm256_loadu_ps(ROWCOL(6,0)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(7,0)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6); ymm15=_mm256_loadu_ps(ROWCOL(8,0)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(9,0)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8); ymm15=_mm256_loadu_ps(ROWCOL(10,0)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    #undef ROWCOL

    _mm256_storeu_ps(dst+0*dst_stride,    ymm0);  _mm256_storeu_ps(dst+1*dst_stride,    ymm1);  _mm256_storeu_ps(dst+2*dst_stride,    ymm2);
    _mm256_storeu_ps(dst+3*dst_stride,    ymm3);  _mm256_storeu_ps(dst+4*dst_stride,    ymm4);  _mm256_storeu_ps(dst+5*dst_stride,    ymm5);
    _mm256_storeu_ps(dst+6*dst_stride,    ymm6);  _mm256_storeu_ps(dst+7*dst_stride,    ymm7);  _mm256_storeu_ps(dst+8*dst_stride,    ymm8);
    _mm256_storeu_ps(dst+9*dst_stride,    ymm9);
}

void kernel_conv3_vert_10x8_upper(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    __m256 ymm0=_mm256_setzero_ps(), ymm1=_mm256_setzero_ps(), ymm2=_mm256_setzero_ps(), ymm3=_mm256_setzero_ps(), ymm4=_mm256_setzero_ps(),
           ymm5=_mm256_setzero_ps(), ymm6=_mm256_setzero_ps(), ymm7=_mm256_setzero_ps(), ymm8=_mm256_setzero_ps(), ymm9=_mm256_setzero_ps();
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4); ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(5,0)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6); ymm15=_mm256_loadu_ps(ROWCOL(6,0)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(7,0)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8); ymm15=_mm256_loadu_ps(ROWCOL(8,0)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4); ymm15=_mm256_loadu_ps(ROWCOL(5,0)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(6,0)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6); ymm15=_mm256_loadu_ps(ROWCOL(7,0)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(8,0)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8); ymm15=_mm256_loadu_ps(ROWCOL(9,0)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(5,0)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4); ymm15=_mm256_loadu_ps(ROWCOL(6,0)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(7,0)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6); ymm15=_mm256_loadu_ps(ROWCOL(8,0)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(9,0)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8); ymm15=_mm256_loadu_ps(ROWCOL(10,0)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    #undef ROWCOL

    _mm256_storeu_ps(dst+0*dst_stride,    ymm0);  _mm256_storeu_ps(dst+1*dst_stride,    ymm1);  _mm256_storeu_ps(dst+2*dst_stride,    ymm2);
    _mm256_storeu_ps(dst+3*dst_stride,    ymm3);  _mm256_storeu_ps(dst+4*dst_stride,    ymm4);  _mm256_storeu_ps(dst+5*dst_stride,    ymm5);
    _mm256_storeu_ps(dst+6*dst_stride,    ymm6);  _mm256_storeu_ps(dst+7*dst_stride,    ymm7);  _mm256_storeu_ps(dst+8*dst_stride,    ymm8);
    _mm256_storeu_ps(dst+9*dst_stride,    ymm9);
}

void kernel_conv3_vert_10x8_lower(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    __m256 ymm0=_mm256_setzero_ps(), ymm1=_mm256_setzero_ps(), ymm2=_mm256_setzero_ps(), ymm3=_mm256_setzero_ps(), ymm4=_mm256_setzero_ps(),
           ymm5=_mm256_setzero_ps(), ymm6=_mm256_setzero_ps(), ymm7=_mm256_setzero_ps(), ymm8=_mm256_setzero_ps(), ymm9=_mm256_setzero_ps();
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    ymm15=_mm256_loadu_ps(ROWCOL(-1,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4); ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(5,0)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6); ymm15=_mm256_loadu_ps(ROWCOL(6,0)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(7,0)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8); ymm15=_mm256_loadu_ps(ROWCOL(8,0)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4); ymm15=_mm256_loadu_ps(ROWCOL(5,0)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(6,0)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6); ymm15=_mm256_loadu_ps(ROWCOL(7,0)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(8,0)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8); ymm15=_mm256_loadu_ps(ROWCOL(9,0)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(5,0)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4); ymm15=_mm256_loadu_ps(ROWCOL(6,0)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(7,0)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6); ymm15=_mm256_loadu_ps(ROWCOL(8,0)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(9,0)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8); ymm15=_mm256_loadu_ps(ROWCOL(9,0)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    #undef ROWCOL

    _mm256_storeu_ps(dst+0*dst_stride,    ymm0);  _mm256_storeu_ps(dst+1*dst_stride,    ymm1);  _mm256_storeu_ps(dst+2*dst_stride,    ymm2);
    _mm256_storeu_ps(dst+3*dst_stride,    ymm3);  _mm256_storeu_ps(dst+4*dst_stride,    ymm4);  _mm256_storeu_ps(dst+5*dst_stride,    ymm5);
    _mm256_storeu_ps(dst+6*dst_stride,    ymm6);  _mm256_storeu_ps(dst+7*dst_stride,    ymm7);  _mm256_storeu_ps(dst+8*dst_stride,    ymm8);
    _mm256_storeu_ps(dst+9*dst_stride,    ymm9);
}
