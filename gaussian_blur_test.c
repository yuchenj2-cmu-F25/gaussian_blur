#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <stddef.h>
#include <string.h>

#define WIDTH  512
#define HEIGHT 600
#define RUNS 100

void kernel_conv3_vert_5x16_lowload(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    __m256 ymm0=_mm256_setzero_ps(), ymm1=_mm256_setzero_ps(), ymm2=_mm256_setzero_ps(), ymm3=_mm256_setzero_ps(), ymm4=_mm256_setzero_ps(),
           ymm5=_mm256_setzero_ps(), ymm6=_mm256_setzero_ps(), ymm7=_mm256_setzero_ps(), ymm8=_mm256_setzero_ps(), ymm9=_mm256_setzero_ps();
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    /* Group 0 */
    ymm15=_mm256_loadu_ps(ROWCOL(-1,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(5,0));  ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(-1,8)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8));  ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(5,8));  ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    /* Group 1 */
    ymm15=_mm256_loadu_ps(ROWCOL(0,0));  ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0));  ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8));  ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8));  ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    /* Group 2 */
    ymm15=_mm256_loadu_ps(ROWCOL(1,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0));  ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8));  ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(4,8));  ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);

    #undef ROWCOL

    _mm256_storeu_ps(dst+0*dst_stride,    ymm0);  _mm256_storeu_ps(dst+1*dst_stride,    ymm1);  _mm256_storeu_ps(dst+2*dst_stride,    ymm2);
    _mm256_storeu_ps(dst+3*dst_stride,    ymm3);  _mm256_storeu_ps(dst+4*dst_stride,    ymm4);
    _mm256_storeu_ps(dst+0*dst_stride+8,  ymm5);  _mm256_storeu_ps(dst+1*dst_stride+8,  ymm6);  _mm256_storeu_ps(dst+2*dst_stride+8,  ymm7);
    _mm256_storeu_ps(dst+3*dst_stride+8,  ymm8);  _mm256_storeu_ps(dst+4*dst_stride+8,  ymm9);
}


void kernel_conv3_vert_5x16_lowload_upper(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    __m256 ymm0=_mm256_setzero_ps(), ymm1=_mm256_setzero_ps(), ymm2=_mm256_setzero_ps(), ymm3=_mm256_setzero_ps(), ymm4=_mm256_setzero_ps(),
           ymm5=_mm256_setzero_ps(), ymm6=_mm256_setzero_ps(), ymm7=_mm256_setzero_ps(), ymm8=_mm256_setzero_ps(), ymm9=_mm256_setzero_ps();
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    /* Group 0 */
    ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(5,0));  ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8));  ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(5,8));  ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    /* Group 1 */
    ymm15=_mm256_loadu_ps(ROWCOL(0,0));  ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0));  ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8));  ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8));  ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    /* Group 2 */
    ymm15=_mm256_loadu_ps(ROWCOL(1,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0));  ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8));  ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(4,8));  ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);

    #undef ROWCOL

    _mm256_storeu_ps(dst+0*dst_stride,    ymm0);  _mm256_storeu_ps(dst+1*dst_stride,    ymm1);  _mm256_storeu_ps(dst+2*dst_stride,    ymm2);
    _mm256_storeu_ps(dst+3*dst_stride,    ymm3);  _mm256_storeu_ps(dst+4*dst_stride,    ymm4);
    _mm256_storeu_ps(dst+0*dst_stride+8,  ymm5);  _mm256_storeu_ps(dst+1*dst_stride+8,  ymm6);  _mm256_storeu_ps(dst+2*dst_stride+8,  ymm7);
    _mm256_storeu_ps(dst+3*dst_stride+8,  ymm8);  _mm256_storeu_ps(dst+4*dst_stride+8,  ymm9);
}

void kernel_conv3_vert_5x16_lowload_lower(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    __m256 ymm0=_mm256_setzero_ps(), ymm1=_mm256_setzero_ps(), ymm2=_mm256_setzero_ps(), ymm3=_mm256_setzero_ps(), ymm4=_mm256_setzero_ps(),
           ymm5=_mm256_setzero_ps(), ymm6=_mm256_setzero_ps(), ymm7=_mm256_setzero_ps(), ymm8=_mm256_setzero_ps(), ymm9=_mm256_setzero_ps();
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    /* Group 0 */
    ymm15=_mm256_loadu_ps(ROWCOL(-1,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0));  ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0));  ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(-1,8)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8));  ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,8));  ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    /* Group 1 */
    ymm15=_mm256_loadu_ps(ROWCOL(0,0));  ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0));  ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8));  ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8));  ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    /* Group 2 */
    ymm15=_mm256_loadu_ps(ROWCOL(1,0));  ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0));  ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8));  ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7);
    ymm15=_mm256_loadu_ps(ROWCOL(4,8));  ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);

    #undef ROWCOL

    _mm256_storeu_ps(dst+0*dst_stride,    ymm0);  _mm256_storeu_ps(dst+1*dst_stride,    ymm1);  _mm256_storeu_ps(dst+2*dst_stride,    ymm2);
    _mm256_storeu_ps(dst+3*dst_stride,    ymm3);  _mm256_storeu_ps(dst+4*dst_stride,    ymm4);
    _mm256_storeu_ps(dst+0*dst_stride+8,  ymm5);  _mm256_storeu_ps(dst+1*dst_stride+8,  ymm6);  _mm256_storeu_ps(dst+2*dst_stride+8,  ymm7);
    _mm256_storeu_ps(dst+3*dst_stride+8,  ymm8);  _mm256_storeu_ps(dst+4*dst_stride+8,  ymm9);
}

void kernel_conv3_vert_5x16(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    __m256 ymm0=_mm256_setzero_ps(), ymm1=_mm256_setzero_ps(), ymm2=_mm256_setzero_ps(), ymm3=_mm256_setzero_ps(), ymm4=_mm256_setzero_ps(),
           ymm5=_mm256_setzero_ps(), ymm6=_mm256_setzero_ps(), ymm7=_mm256_setzero_ps(), ymm8=_mm256_setzero_ps(), ymm9=_mm256_setzero_ps();
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    ymm15=_mm256_loadu_ps(ROWCOL(-1,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(-1,8)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(0,8)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(2,8)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(1,8)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(3,8)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,8)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(5,0)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(2,8)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(4,8)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(5,8)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    #undef ROWCOL

    _mm256_storeu_ps(dst+0*dst_stride,    ymm0);  _mm256_storeu_ps(dst+1*dst_stride,    ymm1);  _mm256_storeu_ps(dst+2*dst_stride,    ymm2);
    _mm256_storeu_ps(dst+3*dst_stride,    ymm3);  _mm256_storeu_ps(dst+4*dst_stride,    ymm4);
    _mm256_storeu_ps(dst+0*dst_stride+8,  ymm5);  _mm256_storeu_ps(dst+1*dst_stride+8,  ymm6);  _mm256_storeu_ps(dst+2*dst_stride+8,  ymm7);
    _mm256_storeu_ps(dst+3*dst_stride+8,  ymm8);  _mm256_storeu_ps(dst+4*dst_stride+8,  ymm9);
}

void kernel_conv3_vert_5x16_upper(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    __m256 ymm0=_mm256_setzero_ps(), ymm1=_mm256_setzero_ps(), ymm2=_mm256_setzero_ps(), ymm3=_mm256_setzero_ps(), ymm4=_mm256_setzero_ps(),
           ymm5=_mm256_setzero_ps(), ymm6=_mm256_setzero_ps(), ymm7=_mm256_setzero_ps(), ymm8=_mm256_setzero_ps(), ymm9=_mm256_setzero_ps();
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(0,8)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(2,8)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(1,8)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(3,8)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,8)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(5,0)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(2,8)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(4,8)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(5,8)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    #undef ROWCOL

    _mm256_storeu_ps(dst+0*dst_stride,    ymm0);  _mm256_storeu_ps(dst+1*dst_stride,    ymm1);  _mm256_storeu_ps(dst+2*dst_stride,    ymm2);
    _mm256_storeu_ps(dst+3*dst_stride,    ymm3);  _mm256_storeu_ps(dst+4*dst_stride,    ymm4);
    _mm256_storeu_ps(dst+0*dst_stride+8,  ymm5);  _mm256_storeu_ps(dst+1*dst_stride+8,  ymm6);  _mm256_storeu_ps(dst+2*dst_stride+8,  ymm7);
    _mm256_storeu_ps(dst+3*dst_stride+8,  ymm8);  _mm256_storeu_ps(dst+4*dst_stride+8,  ymm9);
}

void kernel_conv3_vert_5x16_lower(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    __m256 ymm0=_mm256_setzero_ps(), ymm1=_mm256_setzero_ps(), ymm2=_mm256_setzero_ps(), ymm3=_mm256_setzero_ps(), ymm4=_mm256_setzero_ps(),
           ymm5=_mm256_setzero_ps(), ymm6=_mm256_setzero_ps(), ymm7=_mm256_setzero_ps(), ymm8=_mm256_setzero_ps(), ymm9=_mm256_setzero_ps();
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    ymm15=_mm256_loadu_ps(ROWCOL(-1,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(-1,8)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(0,8)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(2,8)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(1,8)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(3,8)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,8)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(1,8)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(2,8)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(3,8)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(4,8)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,8)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    #undef ROWCOL

    _mm256_storeu_ps(dst+0*dst_stride,    ymm0);  _mm256_storeu_ps(dst+1*dst_stride,    ymm1);  _mm256_storeu_ps(dst+2*dst_stride,    ymm2);
    _mm256_storeu_ps(dst+3*dst_stride,    ymm3);  _mm256_storeu_ps(dst+4*dst_stride,    ymm4);
    _mm256_storeu_ps(dst+0*dst_stride+8,  ymm5);  _mm256_storeu_ps(dst+1*dst_stride+8,  ymm6);  _mm256_storeu_ps(dst+2*dst_stride+8,  ymm7);
    _mm256_storeu_ps(dst+3*dst_stride+8,  ymm8);  _mm256_storeu_ps(dst+4*dst_stride+8,  ymm9);
}

void kernel_conv3_horiz_5x16(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    __m256 ymm0=_mm256_setzero_ps(), ymm1=_mm256_setzero_ps(), ymm2=_mm256_setzero_ps(), ymm3=_mm256_setzero_ps(), ymm4=_mm256_setzero_ps(),
           ymm5=_mm256_setzero_ps(), ymm6=_mm256_setzero_ps(), ymm7=_mm256_setzero_ps(), ymm8=_mm256_setzero_ps(), ymm9=_mm256_setzero_ps();
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    ymm15=_mm256_loadu_ps(ROWCOL(0,-1)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,-1)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,-1)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,-1)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,-1)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,7)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(1,7)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,7)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(3,7)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,7)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(1,8)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(3,8)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,8)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,1)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,1)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,1)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,1)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,1)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,9)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(1,9)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,9)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(3,9)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,9)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    #undef ROWCOL

    _mm256_storeu_ps(dst+0*dst_stride,    ymm0);  _mm256_storeu_ps(dst+1*dst_stride,    ymm1);  _mm256_storeu_ps(dst+2*dst_stride,    ymm2);
    _mm256_storeu_ps(dst+3*dst_stride,    ymm3);  _mm256_storeu_ps(dst+4*dst_stride,    ymm4);
    _mm256_storeu_ps(dst+0*dst_stride+8,  ymm5);  _mm256_storeu_ps(dst+1*dst_stride+8,  ymm6);  _mm256_storeu_ps(dst+2*dst_stride+8,  ymm7);
    _mm256_storeu_ps(dst+3*dst_stride+8,  ymm8);  _mm256_storeu_ps(dst+4*dst_stride+8,  ymm9);
}

void kernel_conv3_horiz_5x16_left(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    __m256 ymm0=_mm256_setzero_ps(), ymm1=_mm256_setzero_ps(), ymm2=_mm256_setzero_ps(), ymm3=_mm256_setzero_ps(), ymm4=_mm256_setzero_ps(),
           ymm5=_mm256_setzero_ps(), ymm6=_mm256_setzero_ps(), ymm7=_mm256_setzero_ps(), ymm8=_mm256_setzero_ps(), ymm9=_mm256_setzero_ps();
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    memcpy(dst+0*dst_stride+1,ROWCOL(0,0),7*sizeof(float)); *(dst+0*dst_stride) = *(dst+0*dst_stride+1);
    memcpy(dst+1*dst_stride+1,ROWCOL(1,0),7*sizeof(float)); *(dst+1*dst_stride) = *(dst+1*dst_stride+1);
    memcpy(dst+2*dst_stride+1,ROWCOL(2,0),7*sizeof(float)); *(dst+2*dst_stride) = *(dst+2*dst_stride+1);
    memcpy(dst+3*dst_stride+1,ROWCOL(3,0),7*sizeof(float)); *(dst+3*dst_stride) = *(dst+3*dst_stride+1);
    memcpy(dst+4*dst_stride+1,ROWCOL(4,0),7*sizeof(float)); *(dst+4*dst_stride) = *(dst+4*dst_stride+1);
    ymm0=_mm256_loadu_ps(dst+0*dst_stride); ymm1=_mm256_loadu_ps(dst+1*dst_stride); ymm2=_mm256_loadu_ps(dst+2*dst_stride);
    ymm3=_mm256_loadu_ps(dst+3*dst_stride); ymm4=_mm256_loadu_ps(dst+4*dst_stride);
    ymm0=_mm256_fmadd_ps(ymm13,ymm0,ymm5); ymm1=_mm256_fmadd_ps(ymm13,ymm1,ymm6); ymm2=_mm256_fmadd_ps(ymm13,ymm2,ymm7);
    ymm3=_mm256_fmadd_ps(ymm13,ymm3,ymm8); ymm4=_mm256_fmadd_ps(ymm13,ymm4,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,7)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(1,7)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,7)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(3,7)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,7)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(1,8)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(3,8)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,8)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,1)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,1)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,1)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,1)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,1)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,9)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(1,9)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,9)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(3,9)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,9)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    #undef ROWCOL

    _mm256_storeu_ps(dst+0*dst_stride,    ymm0);  _mm256_storeu_ps(dst+1*dst_stride,    ymm1);  _mm256_storeu_ps(dst+2*dst_stride,    ymm2);
    _mm256_storeu_ps(dst+3*dst_stride,    ymm3);  _mm256_storeu_ps(dst+4*dst_stride,    ymm4);
    _mm256_storeu_ps(dst+0*dst_stride+8,  ymm5);  _mm256_storeu_ps(dst+1*dst_stride+8,  ymm6);  _mm256_storeu_ps(dst+2*dst_stride+8,  ymm7);
    _mm256_storeu_ps(dst+3*dst_stride+8,  ymm8);  _mm256_storeu_ps(dst+4*dst_stride+8,  ymm9);
}

void kernel_conv3_horiz_5x16_right(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride)
{
    __m256 ymm0=_mm256_setzero_ps(), ymm1=_mm256_setzero_ps(), ymm2=_mm256_setzero_ps(), ymm3=_mm256_setzero_ps(), ymm4=_mm256_setzero_ps(),
           ymm5=_mm256_setzero_ps(), ymm6=_mm256_setzero_ps(), ymm7=_mm256_setzero_ps(), ymm8=_mm256_setzero_ps(), ymm9=_mm256_setzero_ps();
    const __m256 ymm13=_mm256_set1_ps(0.25f), ymm14=_mm256_set1_ps(0.5f);
    __m256 ymm15;

    #define ROWCOL(s,t) (src + (s)*(ptrdiff_t)src_stride + (t)*(ptrdiff_t)1)

    memcpy(dst+0*dst_stride+8,ROWCOL(0,9),7*sizeof(float)); *(dst+0*dst_stride+15) = *(dst+0*dst_stride+14);
    memcpy(dst+1*dst_stride+8,ROWCOL(1,9),7*sizeof(float)); *(dst+1*dst_stride+15) = *(dst+1*dst_stride+14);
    memcpy(dst+2*dst_stride+8,ROWCOL(2,9),7*sizeof(float)); *(dst+2*dst_stride+15) = *(dst+2*dst_stride+14);
    memcpy(dst+3*dst_stride+8,ROWCOL(3,9),7*sizeof(float)); *(dst+3*dst_stride+15) = *(dst+3*dst_stride+14);
    memcpy(dst+4*dst_stride+8,ROWCOL(4,9),7*sizeof(float)); *(dst+4*dst_stride+15) = *(dst+4*dst_stride+14);
    ymm5=_mm256_loadu_ps(dst+0*dst_stride+8); ymm6=_mm256_loadu_ps(dst+1*dst_stride+8); ymm7=_mm256_loadu_ps(dst+2*dst_stride+8);
    ymm8=_mm256_loadu_ps(dst+3*dst_stride+8); ymm9=_mm256_loadu_ps(dst+4*dst_stride+8);
    ymm5=_mm256_fmadd_ps(ymm13,ymm5,ymm0); ymm6=_mm256_fmadd_ps(ymm13,ymm6,ymm1); ymm7=_mm256_fmadd_ps(ymm13,ymm7,ymm2);
    ymm8=_mm256_fmadd_ps(ymm13,ymm8,ymm3); ymm9=_mm256_fmadd_ps(ymm13,ymm9,ymm4);

    ymm15=_mm256_loadu_ps(ROWCOL(0,-1)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,-1)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,-1)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,-1)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,-1)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,7)); ymm5=_mm256_fmadd_ps(ymm13,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(1,7)); ymm6=_mm256_fmadd_ps(ymm13,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,7)); ymm7=_mm256_fmadd_ps(ymm13,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(3,7)); ymm8=_mm256_fmadd_ps(ymm13,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,7)); ymm9=_mm256_fmadd_ps(ymm13,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,0)); ymm0=_mm256_fmadd_ps(ymm14,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,0)); ymm1=_mm256_fmadd_ps(ymm14,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,0)); ymm2=_mm256_fmadd_ps(ymm14,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,0)); ymm3=_mm256_fmadd_ps(ymm14,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,0)); ymm4=_mm256_fmadd_ps(ymm14,ymm15,ymm4);
    ymm15=_mm256_loadu_ps(ROWCOL(0,8)); ymm5=_mm256_fmadd_ps(ymm14,ymm15,ymm5); ymm15=_mm256_loadu_ps(ROWCOL(1,8)); ymm6=_mm256_fmadd_ps(ymm14,ymm15,ymm6);
    ymm15=_mm256_loadu_ps(ROWCOL(2,8)); ymm7=_mm256_fmadd_ps(ymm14,ymm15,ymm7); ymm15=_mm256_loadu_ps(ROWCOL(3,8)); ymm8=_mm256_fmadd_ps(ymm14,ymm15,ymm8);
    ymm15=_mm256_loadu_ps(ROWCOL(4,8)); ymm9=_mm256_fmadd_ps(ymm14,ymm15,ymm9);

    ymm15=_mm256_loadu_ps(ROWCOL(0,1)); ymm0=_mm256_fmadd_ps(ymm13,ymm15,ymm0); ymm15=_mm256_loadu_ps(ROWCOL(1,1)); ymm1=_mm256_fmadd_ps(ymm13,ymm15,ymm1);
    ymm15=_mm256_loadu_ps(ROWCOL(2,1)); ymm2=_mm256_fmadd_ps(ymm13,ymm15,ymm2); ymm15=_mm256_loadu_ps(ROWCOL(3,1)); ymm3=_mm256_fmadd_ps(ymm13,ymm15,ymm3);
    ymm15=_mm256_loadu_ps(ROWCOL(4,1)); ymm4=_mm256_fmadd_ps(ymm13,ymm15,ymm4);


    #undef ROWCOL

    _mm256_storeu_ps(dst+0*dst_stride,    ymm0);  _mm256_storeu_ps(dst+1*dst_stride,    ymm1);  _mm256_storeu_ps(dst+2*dst_stride,    ymm2);
    _mm256_storeu_ps(dst+3*dst_stride,    ymm3);  _mm256_storeu_ps(dst+4*dst_stride,    ymm4);
    _mm256_storeu_ps(dst+0*dst_stride+8,  ymm5);  _mm256_storeu_ps(dst+1*dst_stride+8,  ymm6);  _mm256_storeu_ps(dst+2*dst_stride+8,  ymm7);
    _mm256_storeu_ps(dst+3*dst_stride+8,  ymm8);  _mm256_storeu_ps(dst+4*dst_stride+8,  ymm9);
}


void gaussian_blur_5x16_lowload(float input[HEIGHT][WIDTH],
                                      float output[HEIGHT][WIDTH])
{
    const int src_stride = WIDTH;  // in floats
    const int dst_stride = WIDTH;  // in floats
    const int block_h    = 5;
    const int block_w    = 16;
    const int num_blocks = HEIGHT / block_h;
    float tmp[HEIGHT][WIDTH];

    int row0 = 0 * block_h;
    for (int c = 0; c < WIDTH; c += block_w) {
        const float *src_block = &input[row0][c];
        float       *dst_block = &tmp[row0][c];

        kernel_conv3_vert_5x16_lowload_upper(
            src_block,
            src_stride,
            dst_block,
            dst_stride
        );
    }

    // Loop over row blocks of height 13
    for (int b = 1; b < num_blocks - 1; ++b) {
        row0 = b * block_h;  // first output row of this 13-row block

        // Now sweep across columns in chunks of 8
        for (int c = 0; c < WIDTH; c += block_w) {
            const float *src_block = &input[row0][c];
            float       *dst_block = &tmp[row0][c];

            kernel_conv3_vert_5x16_lowload(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }
    }

    row0 = (num_blocks - 1) * block_h;
    for (int c = 0; c < WIDTH; c += block_w) {
        const float *src_block = &input[row0][c];
        float       *dst_block = &tmp[row0][c];

        kernel_conv3_vert_5x16_lowload_lower(
            src_block,
            src_stride,
            dst_block,
            dst_stride
        );
    }

    // Loop over row blocks of height 13
    for (int b = 0; b < num_blocks; ++b) {
        row0 = b * block_h;  // first output row of this 13-row block

        // Now sweep across columns in chunks of 8
        for (int c = 0; c < block_w; c += block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16_left(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Now sweep across columns in chunks of 8
        for (int c = block_w; c < WIDTH - block_w; c += block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Now sweep across columns in chunks of 8
        for (int c = WIDTH - block_w; c < WIDTH; c += block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16_right(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

    }
}

void gaussian_blur_5x16(float input[HEIGHT][WIDTH],
                                      float output[HEIGHT][WIDTH])
{
    const int src_stride = WIDTH;  // in floats
    const int dst_stride = WIDTH;  // in floats
    const int block_h    = 5;
    const int block_w    = 16;
    const int num_blocks = HEIGHT / block_h;
    float tmp[HEIGHT][WIDTH];

    int row0 = 0 * block_h;
    for (int c = 0; c < WIDTH; c += block_w) {
        const float *src_block = &input[row0][c];
        float       *dst_block = &tmp[row0][c];

        kernel_conv3_vert_5x16_upper(
            src_block,
            src_stride,
            dst_block,
            dst_stride
        );
    }

    // Loop over row blocks of height 13
    for (int b = 1; b < num_blocks - 1; ++b) {
        row0 = b * block_h;  // first output row of this 13-row block

        // Now sweep across columns in chunks of 8
        for (int c = 0; c < WIDTH; c += block_w) {
            const float *src_block = &input[row0][c];
            float       *dst_block = &tmp[row0][c];

            kernel_conv3_vert_5x16(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }
    }

    row0 = (num_blocks - 1) * block_h;
    for (int c = 0; c < WIDTH; c += block_w) {
        const float *src_block = &input[row0][c];
        float       *dst_block = &tmp[row0][c];

        kernel_conv3_vert_5x16_lower(
            src_block,
            src_stride,
            dst_block,
            dst_stride
        );
    }

    // Loop over row blocks of height 13
    for (int b = 0; b < num_blocks; ++b) {
        row0 = b * block_h;  // first output row of this 13-row block

        // Now sweep across columns in chunks of 8
        for (int c = 0; c < block_w; c += block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16_left(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Now sweep across columns in chunks of 8
        for (int c = block_w; c < WIDTH - block_w; c += block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

        // Now sweep across columns in chunks of 8
        for (int c = WIDTH - block_w; c < WIDTH; c += block_w) {
            const float *src_block = &tmp[row0][c];
            float       *dst_block = &output[row0][c];

            kernel_conv3_horiz_5x16_right(
                src_block,
                src_stride,
                dst_block,
                dst_stride
            );
        }

    }
}


// --------------------------------------------------------
// Timing function
// --------------------------------------------------------
unsigned long long rdtsc() {
    unsigned a, d;
    __asm__ volatile("rdtsc" : "=a" (a), "=d" (d));
    return ((unsigned long long)a) | (((unsigned long long)d) << 32);
}

// --------------------------------------------------------
// Function prototype for each person's implementation
// --------------------------------------------------------
static inline void gaussian_blur(float input[HEIGHT][WIDTH], float output[HEIGHT][WIDTH]);

// --------------------------------------------------------
// Utility: fill image with random values between 0–255
// --------------------------------------------------------
void fill_random_image(float img[HEIGHT][WIDTH]) {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            img[i][j] = (float)(rand() % 256);
        }
    }
}

// --------------------------------------------------------
// Utility: compare results (for correctness check)
// --------------------------------------------------------
float compare_images(float ref[HEIGHT][WIDTH], float test[HEIGHT][WIDTH]) {
    float diff = 0.0f;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            diff += fabsf(ref[i][j] - test[i][j]);
        }
    }
    return diff / (WIDTH * HEIGHT);
}

// --------------------------------------------------------
// Reference Gaussian blur (3x3 separable kernel, sigma ≈ 1.0)
// --------------------------------------------------------
static inline void gaussian_blur_reference(float input[HEIGHT][WIDTH], float output[HEIGHT][WIDTH]) {
    const float k0f = 0.25f, k1f = 0.5f, k2f = 0.25f;
    static float tmp[HEIGHT][WIDTH];

    for (int r = 0; r < HEIGHT; ++r) {
        const float *row = &input[r][0];
        float *dst = &tmp[r][0];

        // Handle left edge
        dst[0] = k0f * row[0] + k1f * row[0] + k2f * row[1];

        // Vectorizable, branchless main body
        for (int c = 1; c < WIDTH - 1; ++c) {
            dst[c] = k0f * row[c-1] + k1f * row[c] + k2f * row[c+1];
        }

        // Handle right edge
        dst[WIDTH - 1] = k0f * row[WIDTH - 2] + k1f * row[WIDTH - 1] + k2f * row[WIDTH - 1];
    }

    for (int r = 0; r < HEIGHT; ++r) {
        int rm1 = (r == 0) ? 0 : (r - 1);
        int rp1 = (r == HEIGHT - 1) ? (HEIGHT - 1) : (r + 1);
        const float *row_m1 = &tmp[rm1][0];
        const float *row_0  = &tmp[r][0];
        const float *row_p1 = &tmp[rp1][0];
        float *dst = &output[r][0];

        for (int c = 0; c < WIDTH; ++c) {
            dst[c] = k0f * row_m1[c] + k1f * row_0[c] + k2f * row_p1[c];
        }
    }
}



// --------------------------------------------------------
// Main testing harness
// --------------------------------------------------------
int main() {
    static float input[HEIGHT][WIDTH];
    static float output[HEIGHT][WIDTH];
    static float reference[HEIGHT][WIDTH];

    srand((unsigned)time(NULL));
    fill_random_image(input);

    // Measure custom implementation
    unsigned long long t0, t1;
    double cycles;
    double sums;



    t0 = rdtsc();
    // Compute reference result
    gaussian_blur_5x16(input, output);
    t1 = rdtsc();

    cycles = (double)(t1 - t0);
    printf("Gaussian blur full took %.0f cycles\n", cycles);
    printf(" %lf\n", (12.0*HEIGHT*WIDTH)/((double)(cycles/(1.0))));

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        // Compute reference result
        gaussian_blur_5x16(input, output);
        t1 = rdtsc();

        sums += (double)(t1 - t0);
    }
    printf("Gaussian blur full took %.0f cycles\n", sums/(1.0*RUNS));
    printf(" %lf\n", (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS))));
    sums = .0f;







    t0 = rdtsc();
    // Compute reference result
    gaussian_blur_5x16_lowload(input, output);
    t1 = rdtsc();

    cycles = (double)(t1 - t0);
    printf("Gaussian blur full took %.0f cycles\n", cycles);
    printf(" %lf\n", (12.0*HEIGHT*WIDTH)/((double)(cycles/(1.0))));

    sums = .0f;
    for (size_t i = 0; i < RUNS; ++i) {
        t0 = rdtsc();
        // Compute reference result
        gaussian_blur_5x16_lowload(input, output);
        t1 = rdtsc();

        sums += (double)(t1 - t0);
    }
    printf("Gaussian blur full took %.0f cycles\n", sums/(1.0*RUNS));
    printf(" %lf\n", (12.0*HEIGHT*WIDTH)/((double)(sums/(1.0*RUNS))));
    sums = .0f;







   

    gaussian_blur_reference(input, reference);

    t0 = rdtsc();
    // Compute reference result
    gaussian_blur_reference(input, reference);
    t1 = rdtsc();

    cycles = (double)(t1 - t0);
    printf("Gaussian blur fast reference took %.0f cycles\n", cycles);

   
    // Check correctness
    float diff = compare_images(reference, output);
    printf("Average difference from reference: %.6f\n", diff);

    return 0;
}


// --------------------------------------------------------
// Example baseline (each person replaces this with their own)
// --------------------------------------------------------
static inline void gaussian_blur(float input[HEIGHT][WIDTH], float output[HEIGHT][WIDTH]) {
    // TODO: Replace this with your optimized version
    // For now, just call reference to verify correctness
    const float kernel[3] = {0.25f, 0.5f, 0.25f}; // normalized 1D kernel

    static float temp[HEIGHT][WIDTH];

    // horiztal pass
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            float acc = 0.0f;
            for (int k = -1; k <= 1; k++) {
                int jj = j + k;
                if (jj < 0) jj = 0;
                if (jj >= WIDTH) jj = WIDTH - 1;
                acc += input[i][jj] * kernel[k + 1];
            }
            temp[i][j] = acc;
        }
    }

    // Vertical pass
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            float acc = 0.0f;
            for (int k = -1; k <= 1; k++) {
                int ii = i + k;
                if (ii < 0) ii = 0;
                if (ii >= HEIGHT) ii = HEIGHT - 1;
                acc += temp[ii][j] * kernel[k + 1];
            }
            output[i][j] = acc;
        }
    }
}
