#include <immintrin.h>
#include <stddef.h>
#include <string.h>
#include "kernels.h"

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
