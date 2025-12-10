#include "sum_of_squares.h"
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// Optimized AVX2 implementation with 12-accumulator chains
void kernel2_sum_of_squares(float* gx, float* gy, float* mag2, int size) {
    int i = 0;
    // Process in blocks of 96 floats = 12 vectors of 8 floats each
    int vec_end = (size / 96) * 96;
    for (; i < vec_end; i += 96) {
        // Chain 0
        __m256 r0  = _mm256_loadu_ps(gx + i +  0);
        __m256 tmp = _mm256_loadu_ps(gy + i +  0);
        tmp = _mm256_mul_ps(tmp, tmp);
        r0  = _mm256_fmadd_ps(r0, r0, tmp);

        // Chain 1
        __m256 r1  = _mm256_loadu_ps(gx + i +  8);
        tmp = _mm256_loadu_ps(gy + i +  8);
        tmp = _mm256_mul_ps(tmp, tmp);
        r1  = _mm256_fmadd_ps(r1, r1, tmp);

        // Chain 2
        __m256 r2  = _mm256_loadu_ps(gx + i + 16);
        tmp = _mm256_loadu_ps(gy + i + 16);
        tmp = _mm256_mul_ps(tmp, tmp);
        r2  = _mm256_fmadd_ps(r2, r2, tmp);

        // Chain 3
        __m256 r3  = _mm256_loadu_ps(gx + i + 24);
        tmp = _mm256_loadu_ps(gy + i + 24);
        tmp = _mm256_mul_ps(tmp, tmp);
        r3  = _mm256_fmadd_ps(r3, r3, tmp);

        // Chain 4
        __m256 r4  = _mm256_loadu_ps(gx + i + 32);
        tmp = _mm256_loadu_ps(gy + i + 32);
        tmp = _mm256_mul_ps(tmp, tmp);
        r4  = _mm256_fmadd_ps(r4, r4, tmp);

        // Chain 5
        __m256 r5  = _mm256_loadu_ps(gx + i + 40);
        tmp = _mm256_loadu_ps(gy + i + 40);
        tmp = _mm256_mul_ps(tmp, tmp);
        r5  = _mm256_fmadd_ps(r5, r5, tmp);

        // Chain 6
        __m256 r6  = _mm256_loadu_ps(gx + i + 48);
        tmp = _mm256_loadu_ps(gy + i + 48);
        tmp = _mm256_mul_ps(tmp, tmp);
        r6  = _mm256_fmadd_ps(r6, r6, tmp);

        // Chain 7
        __m256 r7  = _mm256_loadu_ps(gx + i + 56);
        tmp = _mm256_loadu_ps(gy + i + 56);
        tmp = _mm256_mul_ps(tmp, tmp);
        r7  = _mm256_fmadd_ps(r7, r7, tmp);

        // Chain 8
        __m256 r8  = _mm256_loadu_ps(gx + i + 64);
        tmp = _mm256_loadu_ps(gy + i + 64);
        tmp = _mm256_mul_ps(tmp, tmp);
        r8  = _mm256_fmadd_ps(r8, r8, tmp);

        // Chain 9
        __m256 r9  = _mm256_loadu_ps(gx + i + 72);
        tmp = _mm256_loadu_ps(gy + i + 72);
        tmp = _mm256_mul_ps(tmp, tmp);
        r9  = _mm256_fmadd_ps(r9, r9, tmp);

        // Chain 10
        __m256 r10 = _mm256_loadu_ps(gx + i + 80);
        tmp = _mm256_loadu_ps(gy + i + 80);
        tmp = _mm256_mul_ps(tmp, tmp);
        r10 = _mm256_fmadd_ps(r10, r10, tmp);

        // Chain 11
        __m256 r11 = _mm256_loadu_ps(gx + i + 88);
        tmp = _mm256_loadu_ps(gy + i + 88);
        tmp = _mm256_mul_ps(tmp, tmp);
        r11 = _mm256_fmadd_ps(r11, r11, tmp);

        // Store all 12 output vectors
        _mm256_storeu_ps(mag2 + i +  0, r0);
        _mm256_storeu_ps(mag2 + i +  8, r1);
        _mm256_storeu_ps(mag2 + i + 16, r2);
        _mm256_storeu_ps(mag2 + i + 24, r3);
        _mm256_storeu_ps(mag2 + i + 32, r4);
        _mm256_storeu_ps(mag2 + i + 40, r5);
        _mm256_storeu_ps(mag2 + i + 48, r6);
        _mm256_storeu_ps(mag2 + i + 56, r7);
        _mm256_storeu_ps(mag2 + i + 64, r8);
        _mm256_storeu_ps(mag2 + i + 72, r9);
        _mm256_storeu_ps(mag2 + i + 80, r10);
        _mm256_storeu_ps(mag2 + i + 88, r11);
    }

    // Scalar tail
    for (; i < size; i++) {
        float gxv = gx[i];
        float gyv = gy[i];
        mag2[i] = gxv * gxv + gyv * gyv;
    }
}

// Parallel driver that calls the 12-accumulator kernel on chunks of the array
void kernel2_sum_of_squares_omp(float* gx, float* gy, float* mag2, int size) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        int chunk = (size + nthreads - 1) / nthreads;
        int start = tid * chunk;
        int end   = start + chunk;
        if (end > size) end = size;

        // Vectorized part per thread
        int i = start;
        int vec_end = start + ((end - start) / 96) * 96;
        for (; i < vec_end; i += 96) {
            __m256 r0  = _mm256_loadu_ps(gx + i +  0);
            __m256 tmp = _mm256_loadu_ps(gy + i +  0);
            tmp = _mm256_mul_ps(tmp, tmp);
            r0  = _mm256_fmadd_ps(r0, r0, tmp);

            __m256 r1  = _mm256_loadu_ps(gx + i +  8);
            tmp = _mm256_loadu_ps(gy + i +  8);
            tmp = _mm256_mul_ps(tmp, tmp);
            r1  = _mm256_fmadd_ps(r1, r1, tmp);

            __m256 r2  = _mm256_loadu_ps(gx + i + 16);
            tmp = _mm256_loadu_ps(gy + i + 16);
            tmp = _mm256_mul_ps(tmp, tmp);
            r2  = _mm256_fmadd_ps(r2, r2, tmp);

            __m256 r3  = _mm256_loadu_ps(gx + i + 24);
            tmp = _mm256_loadu_ps(gy + i + 24);
            tmp = _mm256_mul_ps(tmp, tmp);
            r3  = _mm256_fmadd_ps(r3, r3, tmp);

            __m256 r4  = _mm256_loadu_ps(gx + i + 32);
            tmp = _mm256_loadu_ps(gy + i + 32);
            tmp = _mm256_mul_ps(tmp, tmp);
            r4  = _mm256_fmadd_ps(r4, r4, tmp);

            __m256 r5  = _mm256_loadu_ps(gx + i + 40);
            tmp = _mm256_loadu_ps(gy + i + 40);
            tmp = _mm256_mul_ps(tmp, tmp);
            r5  = _mm256_fmadd_ps(r5, r5, tmp);

            __m256 r6  = _mm256_loadu_ps(gx + i + 48);
            tmp = _mm256_loadu_ps(gy + i + 48);
            tmp = _mm256_mul_ps(tmp, tmp);
            r6  = _mm256_fmadd_ps(r6, r6, tmp);

            __m256 r7  = _mm256_loadu_ps(gx + i + 56);
            tmp = _mm256_loadu_ps(gy + i + 56);
            tmp = _mm256_mul_ps(tmp, tmp);
            r7  = _mm256_fmadd_ps(r7, r7, tmp);

            __m256 r8  = _mm256_loadu_ps(gx + i + 64);
            tmp = _mm256_loadu_ps(gy + i + 64);
            tmp = _mm256_mul_ps(tmp, tmp);
            r8  = _mm256_fmadd_ps(r8, r8, tmp);

            __m256 r9  = _mm256_loadu_ps(gx + i + 72);
            tmp = _mm256_loadu_ps(gy + i + 72);
            tmp = _mm256_mul_ps(tmp, tmp);
            r9  = _mm256_fmadd_ps(r9, r9, tmp);

            __m256 r10 = _mm256_loadu_ps(gx + i + 80);
            tmp = _mm256_loadu_ps(gy + i + 80);
            tmp = _mm256_mul_ps(tmp, tmp);
            r10 = _mm256_fmadd_ps(r10, r10, tmp);

            __m256 r11 = _mm256_loadu_ps(gx + i + 88);
            tmp = _mm256_loadu_ps(gy + i + 88);
            tmp = _mm256_mul_ps(tmp, tmp);
            r11 = _mm256_fmadd_ps(r11, r11, tmp);

            _mm256_storeu_ps(mag2 + i +  0, r0);
            _mm256_storeu_ps(mag2 + i +  8, r1);
            _mm256_storeu_ps(mag2 + i + 16, r2);
            _mm256_storeu_ps(mag2 + i + 24, r3);
            _mm256_storeu_ps(mag2 + i + 32, r4);
            _mm256_storeu_ps(mag2 + i + 40, r5);
            _mm256_storeu_ps(mag2 + i + 48, r6);
            _mm256_storeu_ps(mag2 + i + 56, r7);
            _mm256_storeu_ps(mag2 + i + 64, r8);
            _mm256_storeu_ps(mag2 + i + 72, r9);
            _mm256_storeu_ps(mag2 + i + 80, r10);
            _mm256_storeu_ps(mag2 + i + 88, r11);
        }

        // Scalar tail for this thread's chunk
        for (; i < end; i++) {
            float gxv = gx[i];
            float gyv = gy[i];
            mag2[i] = gxv * gxv + gyv * gyv;
        }
    }
}

// Scalar reference implementation
// Disable auto-vectorization for true scalar baseline
#ifdef __GNUC__
#pragma GCC push_options
#pragma GCC optimize ("no-tree-vectorize")
#endif

#ifdef __clang__
#pragma clang optimize off
#endif

void kernel2_sum_of_squares_reference(float* gx, float* gy, float* mag2, int size) {
    volatile float *vgx = gx;
    volatile float *vgy = gy;
    volatile float *vmag2 = mag2;

    #ifdef __clang__
    #pragma clang loop vectorize(disable)
    #endif
    for (int i = 0; i < size; i++) {
        float gxv = vgx[i];
        float gyv = vgy[i];
        vmag2[i] = gxv * gxv + gyv * gyv;
    }
}

#ifdef __clang__
#pragma clang optimize on
#endif

#ifdef __GNUC__
#pragma GCC pop_options
#endif

