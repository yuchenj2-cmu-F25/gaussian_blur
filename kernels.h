#ifndef KERNELS_H
#define KERNELS_H

/* Vertical convolution kernels */
void kernel_conv3_vert_5x16_lowload(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

void kernel_conv3_vert_5x16_lowload_upper(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

void kernel_conv3_vert_5x16_lowload_lower(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

void kernel_conv3_vert_5x16(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

void kernel_conv3_vert_5x16_upper(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

void kernel_conv3_vert_5x16_lower(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

/* Horizontal convolution kernels */
void kernel_conv3_horiz_5x16(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

void kernel_conv3_horiz_5x16_left(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

void kernel_conv3_horiz_5x16_right(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

/* 10x8 Vertical convolution kernels */
void kernel_conv3_vert_10x8(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

void kernel_conv3_vert_10x8_upper(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

void kernel_conv3_vert_10x8_lower(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

/* 4x80 Vertical convolution kernels (unrolled) */
void kernel_conv3_vert_4x80(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

void kernel_conv3_vert_4x80_upper(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

void kernel_conv3_vert_4x80_lower(const float *restrict src, int src_stride,
                        float *restrict dst, int dst_stride);

/* 2D 24x80 convolution kernel (combined horizontal and vertical) */
void kernel_conv3_2d_24x80(const float *restrict src, int src_stride,
                           float *restrict dst, int dst_stride);

#endif /* KERNELS_H */
