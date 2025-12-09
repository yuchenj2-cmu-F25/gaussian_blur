#ifndef BLUR_H
#define BLUR_H

#include "config.h"

/* Gaussian blur implementations */
void gaussian_blur_10x8(float input[HEIGHT][WIDTH],
                        float output[HEIGHT][WIDTH]);

void gaussian_blur_5x16_lowload(float input[HEIGHT][WIDTH],
                                float output[HEIGHT][WIDTH]);

void gaussian_blur_5x16(float input[HEIGHT][WIDTH],
                        float output[HEIGHT][WIDTH]);

void gaussian_blur_4x80(float input[HEIGHT][WIDTH],
                        float output[HEIGHT][WIDTH]);

void gaussian_blur_2d_24x80(float input[HEIGHT][WIDTH],
                             float output[HEIGHT][WIDTH]);

void gaussian_blur_separable_24x80(float input[HEIGHT][WIDTH],
                                     float output[HEIGHT][WIDTH]);

void gaussian_blur_reference(float input[HEIGHT][WIDTH],
                             float output[HEIGHT][WIDTH]);

void gaussian_blur(float input[HEIGHT][WIDTH],
                   float output[HEIGHT][WIDTH]);

#endif /* BLUR_H */
