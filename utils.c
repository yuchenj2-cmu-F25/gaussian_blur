#include "utils.h"
#include <stdlib.h>
#include <math.h>

unsigned long long rdtsc(void) {
    unsigned a, d;
    __asm__ volatile("rdtsc" : "=a" (a), "=d" (d));
    return ((unsigned long long)a) | (((unsigned long long)d) << 32);
}

void fill_random_image(float img[HEIGHT][WIDTH]) {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            img[i][j] = (float)(rand() % 256);
        }
    }
}

float compare_images(float ref[HEIGHT][WIDTH], float test[HEIGHT][WIDTH]) {
    float diff = 0.0f;
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            diff += fabsf(ref[i][j] - test[i][j]);
        }
    }
    return diff / (WIDTH * HEIGHT);
}
