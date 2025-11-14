#ifndef UTILS_H
#define UTILS_H

#include "config.h"

/* Timing function */
unsigned long long rdtsc(void);

/* Utility functions */
void fill_random_image(float img[HEIGHT][WIDTH]);
float compare_images(float ref[HEIGHT][WIDTH], float test[HEIGHT][WIDTH]);

#endif /* UTILS_H */
