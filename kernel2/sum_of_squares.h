#ifndef SUM_OF_SQUARES_H
#define SUM_OF_SQUARES_H

#include "config.h"

/* Sum of squares implementations */
void kernel2_sum_of_squares(float* gx, float* gy, float* mag2, int size);

void kernel2_sum_of_squares_omp(float* gx, float* gy, float* mag2, int size);

void kernel2_sum_of_squares_reference(float* gx, float* gy, float* mag2, int size);

#endif /* SUM_OF_SQUARES_H */

