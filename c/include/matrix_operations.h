#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H
#include <rocblas/rocblas.h>
void initialize_matrices(float *A, float *B, int m, int k, int n);
void transpose_matrix(const float *src, float *dst, int rows, int cols);
void perform_matrix_multiplication(rocblas_handle handle, float *d_A, float *d_B, float *d_C,
                                 int m, int n, int k, int NUM_RUNS);
#endif // MATRIX_OPERATIONS_H
