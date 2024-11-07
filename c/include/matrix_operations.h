#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <rocblas/rocblas.h>
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>

// Existing function declarations
void initialize_matrices(float *A, float *B, float *bias, int m, int k, int n);
void transpose_matrix(const float *src, float *dst, int rows, int cols);
void perform_matrix_multiplication(rocblas_handle handle, float *d_A, float *d_B, float *d_C,
                                 float *d_bias, int m, int n, int k, int NUM_RUNS);

// New hipblasLt function declaration
void perform_matrix_multiplication_hipblaslt(hipblasLtHandle_t handle, float *d_A, float *d_B, float *d_C,
                                           int m, int n, int k, int NUM_RUNS);

void print_debug(const char* name, const float* data, int num_elements);

#endif // MATRIX_OPERATIONS_H
