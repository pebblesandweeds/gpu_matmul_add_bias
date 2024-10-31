#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "../include/timer.h"
#include "../include/matrix_operations.h"
#include "../include/spot_check.h"
#include "../include/utils.h"

// Matrix dimensions
#define M 16384
#define K 4096
#define N 16384
#define NUM_RUNS 25

int main() {
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    size_t size_bias = N * sizeof(float);

    print_gpu_info();
    print_precision();

    float *h_A, *h_B, *h_C, *h_A_trans, *h_B_trans, *h_C_trans, *h_bias;
    float *d_A, *d_B, *d_C, *d_bias;

    // Allocate host memory
    h_A = (float*)malloc(size_a);
    h_B = (float*)malloc(size_b);
    h_C = (float*)malloc(size_c);
    h_A_trans = (float*)malloc(size_a);
    h_B_trans = (float*)malloc(size_b);
    h_C_trans = (float*)malloc(size_c);
    h_bias = (float*)malloc(size_bias);

    // Initialize matrices with dimensions
    initialize_matrices(h_A, h_B, h_bias, M, K, N);

    // Transpose matrices
    transpose_matrix(h_A, h_A_trans, M, K);
    transpose_matrix(h_B, h_B_trans, K, N);

    // Allocate device memory
    CHECK_HIP(hipMalloc(&d_A, size_a));
    CHECK_HIP(hipMalloc(&d_B, size_b));
    CHECK_HIP(hipMalloc(&d_C, size_c));
    CHECK_HIP(hipMalloc(&d_bias, size_bias));

    // Transfer data to device
    float transfer_time = time_memory_transfer(h_A_trans, h_B_trans, h_bias,
                                             d_A, d_B, d_bias,
                                             size_a, size_b, size_bias);
    printf("Memory transfer to device time: %f ms\n", transfer_time);

    // Perform matrix multiplication
    rocblas_handle handle;
    CHECK_ROCBLAS(rocblas_create_handle(&handle));
    perform_matrix_multiplication(handle, d_A, d_B, d_C, d_bias, M, N, K, NUM_RUNS);

    // Transfer results back
    float transfer_back_time = time_memory_transfer_back(h_C, d_C, size_c);
    printf("Memory transfer from device time: %f ms\n", transfer_back_time);

    spot_check(h_A, h_B, h_bias, h_C, M, N, K);

    free(h_bias);
    CHECK_HIP(hipFree(d_bias));
    cleanup(handle, d_A, d_B, d_C, h_A, h_B, h_C, h_A_trans, h_B_trans, h_C_trans);
    return 0;
}
