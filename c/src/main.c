#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include "../include/timer.h"
#include "../include/matrix_operations.h"
#include "../include/spot_check.h"
#include "../include/utils.h"

// Matrix dimensions
#define M 2
#define K 2
#define N 2
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

    // Debug print initial matrices
    print_debug("Matrix A initial", h_A, M * K);
    print_debug("Matrix B initial", h_B, K * N);
    print_debug("Bias initial", h_bias, N);  // New line for bias debug print

    // Transpose matrices
    transpose_matrix(h_A, h_A_trans, M, K);
    transpose_matrix(h_B, h_B_trans, K, N);

    // Debug print transposed matrices
    print_debug("Matrix A after transpose", h_A_trans, M * K);
    print_debug("Matrix B after transpose", h_B_trans, K * N);

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

    // Create handles for both rocBLAS and hipblasLt
    rocblas_handle rocblas_handle;
    hipblasLtHandle_t hipblaslt_handle;

    CHECK_ROCBLAS(rocblas_create_handle(&rocblas_handle));
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&hipblaslt_handle));

    // Perform matrix multiplication using rocBLAS
    printf("\n--- Running rocBLAS implementation ---\n");
    perform_matrix_multiplication(rocblas_handle, d_A, d_B, d_C, d_bias, M, N, K, NUM_RUNS);

    CHECK_HIP(hipMemcpy(h_C, d_C, size_c, hipMemcpyDeviceToHost));

    // Debug print final C matrix (after matmul and bias)
    print_debug("C matrix after rocBLAS (with bias)", h_C, M * N);

    // Verify results
    spot_check(h_A, h_B, h_bias, h_C, M, N, K);

    // Perform matrix multiplication using hipblasLt
    printf("\n--- Running hipblasLt implementation ---\n");
    perform_matrix_multiplication_hipblaslt(hipblaslt_handle, d_A, d_B, d_C, M, N, K, NUM_RUNS);

    // Transfer results back
    float transfer_back_time = time_memory_transfer_back(h_C, d_C, size_c);
    printf("\nMemory transfer from device time: %f ms\n", transfer_back_time);

    // Debug print C matrix from hipblasLt
    print_debug("C matrix after hipblasLt", h_C, M * N);

    // Cleanup
    CHECK_HIPBLASLT_ERROR(hipblasLtDestroy(hipblaslt_handle));
    free(h_bias);
    CHECK_HIP(hipFree(d_bias));
    cleanup(rocblas_handle, d_A, d_B, d_C, h_A, h_B, h_C, h_A_trans, h_B_trans, h_C_trans);

    return 0;
}
