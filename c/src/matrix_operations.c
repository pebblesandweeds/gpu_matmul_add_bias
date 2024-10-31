#include <stdlib.h>
#include <stdio.h>
#include "../include/matrix_operations.h"
#include "../include/utils.h"

static float random_float() {
    return (float)rand() / ((float)RAND_MAX + 1.0f) * 2.0f - 1.0f;
}

void initialize_matrices(float *A, float *B, int m, int k, int n) {
    for (int i = 0; i < m * k; i++) {
        A[i] = random_float();
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = random_float();
    }
}

void transpose_matrix(const float *src, float *dst, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

void perform_matrix_multiplication(rocblas_handle handle, float *d_A, float *d_B, float *d_C,
                                 int m, int n, int k, int NUM_RUNS) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    double total_flops = 2.0 * m * n * k;  // Updated FLOPS calculation for rectangular matrices

    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    for (int run = 0; run < NUM_RUNS; run++) {
        CHECK_HIP(hipEventRecord(start));
        CHECK_ROCBLAS(rocblas_sgemm(handle,
                                   rocblas_operation_none, rocblas_operation_none,
                                   m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));
        CHECK_HIP(hipEventRecord(stop));
        CHECK_HIP(hipEventSynchronize(stop));

        float compute_time;
        CHECK_HIP(hipEventElapsedTime(&compute_time, start, stop));
        double seconds = compute_time / 1000.0;
        double tflops = total_flops / (seconds * 1e12);
        printf("Run %d: Matrix multiplication time: %f ms, Performance: %.2f TFLOPS\n",
               run+1, compute_time, tflops);
    }

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
}
