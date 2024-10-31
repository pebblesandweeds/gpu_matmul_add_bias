#include <stdlib.h>
#include <stdio.h>
#include "../include/matrix_operations.h"
#include "../include/utils.h"

static float random_float() {
    return (float)rand() / ((float)RAND_MAX + 1.0f) * 2.0f - 1.0f;
}

__global__ void add_bias_kernel(float *C, const float *bias, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        int col = idx / m;
        C[idx] += bias[col];
    }
}

void initialize_matrices(float *A, float *B, float *bias, int m, int k, int n) {
    for (int i = 0; i < m * k; i++) {
        A[i] = random_float();
    }
    for (int i = 0; i < k * n; i++) {
        B[i] = random_float();
    }
    for (int i = 0; i < n; i++) {
        bias[i] = random_float();
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
                                 float *d_bias, int m, int n, int k, int NUM_RUNS) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    double matmul_flops = 2.0 * m * n * k;  // Each multiply-add in matmul is 2 operations
    double bias_flops = m * n;              // One add per element

    hipEvent_t start_matmul, stop_matmul, start_bias, stop_bias;
    CHECK_HIP(hipEventCreate(&start_matmul));
    CHECK_HIP(hipEventCreate(&stop_matmul));
    CHECK_HIP(hipEventCreate(&start_bias));
    CHECK_HIP(hipEventCreate(&stop_bias));

    int blockSize = 256;
    int numBlocks = (m * n + blockSize - 1) / blockSize;

    for (int run = 0; run < NUM_RUNS; run++) {
        // Time MatMul
        CHECK_HIP(hipEventRecord(start_matmul));
        CHECK_ROCBLAS(rocblas_sgemm(handle,
                                   rocblas_operation_none, rocblas_operation_none,
                                   m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));
        CHECK_HIP(hipEventRecord(stop_matmul));
        CHECK_HIP(hipEventSynchronize(stop_matmul));

        float matmul_time;
        CHECK_HIP(hipEventElapsedTime(&matmul_time, start_matmul, stop_matmul));
        double matmul_seconds = matmul_time / 1000.0;
        double matmul_tflops = matmul_flops / (matmul_seconds * 1e12);

        // Time Bias Addition
        CHECK_HIP(hipEventRecord(start_bias));
        hipLaunchKernelGGL(add_bias_kernel,
                          dim3(numBlocks), dim3(blockSize),
                          0, 0,
                          d_C, d_bias, m, n);
        CHECK_HIP(hipEventRecord(stop_bias));
        CHECK_HIP(hipEventSynchronize(stop_bias));

        float bias_time;
        CHECK_HIP(hipEventElapsedTime(&bias_time, start_bias, stop_bias));
        double bias_seconds = bias_time / 1000.0;
        double bias_tflops = bias_flops / (bias_seconds * 1e12);

        // Total time and TFLOPS
        float total_time = matmul_time + bias_time;
        double total_seconds = total_time / 1000.0;
        double combined_tflops = (matmul_flops + bias_flops) / (total_seconds * 1e12);

        printf("Run %d:\n", run+1);
        printf("  MatMul time: %.3f ms, Performance: %.2f TFLOPS\n",
               matmul_time, matmul_tflops);
        printf("  Bias time:   %.3f ms, Performance: %.2f TFLOPS\n",
               bias_time, bias_tflops);
        printf("  Total time:  %.3f ms, Combined: %.2f TFLOPS\n",
               total_time, combined_tflops);
    }

    CHECK_HIP(hipEventDestroy(start_matmul));
    CHECK_HIP(hipEventDestroy(stop_matmul));
    CHECK_HIP(hipEventDestroy(start_bias));
    CHECK_HIP(hipEventDestroy(stop_bias));
}
