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

void print_debug(const char* name, const float* data, int num_elements) {
    printf("\nDEBUG %s - First %d elements in memory:\n", name, num_elements);
    for (int i = 0; i < num_elements; i++) {
        printf("%8.4f ", data[i]);
        if ((i + 1) % 8 == 0) printf("\n");  // New line every 8 elements
    }
    printf("\n");
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

        // Debug print C matrix after sgemm (only on first run)
        if (run == 0) {
            float* h_C_debug = (float*)malloc(m * n * sizeof(float));
            CHECK_HIP(hipMemcpy(h_C_debug, d_C, m * n * sizeof(float), hipMemcpyDeviceToHost));
            print_debug("C matrix after sgemm before bias", h_C_debug, m * n);
            free(h_C_debug);
        }

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

// Add to matrix_operations.c

void perform_matrix_multiplication_hipblaslt(hipblasLtHandle_t handle, float *d_A, float *d_B, float *d_C,
                                           int m, int n, int k, int NUM_RUNS) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    double total_flops = 2.0 * m * n * k;
    hipEvent_t start, stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    // Create matrix layouts
    hipblasLtMatrixLayout_t matA, matB, matC, matD;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matA, HIP_R_32F, m, k, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matB, HIP_R_32F, k, n, k));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matC, HIP_R_32F, m, n, m));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutCreate(&matD, HIP_R_32F, m, n, m));

    // Create matmul descriptor
    hipblasLtMatmulDesc_t matmul;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescCreate(&matmul, HIPBLAS_COMPUTE_32F, HIP_R_32F));

    // Set operations for matrices A and B (no transpose)
    hipblasOperation_t trans_a = HIPBLAS_OP_N;
    hipblasOperation_t trans_b = HIPBLAS_OP_N;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(int32_t)));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(int32_t)));

    // Set default epilogue
    hipblasLtEpilogue_t epilogue = HIPBLASLT_EPILOGUE_DEFAULT;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescSetAttribute(
        matmul, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // Create preference with workspace
    hipblasLtMatmulPreference_t pref;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceCreate(&pref));
    size_t max_workspace_size = 32 * 1024 * 1024; // 32MB workspace
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceSetAttribute(
        pref,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &max_workspace_size,
        sizeof(max_workspace_size)));

    // Get heuristic result
    hipblasLtMatmulHeuristicResult_t heuristicResult[1];
    int returnedAlgoCount = 0;
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulAlgoGetHeuristic(
        handle, matmul, matA, matB, matC, matD,
        pref, 1, heuristicResult, &returnedAlgoCount));

    if (returnedAlgoCount == 0) {
        fprintf(stderr, "No valid solution found!\n");
        return;
    }

    // Allocate workspace
    void* d_workspace = NULL;
    CHECK_HIP(hipMalloc(&d_workspace, heuristicResult[0].workspaceSize));

    // Main computation loop
    for (int run = 0; run < NUM_RUNS; run++) {
        CHECK_HIP(hipEventRecord(start));

        CHECK_HIPBLASLT_ERROR(hipblasLtMatmul(
            handle, matmul,
            &alpha,
            d_A, matA,
            d_B, matB,
            &beta,
            d_C, matC,
            d_C, matD,
            &heuristicResult[0].algo,
            d_workspace,
            heuristicResult[0].workspaceSize,
            0));  // stream = 0 for default stream

        CHECK_HIP(hipEventRecord(stop));
        CHECK_HIP(hipEventSynchronize(stop));

        float compute_time;
        CHECK_HIP(hipEventElapsedTime(&compute_time, start, stop));
        double seconds = compute_time / 1000.0;
        double tflops = total_flops / (seconds * 1e12);
        printf("Run %d: Matrix multiplication time: %f ms, Performance: %.2f TFLOPS\n",
               run+1, compute_time, tflops);
    }

    // Cleanup
    CHECK_HIP(hipFree(d_workspace));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matA));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matB));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matC));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatrixLayoutDestroy(matD));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulDescDestroy(matmul));
    CHECK_HIPBLASLT_ERROR(hipblasLtMatmulPreferenceDestroy(pref));
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
}
