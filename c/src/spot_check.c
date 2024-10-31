#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../include/spot_check.h"

void spot_check(const float *A, const float *B, const float *bias, const float *C_gpu, int m, int n, int k) {
    printf("\nPerforming %d random spot checks between CPU and GPU results...\n", NUM_SPOT_CHECKS);
    srand(time(NULL));

    // First transpose C_gpu from column-major to row-major
    float *C_gpu_transposed = (float *)malloc(m * n * sizeof(float));
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            C_gpu_transposed[i * n + j] = C_gpu[j * m + i];
        }
    }

    int mismatch_count = 0;
    for(int check = 0; check < NUM_SPOT_CHECKS; check++) {
        int row = rand() % m;
        int col = rand() % n;

        float cpu_val = 0.0f;
        for(int p = 0; p < k; p++) {
            float a_val = A[row * k + p];
            float b_val = B[p * n + col];
            cpu_val += a_val * b_val;
        }
        cpu_val += bias[col];

        float gpu_val = C_gpu_transposed[row * n + col];
        float rel_error = fabsf(cpu_val - gpu_val) / (fabsf(cpu_val) + 1e-8f);

        if(rel_error > 1e-3f) {
            printf("Error at C[%d,%d]: CPU = %.12f, GPU = %.12f, Rel Error = %.12f\n",
                   row, col, cpu_val, gpu_val, rel_error);
            mismatch_count++;
        }
    }

    free(C_gpu_transposed);
    if(mismatch_count == 0) {
        printf("\nValidation PASSED - All %d random spot checks match\n", NUM_SPOT_CHECKS);
    } else {
        printf("\nValidation FAILED - %d elements out of %d spot checks don't match\n",
               mismatch_count, NUM_SPOT_CHECKS);
    }
}
~     
