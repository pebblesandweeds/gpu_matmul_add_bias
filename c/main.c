#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define SEQ_LEN 2048
#define HIDDEN_SIZE 4096
#define OUT_FEATURES 16384
#define CHUNK_SIZE 64
#define NUM_RUNS 5000

#define CHECK_HIP(cmd) \
{\
   hipError_t error  = cmd;\
   if (error != hipSuccess) { \
       fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
       exit(EXIT_FAILURE);\
   }\
}

#define CHECK_ROCBLAS(cmd) \
{\
   rocblas_status status = cmd;\
   if (status != rocblas_status_success) { \
       fprintf(stderr, "rocBLAS error: %d at %s:%d\n", status,__FILE__, __LINE__); \
       exit(EXIT_FAILURE);\
   }\
}

void transpose_matrix(const float *src, float *dst, int rows, int cols) {
   for (int i = 0; i < rows; ++i) {
       for (int j = 0; j < cols; ++j) {
           dst[j * rows + i] = src[i * cols + j];
       }
   }
}

void perform_matrix_multiplication(rocblas_handle handle, float *d_sequence, float *d_weights, float *d_output,
                                float *d_sequence_trans, float *d_weights_trans,
                                int chunk_size, int hidden_size, int out_features, int seq_len, int num_runs) {
   const float alpha = 1.0f;
   const float beta = 0.0f;
   double total_flops = 2.0 * chunk_size * hidden_size * out_features;
   size_t chunk_bytes = chunk_size * hidden_size * sizeof(float);

   hipEvent_t start, stop;
   CHECK_HIP(hipEventCreate(&start));
   CHECK_HIP(hipEventCreate(&stop));

   for (int run = 0; run < num_runs; run++) {
       size_t start_idx = rand() % (seq_len - chunk_size);

       CHECK_HIP(hipEventRecord(start));

       CHECK_ROCBLAS(rocblas_sgemm(handle,
                                  rocblas_operation_none, rocblas_operation_none,
                                  out_features,
                                  chunk_size,
                                  hidden_size,
                                  &alpha,
                                  d_weights_trans,
                                  out_features,
                                  d_sequence_trans + (start_idx * hidden_size),
                                  hidden_size,
                                  &beta,
                                  d_output,
                                  out_features));

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

static float randn() {
   float u = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
   float v = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
   return u * v;
}

static float uniform(float min, float max) {
   float random = (float)rand() / (float)RAND_MAX;
   return min + random * (max - min);
}

static float kaiming_uniform(float a, int fan_in) {
   float bound = a * sqrtf(3.0f / fan_in);
   return uniform(-bound, bound);
}

void initialize_matrices(float *sequence, float *weights, float *bias,
                      float *sequence_trans, float *weights_trans,
                      int hidden_size, int out_features, int seq_len) {
   // Initialize original matrices
   for (int i = 0; i < seq_len * hidden_size; i++) {
       sequence[i] = randn() * 0.02f;
   }

   float a = sqrtf(5.0f);
   for (int i = 0; i < out_features * hidden_size; i++) {
       weights[i] = kaiming_uniform(a, hidden_size);
   }

   float bound = 1.0f / sqrtf(hidden_size);
   for (int i = 0; i < out_features; i++) {
       bias[i] = uniform(-bound, bound);
   }

   // Create transposed copies
   transpose_matrix(sequence, sequence_trans, seq_len, hidden_size);
   transpose_matrix(weights, weights_trans, out_features, hidden_size);
}

void verify_matrices(float *sequence, float *weights, float *bias, size_t seq_len, size_t hidden_size, size_t out_features) {
   int errors = 0;

   for (size_t i = 0; i < seq_len * hidden_size; i++) {
       if (fabsf(sequence[i]) > 0.1f) {
           printf("Error: Sequence value out of expected range at index %zu: %f\n", i, sequence[i]);
           errors++;
           if (errors > 5) break;
       }
   }

   float weight_bound = sqrtf(5.0f) * sqrtf(3.0f / hidden_size);
   for (size_t i = 0; i < out_features * hidden_size; i++) {
       if (fabsf(weights[i]) > weight_bound) {
           printf("Error: Weight value out of expected range at index %zu: %f\n", i, weights[i]);
           errors++;
           if (errors > 5) break;
       }
   }

   float bias_bound = 1.0f / sqrtf(hidden_size);
   for (size_t i = 0; i < out_features; i++) {
       if (fabsf(bias[i]) > bias_bound) {
           printf("Error: Bias value out of expected range at index %zu: %f\n", i, bias[i]);
           errors++;
           if (errors > 5) break;
       }
   }

   if (errors == 0) {
       printf("Matrix initialization verification passed!\n");
   } else {
       printf("Matrix initialization verification failed with %d errors\n", errors);
   }
}

void print_gpu_info() {
   hipDeviceProp_t prop;
   int device;
   CHECK_HIP(hipGetDevice(&device));
   CHECK_HIP(hipGetDeviceProperties(&prop, device));
   printf("\nGPU Info:\n");
   printf("Device Name: %s\n", prop.name);
   printf("Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
   printf("Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
   printf("Clock Rate: %.2f GHz\n", prop.clockRate / 1000000.0);
}

int main() {
   srand(42);
   printf("Running matrix initialization verification...\n");
   print_gpu_info();

   size_t sequence_size = SEQ_LEN * HIDDEN_SIZE * sizeof(float);
   size_t weights_size = OUT_FEATURES * HIDDEN_SIZE * sizeof(float);
   size_t bias_size = OUT_FEATURES * sizeof(float);
   size_t output_size = CHUNK_SIZE * OUT_FEATURES * sizeof(float);

   // Allocate both original and transposed on host
   float *h_sequence = (float*)malloc(sequence_size);
   float *h_weights = (float*)malloc(weights_size);
   float *h_bias = (float*)malloc(bias_size);
   float *h_sequence_trans = (float*)malloc(sequence_size);
   float *h_weights_trans = (float*)malloc(weights_size);

   if (!h_sequence || !h_weights || !h_bias || !h_sequence_trans || !h_weights_trans) {
       fprintf(stderr, "Failed to allocate host memory!\n");
       exit(EXIT_FAILURE);
   }

   // Initialize and create transposed copies
   initialize_matrices(h_sequence, h_weights, h_bias, h_sequence_trans, h_weights_trans,
                      HIDDEN_SIZE, OUT_FEATURES, SEQ_LEN);
   verify_matrices(h_sequence, h_weights, h_bias, SEQ_LEN, HIDDEN_SIZE, OUT_FEATURES);

   // Allocate both original and transposed on device
   float *d_sequence, *d_weights, *d_bias, *d_output;
   float *d_sequence_trans, *d_weights_trans;
   CHECK_HIP(hipMalloc(&d_sequence, sequence_size));
   CHECK_HIP(hipMalloc(&d_weights, weights_size));
   CHECK_HIP(hipMalloc(&d_bias, bias_size));
   CHECK_HIP(hipMalloc(&d_output, output_size));
   CHECK_HIP(hipMalloc(&d_sequence_trans, sequence_size));
   CHECK_HIP(hipMalloc(&d_weights_trans, weights_size));

   hipEvent_t start, stop;
   CHECK_HIP(hipEventCreate(&start));
   CHECK_HIP(hipEventCreate(&stop));

   CHECK_HIP(hipEventRecord(start));
   // Copy both original and transposed to device
   CHECK_HIP(hipMemcpy(d_sequence, h_sequence, sequence_size, hipMemcpyHostToDevice));
   CHECK_HIP(hipMemcpy(d_weights, h_weights, weights_size, hipMemcpyHostToDevice));
   CHECK_HIP(hipMemcpy(d_bias, h_bias, bias_size, hipMemcpyHostToDevice));
   CHECK_HIP(hipMemcpy(d_sequence_trans, h_sequence_trans, sequence_size, hipMemcpyHostToDevice));
   CHECK_HIP(hipMemcpy(d_weights_trans, h_weights_trans, weights_size, hipMemcpyHostToDevice));
   CHECK_HIP(hipEventRecord(stop));

   CHECK_HIP(hipEventSynchronize(stop));
   float transfer_time;
   CHECK_HIP(hipEventElapsedTime(&transfer_time, start, stop));
   printf("\nMemory transfer to device time: %f ms\n", transfer_time);

   rocblas_handle handle;
   CHECK_ROCBLAS(rocblas_create_handle(&handle));

   perform_matrix_multiplication(handle,
                               d_sequence,
                               d_weights,
                               d_output,
                               d_sequence_trans,
                               d_weights_trans,
                               CHUNK_SIZE,
                               HIDDEN_SIZE,
                               OUT_FEATURES,
                               SEQ_LEN,
                               NUM_RUNS);

   CHECK_HIP(hipEventDestroy(start));
   CHECK_HIP(hipEventDestroy(stop));
   CHECK_ROCBLAS(rocblas_destroy_handle(handle));

   CHECK_HIP(hipFree(d_sequence));
   CHECK_HIP(hipFree(d_weights));
   CHECK_HIP(hipFree(d_bias));
   CHECK_HIP(hipFree(d_output));
   CHECK_HIP(hipFree(d_sequence_trans));
   CHECK_HIP(hipFree(d_weights_trans));

   free(h_sequence);
   free(h_weights);
   free(h_bias);
   free(h_sequence_trans);
   free(h_weights_trans);

   return 0;
}
