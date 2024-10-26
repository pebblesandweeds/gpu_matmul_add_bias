import torch
import math
import time

chunk_size = 64
hidden_size = 4096
out_features = 16384
NUM_RUNS = 5000

seq_len = 2048
sequence = torch.randn(seq_len, hidden_size, device='cuda') * 0.02  # (2048, 4096)

weights = torch.randn(out_features, hidden_size, device='cuda')
bias = torch.randn(out_features, device='cuda')

torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
bound = 1 / math.sqrt(hidden_size)
torch.nn.init.uniform_(bias, -bound, bound)

layer = torch.nn.Linear(hidden_size, out_features, device='cuda')
layer.weight = torch.nn.Parameter(weights)
layer.bias = torch.nn.Parameter(bias)

matmul_flops = 2 * chunk_size * hidden_size * out_features
bias_flops = chunk_size * out_features
flops = matmul_flops + bias_flops

print(f"Total sequence length: {seq_len}")
print(f"Processing in chunks of: {chunk_size}")
print(f"Matrix sizes per chunk: {chunk_size}x{hidden_size} @ {hidden_size}x{out_features}")
print(f"Number of runs: {NUM_RUNS}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("\nRun\tTime (s)\tTFLOPS")
print("-" * 30)

for i in range(NUM_RUNS):
   start_idx = torch.randint(0, seq_len - chunk_size, (1,)).item()
   x = sequence[start_idx:start_idx + chunk_size]

   torch.cuda.synchronize()
   start = time.perf_counter()
   output = layer(x)
   torch.cuda.synchronize()
   end = time.perf_counter()

   run_time = end - start
   tflops = (flops / run_time) / 1e12
   print(f"{i+1}\t{run_time:.6f}\t{tflops:.2f}")

print(f"\nTensor shapes:")
print(f"Full sequence: {sequence.shape}")    # (2048, 4096)
print(f"Chunk size: {x.shape}")             # (64, 4096)
print(f"weights: {weights.shape}")          # (16384, 4096)
print(f"bias: {bias.shape}")                # (16384)
print(f"chunk output: {output.shape}")      # (64, 16384)
