import torch
import time

# Define a smaller matrix size for testing
matrix_size = 20000  

# Run on CPU
start = time.time()
x = torch.rand(matrix_size, matrix_size)
y = torch.rand(matrix_size, matrix_size)
result = torch.mm(x, y)
print("CPU time:", time.time() - start)

# Run on GPU
if torch.cuda.is_available():
    start = time.time()
    # Move tensors to GPU
    x = x.to("cuda")
    y = y.to("cuda")
    
    # Perform matrix multiplication on the GPU
    result = torch.mm(x, y)
    print("GPU time:", time.time() - start)

    # Optional: Move the result back to CPU if needed
    result_cpu = result.cpu()
else:
    print("CUDA is not available")
