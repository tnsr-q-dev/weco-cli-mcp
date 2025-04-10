# Writing In-line CUDA Kernels: 101

This document outlines the strategy to improve speedup by writing fused and optimized CUDA kernels using a single-file implementation.

## Requirements

- **Single-File Implementation:** Develop fused CUDA kernels within one file.
- **No Fallback Implementation:** Do not include any alternative or fallback code.
- **Simplicity & Readability:** Write simple, easy-to-understand code and include clear comments.
- **Avoid Templates:** Use plain fused kernel functions without templates.
- **Multiple Kernels Allowed:** You can define more than one kernel in the file if needed.
- **Model Class Requirement:** The solution must include a class `Model` (an instance of `nn.Module`), with the main computation in its `forward` method.
- **Preserve Initialization:** Do not change the initialization of the `Model` class.
- **Focus on Efficiency:** Concentrate solely on efficient PyTorch and CUDA coding without capturing logs.
- **Error Handling:** Any terminal output or errors will be reviewed by an LLM for feedback.

## GPU Hardware Specifications

Here are some details on the hardware you have access to.

```json
{
    "GPU Architecture": "Ampere",
    "GPU Memory": "40GB",
    "Memory Bandwidth": "1935 GB/s",
    "FP64 TFLOPS": "9.7",
    "FP64 Tensor Core TFLOPS": "19.5",
    "FP32 TFLOPS": "19.5",
    "TF32 Tensor Core TFLOPS": "156 (312 with sparsity)",
    "BFLOAT16 Tensore Core TFLOPS": "312 (624 with sparsity)",
    "FP16 Tensor Core TFLOPS": "312 (624 with sparsity)",
    "INT8 Tensor Core TOPS": "624 (1248 with sparsity)",
    "Register File Size": "64K 32-bit registers per SM",
    "Maximum number of registers per thread": "255",
    "Maximum number of thread blocks per SM": "32",
    "Shared memory capacity per SM": "164 KB",
    "Maximum shared memory per thread block": "163 KB"
}
```

## Baseline Code

The baseline implementation of the `Model` class simply performs an element-wise addition.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b
```

## Optimized Code

The optimized version employs a custom CUDA kernel for fused element-wise addition. The kernel is defined and compiled inline using PyTorch's `load_inline`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for element-wise addition
__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// Launch function for the CUDA kernel
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);
    return out;
}
'''

# C++ function prototype declaration
elementwise_add_cpp_source = "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"

# Compile the inline CUDA code for element-wise addition
elementwise_add = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.elementwise_add = elementwise_add

    def forward(self, a, b):
        return self.elementwise_add.elementwise_add_cuda(a, b)
```