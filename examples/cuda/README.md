# Example: Optimizing PyTorch Self-Attention with CUDA

This example showcases using Weco to optimize a PyTorch causal multi-head self-attention implementation by generating custom [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) kernels. This approach aims for low-level optimization beyond standard PyTorch or even Triton for potentially higher performance on NVIDIA GPUs.

This example uses a separate Markdown file (`guide.md`) to provide detailed instructions and context to the LLM.

## Setup

1.  Ensure you are in the `examples/cuda` directory.
2.  Install the required dependency:
    ```bash
    pip install torch
    ```
    *(Note: This example requires a compatible NVIDIA GPU and the CUDA Toolkit installed on your system for compiling and running the generated CUDA code.)*

## Optimization Command

Run the following command to start the optimization process:

```bash
weco run --source optimize.py \
         --eval-command "python evaluate.py --solution-path optimize.py" \
         --metric speedup \
         --maximize true \
         --steps 30 \
         --model gemini-2.5-pro-exp-03-25 \
         --additional-instructions guide.md
```

### Explanation

*   `--source optimize.py`: The initial PyTorch self-attention code to be optimized with CUDA.
*   `--eval-command "python evaluate.py --solution-path optimize.py"`: Runs the evaluation script, which compiles (if necessary) and benchmarks the CUDA-enhanced code in `optimize.py` against a baseline, printing the `speedup`.
*   `--metric speedup`: The optimization target metric.
*   `--maximize true`: Weco aims to increase the speedup.
*   `--steps 30`: The number of optimization iterations.
*   `--model gemini-2.5-pro-exp-03-25`: The LLM used for code generation.
*   `--additional-instructions guide.md`: Points Weco to a file containing detailed instructions for the LLM on how to write the CUDA kernels, handle compilation (e.g., using `torch.utils.cpp_extension`), manage data types, and ensure correctness.

Weco will iteratively modify `optimize.py`, potentially generating and integrating CUDA C++ code, guided by the evaluation results and the instructions in `guide.md`.