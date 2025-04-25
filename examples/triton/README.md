# Example: Optimizing PyTorch Self-Attention with Triton

This example demonstrates using Weco to optimize a causal multi-head self-attention mechanism, a core component of Transformer models, implemented in PyTorch. The optimization target is to leverage [Triton](https://github.com/triton-lang/triton), a language and compiler for writing highly efficient GPU code, to accelerate the operation.

## Setup

1.  Ensure you are in the `examples/triton` directory.
2.  Install the required dependencies:
    ```bash
    pip install torch triton
    ```
    *(Note: Triton installation might require specific CUDA versions. Refer to the official Triton documentation if you encounter issues.)*

## Optimization Command

Run the following command to start the optimization process:

```bash
weco run --source optimize.py \
         --eval-command "python evaluate.py --solution-path optimize.py" \
         --metric speedup \
         --maximize true \
         --steps 30 \
         --model gemini-2.5-pro-exp-03-25 \
         --additional-instructions "Use triton to optimize the code while ensuring a small max float diff. Maintain the same code format."
```

### Explanation

*   `--source optimize.py`: The PyTorch self-attention implementation to be optimized.
*   `--eval-command "python evaluate.py --solution-path optimize.py"`: Executes the evaluation script, which benchmarks the `optimize.py` code against a baseline and prints the `speedup`.
*   `--metric speedup`: The target metric for optimization.
*   `--maximize true`: Weco should maximize the speedup.
*   `--steps 30`: The number of optimization iterations.
*   `--model gemini-2.5-pro-exp-03-25`: The LLM driving the optimization.
*   `--additional-instructions "..."`: Provides specific guidance to the LLM, instructing it to use Triton, maintain numerical accuracy ("small max float diff"), and preserve the code structure.

Weco will iteratively refine `optimize.py` using Triton, guided by the evaluation results and the provided instructions.