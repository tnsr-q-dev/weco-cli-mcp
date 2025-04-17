# Example: Optimizing MLX Convolution with Metal

This example demonstrates how to use Weco to optimize a 2D convolution operation implemented in [`mlx`](https://github.com/ml-explore/mlx), targeting Apple's [Metal](https://developer.apple.com/documentation/metal/) framework for execution on Apple Silicon GPUs.

It showcases using a separate file (`examples.rst`) to provide detailed context and instructions to the optimizing LLM.

## Setup

1.  Ensure you are in the `examples/metal` directory.
2.  Install the required dependency:
    ```bash
    pip install mlx
    ```

## Optimization Command

Run the following command to start the optimization process:

```bash
weco --source optimize.py \
     --eval-command "python evaluate.py --solution-path optimize.py" \
     --metric speedup \
     --maximize true \
     --steps 30 \
     --model gemini-2.5-pro-exp-03-25 \
     --additional-instructions examples.rst
```

### Explanation

*   `--source optimize.py`: Specifies the Python file containing the MLX convolution code to be optimized.
*   `--eval-command "python evaluate.py --solution-path optimize.py"`: Runs the evaluation script. `evaluate.py` executes the code in `optimize.py`, measures its performance against a baseline, and prints the `speedup` metric.
*   `--metric speedup`: Tells Weco to target the 'speedup' value printed by the evaluation command.
*   `--maximize true`: Instructs Weco to aim for a higher speedup value.
*   `--steps 30`: Defines the number of iterative optimization steps Weco will perform.
*   `--model gemini-2.5-pro-exp-03-25`: Selects the LLM used for proposing code modifications.
*   `--additional-instructions examples.rst`: Provides a path to a file containing detailed guidance for the LLM during optimization (e.g., constraints, preferred Metal techniques).

Weco will iteratively modify `optimize.py`, run `evaluate.py`, parse the `speedup`, and generate new code versions based on the results and the instructions in `examples.rst`.