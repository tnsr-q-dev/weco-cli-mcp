import time
import sys
import pathlib
import importlib
import traceback
import mlx.core as mx
import mlx.nn as nn


########################################################
# Baseline
########################################################
class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = mx.random.normal(shape=(hidden_size, input_size))
        self.scaling_factor = scaling_factor

    def __call__(self, x):
        """
        Args:
            x (mx.array): Input tensor of shape (batch_size, input_size).
        Returns:
            mx.array: Output tensor of shape (batch_size, hidden_size).
        """
        x = mx.matmul(x, mx.transpose(self.weight))  # Gemm
        x = x / 2  # Divide
        x = mx.sum(x, axis=1, keepdims=True)  # Sum
        x = x * self.scaling_factor  # Scaling
        return x


########################################################
# Weco Solution
########################################################
def load_module_from_path(module_path: str, add_to_sys_modules: bool = False):
    # Clean out all old compiled extensions to prevent namespace collisions during build
    module_path = pathlib.Path(module_path)
    name = module_path.stem
    spec = importlib.util.spec_from_file_location(name, module_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    if add_to_sys_modules:
        sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod


########################################################
# Benchmark
########################################################
def get_inputs(B, N):
    # MLX doesn't use device parameter like PyTorch, as it automatically uses Metal
    return mx.random.normal(shape=(B, N), dtype=mx.float32)


def bench(f, inputs, n_warmup, n_rep):
    # Warm up
    for _ in range(n_warmup):
        result = f(inputs)
        mx.eval(result)  # Force computation due to lazy evaluation

    t_avg = 0.0
    for _ in range(n_rep):
        # Clear cache before timing
        mx.clear_cache()

        start_time = time.time()
        result = f(inputs)
        mx.eval(result)  # Force computation
        mx.synchronize()  # Wait for all computations to complete
        t_avg += time.time() - start_time

    t_avg /= n_rep * 1e-3
    return t_avg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-path", type=str, required=True)
    args = parser.parse_args()

    # init and input parameters
    B, N, H, S = 128, 10, 20, 1.5

    # Set the default device to 0
    mx.set_default_device(mx.gpu)

    # load solution module
    try:
        mx.random.seed(0)
        solution_module = load_module_from_path(args.solution_path, add_to_sys_modules=False)
        solution_model = solution_module.Model(N, H, S)
        assert hasattr(solution_model, "__call__")
    except Exception:
        print(f"Candidate module initialization failed: {traceback.format_exc()}")
        exit(1)

    mx.random.seed(0)
    baseline_model = Model(N, H, S)

    # measure correctness
    n_correctness_trials = 10
    max_diff_avg = 0
    for _ in range(n_correctness_trials):
        inputs = get_inputs(B, N)
        baseline_output = baseline_model(inputs)
        optimized_output = solution_model(inputs)
        max_diff = mx.max(mx.abs(optimized_output - baseline_output))
        mx.eval(max_diff)  # Force computation
        max_diff_avg += max_diff.item()  # Convert to Python scalar
    max_diff_avg /= n_correctness_trials
    print(f"max float diff between values of baseline and optimized model: {max_diff_avg}")

    # measure performance
    inputs = get_inputs(B, N)
    n_warmup = 100
    n_rep = 500

    # baseline
    t_avg_baseline = bench(baseline_model, inputs, n_warmup, n_rep)
    print(f"baseline time: {t_avg_baseline:.2f}ms")

    # optimized
    t_avg_optimized = bench(solution_model, inputs, n_warmup, n_rep)
    print(f"optimized time: {t_avg_optimized:.2f}ms")

    print(f"speedup: {t_avg_baseline / t_avg_optimized:.2f}x")
