# Weco CLI – Code Optimizer for Machine Learning Engineers

[![Python](https://img.shields.io/badge/Python-3.12.0-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI version](https://badge.fury.io/py/weco.svg)](https://badge.fury.io/py/weco)

`weco` is a command-line interface for interacting with Weco AI's code optimizer, powered by [AI-Driven Exploration](https://arxiv.org/abs/2502.13138). It helps you automate the improvement of your code for tasks like GPU kernel optimization, feature engineering, model development, and prompt engineering.

https://github.com/user-attachments/assets/cb724ef1-bff6-4757-b457-d3b2201ede81

---

## Overview

The `weco` CLI leverages a tree search approach guided by Large Language Models (LLMs) to iteratively explore and refine your code. It automatically applies changes, runs your evaluation script, parses the results, and proposes further improvements based on the specified goal.

![image](https://github.com/user-attachments/assets/a6ed63fa-9c40-498e-aa98-a873e5786509)

---

## Example Use Cases

Here's how `weco` can be applied to common ML engineering tasks:

- **GPU Kernel Optimization:**

  - **Goal:** Improve the speed or efficiency of low-level GPU code.
  - **How:** `weco` iteratively refines CUDA, Triton, Metal, or other kernel code specified in your `--source` file.
  - **`--eval-command`:** Typically runs a script that compiles the kernel, executes it, and benchmarks performance (e.g., latency, throughput).
  - **`--metric`:** Examples include `latency`, `throughput`, `TFLOPS`, `memory_bandwidth`. Optimize to `minimize` latency or `maximize` throughput.

- **Feature Engineering:**

  - **Goal:** Discover better data transformations or feature combinations for your machine learning models.
  - **How:** `weco` explores different processing steps or parameters within your feature transformation code (`--source`).
  - **`--eval-command`:** Executes a script that applies the features, trains/validates a model using those features, and prints a performance score.
  - **`--metric`:** Examples include `accuracy`, `AUC`, `F1-score`, `validation_loss`. Usually optimized to `maximize` accuracy/AUC/F1 or `minimize` loss.

- **Model Development:**

  - **Goal:** Tune hyperparameters or experiment with small architectural changes directly within your model's code.
  - **How:** `weco` modifies hyperparameter values (like learning rate, layer sizes if defined in the code) or structural elements in your model definition (`--source`).
  - **`--eval-command`:** Runs your model training and evaluation script, printing the key performance indicator.
  - **`--metric`:** Examples include `validation_accuracy`, `test_loss`, `inference_time`, `perplexity`. Optimize according to the metric's nature (e.g., `maximize` accuracy, `minimize` loss).

- **Prompt Engineering:**
  - **Goal:** Refine prompts used within larger systems (e.g., for LLM interactions) to achieve better or more consistent outputs.
  - **How:** `weco` modifies prompt templates, examples, or instructions stored in the `--source` file.
  - **`--eval-command`:** Executes a script that uses the prompt, generates an output, evaluates that output against desired criteria (e.g., using another LLM, checking for keywords, format validation), and prints a score.
  - **`--metric`:** Examples include `quality_score`, `relevance`, `task_success_rate`, `format_adherence`. Usually optimized to `maximize`.

---

## Setup

1.  **Install the Package:**

    ```bash
    pip install weco
    ```

2.  **Set Up LLM API Keys (Required):**

    `weco` requires API keys for the Large Language Models (LLMs) it uses internally. You **must** provide these keys via environment variables:

    - **OpenAI:** `export OPENAI_API_KEY="your_key_here"`
    - **Anthropic:** `export ANTHROPIC_API_KEY="your_key_here"`
    - **Google DeepMind:** `export GEMINI_API_KEY="your_key_here"` (Google AI Studio has a free API usage quota. Create a key [here](https://aistudio.google.com/apikey) to use `weco` for free.)

    The optimization process will fail if the necessary keys for the chosen model are not found in your environment.

3.  **Log In to Weco (Optional):**

    To associate your optimization runs with your Weco account and view them on the Weco dashboard, you can log in. `weco` uses a device authentication flow:

    - When you first run `weco run`, you'll be prompted if you want to log in.
    - If you agree (by pressing `y` or `yes`), you'll be asked to visit a URL in your browser and authenticate.
    - This saves a Weco-specific API key locally (typically at `~/.weco/api_key`).

    If you choose not to log in (by pressing Enter or `n`), `weco` will still function using the environment variable LLM keys, but the run history will not be linked to a Weco account.

    To log out and remove your saved Weco API key, use the `weco logout` command.

---

## Usage

The CLI has two main commands:

- `weco run`: Initiates the code optimization process.
- `weco logout`: Logs you out of your Weco account.

<div style="background-color: #fff3cd; border: 1px solid #ffeeba; padding: 15px; border-radius: 4px; margin-bottom: 15px;">
  <strong>⚠️ Warning: Code Modification</strong><br>
  <code>weco</code> directly modifies the file specified by <code>--source</code> during the optimization process. It is <strong>strongly recommended</strong> to use version control (like Git) to track changes and revert if needed. Alternatively, ensure you have a backup of your original file before running the command. Upon completion, the file will contain the best-performing version of the code found during the run.
</div>

---

### `weco run` Command

This command starts the optimization process.

**Examples:**

**Example 1: Optimizing PyTorch operations**

```bash
weco run --source examples/simple-torch/optimize.py \
         --eval-command "python examples/simple-torch/evaluate.py --solution-path examples/simple-torch/optimize.py --device mps" \
         --metric speedup \
         --maximize true \
         --steps 15 \
         --model o3-mini \
         --additional-instructions "Fuse operations in the forward method while ensuring the max float deviation remains small. Maintain the same format of the code."
```

**Example 2: Optimizing MLX operations with instructions from a file**

Sometimes, additional context or instructions are too complex for a single command-line string. You can provide a path to a file containing these instructions.

```bash
weco run --source examples/simple-mlx/optimize.py \
         --eval-command "python examples/simple-mlx/evaluate.py --solution-path examples/simple-mlx/optimize.py" \
         --metric speedup \
         --maximize true \
         --steps 30 \
         --model o3-mini \
         --additional-instructions examples/simple-mlx/metal-examples.rst
```

**Arguments for `weco run`:**

| Argument                    | Description                                                                                                                                                               | Required |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------- |
| `--source`                  | Path to the source code file that will be optimized (e.g., `optimize.py`).                                                                                                | Yes      |
| `--eval-command`            | Command to run for evaluating the code in `--source`. This command should print the target `--metric` and its value to the terminal (stdout/stderr). See note below.      | Yes      |
| `--metric`                  | The name of the metric you want to optimize (e.g., 'accuracy', 'speedup', 'loss'). This metric name should match what's printed by your `--eval-command`.                 | Yes      |
| `--maximize`                | Whether to maximize (`true`) or minimize (`false`) the metric.                                                                                                            | Yes      |
| `--steps`                   | Number of optimization steps (LLM iterations) to run.                                                                                                                     | Yes      |
| `--model`                   | Model identifier for the LLM to use (e.g., `gpt-4o`, `claude-3.5-sonnet`). Recommended models to try include `o3-mini`, `claude-3-haiku`, and `gemini-2.5-pro-exp-03-25`. | Yes      |
| `--additional-instructions` | (Optional) Natural language description of specific instructions OR path to a file containing detailed instructions to guide the LLM.                                     | No       |

---

### `weco logout` Command

This command logs you out by removing the locally stored Weco API key.

```bash
weco logout
```

---

### Important Note on Evaluation

The command specified by `--eval-command` is crucial. It's responsible for executing the potentially modified code from `--source` and assessing its performance. **This command MUST print the metric you specified with `--metric` along with its numerical value to the terminal (standard output or standard error).** Weco reads this output to understand how well each code version performs and guide the optimization process.

For example, if you set `--metric speedup`, your evaluation script (`eval.py` in the examples) should output a line like:

```
speedup: 1.5
```

or

```
Final speedup value = 1.5
```

Weco will parse this output to extract the numerical value (1.5 in this case) associated with the metric name ('speedup').

## Contributing

We welcome contributions! To get started:

1.  **Fork and Clone the Repository:**

    ```bash
    git clone https://github.com/WecoAI/weco-cli.gifully review @/weco/cli.py   and @/README.md  then update the readme to reflect changes.t
    cd weco-cli
    ```

2.  **Install Development Dependencies:**

    ```bash
    pip install -e ".[dev]"
    ```

3.  **Create a Feature Branch:**

    ```bash
    git checkout -b feature/your-feature-name
    ```

4.  **Make Your Changes:** Ensure your code adheres to our style guidelines and includes relevant tests.

5.  **Commit and Push** your changes, then open a pull request with a clear description of your enhancements.

---
