<div align="center">

# Weco: The AI Code Optimizer

[![Python](https://img.shields.io/badge/Python-3.8.0+-blue)](https://www.python.org)
[![docs](https://img.shields.io/website?url=https://docs.weco.ai/&label=docs)](https://docs.weco.ai/)
[![PyPI version](https://badge.fury.io/py/weco.svg)](https://badge.fury.io/py/weco)
[![AIDE](https://img.shields.io/badge/AI--Driven_Exploration-arXiv-orange?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2502.13138)

<code>pip install weco</code>

</div>

---

Weco systematically optimizes your code, guided directly by your evaluation metrics.

Example applications include:

- **GPU Kernel Optimization**: Reimplement PyTorch functions using CUDA or Triton optimizing for `latency`, `throughput`, or `memory_bandwidth`.
- **Model Development**: Tune feature transformations or architectures, optimizing for `validation_accuracy`, `AUC`, or `Sharpe Ratio`.
- **Prompt Engineering**: Refine prompts for LLMs, optimizing for `win_rate`, `relevance`, or `format_adherence`

![image](assets/example-optimization.gif)

---

## Overview

The `weco` CLI leverages a tree search approach guided by Large Language Models (LLMs) to iteratively explore and refine your code. It automatically applies changes, runs your evaluation script, parses the results, and proposes further improvements based on the specified goal.

![image](https://github.com/user-attachments/assets/a6ed63fa-9c40-498e-aa98-a873e5786509)

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

    - When you first run `weco run`, you'll be prompted if you want to log in or proceed anonymously.
    - If you choose to log in (by pressing `l`), you'll be shown a URL and `weco` will attempt to open it in your default web browser.
    - You then authenticate in the browser. Once authenticated, the CLI will detect this and complete the login.
    - This saves a Weco-specific API key locally (typically at `~/.config/weco/credentials.json`).

    If you choose to skip login (by pressing Enter or `s`), `weco` will still function using the environment variable LLM keys, but the run history will not be linked to a Weco account.

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

**Example: Optimizing Simple PyTorch Operations**

This basic example shows how to optimize a simple PyTorch function for speedup.

For more advanced examples, including [Triton](/examples/triton/README.md), [CUDA kernel optimization](/examples/cuda/README.md), [ML model optimization](/examples/spaceship-titanic/README.md), and [prompt engineering for math problems](https://github.com/WecoAI/weco-cli/tree/main/examples/prompt), please see the `README.md` files within the corresponding subdirectories under the [`examples/`](./examples/) folder.

```bash
# Navigate to the example directory
cd examples/hello-kernel-world

# Install dependencies
pip install torch

# Run Weco
weco run --source optimize.py \
     --eval-command "python evaluate.py --solution-path optimize.py --device cpu" \
     --metric speedup \
     --maximize true \
     --steps 15 \
     --model gemini-2.5-pro-exp-03-25 \
     --additional-instructions "Fuse operations in the forward method while ensuring the max float deviation remains small. Maintain the same format of the code."
```

**Note:** If you have an NVIDIA GPU, change the device in the `--eval-command` to `cuda`. If you are running this on Apple Silicon, set it to `mps`.

---

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
| `--log-dir`                 | (Optional) Path to the directory to log intermediate steps and final optimization result. Defaults to `.runs/`.                                                           | No       |

---

### `weco logout` Command

This command logs you out by removing the locally stored Weco API key.

```bash
weco logout
```

---

### Performance & Expectations

Weco, powered by the AIDE algorithm, optimizes code iteratively based on your evaluation results. Achieving significant improvements, especially on complex research-level tasks, often requires substantial exploration time.

The following plot from the independent [Research Engineering Benchmark (RE-Bench)](https://metr.org/AI_R_D_Evaluation_Report.pdf) report shows the performance of AIDE (the algorithm behind Weco) on challenging ML research engineering tasks over different time budgets.

<p align="center">
<img src="https://github.com/user-attachments/assets/ff0e471d-2f50-4e2d-b718-874862f533df" alt="RE-Bench Performance Across Time" width="60%"/>
</p>

As shown, AIDE demonstrates strong performance gains over time, surpassing lower human expert percentiles within hours and continuing to improve. This highlights the potential of evaluation-driven optimization but also indicates that reaching high levels of performance comparable to human experts on difficult benchmarks can take considerable time (tens of hours in this specific benchmark, corresponding to many `--steps` in the Weco CLI). Factor this into your planning when setting the number of `--steps` for your optimization runs.

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
    git clone https://github.com/WecoAI/weco-cli.git
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
