<div align="center">

# Weco: The Platform for Self-Improving Code

[![Python](https://img.shields.io/badge/Python-3.8.0+-blue)](https://www.python.org)
[![docs](https://img.shields.io/website?url=https://docs.weco.ai/&label=docs)](https://docs.weco.ai/)
[![PyPI version](https://badge.fury.io/py/weco.svg)](https://badge.fury.io/py/weco)
[![AIDE](https://img.shields.io/badge/AI--Driven_Exploration-arXiv-orange?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2502.13138)

</div>

---

Weco systematically optimizes your code, guided directly by your evaluation metrics.

Example applications include:

- **GPU Kernel Optimization**: Reimplement PyTorch functions using [CUDA](/examples/cuda/README.md) or [Triton](/examples/triton/README.md), optimizing for `latency`, `throughput`, or `memory_bandwidth`.
- **Model Development**: Tune feature transformations, architectures or [the whole training pipeline](/examples/spaceship-titanic/README.md), optimizing for `validation_accuracy`, `AUC`, or `Sharpe Ratio`.
- **Prompt Engineering**: Refine prompts for LLMs (e.g., for [math problems](/examples/prompt/README.md)), optimizing for `win_rate`, `relevance`, or `format_adherence`

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

---

## Get Started

<div style="background-color: #fff3cd; border: 1px solid #ffeeba; padding: 15px; border-radius: 4px; margin-bottom: 15px;">
  <strong>⚠️ Warning: Code Modification</strong><br>
  <code>weco</code> directly modifies the file specified by <code>--source</code> during the optimization process. It is <strong>strongly recommended</strong> to use version control (like Git) to track changes and revert if needed. Alternatively, ensure you have a backup of your original file before running the command. Upon completion, the file will contain the best-performing version of the code found during the run.
</div>

---

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
     --goal maximize \
     --steps 15 \
     --additional-instructions "Fuse operations in the forward method while ensuring the max float deviation remains small. Maintain the same format of the code."
```

**Note:** If you have an NVIDIA GPU, change the device in the `--eval-command` to `cuda`. If you are running this on Apple Silicon, set it to `mps`.

---

### Arguments for `weco run`

**Required:**

| Argument            | Description                                                                                                                                                                                  |
| :------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-s, --source`      | Path to the source code file that will be optimized (e.g., `optimize.py`).                                                                                                                   |
| `-c, --eval-command`| Command to run for evaluating the code in `--source`. This command should print the target `--metric` and its value to the terminal (stdout/stderr). See note below.                        |
| `-m, --metric`      | The name of the metric you want to optimize (e.g., 'accuracy', 'speedup', 'loss'). This metric name should match what's printed by your `--eval-command`.                                    |
| `-g, --goal`   | `maximize`/`max` to maximize the `--metric` or `minimize`/`min` to minimize it. |

<br>

**Optional:**

| Argument                       | Description                                                                                                                                                                                                                | Default |
| :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------ |
| `-n, --steps`                  | Number of optimization steps (LLM iterations) to run.                                                                                                                                                                      | 100 |
| `-M, --model`                  | Model identifier for the LLM to use (e.g., `gpt-4o`, `claude-3.5-sonnet`).                                                                                                                                                 | `o4-mini` when `OPENAI_API_KEY` is set; `claude-3-7-sonnet-20250219` when `ANTHROPIC_API_KEY` is set; `gemini-2.5-pro-exp-03-25` when `GEMINI_API_KEY` is set (priority: `OPENAI_API_KEY` > `ANTHROPIC_API_KEY` > `GEMINI_API_KEY`). |
| `-i, --additional-instructions`| Natural language description of specific instructions **or** path to a file containing detailed instructions to guide the LLM.                                                                                             | `None` |
| `-l, --log-dir`                | Path to the directory to log intermediate steps and final optimization result.                                                                                                                                             | `.runs/` |

---

### Weco Dashboard
To associate your optimization runs with your Weco account and view them on the Weco dashboard, you can log in. `weco` uses a device authentication flow
![image (16)](https://github.com/user-attachments/assets/8a0a285b-4894-46fa-b6a2-4990017ca0c6)

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
