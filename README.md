# Weco CLI – Optimize Your Code Effortlessly

[![Python](https://img.shields.io/badge/Python-3.12.0-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`weco` is a powerful command-line interface for interacting with Weco AI's code optimizer. Whether you are looking to improve performance or refine code quality, our CLI streamlines your workflow for a better development experience.

---

## Overview

The `weco` CLI leverages advanced optimization techniques and language model strategies to iteratively improve your source code. It supports multiple language models and offers a flexible configuration to suit different optimization tasks.

---

## Setup

1. **Install the Package:**

   ```bash
   pip install weco
   ```

2. **Configure API Keys:**

   Set the appropriate environment variables for your language model provider:
   
   - **OpenAI:** `export OPENAI_API_KEY="your_key_here"`
   - **Anthropic:** `export ANTHROPIC_API_KEY="your_key_here"`

---

## Usage

### Command Line Arguments

| Argument                    | Description                                                                                                                                   | Required |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `--source`                  | Path to the Python source code that will be optimized (e.g. optimize.py).                                                                    | Yes      |
| `--eval-command`            | Command to run for evaluation (e.g. 'python eval.py --arg1=val1').                                                                            | Yes      |
| `--metric`                  | Metric to optimize.                                                                                                                           | Yes      |
| `--maximize`                | Boolean flag indicating whether to maximize the metric.                                                                                       | Yes      |
| `--steps`                   | Number of optimization steps to run.                                                                                                          | Yes      |
| `--model`                   | Model to use for optimization.                                                                                                                | Yes      |
| `--additional-instructions` | (Optional) Description of additional instructions or path to a file containing additional instructions.                                       | No       |

---

### Example

Optimizing common operations in pytorch:
```bash
weco --source examples/simple-torch/optimize.py \
     --eval-command "python examples/simple-torch/evaluate.py --solution-path examples/simple-torch/optimize.py --device mps" \
     --metric "speedup" \
     --maximize true \
     --steps 15 \
     --model "o3-mini" \
     --additional-instructions "Fuse operations in the forward method while ensuring the max float deviation remains small. Maintain the same format of the code."
```

Optimizing these same using mlx and metal:
```bash
weco --source examples/simple-mlx/optimize.py \
     --eval-command "python examples/simple-mlx/evaluate.py --solution-path examples/simple-mlx/optimize.py" \
     --metric "speedup" \
     --maximize true \
     --steps 30 \
     --model "o3-mini" \
     --additional-instructions "examples/simple-mlx/metal-examples.rst"
```
---

## Supported Providers

The CLI supports the following model providers:

- **OpenAI:** Set your API key using `OPENAI_API_KEY`.
- **Anthropic:** Set your API key using `ANTHROPIC_API_KEY`.

---

## Contributing

We welcome contributions! To get started:

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/WecoAI/weco-cli.git
   cd weco-cli
   ```

2. **Install Development Dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes:** Ensure your code adheres to our style guidelines and includes relevant tests.

5. **Commit and Push** your changes, then open a pull request with a clear description of your enhancements.

---
