# Weco CLI – Code Optimizer for Machine Learning Engineers

[![Python](https://img.shields.io/badge/Python-3.12.0-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`weco` is a command-line interface for interacting with Weco AI's code optimizer, powerred by [AI-Driven Exploration](https://arxiv.org/abs/2502.13138).



https://github.com/user-attachments/assets/cb724ef1-bff6-4757-b457-d3b2201ede81



---

## Overview

The weco CLI leverages a tree search approach with LLMs to iteratively improve your code.

![image](https://github.com/user-attachments/assets/a6ed63fa-9c40-498e-aa98-a873e5786509)



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
   - **Google DeepMind:** `export GEMINI_API_KEY="your_key_here"`

---

## Usage

### Command Line Arguments

| Argument                    | Description                                                                                                                                   | Required |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|----------|
| `--source`                  | Path to the Python source code that will be optimized (e.g. optimize.py).                                                                     | Yes      |
| `--eval-command`            | Command to run for evaluation (e.g. 'python eval.py --arg1=val1').                                                                            | Yes      |
| `--metric`                  | Metric to optimize.                                                                                                                           | Yes      |
| `--maximize`                | Whether to maximize ('true') or minimize ('false') the metric.                                                                                | Yes      |
| `--steps`                   | Number of optimization steps to run.                                                                                                          | Yes      |
| `--model`                   | Model to use for optimization.                                                                                                                | Yes      |
| `--additional-instructions` | (Optional) Description of additional instructions OR path to a file containing additional instructions.                                       | No       |

---

### Example

Optimizing common operations in pytorch:
```bash
weco --source examples/simple-torch/optimize.py \
     --eval-command "python examples/simple-torch/evaluate.py --solution-path examples/simple-torch/optimize.py --device mps" \
     --metric speedup \
     --maximize true \
     --steps 15 \
     --model o3-mini \
     --additional-instructions "Fuse operations in the forward method while ensuring the max float deviation remains small. Maintain the same format of the code."
```

Sometimes we have a bit more context we'd like to provide. Its not easy to fit all of this in a string like shown above with `additional-instructions`. Thats why you can also provide a path to any file you'd like to me read as in context. In this example, we optimize the same operations using mlx and metal with additional instructions:
```bash
weco --source examples/simple-mlx/optimize.py \
     --eval-command "python examples/simple-mlx/evaluate.py --solution-path examples/simple-mlx/optimize.py" \
     --metric speedup \
     --maximize true \
     --steps 30 \
     --model o3-mini \
     --additional-instructions examples/simple-mlx/metal-examples.rst
```
---

## Supported Providers

The CLI supports the following model providers:

- **OpenAI:** Set your API key using `OPENAI_API_KEY`.
- **Anthropic:** Set your API key using `ANTHROPIC_API_KEY`.
- **Google DeepMind:** Set your API key using `GEMINI_API_KEY`.

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
