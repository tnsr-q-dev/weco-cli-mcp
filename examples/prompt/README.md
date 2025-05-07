# AIME Prompt Engineering Example with Weco

This example shows how **Weco** can iteratively improve a prompt for solving American Invitational Mathematics Examination (AIME) problems. The experiment runs locally, requires only two short Python files, and aims to improve the accuracy metric.

This example uses `gpt-4o-mini` via the OpenAI API by default. Ensure your `OPENAI_API_KEY` environment variable is set.

## Files in this folder

| File          | Purpose                                                                                                                                                           |
| :------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `optimize.py` | Holds the prompt template (instructing the LLM to reason step-by-step and use `\\boxed{}` for the final answer) and the mutable `EXTRA_INSTRUCTIONS` string. Weco edits **only** this file during the search. |
| `eval.py`     | Downloads a small slice of the 2024 AIME dataset, calls `optimize.solve` in parallel, parses the LLM output (looking for `\\boxed{}`), compares it to the ground truth, prints progress logs, and finally prints an `accuracy:` line that Weco reads. |

## Quick start

1.  **Clone the repository and enter the folder.**
    ```bash
    # If you cloned the main weco-cli repo already:
    cd examples/prompt

    # Otherwise:
    # git clone https://github.com/WecoAI/weco-cli.git
    # cd weco-cli/examples/prompt
    ```
2.  **Install dependencies.**
    ```bash
    # Ensure you have weco installed: pip install weco
    pip install openai datasets # Add any other dependencies if needed
    ```
3.  **Set your OpenAI API Key.**
    ```bash
    export OPENAI_API_KEY="your_openai_api_key_here"
    ```
4.  **Run Weco.** The command below iteratively modifies `EXTRA_INSTRUCTIONS` in `optimize.py`, runs `eval.py` to evaluate the prompt's effectiveness, reads the printed accuracy, and keeps the best prompt variations found.
    ```bash
    weco run --source optimize.py \
             --eval-command "python eval.py" \
             --metric accuracy \
             --maximize true \
             --steps 40 \
             --model gemini-2.5-pro-exp-03-25
    ```
    *Note: You can replace `--model gemini-2.5-pro-exp-03-25` with another powerful model like `o3` if you have the respective API keys set.*

During each evaluation round, you will see log lines similar to the following:

```text
[setup] loading 20 problems from AIME 2024 …
[progress] 5/20 completed, accuracy: 0.0000, elapsed 7.3 s
[progress] 10/20 completed, accuracy: 0.1000, elapsed 14.6 s
[progress] 15/20 completed, accuracy: 0.0667, elapsed 21.8 s
[progress] 20/20 completed, accuracy: 0.0500, elapsed 28.9 s
accuracy: 0.0500# AIME 2024 Prompt‑Engineering Example
This example shows how **Weco** can iteratively improve a prompt for solving American Invitational Mathematics Examination (AIME) problems. The experiment runs locally, requires only two short Python files, and finishes in a few hours on a laptop.

## Files in this folder

| File          | Purpose                                                                                                                                                           |
| :------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `optimize.py` | Holds the prompt template (instructing the LLM to reason step-by-step and use `\\boxed{}` for the final answer) and the function to call the LLM. Weco edits **only** this file during the search to refine the prompt template. |
| `eval.py`     | Defines the LLM model to use (`MODEL_TO_USE`). Downloads a small slice of the 2024 AIME dataset, calls `optimize.solve` in parallel (passing the chosen model), parses the LLM output, compares it to the ground truth, prints progress logs, and finally prints an `accuracy:` line that Weco reads. |

## Quick start

1. **Clone the repository and enter the folder.**
   ```bash
   git clone https://github.com/your‑fork/weco‑examples.git
   cd weco‑examples/aime‑2024
   ```
2. **Run Weco.**  The command below edits `EXTRA_INSTRUCTIONS` in `optimize.py`, invokes `eval.py` on every iteration, reads the printed accuracy, and keeps the best variants.
   ```bash
   weco --source optimize.py \
        --eval-command "python eval.py" \
        --metric accuracy \
        --maximize true \
        --steps 40 \
        --model gemini-2.5-flash-preview-04-17 \
        --addtional-instructions prompt_guide.md
   ```

During each evaluation round you will see log lines similar to the following.

```text
[setup] loading 20 problems from AIME 2024 …
[progress] 5/20 completed, elapsed 7.3 s
[progress] 10/20 completed, elapsed 14.6 s
[progress] 15/20 completed, elapsed 21.8 s
[progress] 20/20 completed, elapsed 28.9 s
accuracy: 0.0500
```

Weco then mutates the config, tries again, and gradually pushes the accuracy higher. On a modern laptop you can usually double the baseline score within thirty to forty iterations.

## How it works

* `eval_aime.py` slices the **Maxwell‑Jia/AIME_2024** dataset to twenty problems for fast feedback. You can change the slice in one line.
* The script sends model calls in parallel via `ThreadPoolExecutor`, so network latency is hidden.
* Every five completed items, the script logs progress and elapsed time.
* The final line `accuracy: value` is the only part Weco needs for guidance.
