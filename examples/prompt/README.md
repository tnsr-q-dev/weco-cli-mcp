# AIME Prompt Engineering Example with Weco

This example shows how **Weco** can iteratively improve a prompt for solving American Invitational Mathematics Examination (AIME) problems. The experiment runs locally, requires only two short Python files, and aims to improve the accuracy metric.

This example uses `gpt-4o-mini` via the OpenAI API by default. Ensure your `OPENAI_API_KEY` environment variable is set.

## Files in this folder

| File          | Purpose                                                                                                                                                           |
| :------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `optimize.py` | Holds the prompt template (instructing the LLM to reason step-by-step and use `\\boxed{}` for the final answer) and the mutable `EXTRA_INSTRUCTIONS` string. Weco edits **only** this file during the search. |
| `eval.py`     | Downloads a small slice of the 2024 AIME dataset, calls `optimize.solve` in parallel, parses the LLM output (looking for `\\boxed{}`), compares it to the ground truth, prints progress logs, and finally prints an `accuracy:` line that Weco reads. |


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
        --goal maximize \
        --steps 40 \
        --model gemini-2.5-flash-preview-04-17 \
        --additional-instructions prompt_guide.md
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
