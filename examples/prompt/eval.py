"""
eval_aime.py  (parallel with progress logs)

Downloads a slice of AIME 2024, calls optimize.solve in parallel,
prints progress every N samples, and finally prints accuracy
in the format that Weco expects.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
import optimize                       # the file Weco mutates

# ---------------------------------------------------------------------
# Configuration
TOTAL_SAMPLES = 20                    # how many problems to load
NUM_WORKERS   = 20                    # concurrent LLM calls
LOG_EVERY     = 5                     # print progress after this many
# ---------------------------------------------------------------------

print(f"[setup] loading {TOTAL_SAMPLES} problems from AIME 2024 …")
DATA = load_dataset(
    "Maxwell-Jia/AIME_2024",
    split=f"train[:{TOTAL_SAMPLES}]",
    cache_dir=".cache"
)

def extract_number(text: str) -> str:
    m = re.search(r"\b(\d{1,3})\b", text)
    return m.group(1) if m else ""

def score_one(row) -> bool:
    guess = extract_number(optimize.solve(row["Problem"]))
    return guess == str(row["Answer"])

def accuracy() -> float:
    correct = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = [pool.submit(score_one, row) for row in DATA]
        for idx, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                correct += 1

            if idx % LOG_EVERY == 0 or idx == TOTAL_SAMPLES:
                elapsed = time.time() - start
                print(
                    f"[progress] {idx}/{TOTAL_SAMPLES} completed, "
                    f"elapsed {elapsed:.1f} s"
                )

    return correct / TOTAL_SAMPLES

if __name__ == "__main__":
    acc = accuracy()
    print(f"accuracy: {acc:.4f}")     # Weco parses this line