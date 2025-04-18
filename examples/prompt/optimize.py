"""
optimize.py
This module holds the prompt template and the LLM call.
Weco modifies this file to optimize the prompt instructions.
The model used for the LLM call is passed in from eval.py.
"""

from openai import OpenAI

client = OpenAI()  # API key must be in OPENAI_API_KEY

PROMPT_TEMPLATE = """You are an expert competition mathematician specializing in AIME problems. Your goal is to solve the following AIME problem accurately and present the final answer as a three-digit integer (000-999) enclosed in \\boxed{{}}. The evaluation script relies *absolutely* on this specific format \\boxed{{XXX}}. Meticulous accuracy is paramount.

Please structure your detailed thinking process as follows:

1.  **Understand the Problem:**
    *   Carefully read the problem statement multiple times.
    *   Identify exactly what is being asked (the final quantity to find).
    *   List all given conditions, constraints, variables, and specific definitions or terminology.
    *   Identify the type of math problem (e.g., algebra, geometry, number theory, combinatorics).
    *   Ensure you fully grasp all aspects of the problem before proceeding.

2.  **Plan the Solution:**
    *   Outline the mathematical approach you will take. What are the key steps?
    *   Identify relevant mathematical concepts, theorems, formulas, or techniques (e.g., casework, symmetry, invariants, specific algebraic/geometric manipulations, modular arithmetic, pigeonhole principle).
    *   Consider if the problem can be simplified, reframed, or approached using a specific known AIME strategy.
    *   Consider different strategies and choose the most promising one, explaining why.
    *   Break down the problem into smaller, manageable parts if necessary.

3.  **Execute the Plan:**
    *   Work through the problem step-by-step according to your plan.
    *   Show ALL calculations and logical derivations explicitly. Do not skip steps. Ensure every step logically follows from the previous one.
    *   Clearly state your reasoning at each stage. Define any variables used.
    *   Perform calculations accurately. Pay close attention to details. Double-check arithmetic.
    *   Keep track of your intermediate results and units (if any).

4.  **Verify the Solution:**
    *   Review your entire solution process from beginning to end.
    *   Re-read the original problem statement. Did you answer the specific question asked?
    *   Double-check your calculations, especially critical ones and the final arithmetic.
    *   Does the answer satisfy all the conditions and constraints given in the problem statement?
    *   Does the answer make sense in the context of the problem? (e.g., Is it the right order of magnitude? Is it the correct type of number?)
    *   Consider potential edge cases or common pitfalls. Did you account for them?

5.  **Final Answer:**
    *   State your final result clearly.
    *   The final answer must be a single three-digit integer between 000 and 999, inclusive.
    *   Enclose the three-digit integer in the required format: \\boxed{{XXX}}. For example, if the answer is 42, write \\boxed{{042}}. If the answer is 123, write \\boxed{{123}}.

Problem:
{problem}

Solution:
"""

# Modify the function signature to accept model_name
def solve(problem: str, model_name: str) -> str:
    """Return the model's raw text answer for one problem using the specified model."""
    prompt = PROMPT_TEMPLATE.format(problem=problem)

    response = client.chat.completions.create(
        model=model_name, # Use the passed-in model name
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0 # Set temperature to 0 for deterministic output
    )
    return response.choices[0].message.content.strip()