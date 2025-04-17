"""
optimize.py
This module holds the prompt template and the LLM call.
Weco edits the EXTRA_INSTRUCTIONS string to search for better prompts.
"""

from openai import OpenAI

client = OpenAI()          # API key must be in OPENAI_API_KEY
MODEL = "gpt-4o-mini"      # change if you have another model

PROMPT_TEMPLATE = """You are an expert competition mathematician.
Solve the following AIME problem.
Think step by step and keep scratch work inside triple hash marks ###.
Give the final answer as a three digit integer only.

Problem:
{problem}

###
{extra_instructions}
"""

# === BEGIN MUTABLE PROMPT ===
EXTRA_INSTRUCTIONS = "Use bullet points in the scratch work."
# === END MUTABLE PROMPT ===


def solve(problem: str) -> str:
    """Return the model's raw text answer for one problem."""
    prompt = PROMPT_TEMPLATE.format(
        problem=problem,
        extra_instructions=EXTRA_INSTRUCTIONS,
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
