"""
AIME Agent implementation using OpenAI directly.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


def get_aime_agent():
    """Create and return an AIME problem-solving agent."""

    class AIMEAgent:
        def __init__(self):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.system_prompt = """You are an expert mathematician specializing in AIME (American Invitational Mathematics Examination) problems.

CRITICAL: Your final answer MUST be a single integer between 0 and 999 (inclusive).

Your approach:
1. Carefully read and understand the problem
2. Identify the key mathematical concepts and techniques needed
3. Work through the solution step by step with clear reasoning
4. Show all calculations and algebraic manipulations
5. Double-check your work
6. State your final answer clearly as "The answer is [number]" or "Final answer: [number]"

Common AIME techniques to consider:
- Algebraic manipulation and clever substitutions
- Geometric properties and coordinate geometry
- Number theory (divisibility, modular arithmetic, congruences)
- Combinatorics and counting principles
- Trigonometric identities and complex numbers
- Sequences, series, and generating functions
- Probability and expected value

Remember:
- Be meticulous with calculations
- Look for elegant solutions and patterns
- The answer must be an integer from 0 to 999
- Always verify your answer makes sense in the context of the problem
"""

        def solve(self, problem):
            """Solve an AIME problem and return the answer."""
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": problem}],
                    temperature=0.1,
                    max_tokens=2000,
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error solving problem: {e!s}"

    return AIMEAgent()
