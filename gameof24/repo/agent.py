"""
GameOf24 Agent implementation using OpenAI directly.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


def get_gameof24_agent():
    """Create and return a GameOf24 problem-solving agent."""

    class GameOf24Agent:
        def __init__(self):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.system_prompt = """You are an expert at solving Game of 24 puzzles.

CRITICAL: Your task is to use four given numbers and basic arithmetic operations (+, -, ×, ÷) to create an expression that equals 24.

Your approach:
1. Carefully read the four numbers provided
2. Try different combinations of operations and groupings
3. Use parentheses to control the order of operations
4. Show your calculation step by step
5. Verify that your expression equals 24
6. If no solution exists, clearly state "No solution"

Rules:
- You must use ALL four numbers exactly once
- You can only use basic arithmetic operations
- You can use parentheses to group operations
- The result must equal exactly 24

Output format:
- Show your solution as a clear mathematical expression
- State your final answer as "[expression] = 24"
- If no solution exists, state "No solution exists for these numbers"
"""

        def solve(self, problem):
            """Solve a GameOf24 problem and return the answer."""
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": problem},
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error solving problem: {e!s}"

    return GameOf24Agent()
