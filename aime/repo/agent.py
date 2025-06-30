"""
AIME Agent implementation using the agents library.
"""

from agents import BasicGenerateAgent


def get_aime_agent():
    """Create and return an AIME problem-solving agent."""

    # Create a comprehensive AIME solver agent
    aime_agent = BasicGenerateAgent(
        "AIME Solver",
        generation_config={
            "model": "gpt-4",
            "temperature": 0.1,
            "max_tokens": 2000,
        },
        system_prompt="""You are an expert mathematician specializing in AIME (American Invitational Mathematics Examination) problems.

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
""",
    )

    return aime_agent
