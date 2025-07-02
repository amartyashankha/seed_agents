"""
Simple ARC Agent - Minimal agent for solving ARC-AGI tasks.
"""

from agents import Agent

# Simple ARC solver agent
arc_agent = Agent(
    name="arc_solver",
    model="gpt-4.1-mini",
    instructions="""You are a top-tier ARC-AGI solver. Your goal is to analyze abstract puzzles and output the solution grid.

Analyze the demonstration pairs to find the transformation rule. Apply that rule to the test inputs.

Your final response MUST contain the solution JSON enclosed in <answer> tags.
For example:
<answer>
[[[0, 1, 2], [1, 2, 0], [2, 0, 1]]]
</answer>

Do NOT include any other text inside the <answer> tags. The JSON should be the only content.
""",
    tools=[],  # No external tools needed
)


def get_arc_agent() -> Agent:
    """Get the ARC solver agent."""
    return arc_agent
