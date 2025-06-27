"""
Simple ARC Agent - Minimal agent for solving ARC-AGI tasks.
"""

from agents import Agent

# Simple ARC solver agent
arc_agent = Agent(
    name="arc_solver",
    model="o3-mini",
    instructions="""You are an ARC-AGI solver agent. Your task is to analyze abstract reasoning puzzles and find patterns.

Given demonstration input/output pairs, identify the transformation rule and apply it to test inputs.

Approach:
1. Carefully analyze each demonstration pair
2. Identify patterns in transformations (colors, shapes, movements, symmetries)
3. Formulate a clear rule
4. Apply the rule to test inputs

Focus on:
- Grid transformations
- Color mappings
- Spatial patterns
- Counting and arithmetic operations
- Symmetries and rotations
""",
    tools=[],  # No external tools needed
)


def get_arc_agent() -> Agent:
    """Get the ARC solver agent."""
    return arc_agent
