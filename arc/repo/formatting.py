"""
Utilities for formatting ARC tasks into prompts.
"""


def format_grid(grid):
    """Format a grid for display."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def format_task_for_agent(demo_pairs, test_inputs):
    """Format ARC task for the agent."""
    prompt = "Analyze these input/output demonstration pairs and find the pattern:\n\n"

    for i, pair in enumerate(demo_pairs):
        prompt += f"Demo {i + 1}:\n"
        prompt += f"Input:\n{format_grid(pair['input'])}\n"
        prompt += f"Output:\n{format_grid(pair['output'])}\n\n"

    prompt += "Now apply the same transformation to these test inputs:\n\n"

    for i, test_input in enumerate(test_inputs):
        prompt += f"Test {i + 1}:\n{format_grid(test_input)}\n\n"

    prompt += """Please provide your answer as a JSON array of grids. Each grid should be a 2D array of integers.
Example format: [[[1,2],[3,4]], [[5,6],[7,8]]] for two test outputs."""

    return prompt
