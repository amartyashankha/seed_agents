"""
Utilities for formatting ARC tasks into prompts.
"""


def format_grid(grid):
    """Format a grid for display as a Python list (multiline as a grid)."""
    lines = []
    lines.append("[")
    for i, row in enumerate(grid):
        if i == len(grid) - 1:
            lines.append(f"    {list(row)}")
        else:
            lines.append(f"    {list(row)},")
    lines.append("]")
    return "\n".join(lines)


def format_pair_for_llm(pair):
    """Format a pair for the LLM."""
    return f"Input:\n{format_grid(pair.input)}\n\nOutput:\n{format_grid(pair.output)}"


def format_task_for_agent(demo_pairs, test_inputs):
    """Format ARC task for the agent."""
    prompt = "Analyze these input/output demonstration pairs and find the pattern:\n\n"

    for i, pair in enumerate(demo_pairs):
        prompt += f"Demo {i + 1}:\n{format_pair_for_llm(pair)}\n\n"

    prompt += "Now apply the same transformation to these test inputs:\n\n"

    for i, test_input in enumerate(test_inputs):
        prompt += f"Test {i + 1}:\n{format_grid(test_input.input)}\n\n"

    prompt += """Please provide your answer as a JSON array of grids. Each grid should be a 2D array of integers.
Example format: [[[1,2],[3,4]], [[5,6],[7,8]]] for two test outputs."""

    return prompt
