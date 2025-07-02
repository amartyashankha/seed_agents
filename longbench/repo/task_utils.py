"""
Task utilities for LongBench agent.
"""

import json
import logging

logger = logging.getLogger(__name__)


def load_task_from_file(file_path):
    """Load a LongBench task from a JSON file.

    Expected format:
    {
        "_id": "task_id",
        "domain": "domain_name",
        "sub_domain": "sub_domain_name",
        "difficulty": "easy|hard",
        "length": "short|medium|long",
        "question": "question text",
        "context": "long context text",
        "choices": ["A", "B", "C", "D"],
        "answer": "A|B|C|D"  # Optional, for evaluation
    }
    """
    logger.info(f"Loading task from {file_path}")

    with open(file_path) as f:
        task = json.load(f)

    # Validate required fields
    required_fields = ["question", "context", "choices"]
    for field in required_fields:
        if field not in task:
            raise ValueError(f"Task file missing required field: {field}")

    # Validate choices format
    if not isinstance(task["choices"], list) or len(task["choices"]) != 4:
        raise ValueError("Task must have exactly 4 choices")

    return task


def write_output(file_path, data):
    """Write the agent's output to a JSON file."""
    logger.info(f"Writing output to {file_path}")

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Output written successfully")
