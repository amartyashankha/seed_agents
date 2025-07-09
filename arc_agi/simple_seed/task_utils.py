"""
Task utilities for loading ARC tasks and evaluating outputs.
"""

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Pair:
    input: list[list[int]]
    output: list[list[int]] | None


def load_task_from_file(file_path):
    """Load an ARC task from a JSON file."""
    logger.debug(f"Attempting to load task from {file_path}")
    with open(file_path) as f:
        data = json.load(f)

    # Validate the format
    if "train" not in data or "test" not in data:
        logger.error(f"Invalid task format in {file_path}: missing 'train' or 'test' keys.")
        raise ValueError("Task file must contain 'train' and 'test' fields")

    # Extract train pairs and test inputs
    train_pairs = [Pair(input=pair["input"], output=pair["output"]) for pair in data["train"]]
    test_inputs = [Pair(input=pair["input"], output=None) for pair in data["test"]]

    logger.debug(f"Successfully loaded task from {file_path}")
    return train_pairs, test_inputs


def evaluate_against_expected(predicted_outputs, expected_outputs):
    """Evaluate predicted outputs against expected outputs."""
    if not predicted_outputs or len(predicted_outputs) != len(expected_outputs):
        return 0.0, f"Expected {len(expected_outputs)} outputs, got {len(predicted_outputs) if predicted_outputs else 0}"

    correct = 0
    explanations = []

    for i, (pred, expected) in enumerate(zip(predicted_outputs, expected_outputs, strict=False)):
        if pred == expected:
            correct += 1
            explanations.append(f"Test {i + 1}: Correct!")
        else:
            explanations.append(f"Test {i + 1}: Incorrect")

    score = correct / len(expected_outputs)
    logger.info(f"Evaluation result: {score:.2f} ({correct}/{len(expected_outputs)} correct)")
    return score, " ".join(explanations)
