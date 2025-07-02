"""
Example script showing how to use the ARC agent repository.

This script demonstrates the task injection format by:
1. Loading a task from the ARC benchmark
2. Saving it as a JSON file (without test outputs)
3. Running the agent on that file
4. Reading the agent's output and evaluating it
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.benchmarks.arc_agi import ARCAGIBenchmark
from benchmarks.core.base_benchmark import Split


def evaluate_predictions(predictions, ground_truth):
    """Compare predicted grids with ground truth."""
    if not predictions:
        print("Evaluation failed: No predictions provided.")
        return 0.0

    if len(predictions) != len(ground_truth):
        print(f"Evaluation failed: Mismatched number of test cases (Expected {len(ground_truth)}, Got {len(predictions)}).")
        return 0.0

    correct_count = 0
    for i, (pred, expected) in enumerate(zip(predictions, ground_truth, strict=False)):
        if pred == expected:
            correct_count += 1
            print(f"  - Test case {i + 1}: Correct")
        else:
            print(f"  - Test case {i + 1}: Incorrect")

    score = correct_count / len(ground_truth)
    return score


def load_arc_benchmark():
    print("ARC Agent Example - Task Injection & Evaluation")
    print("=" * 60)

    # 1. Load benchmark and select a task
    print("\n1. Loading ARC-AGI benchmark...")
    benchmark = ARCAGIBenchmark()
    train_ids = benchmark.get_task_ids(Split.TRAIN)
    if not train_ids:
        print("Error: No training tasks found in benchmark.")
        return

    return train_ids, benchmark


def get_task_info(benchmark, task_id):
    print(f"   - Selected task: {task_id}")

    # 2. Prepare and save the task file for the agent
    task_info = benchmark.get_task_info(task_id)
    ground_truth_outputs = task_info["test_outputs"]
    demo_pairs = benchmark.get_demonstration_pairs(task_id)

    return task_info, ground_truth_outputs, demo_pairs


async def run_arc_agent(task_info, ground_truth_outputs, demo_pairs):
    """Demonstrate the ARC agent by extracting a task, running it, and evaluating the output."""

    task_for_agent = {
        "train": demo_pairs,
        "test": [{"input": test_input} for test_input in task_info["test_inputs"]],
    }

    repo_dir = Path(__file__).parent / "repo"
    task_file = repo_dir / "task.json"
    print(f"\n2. Saving task for agent to: {task_file}")
    with open(task_file, "w") as f:
        json.dump(task_for_agent, f, indent=2)
    print(f"   - Task file created with {len(task_for_agent['train'])} training pairs and {len(task_for_agent['test'])} test inputs.")

    # 3. Run the agent
    print("\n3. Running ARC agent on the task file...")
    print("-" * 40)

    original_dir = os.getcwd()
    try:
        os.chdir(repo_dir)
        result = subprocess.run([sys.executable, "main.py", "task.json"], capture_output=True, text=True, check=False)

        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

    finally:
        os.chdir(original_dir)
    print("-" * 40)

    # 4. Evaluate the agent's output
    print("\n4. Evaluating agent's predictions...")
    output_file = repo_dir / "output.json"
    if not output_file.exists():
        print(f"Error: Output file '{output_file}' not found. The agent might have failed.")
        return

    with open(output_file) as f:
        predictions = json.load(f)

    score = evaluate_predictions(predictions, ground_truth_outputs)
    print(f"\nFinal Score: {score:.2f}")

    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == "__main__":
    asyncio.run(run_arc_agent_example())
