"""
Simple ARC Agent - Main entry point for running ARC-AGI tasks.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from formatting import format_task_for_agent
from task_utils import load_task_from_file

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def solve_arc_task(demo_pairs, test_inputs):
    """Solve a single ARC task using the agent."""

    # Format the task for the agent
    prompt = format_task_for_agent(demo_pairs, test_inputs)
    logger.debug(f"Formatted prompt for agent:\n{prompt}")

    for pair in demo_pairs:
        formatted_pair = format_pair_for_llm(pair)
        pass


def write_output(file_path, data):
    """Write the predicted output to a JSON file."""
    logger.info(f"Writing agent predictions to {file_path}")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ARC agent for solving tasks from JSON files")
    parser.add_argument("task_file", type=str, nargs="?", default="task.json", help="Path to a JSON file containing an ARC task (default: task.json)")
    args = parser.parse_args()

    logger.info("--- ARC Agent Run Started ---")
    logger.info(f"Processing task file: {args.task_file}")

    task_file_path = Path(args.task_file)

    try:
        train_pairs, test_inputs = load_task_from_file(task_file_path)

        logger.info(f"Task '{task_file_path.stem}' loaded: {len(train_pairs)} training examples, {len(test_inputs)} test cases.")
        print(f"Loaded task with {len(train_pairs)} training examples and {len(test_inputs)} test cases")
        print("-" * 40)

        predicted_outputs = await solve_arc_task(train_pairs, test_inputs)

        output_file = task_file_path.parent / f"{task_file_path.stem}_output.json"
        if predicted_outputs:
            write_output(output_file, predicted_outputs)
            print(f"\nAgent predictions written to {output_file}")
        else:
            logger.warning("Agent did not produce a valid output.")
            print("\nFailed to get valid output from agent")

    except FileNotFoundError:
        logger.error(f"Task file not found: {task_file_path}")
        print(f"Error: Task file not found at '{task_file_path}'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    logger.info("--- ARC Agent Run Finished ---")


if __name__ == "__main__":
    asyncio.run(main())
