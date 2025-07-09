# Main entrypoint
"""
Simple LongBench Agent - Main entry point for solving LongBench tasks.
"""

import argparse
import logging
import sys
from pathlib import Path

from agent import solve_longbench_task
from task_utils import load_task_from_file, write_output

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LongBench agent for solving long-context QA tasks")
    parser.add_argument("task_id", type=str, help="Task ID to process (looks for inputs/{task_id}.json)")
    args = parser.parse_args()

    logger.info("--- LongBench Agent Run Started ---")
    logger.info(f"Processing task ID: {args.task_id}")

    # Set up paths
    task_file_path = Path("inputs") / f"{args.task_id}.json"
    output_file_path = Path("outputs") / f"{args.task_id}.json"

    try:
        # Load task from file
        task = load_task_from_file(task_file_path)

        logger.info(f"Task loaded: {task['domain']} - {task['sub_domain']}")
        logger.info(f"Difficulty: {task['difficulty']}, Length: {task['length']}")
        logger.info(f"Context length: {len(task['context'])} characters")

        print(f"Loaded task: {task['domain']} - {task['sub_domain']}")
        print(f"Context length: {len(task['context'])} characters")
        print("-" * 40)

        # Solve the task
        predicted_answer = solve_longbench_task(question=task["question"], context_str=task["context"], choices=task["choices"])

        # Write output
        output_dir = output_file_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        output_data = {"task_id": task.get("_id", args.task_id), "predicted_answer": predicted_answer, "choices": task["choices"]}

        write_output(output_file_path, output_data)
        print(f"\nAgent prediction written to {output_file_path}")
        print(f"Predicted answer: {predicted_answer}")

    except FileNotFoundError:
        logger.error(f"Task file not found: {task_file_path}")
        print(f"Error: Task file not found at '{task_file_path}'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    logger.info("--- LongBench Agent Run Finished ---")


if __name__ == "__main__":
    main()
