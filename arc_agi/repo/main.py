"""
Simple ARC Agent - Main entry point for running ARC-AGI tasks.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from agent import get_arc_agent
from agents import Runner
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


async def solve_arc_task(agent, task_name, demo_pairs, test_inputs):
    """Solve a single ARC task using the agent."""
    logger.info(f"Invoking agent for task: '{task_name}'")

    # Format the task for the agent
    prompt = format_task_for_agent(demo_pairs, test_inputs)
    logger.debug(f"Formatted prompt for agent:\n{prompt}")

    # Run the agent
    result = await Runner.run(agent, prompt)
    output = result.final_output
    logger.info("Agent returned a response.")
    logger.info(f"Raw LLM response length: {len(output)} characters")
    logger.info(f"Raw LLM response (first 500 chars): {output[:500]}...")

    # Parse the agent's response to extract grids
    json_candidate = ""
    try:
        logger.info("Looking for JSON within <answer> tags...")

        # Extract content between <answer> tags
        parts = output.split("<answer>")
        if len(parts) < 2:
            logger.error("No <answer> start tag found in the response.")
            return None

        content = parts[1].split("</answer>")[0]
        json_candidate = content.strip()

        logger.info(f"Extracted JSON candidate: {json_candidate[:200]}...")

        # Try to parse it
        grids = json.loads(json_candidate)

        # Validate structure
        if not isinstance(grids, list):
            logger.error(f"JSON is not a list, but a {type(grids)}")
            return None

        if len(grids) == 0:
            logger.error("JSON list is empty")
            return None

        # Check if it's a single grid or multiple grids
        if isinstance(grids[0], list) and len(grids[0]) > 0 and isinstance(grids[0][0], int):
            # Single grid format: [[1,2,3],[4,5,6]] -> [[[1,2,3],[4,5,6]]]
            grids = [grids]
        elif not (isinstance(grids[0], list) and len(grids[0]) > 0 and isinstance(grids[0][0], list)):
            logger.error(f"Invalid grid structure: The first element is of type {type(grids[0])}")
            return None

        logger.info(f"Successfully parsed {len(grids)} grid(s)")
        return grids

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.error(f"Failed to parse: {json_candidate}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during parsing: {e}", exc_info=True)
        return None


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

    agent = get_arc_agent()
    task_file_path = Path(args.task_file)

    try:
        train_pairs, test_inputs = load_task_from_file(task_file_path)

        logger.info(f"Task '{task_file_path.stem}' loaded: {len(train_pairs)} training examples, {len(test_inputs)} test cases.")
        print(f"Loaded task with {len(train_pairs)} training examples and {len(test_inputs)} test cases")
        print("-" * 40)

        task_name = task_file_path.stem
        predicted_outputs = await solve_arc_task(agent, task_name, train_pairs, test_inputs)

        output_file = task_file_path.parent / "output.json"
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
