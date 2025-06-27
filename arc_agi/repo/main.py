"""
Simple ARC Agent - Main entry point for running ARC-AGI tasks.
"""

import argparse
import asyncio
import json
import logging
import re
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
    try:
        # Look for JSON array pattern
        json_match = re.search(r"\[\s*\[.*?\]\s*\]", output, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            logger.info(f"Found JSON pattern: {json_str[:200]}...")

            try:
                grids = json.loads(json_str)
                if isinstance(grids, list) and len(grids) > 0 and isinstance(grids[0], list) and isinstance(grids[0][0], int):
                    grids = [grids]
                logger.info(f"Successfully parsed {len(grids)} grid(s) from agent output.")
                return grids
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing failed: {json_err}")
                logger.error(f"Problematic JSON string: {json_str}")

                # Try to fix common JSON issues
                fixed_json = json_str.replace("\n", "").replace("\r", "").strip()
                # Remove trailing commas
                fixed_json = re.sub(r",\s*([}\]])", r"\1", fixed_json)

                try:
                    grids = json.loads(fixed_json)
                    if isinstance(grids, list) and len(grids) > 0 and isinstance(grids[0], list) and isinstance(grids[0][0], int):
                        grids = [grids]
                    logger.info(f"Successfully parsed {len(grids)} grid(s) after JSON cleanup.")
                    return grids
                except json.JSONDecodeError:
                    logger.error("JSON cleanup failed, returning None")
                    return None
        else:
            logger.error("No JSON array pattern found in agent output")
            logger.error(f"Full output: {output}")
            return None
    except Exception as e:
        logger.error(f"Error parsing agent output: {e}", exc_info=True)
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
