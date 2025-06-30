"""
Simple AIME Agent - Main entry point for solving AIME mathematics problems.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from agent import get_aime_agent
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_answer(text):
    """Extract numerical answer from agent response."""
    # Look for patterns like "answer is 123" or "final answer: 456"
    patterns = [
        r"answer[:\s]+(\d{1,3})",
        r"final answer[:\s]+(\d{1,3})",
        r"result[:\s]+(\d{1,3})",
        r"\b(\d{1,3})\b(?!.*\b\d{1,3}\b)",  # Last number in text
    ]

    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            answer = int(match.group(1))
            if 0 <= answer <= 999:
                return answer

    # Try to find any 3-digit or less number
    numbers = re.findall(r"\b(\d{1,3})\b", text)
    if numbers:
        # Return the last valid number found
        for num in reversed(numbers):
            answer = int(num)
            if 0 <= answer <= 999:
                return answer

    return None


def solve_aime_problem(agent, problem_text):
    """Solve a single AIME problem using the agent."""
    logger.info("Invoking agent for AIME problem")
    logger.info(f"Problem text length: {len(problem_text)} characters")
    logger.debug(f"Problem: {problem_text[:200]}...")

    # Run the agent
    output = agent.solve(problem_text)
    logger.info("Agent returned a response.")
    logger.info(f"Raw LLM response length: {len(output)} characters")
    logger.debug(f"Raw LLM response (first 500 chars): {output[:500]}...")

    # Extract the numerical answer
    answer = extract_answer(output)

    if answer is not None:
        logger.info(f"Extracted answer: {answer}")
        return answer
    else:
        logger.error("Failed to extract a valid answer from the response.")
        return None


def write_output(file_path, answer):
    """Write the predicted answer to a JSON file."""
    logger.info(f"Writing agent prediction to {file_path}")
    data = {"answer": answer}
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AIME agent for solving math problems")
    parser.add_argument("task_file", type=str, nargs="?", default="task.json", help="Path to a JSON file containing the AIME task (default: task.json)")
    args = parser.parse_args()

    logger.info("--- AIME Agent Run Started ---")
    logger.info(f"Processing task file: {args.task_file}")

    agent = get_aime_agent()
    task_file_path = Path(args.task_file)

    try:
        # Read task from JSON file
        with open(task_file_path) as f:
            task_data = json.load(f)

        # Extract problem text from task_input field
        if "task_input" in task_data:
            problem_text = task_data["task_input"]
        else:
            # Fallback to other possible field names
            problem_text = task_data.get("problem", task_data.get("question", ""))

        if not problem_text:
            logger.error("No problem text found in task file")
            sys.exit(1)

        logger.info(f"Loaded problem: {problem_text[:100]}...")
        print(f"Loaded AIME problem from {task_file_path}")
        print("-" * 40)

        answer = solve_aime_problem(agent, problem_text)

        output_file = Path("output.json")
        if answer is not None:
            write_output(output_file, answer)
            print(f"\nAgent prediction: {answer}")
            print(f"Result written to {output_file}")
        else:
            logger.warning("Agent did not produce a valid answer.")
            print("\nFailed to get valid answer from agent")
            # Write null answer
            write_output(output_file, None)

    except FileNotFoundError:
        logger.error(f"Task file not found: {task_file_path}")
        print(f"Error: Task file not found at '{task_file_path}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in task file: {e}")
        print(f"Error: Invalid JSON in task file: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    logger.info("--- AIME Agent Run Finished ---")


if __name__ == "__main__":
    main()
