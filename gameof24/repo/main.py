"""
Simple GameOf24 Agent - Main entry point for solving Game of 24 puzzles.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from agent import get_gameof24_agent
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
    """Extract the solution expression from agent response."""
    # Look for patterns like "solution is: expression = 24"
    patterns = [
        r"solution is[:\s]+([^=]+)\s*=\s*24",
        r"answer is[:\s]+([^=]+)\s*=\s*24",
        r"expression[:\s]+([^=]+)\s*=\s*24",
    ]

    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            solution = match.group(1).strip()
            return solution

    # Check if no solution exists
    if "no solution" in text_lower:
        return "No solution"

    # Try to find any expression that equals 24
    expression_pattern = r"([0-9+\-×÷*/().\s]+)\s*=\s*24"
    matches = re.findall(expression_pattern, text)
    if matches:
        # Return the last valid expression found
        return matches[-1].strip()

    return None


def solve_gameof24_problem(agent, problem_text):
    """Solve a single GameOf24 problem using the agent."""
    logger.info("Invoking agent for GameOf24 problem")
    logger.info(f"Problem text length: {len(problem_text)} characters")
    logger.debug(f"Problem: {problem_text[:200]}...")

    # Run the agent
    output = agent.solve(problem_text)
    logger.info("Agent returned a response.")
    logger.info(f"Raw LLM response length: {len(output)} characters")
    logger.debug(f"Raw LLM response (first 500 chars): {output[:500]}...")

    # Extract the solution expression
    answer = extract_answer(output)

    if answer is not None:
        logger.info(f"Extracted solution: {answer}")
        return answer
    else:
        logger.error("Failed to extract a valid solution from the response.")
        return None


def write_output(file_path, answer):
    """Write the predicted answer to a JSON file."""
    logger.info(f"Writing agent prediction to {file_path}")
    data = {"answer": answer}
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GameOf24 agent for solving puzzles")
    parser.add_argument(
        "task",
        type=str,
        help="The GameOf24 task string (e.g., 'Use the numbers 2, 3, 8, 8 to make 24')",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.json",
        help="Path to output JSON file (default: output.json)",
    )
    args = parser.parse_args()

    logger.info("--- GameOf24 Agent Run Started ---")
    logger.info(f"Processing task: {args.task}")

    agent = get_gameof24_agent()
    problem_text = args.task

    try:
        logger.info(f"Loaded problem: {problem_text[:100]}...")
        print(f"GameOf24 problem: {problem_text}")
        print("-" * 40)

        answer = solve_gameof24_problem(agent, problem_text)

        output_file = Path(args.output)
        if answer is not None:
            write_output(output_file, answer)
            print(f"\nAgent solution: {answer}")
            print(f"Result written to {output_file}")
        else:
            logger.warning("Agent did not produce a valid solution.")
            print("\nFailed to get valid solution from agent")
            # Write null answer
            write_output(output_file, None)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    logger.info("--- GameOf24 Agent Run Finished ---")


if __name__ == "__main__":
    main()
