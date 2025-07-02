"""
Example script showing how to use the LongBench agent on real benchmark tasks.

This script demonstrates running multiple LongBench tasks in parallel by:
1. Loading tasks from the LongBench benchmark
2. Sampling k random tasks
3. Running the agent on multiple tasks in parallel
4. Evaluating the results
"""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from benchmarks.benchmarks.longbench.longbench_benchmark import LongBenchBenchmark
from benchmarks.core.base_benchmark import Split


async def run_agent_on_task(script_dir: Path, task_id: str, show_output: bool = True) -> dict:
    """Run the agent on a single task asynchronously."""
    main_py = script_dir / "main.py"

    print(f"   Starting task {task_id}...")

    if show_output:
        # Let output go to terminal for real-time feedback
        process = await asyncio.create_subprocess_exec(sys.executable, str(main_py), task_id, cwd=script_dir)
        await process.wait()
        return_code = process.returncode
    else:
        # Capture output for silent execution
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(main_py), task_id, cwd=script_dir, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return_code = process.returncode

    # Read the output file
    output_file = script_dir / "outputs" / f"{task_id}.json"
    result = {
        "task_id": task_id,
        "success": False,
        "predicted_answer": None,
        "return_code": return_code,
    }

    if output_file.exists():
        try:
            with open(output_file) as f:
                output_data = json.load(f)
            result["success"] = True
            result["predicted_answer"] = output_data.get("predicted_answer")
        except Exception as e:
            result["error"] = str(e)

    print(f"   Completed task {task_id} (success: {result['success']})")
    return result


def save_task_to_file(task_data: dict, task_id: str, inputs_dir: Path):
    """Save a task to the inputs directory."""
    task_file = inputs_dir / f"{task_id}.json"

    # Convert benchmark task format to agent format
    agent_task = {
        "_id": task_data.get("_id", task_id),
        "domain": task_data.get("domain", ""),
        "sub_domain": task_data.get("sub_domain", ""),
        "difficulty": task_data.get("difficulty", ""),
        "length": task_data.get("length", ""),
        "question": task_data.get("question", ""),
        "context": task_data.get("context", ""),
        "choices": task_data.get("choices", []),
        "answer": task_data.get("correct_answer_letter", ""),  # For evaluation
    }

    with open(task_file, "w") as f:
        json.dump(agent_task, f, indent=2)

    return agent_task


async def run_parallel_evaluation(limit: int = 5, sequential: bool = False):
    """Run parallel evaluation on LongBench tasks."""
    print(f"LongBench Parallel Agent Evaluation (limit={limit})")
    print("=" * 60)

    # 1. Load benchmark and sample tasks
    print("\n1. Loading LongBench benchmark...")
    benchmark = LongBenchBenchmark(validation_size=50)

    # Get validation tasks for testing
    val_task_ids = benchmark.get_task_ids(split=Split.VALIDATION)
    print(f"   Available validation tasks: {len(val_task_ids)}")

    # Sample random tasks
    if limit > len(val_task_ids):
        limit = len(val_task_ids)
        print(f"   Limiting to available tasks: {limit}")

    selected_task_ids = random.sample(val_task_ids, limit)
    print(f"   Selected {len(selected_task_ids)} random tasks")

    # 2. Prepare task files
    print("\n2. Preparing task files...")
    script_dir = Path(__file__).parent / "repo"
    inputs_dir = script_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)

    tasks_data = {}
    for task_id in selected_task_ids:
        task_info = benchmark.get_task_info(task_id)
        agent_task = save_task_to_file(task_info, task_id, inputs_dir)
        tasks_data[task_id] = {"agent_task": agent_task, "benchmark_task": task_info}

    print(f"   Saved {len(tasks_data)} task files to {inputs_dir}")

    # 3. Run agents
    mode = "sequentially" if sequential else "in parallel"
    print(f"\n3. Running agents {mode}...")
    start_time = time.time()

    if sequential:
        # Run tasks one by one for easier debugging
        results = []
        for task_id in selected_task_ids:
            result = await run_agent_on_task(script_dir, task_id, show_output=True)
            results.append(result)
    else:
        # Create tasks for async execution
        agent_tasks = [run_agent_on_task(script_dir, task_id, show_output=True) for task_id in selected_task_ids]
        # Run all tasks in parallel
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"   Completed {len(results)} tasks in {total_time:.2f} seconds")
    print(f"   Average time per task: {total_time / len(results):.2f} seconds")

    # 4. Evaluate results
    print("\n4. Evaluating results...")
    correct_count = 0
    successful_count = 0
    failed_tasks = []

    for result in results:
        if isinstance(result, Exception):
            print(f"   Exception for task: {result}")
            continue

        # Type check to ensure result is a dict
        if not isinstance(result, dict):
            print(f"   Invalid result type: {type(result)}")
            continue

        task_id = result["task_id"]
        success = result["success"]
        predicted = result["predicted_answer"]

        if success:
            successful_count += 1
            expected = tasks_data[task_id]["agent_task"]["answer"]

            # Evaluate using benchmark
            eval_result = benchmark.evaluate(task_id, predicted)
            is_correct = eval_result.score == 1.0

            if is_correct:
                correct_count += 1
                print(f"   ✓ {task_id}: {predicted} (correct)")
            else:
                print(f"   ✗ {task_id}: {predicted} (expected {expected})")
        else:
            failed_tasks.append(task_id)
            print(f"   ✗ {task_id}: FAILED")

    # 5. Summary
    print("\n5. Summary:")
    print(f"   Total tasks: {len(results)}")
    print(f"   Successful: {successful_count}")
    print(f"   Failed: {len(failed_tasks)}")
    print(f"   Correct: {correct_count}")
    print(f"   Accuracy: {correct_count / successful_count * 100:.1f}%" if successful_count > 0 else "   Accuracy: N/A")
    print(f"   Total time: {total_time:.2f}s")

    if failed_tasks:
        print(f"\n   Failed tasks: {failed_tasks}")

    # Show some example results
    print("\n6. Example results:")
    for i, result in enumerate(results[:3]):  # Show first 3 results
        if isinstance(result, Exception):
            continue

        # Type check to ensure result is a dict
        if not isinstance(result, dict):
            continue

        task_id = result["task_id"]
        if result["success"]:
            task_info = tasks_data[task_id]["benchmark_task"]
            print(f"\n   Task {i + 1} ({task_id}):")
            print(f"   Domain: {task_info['domain']}")
            print(f"   Question: {task_info['question'][:100]}...")
            print(f"   Predicted: {result['predicted_answer']}")
            print(f"   Expected: {tasks_data[task_id]['agent_task']['answer']}")

    print("\n" + "=" * 60)
    print("Parallel evaluation complete!")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Run LongBench agent on multiple tasks in parallel")
    parser.add_argument("--limit", type=int, default=5, help="Number of tasks to run (default: 5)")
    parser.add_argument("--sequential", action="store_true", help="Run tasks sequentially instead of in parallel (for debugging)")
    args = parser.parse_args()

    # Run the async evaluation
    asyncio.run(run_parallel_evaluation(args.limit, args.sequential))


if __name__ == "__main__":
    main()
