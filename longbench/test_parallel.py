#!/usr/bin/env python3
"""
Simple test script to verify parallel execution of the LongBench agent.
"""

import asyncio
import time

from multi_agent.genes.seed_agents.longbench.simple_tool_calling.agent import solve_longbench_task

# Sample test data
SAMPLE_TASKS = [
    {
        "task_id": "test_1",
        "question": "What is the capital of France?",
        "context": "France is a country in Europe. Its capital city is Paris, which is located in the north-central part of the country. Paris is known for the Eiffel Tower and is a major cultural center.",
        "choices": ["A) London", "B) Paris", "C) Berlin", "D) Madrid"],
    },
    {
        "task_id": "test_2",
        "question": "What is 2 + 2?",
        "context": "Mathematics is the study of numbers and operations. Addition is one of the basic operations. When we add 2 and 2, we get 4.",
        "choices": ["A) 3", "B) 4", "C) 5", "D) 6"],
    },
    {
        "task_id": "test_3",
        "question": "What color is the sky?",
        "context": "The sky appears blue during the day due to light scattering. This phenomenon is called Rayleigh scattering. The atmosphere scatters shorter wavelengths of light more than longer ones.",
        "choices": ["A) Red", "B) Green", "C) Blue", "D) Yellow"],
    },
]


async def run_single_task(task_data: dict) -> dict:
    """Run a single task and return the result with timing."""
    start_time = time.time()

    try:
        result = await solve_longbench_task(question=task_data["question"], context_str=task_data["context"], choices=task_data["choices"])

        end_time = time.time()
        return {"task_id": task_data["task_id"], "success": True, "result": result, "time": end_time - start_time}
    except Exception as e:
        end_time = time.time()
        return {"task_id": task_data["task_id"], "success": False, "error": str(e), "time": end_time - start_time}


async def test_sequential():
    """Test running tasks sequentially."""
    print("Testing Sequential Execution...")
    start_time = time.time()

    results = []
    for task in SAMPLE_TASKS:
        result = await run_single_task(task)
        results.append(result)
        print(f"  Completed {result['task_id']}: {result.get('result', 'FAILED')} ({result['time']:.2f}s)")

    total_time = time.time() - start_time
    print(f"Sequential total time: {total_time:.2f}s\n")
    return results, total_time


async def test_parallel():
    """Test running tasks in parallel using asyncio.gather."""
    print("Testing Parallel Execution (asyncio.gather)...")
    start_time = time.time()

    # Create all tasks
    tasks = [run_single_task(task_data) for task_data in SAMPLE_TASKS]

    # Run all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_result = {"task_id": SAMPLE_TASKS[i]["task_id"], "success": False, "error": str(result), "time": 0}
        else:
            processed_result = result

        processed_results.append(processed_result)
        print(f"  Completed {processed_result['task_id']}: {processed_result.get('result', 'FAILED')} ({processed_result['time']:.2f}s)")

    total_time = time.time() - start_time
    print(f"Parallel total time: {total_time:.2f}s\n")
    return processed_results, total_time


async def main():
    """Main test function."""
    print("=" * 60)
    print("LongBench Agent Parallel Execution Test")
    print("=" * 60)

    # Test sequential execution
    seq_results, seq_time = await test_sequential()

    # Test parallel execution
    par_results, par_time = await test_parallel()

    # Compare results
    print("=" * 60)
    print("Comparison:")
    print(f"Sequential time: {seq_time:.2f}s")
    print(f"Parallel time:   {par_time:.2f}s")
    print(f"Speedup:         {seq_time / par_time:.2f}x")

    # Verify results are the same
    print("\nResult verification:")
    for seq_result, par_result in zip(seq_results, par_results, strict=False):
        seq_answer = seq_result.get("result", "FAILED")
        par_answer = par_result.get("result", "FAILED")
        match = "✓" if seq_answer == par_answer else "✗"
        print(f"  {seq_result['task_id']}: {match} Sequential={seq_answer}, Parallel={par_answer}")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
