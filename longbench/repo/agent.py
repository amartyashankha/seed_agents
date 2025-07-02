"""
LongBench agent using rolling window strategy.
"""

import logging
from functools import partial

from context_manager import process_with_rolling_context
from llm_ops import answer_question, compress_context, process_chunk_with_context

logger = logging.getLogger(__name__)

# Configuration
CHUNK_SIZE = 1000000  # Size of each chunk to process
CONTEXT_SIZE = 500000  # Maximum maintained context size


def solve_longbench_task(question: str, context: str, choices: list[str]) -> str:
    """
    Solve a LongBench task using rolling window strategy.

    The strategy maintains a rolling context window that accumulates
    information as we process through the document.
    """
    logger.info(f"Processing document of {len(context)} characters")
    logger.info(f"Using chunk size: {CHUNK_SIZE}, context size: {CONTEXT_SIZE}")

    # Define how to process each chunk with context
    process_fn = partial(process_chunk_with_context, question=question)

    # Define how to compress context when it gets too large
    compress_fn = partial(compress_context, question=question)

    # Process the document with rolling context
    final_context = process_with_rolling_context(
        text=context, chunk_size=CHUNK_SIZE, context_size=CONTEXT_SIZE, process_chunk_fn=process_fn, compress_context_fn=compress_fn
    )

    logger.info(f"Final context size: {len(final_context)} characters")

    # Answer the question using the final context
    return answer_question(question, final_context, choices)
