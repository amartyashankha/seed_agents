"""
Explicit context management for rolling window processing.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """Represents a window of context with explicit size management."""

    content: str
    max_size: int

    @property
    def size(self) -> int:
        return len(self.content)

    @property
    def is_full(self) -> bool:
        return self.size >= self.max_size

    def can_fit(self, text: str) -> bool:
        """Check if text can fit in the remaining space."""
        return self.size + len(text) <= self.max_size

    def add(self, text: str) -> "ContextWindow":
        """Create new window with added text (immutable)."""
        return ContextWindow(content=self.content + "\n\n" + text if self.content else text, max_size=self.max_size)

    def compress(self, compression_fn) -> "ContextWindow":
        """Compress content using provided function."""
        compressed = compression_fn(self.content)
        return ContextWindow(content=compressed, max_size=self.max_size)


def sliding_windows(text: str, window_size: int, stride: int) -> Iterator[str]:
    """Generate sliding windows over text with given stride."""
    start = 0
    while start < len(text):
        end = min(start + window_size, len(text))
        yield text[start:end]
        start += stride
        if start >= len(text):
            break


def process_with_rolling_context(text: str, chunk_size: int, context_size: int, process_chunk_fn, compress_context_fn) -> str:
    """
    Process text with explicit rolling context management.

    Args:
        text: Full text to process
        chunk_size: Size of each chunk to process
        context_size: Maximum size of maintained context
        process_chunk_fn: Function to process (context, chunk) -> summary
        compress_context_fn: Function to compress context when full

    Returns:
        Final processed context
    """
    # Initialize context window
    context = ContextWindow(content="", max_size=context_size)

    # Process chunks with stride = chunk_size (no overlap)
    for i, chunk in enumerate(sliding_windows(text, chunk_size, chunk_size)):
        logger.info(f"Processing chunk {i + 1}, size: {len(chunk)}")

        # Process chunk with current context
        summary = process_chunk_fn(context.content, chunk)

        # Check if summary fits in context
        if context.can_fit(summary):
            context = context.add(summary)
        else:
            # Compress context first, then add summary
            logger.info(f"Context full ({context.size}/{context.max_size}), compressing...")
            context = context.compress(compress_context_fn)
            context = context.add(summary)

        logger.info(f"Context size: {context.size}/{context.max_size}")

    return context.content
