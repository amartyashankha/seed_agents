#!/usr/bin/env python3
"""
Simple profiling script for search engine performance.
No external dependencies required.
"""

import asyncio
import gc
import random
import time

from .agent import LongContext, search_context
from .search_engine import AdvancedSearchEngine


def generate_test_document(size_chars: int) -> str:
    """Generate a test document of specified size."""
    # Create a realistic document with repeated patterns
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "lorem",
        "ipsum",
        "dolor",
        "sit",
        "amet",
        "consectetur",
        "adipiscing",
        "elit",
        "sed",
        "do",
        "eiusmod",
        "tempor",
        "incididunt",
        "labore",
    ]

    # Add some unique words for search testing
    unique_words = [
        "blackstone",
        "financial",
        "reports",
        "management",
        "assets",
        "magnus",
        "pye",
        "murder",
        "blakiston",
        "clarissa",
        "investment",
        "portfolio",
        "returns",
        "equity",
        "revenue",
    ]

    doc_parts = []
    current_size = 0

    while current_size < size_chars:
        # Occasionally insert unique words
        if random.random() < 0.1:
            word = random.choice(unique_words)
        else:
            word = random.choice(words)

        doc_parts.append(word)
        current_size += len(word) + 1  # +1 for space

        # Add some sentence structure
        if random.random() < 0.1:
            doc_parts.append(".")
            current_size += 1

    return " ".join(doc_parts)[:size_chars]


def test_initialization_performance():
    """Test how long it takes to initialize the search engine."""
    print("\n=== Testing Search Engine Initialization ===")

    sizes = [10_000, 50_000, 100_000, 250_000, 500_000]

    for size in sizes:
        doc = generate_test_document(size)

        # Force garbage collection for clean measurement
        gc.collect()

        # Measure initialization time
        start_time = time.time()
        engine = AdvancedSearchEngine(doc)
        init_time = time.time() - start_time

        # Measure components separately
        start_time = time.time()
        tokens = engine._tokenize(doc)
        tokenize_time = time.time() - start_time

        print(f"\nDocument size: {size:,} chars")
        print(f"  Total init time: {init_time:.3f}s")
        print(f"  Tokenization only: {tokenize_time:.3f}s ({tokenize_time / init_time * 100:.1f}%)")
        print(f"  Number of tokens: {len(tokens):,}")
        print(f"  Unique words: {len(engine.word_positions):,}")


def test_search_performance():
    """Test the performance of different search algorithms."""
    print("\n=== Testing Search Algorithm Performance ===")

    # Create a medium-sized document
    doc_size = 100_000
    doc = generate_test_document(doc_size)

    print(f"\nInitializing engine with {doc_size:,} char document...")
    start_time = time.time()
    engine = AdvancedSearchEngine(doc)
    print(f"  Initialization took: {time.time() - start_time:.3f}s")

    # Test different keyword sets
    keyword_sets = [
        (["financial"], "Single keyword"),
        (["financial", "reports"], "Two keywords"),
        (["financial", "reports", "management", "assets"], "Four keywords"),
        (["blackstone", "investment", "portfolio", "returns", "equity", "revenue"], "Six keywords"),
    ]

    for keywords, description in keyword_sets:
        print(f"\n{description}: {keywords}")

        # Test each algorithm
        algorithms = [
            ("TF-IDF", engine.tf_idf_search),
            ("Boolean AND", engine.boolean_search),
            ("Fuzzy", engine.fuzzy_search),
            ("Phrase", engine.phrase_search),
        ]

        for algo_name, algo_func in algorithms:
            gc.collect()

            start_time = time.time()
            results = algo_func(keywords, max_results=20, context_chars=1000)
            search_time = time.time() - start_time

            print(f"  {algo_name:12} - Time: {search_time:.3f}s, Results: {len(results)}")


def test_context_extraction_performance():
    """Test how context window size affects performance."""
    print("\n=== Testing Context Extraction Performance ===")

    doc = generate_test_document(500_000)
    engine = AdvancedSearchEngine(doc)

    # Get some random positions
    positions = [random.randint(10000, 490000) for _ in range(100)]

    context_sizes = [100, 500, 1000, 5000, 10000, 25000]

    print(f"\nExtracting context from {len(positions)} positions:")
    for context_size in context_sizes:
        gc.collect()

        start_time = time.time()
        for pos in positions:
            _ = engine.get_context_at_cursor(pos, context_size, context_size)

        total_time = time.time() - start_time
        avg_time = total_time / len(positions)

        print(f"  Context size {context_size:,} chars: {avg_time * 1000:.2f}ms per extraction")


async def test_search_tool_performance():
    """Test the actual search tool as used by the agent."""
    print("\n=== Testing Search Tool Performance ===")

    doc = generate_test_document(100_000)
    context_obj = LongContext(context=doc)

    # Create a mock context wrapper
    class MockContext:
        def __init__(self, context_obj):
            self.context = context_obj

    ctx = MockContext(context_obj)

    # Test different parameter combinations
    test_cases = [
        (["financial", "reports"], 10, 500, "Small context"),
        (["financial", "reports"], 20, 1000, "Medium context"),
        (["financial", "reports"], 20, 5000, "Large context"),
        (["financial", "reports", "management", "assets"], 30, 10000, "Very large context"),
    ]

    print("\nTesting search_context tool with different parameters:")
    for keywords, max_results, context_chars, description in test_cases:
        gc.collect()

        start_time = time.time()
        result = await search_context(ctx, keywords, max_results, context_chars)
        search_time = time.time() - start_time

        print(f"\n{description}:")
        print(f"  Keywords: {keywords}")
        print(f"  Max results: {max_results}, Context chars: {context_chars}")
        print(f"  Time: {search_time:.3f}s")
        print(f"  Result size: {len(result):,} chars")


def test_scaling_behavior():
    """Test how search scales with document size."""
    print("\n=== Testing Scaling Behavior ===")

    doc_sizes = [10_000, 50_000, 100_000, 250_000]
    keywords = ["financial", "reports", "management"]

    print("\nTesting search time vs document size:")
    print("(Using TF-IDF search with 20 results, 1000 char context)")

    for size in doc_sizes:
        doc = generate_test_document(size)

        # Initialize engine
        start_time = time.time()
        engine = AdvancedSearchEngine(doc)
        init_time = time.time() - start_time

        # Run search
        start_time = time.time()
        results = engine.tf_idf_search(keywords, max_results=20, context_chars=1000)
        search_time = time.time() - start_time

        print(f"\nDocument size: {size:,} chars")
        print(f"  Init time: {init_time:.3f}s")
        print(f"  Search time: {search_time:.3f}s")
        print(f"  Results found: {len(results)}")
        print(f"  Time per 1K chars: {search_time / (size / 1000):.3f}s")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Search Engine Performance Profiling")
    print("=" * 60)

    # Run tests
    test_initialization_performance()
    test_search_performance()
    test_context_extraction_performance()
    test_scaling_behavior()

    # Run async tests
    await test_search_tool_performance()

    print("\n" + "=" * 60)
    print("Profiling complete!")
    print("\nKey insights will help identify bottlenecks.")


if __name__ == "__main__":
    asyncio.run(main())
