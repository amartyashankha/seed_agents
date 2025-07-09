#!/usr/bin/env python3
"""
Profile the search engine performance to identify bottlenecks.
Tests each hypothesis about performance issues.
"""

import asyncio
import cProfile
import io
import pstats
import random
import time

from .agent import LongContext, search_context, solve_longbench_task
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
    unique_words = ["blackstone", "financial", "reports", "management", "assets", "magnus", "pye", "murder", "blakiston", "clarissa"]

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


def profile_hypothesis_1_tokenization():
    """Test Hypothesis 1: Document tokenization overhead."""
    print("\n=== Hypothesis 1: Document Tokenization Overhead ===")

    sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]

    for size in sizes:
        doc = generate_test_document(size)

        # Profile tokenization
        start_time = time.time()
        engine = AdvancedSearchEngine(doc)
        tokens = engine._tokenize(doc)
        tokenize_time = time.time() - start_time

        print(f"Document size: {size:,} chars")
        print(f"  Tokenization time: {tokenize_time:.3f}s")
        print(f"  Number of tokens: {len(tokens):,}")
        print(f"  Tokens per second: {len(tokens) / tokenize_time:,.0f}")


def profile_hypothesis_2_word_positions():
    """Test Hypothesis 2: Word position building overhead."""
    print("\n=== Hypothesis 2: Word Position Building Overhead ===")

    sizes = [10_000, 50_000, 100_000, 500_000]

    for size in sizes:
        doc = generate_test_document(size)

        # First tokenize
        start_time = time.time()
        engine = AdvancedSearchEngine(doc)
        tokenize_time = time.time() - start_time

        # Profile position building separately
        start_time = time.time()
        positions = engine._build_word_positions()
        position_time = time.time() - start_time

        print(f"Document size: {size:,} chars")
        print(f"  Total init time: {tokenize_time:.3f}s")
        print(f"  Position building time: {position_time:.3f}s")
        print(f"  Unique words: {len(positions):,}")
        print(f"  Total word occurrences: {sum(len(p) for p in positions.values()):,}")


def profile_hypothesis_3_multiple_searches():
    """Test Hypothesis 3: Multiple search algorithm overhead."""
    print("\n=== Hypothesis 3: Multiple Search Algorithm Overhead ===")

    doc = generate_test_document(100_000)
    engine = AdvancedSearchEngine(doc)

    keywords = ["financial", "reports", "management"]

    # Profile individual search algorithms
    algorithms = [
        ("TF-IDF", engine.tf_idf_search),
        ("Boolean", engine.boolean_search),
        ("Fuzzy", engine.fuzzy_search),
        ("Phrase", engine.phrase_search),
    ]

    for name, search_func in algorithms:
        start_time = time.time()
        results = search_func(keywords, max_results=20, context_chars=1000)
        search_time = time.time() - start_time

        print(f"{name} search:")
        print(f"  Time: {search_time:.3f}s")
        print(f"  Results found: {len(results)}")

    # Profile combined search (as used in agent)
    print("\nCombined search (with fallbacks):")
    start_time = time.time()

    # Simulate the agent's search logic
    results = engine.tf_idf_search(keywords, max_results=20, context_chars=1000)
    if len(results) < 10:
        boolean_results = engine.boolean_search(keywords, max_results=20, context_chars=1000)
        results.extend(boolean_results)
    if len(results) < 10:
        fuzzy_results = engine.fuzzy_search(keywords, max_results=20, context_chars=1000)
        results.extend(fuzzy_results)

    combined_time = time.time() - start_time
    print(f"  Time: {combined_time:.3f}s")
    print(f"  Total results: {len(results)}")


def profile_hypothesis_4_context_extraction():
    """Test Hypothesis 4: Large context window extraction overhead."""
    print("\n=== Hypothesis 4: Context Window Extraction Overhead ===")

    doc = generate_test_document(500_000)
    engine = AdvancedSearchEngine(doc)

    # Find some positions to extract context from
    keywords = ["financial", "reports"]
    results = engine.tf_idf_search(keywords, max_results=10, context_chars=100)

    if not results:
        print("No search results found, using random positions")
        positions = [random.randint(1000, len(doc) - 1000) for _ in range(10)]
    else:
        positions = [r.position for r in results]

    context_sizes = [100, 500, 1000, 5000, 10000, 50000]

    for context_size in context_sizes:
        start_time = time.time()

        for pos in positions:
            text = engine.get_context_at_cursor(pos, context_size, context_size)

        extract_time = time.time() - start_time
        avg_time = extract_time / len(positions)

        print(f"Context size: {context_size:,} chars (before + after)")
        print(f"  Total time for {len(positions)} extractions: {extract_time:.3f}s")
        print(f"  Average time per extraction: {avg_time * 1000:.1f}ms")


async def profile_hypothesis_5_llm_vs_search():
    """Test Hypothesis 5: Compare search time vs LLM API time."""
    print("\n=== Hypothesis 5: Search vs LLM API Performance ===")

    # Create a test context
    doc = generate_test_document(100_000)
    context_obj = LongContext(context=doc)

    # Create a mock context wrapper
    class MockContext:
        def __init__(self, context_obj):
            self.context = context_obj

    ctx = MockContext(context_obj)

    # Test search performance
    print("Search performance:")
    keywords = ["financial", "reports", "management", "assets"]

    start_time = time.time()
    search_result = await search_context(ctx=ctx, keywords=keywords, max_results=20, context_chars=5000)
    search_time = time.time() - start_time
    print(f"  Search time: {search_time:.3f}s")
    print(f"  Result length: {len(search_result)} chars")

    # Test full agent (with LLM calls)
    print("\nFull agent performance (includes LLM calls):")
    question = "What are the financial reports about management assets?"
    choices = ["A) Investment reports", "B) Annual reports", "C) Management fees", "D) Asset allocation"]

    start_time = time.time()
    try:
        result = await solve_longbench_task(question=question, context_str=doc, choices=choices)
        agent_time = time.time() - start_time
        print(f"  Total agent time: {agent_time:.3f}s")
        print(f"  LLM overhead: ~{agent_time - search_time:.3f}s")
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Error running agent: {e}")


def detailed_profiling():
    """Run detailed profiling with cProfile."""
    print("\n=== Detailed Profiling with cProfile ===")

    doc = generate_test_document(100_000)

    # Profile engine initialization
    print("\nProfiling engine initialization:")
    pr = cProfile.Profile()
    pr.enable()
    engine = AdvancedSearchEngine(doc)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(10)
    print(s.getvalue())

    # Profile search operation
    print("\nProfiling search operation:")
    pr = cProfile.Profile()
    pr.enable()
    results = engine.tf_idf_search(["financial", "reports"], max_results=20, context_chars=1000)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(10)
    print(s.getvalue())


def main():
    """Run all profiling tests."""
    print("=" * 60)
    print("Search Engine Performance Profiling")
    print("=" * 60)

    # Run synchronous tests
    profile_hypothesis_1_tokenization()
    profile_hypothesis_2_word_positions()
    profile_hypothesis_3_multiple_searches()
    profile_hypothesis_4_context_extraction()

    # Run async test
    print("\nRunning async tests...")
    asyncio.run(profile_hypothesis_5_llm_vs_search())

    # Run detailed profiling
    detailed_profiling()

    print("\n" + "=" * 60)
    print("Profiling complete!")


if __name__ == "__main__":
    main()
