"""
LongBench agent using tool-based search strategy.
"""

import logging
import time
from dataclasses import dataclass, field

from agents import Agent, RunContextWrapper, Runner, function_tool

from .hooks import create_logging_tool_wrapper
from .search_engine import AdvancedSearchEngine, format_search_results

# Set up logging to see hook messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class LongContext:
    context: str
    search_engine: AdvancedSearchEngine | None = field(default=None, init=False)

    def __post_init__(self):
        """Initialize the search engine after the context is set."""
        self.search_engine = AdvancedSearchEngine(self.context)


@function_tool
def search_context(ctx: RunContextWrapper[LongContext], keywords: list[str], max_results: int = 15, context_chars: int = 1000) -> str:
    """Search the context for keywords using optimized search algorithms.

    This tool searches through the document to find relevant passages containing your keywords.
    For best results, use 3-6 distinctive keywords that capture the essence of what you're looking for.

    Args:
        keywords: A list of keywords to search for (recommended: 3-6 distinctive terms).
        max_results: Maximum number of results to return (default: 15).
        context_chars: Characters of context around each match (default: 1000).

    Returns:
        Search results with context and cursor positions for further exploration.

    Tips:
        - Use distinctive, specific terms rather than common words
        - If searching for a phrase, include the key words from that phrase
        - For complex queries, do multiple searches with different keyword sets
        - Each result includes a cursor position for deeper exploration
    """
    search_engine = ctx.context.search_engine
    assert search_engine is not None, "Search engine not initialized"

    # Remove the explicit limit - let prompting handle this
    # Just log if there are many keywords
    if len(keywords) > 8:
        logger.warning(f"Search called with {len(keywords)} keywords - this may be slow. Consider using fewer, more specific keywords.")

    # Log search parameters
    logger.info(f"search_context called with {len(keywords)} keywords: {keywords}")
    logger.info(f"max_results={max_results}, context_chars={context_chars}")

    # Use Boolean AND search as the primary algorithm (fastest for multiple keywords)
    start_time = time.time()
    results = search_engine.boolean_search(keywords, max_results, context_chars)
    boolean_time = time.time() - start_time
    logger.info(f"Boolean AND search completed in {boolean_time:.2f}s, found {len(results)} results")

    # If we need more results, try fuzzy search (more permissive)
    if len(results) < max_results // 2:
        logger.info(f"Boolean AND found only {len(results)} results, trying fuzzy search...")
        start_time = time.time()
        fuzzy_results = search_engine.fuzzy_search(keywords, max_results, context_chars)
        fuzzy_time = time.time() - start_time
        logger.info(f"Fuzzy search completed in {fuzzy_time:.2f}s, found {len(fuzzy_results)} results")

        # Merge results, avoiding duplicates
        existing_positions = {r.position for r in results}
        added = 0
        for result in fuzzy_results:
            if result.position not in existing_positions and len(results) < max_results:
                results.append(result)
                existing_positions.add(result.position)
                added += 1
        logger.info(f"Added {added} unique fuzzy results")

    # Sort all results by score
    results.sort(key=lambda x: x.score, reverse=True)
    logger.info(f"Returning {len(results)} total results")

    return format_search_results(results[:max_results], keywords)


@function_tool
def get_context_at_cursor(ctx: RunContextWrapper[LongContext], cursor: int, chars_before: int = 10000, chars_after: int = 10000) -> str:
    """Get expanded context around a specific cursor position.

    Use this after finding interesting results with search_context to read more of the document
    around a specific position.

    Args:
        cursor: The cursor position from a search result.
        chars_before: Characters to include before the cursor (default: 10000).
        chars_after: Characters to include after the cursor (default: 10000).

    Returns:
        Extended text context centered on the cursor position.
    """
    search_engine = ctx.context.search_engine
    assert search_engine is not None, "Search engine not initialized"

    context_text = search_engine.get_context_at_cursor(cursor, chars_before, chars_after)

    return f"===== Context at cursor {cursor} =====\n[{chars_before} chars before, {chars_after} chars after]\n\n{context_text}"


long_context_agent = Agent[LongContext](
    name="long_context_agent",
    model="gpt-4.1-nano",
    instructions="""You are an expert at finding information in long documents to answer questions accurately.

AVAILABLE TOOLS:
1. search_context: Search for keywords in the document (returns matches with cursor positions)
2. get_context_at_cursor: Read more text around a specific cursor position

SEARCH STRATEGY:

1. ANALYZE THE QUESTION
   - Identify the key concepts, entities, and distinctive terms
   - Focus on what makes this question unique

2. SEARCH EFFECTIVELY
   - Use 3-6 distinctive keywords per search
   - Choose specific, uncommon terms over generic words
   - Examples:
     * Good: ["Blackstone", "revenue", "2023"] 
     * Bad: ["the", "company", "financial", "report", "shows", "that", "revenue", "increased"]
   - If you need to find multiple concepts, do separate searches

3. EXPLORE RESULTS
   - Each search result includes a cursor position
   - Use get_context_at_cursor to read more around promising results
   - Be generous with context (10,000+ characters) to ensure you don't miss anything

4. ITERATE IF NEEDED
   - If first search doesn't find what you need, try different keywords
   - Consider synonyms or related terms
   - Break complex questions into parts

IMPORTANT GUIDELINES:
- Quality over quantity: Better to do 3 focused searches than 1 search with 15 keywords
- The search tool works best with distinctive terms that are likely to appear together
- Always read enough context around matches to understand the full picture
- For multiple choice questions: provide reasoning, then state the letter clearly
- Base answers only on what you find in the search results

Remember: Effective searching is about choosing the RIGHT keywords, not ALL possible keywords.""",
    tools=[
        create_logging_tool_wrapper(search_context),
        create_logging_tool_wrapper(get_context_at_cursor),
    ],
)


# Simple extraction agent for choice extraction
choice_extractor = Agent(
    name="choice_extractor",
    model="gpt-4.1-mini",
    instructions="""You are a choice extraction assistant. Your job is to extract the final answer choice from a response.

Given a response to a multiple choice question, extract ONLY the letter of the final answer choice (A, B, C, or D).

Rules:
- Return only a single letter: A, B, C, or D
- Look for the final answer in the response
- If multiple choices are mentioned, return the one that appears to be the final conclusion
- If no clear choice can be determined, return the first choice mentioned
- Do not include any explanation or additional text""",
)


async def extract_choice_from_answer(answer: str, choices: list[str] | None = None) -> str:
    """Extract the choice letter from a verbose answer using a simple LLM call."""
    if not choices:
        return answer

    prompt = f"""Extract the final answer choice from this response:

Response: {answer}

Available choices: {", ".join(choices)}

Return only the letter (A, B, C, or D):"""

    try:
        result = await Runner.run(choice_extractor, input=prompt)
        extracted = result.final_output.strip().upper()

        # Validate the extracted choice
        if extracted in ["A", "B", "C", "D"]:
            return extracted
        else:
            # Fallback: return first character if it's a valid choice
            first_char = extracted[0] if extracted else "A"
            return first_char if first_char in ["A", "B", "C", "D"] else "A"

    except Exception as e:
        logger.warning(f"Failed to extract choice with LLM: {e}")
        # Fallback to original answer
        return answer


async def solve_longbench_task(question: str, context_str: str, choices: list[str] | None = None) -> str:
    """Solve a LongBench task using the tool-based agent.

    Args:
        question: The question to answer
        context_str: The long context document
        choices: Optional list of multiple choice options

    Returns:
        The predicted answer
    """
    logger.info(f"Starting LongBench task with question length: {len(question)}")
    logger.info(f"Context length: {len(context_str)} characters")

    context_obj = LongContext(context=context_str)

    # Add choices information to the question if provided
    if choices:
        question_with_choices = f"{question}\n\nChoices: {', '.join(choices)}\n\nPlease provide your reasoning and then clearly state your final answer as just the letter (A, B, C, or D)."
    else:
        question_with_choices = question

    logger.info("Running agent with advanced search capabilities...")
    result = await Runner.run(long_context_agent, input=question_with_choices, context=context_obj)

    # Extract the choice from the verbose answer if it's a multiple choice question
    if choices:
        extracted_choice = await extract_choice_from_answer(result.final_output, choices)
        logger.info(f"Extracted choice: {extracted_choice} from answer: {result.final_output[:100]}...")
        return extracted_choice

    logger.info(f"Agent completed with result: {result.final_output[:100]}...")
    return result.final_output
