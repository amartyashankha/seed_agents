"""
LongBench agent using tool-based search strategy.
"""

import logging
import re
from dataclasses import dataclass

from agents import Agent, RunContextWrapper, Runner, function_tool

logger = logging.getLogger(__name__)


@dataclass
class LongContext:
    context: str

    def search(self, keywords: list[str], limit: int = 20) -> str:
        """Search the context for the keywords."""

        results = []

        for keyword in keywords:
            # Use regex to find all matches with some context
            pattern = re.compile(rf".{{0,50}}{re.escape(keyword)}.{{0,50}}", re.IGNORECASE)
            matches = pattern.findall(self.context)

            for match in matches[:limit]:
                results.append(f"Match for '{keyword}': ...{match}...")

                if len(results) >= limit:
                    break

            if len(results) >= limit:
                break

        if not results:
            return f"No matches found for {keywords}"

        return f"===== Search Results for {keywords} =====\n" + "\n".join(results)


@function_tool
def search_context(ctx: RunContextWrapper[LongContext], keywords: list[str], limit: int = 20) -> str:
    """Search the context for the keywords.

    Args:
        keywords: A python list of keywords to search for.
        limit: The maximum number of matches to return.

    Returns:
        A list of matches separated by ===== separators, where each entry contains the keyword with surrounding context.
    """
    return ctx.context.search(keywords, limit)


long_context_agent = Agent[LongContext](
    name="long_context_agent",
    model="gpt-4.1-mini",
    instructions="""You are a top-tier LongBench agent designed to solve long-context QA tasks.

TASK: You will be given a question and have access to a long context document. Your goal is to find the correct answer using the context.

STRATEGY:
1. First, carefully analyze the question to identify key terms, entities, and concepts
2. Use the search_context tool to find relevant passages in the context
3. Search for multiple related terms if needed to gather comprehensive information
4. Synthesize the information from search results to determine the correct answer
5. If this is a multiple choice question, return ONLY the letter (A, B, C, or D)
6. If this is an open-ended question, provide a concise, accurate answer

SEARCH TIPS:
- Extract key nouns, names, dates, and specific terms from the question
- Try variations of terms (synonyms, related concepts)
- If initial searches don't yield results, try broader or more specific terms
- Look for context clues that might help narrow down the search

IMPORTANT:
- Always use the search tool before answering
- Base your answer strictly on the information found in the context
- For multiple choice questions, return ONLY the letter of the correct choice
- If you cannot find sufficient information, state that clearly

Remember: The context is very long, so strategic searching is crucial for success.""",
    tools=[search_context],
)


async def solve_longbench_task(question: str, context_str: str, choices: list[str] | None = None) -> str:
    """Solve a LongBench task using the tool-based agent.

    Args:
        question: The question to answer
        context_str: The long context document
        choices: Optional list of multiple choice options

    Returns:
        The predicted answer
    """
    context_obj = LongContext(context=context_str)

    # Add choices information to the question if provided
    if choices:
        question_with_choices = f"{question}\n\nChoices: {', '.join(choices)}\n\nPlease return only the letter of the correct choice."
    else:
        question_with_choices = question

    result = await Runner.run(long_context_agent, input=question_with_choices, context=context_obj)
    return result.final_output
