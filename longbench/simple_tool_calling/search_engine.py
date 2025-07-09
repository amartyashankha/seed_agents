"""
Advanced search engine for long context documents.
Implements various classical search algorithms with ranking.
"""

import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result with context and scoring."""

    text: str
    score: float
    position: int  # Character position in document (acts as cursor)
    matched_keywords: list[str]
    keyword_positions: dict[str, list[int]]

    @property
    def cursor(self) -> int:
        """Alias for position to make cursor concept clearer."""
        return self.position


class AdvancedSearchEngine:
    """Advanced search engine with multiple ranking algorithms."""

    def __init__(self, document: str):
        self.document = document
        self.words = self._tokenize(document)
        self.word_positions = self._build_word_positions()

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        return re.findall(r"\b\w+\b", text.lower())

    def _build_word_positions(self) -> dict[str, list[int]]:
        """Build a mapping of words to their positions in the document."""
        positions = defaultdict(list)
        for i, word in enumerate(self.words):
            positions[word].append(i)
        return positions

    def _get_text_around_position(self, position: int, context_chars: int = 100) -> tuple[str, int, int]:
        """Get text around a character position with specified context."""
        start = max(0, position - context_chars)
        end = min(len(self.document), position + context_chars)
        return self.document[start:end], start, end

    def get_context_at_cursor(self, cursor: int, chars_before: int = 10000, chars_after: int = 10000) -> str:
        """Get context around a specific cursor position with custom before/after ranges."""
        start = max(0, cursor - chars_before)
        end = min(len(self.document), cursor + chars_after)
        return self.document[start:end]

    def _word_position_to_char_position(self, word_pos: int) -> int:
        """Convert word position to character position in original document."""
        # This is an approximation - we'll find the word in the document
        words_before = " ".join(self.words[:word_pos])
        return len(words_before) + (1 if word_pos > 0 else 0)

    def boolean_search(self, keywords: list[str], max_results: int = 10, context_chars: int = 100) -> list[SearchResult]:
        """Search using boolean AND logic (all keywords must be present within proximity)."""
        start_time = time.time()
        keywords = [kw.lower() for kw in keywords]
        logger.debug(f"Boolean search started with {len(keywords)} keywords")

        # Find positions where all keywords appear within a window
        results = []
        window_size = context_chars // 5  # Convert chars to approximate word window

        if not keywords:
            return []

        # Start with positions of the first keyword (least common first would be better)
        base_keyword = keywords[0]
        if base_keyword not in self.word_positions:
            logger.debug(f"Base keyword '{base_keyword}' not found in document")
            return []

        base_positions = self.word_positions[base_keyword]
        logger.debug(f"Base keyword '{base_keyword}' has {len(base_positions)} positions")

        # Optimization: limit positions for very common words
        max_positions_to_check = 2000
        if len(base_positions) > max_positions_to_check:
            logger.info(f"Base keyword has {len(base_positions)} positions, sampling {max_positions_to_check}")
            # Sample evenly across the document
            step = len(base_positions) // max_positions_to_check
            base_positions = base_positions[::step][:max_positions_to_check]

        checked_positions = 0
        for base_pos in base_positions:
            checked_positions += 1
            if checked_positions % 500 == 0:
                logger.debug(f"Checked {checked_positions}/{len(base_positions)} positions...")

            # Check if all other keywords appear within the window
            matched_keywords = [base_keyword]
            kw_positions = {base_keyword: [base_pos]}

            for keyword in keywords[1:]:
                if keyword in self.word_positions:
                    nearby_positions = [p for p in self.word_positions[keyword] if abs(p - base_pos) <= window_size]
                    if nearby_positions:
                        matched_keywords.append(keyword)
                        kw_positions[keyword] = nearby_positions

            # Only include if all keywords are found
            if len(matched_keywords) == len(keywords):
                char_pos = self._word_position_to_char_position(base_pos)
                context_text, _, _ = self._get_text_around_position(char_pos, context_chars)

                # Score based on keyword proximity
                score = len(matched_keywords) * 10  # Base score

                # Bonus for keyword proximity
                all_positions = []
                for positions in kw_positions.values():
                    all_positions.extend(positions)

                if len(all_positions) > 1:
                    position_range = max(all_positions) - min(all_positions)
                    proximity_bonus = max(0, 50 - position_range)  # Closer keywords get higher score
                    score += proximity_bonus

                results.append(
                    SearchResult(text=context_text, score=score, position=char_pos, matched_keywords=matched_keywords, keyword_positions=kw_positions)
                )

                # Early exit if we have plenty of results
                if len(results) >= max_results * 2:
                    logger.debug(f"Early exit: found {len(results)} results")
                    break

        # Remove duplicates and sort by score
        seen_positions = set()
        unique_results = []
        for result in results:
            if result.position not in seen_positions:
                seen_positions.add(result.position)
                unique_results.append(result)

        unique_results.sort(key=lambda x: x.score, reverse=True)

        elapsed = time.time() - start_time
        logger.debug(f"Boolean search completed in {elapsed:.2f}s, found {len(unique_results)} unique results")

        return unique_results[:max_results]

    def fuzzy_search(self, keywords: list[str], max_results: int = 10, context_chars: int = 100) -> list[SearchResult]:
        """Search using fuzzy matching (OR logic with partial matches)."""
        keywords = [kw.lower() for kw in keywords]
        results = []

        # Find all words that partially match any keyword
        matching_words = defaultdict(list)
        for word in self.word_positions:
            for keyword in keywords:
                # Only match if keyword is substantial part of word
                if len(keyword) >= 3 and (keyword in word or word in keyword):
                    matching_words[keyword].append(word)

        # Collect positions for matching words
        all_positions = set()
        keyword_matches = defaultdict(list)

        for keyword, words in matching_words.items():
            for word in words[:10]:  # Limit matches per keyword
                positions = self.word_positions[word]
                # Sample positions if too many
                if len(positions) > 100:
                    positions = positions[:: len(positions) // 100][:100]
                all_positions.update(positions)
                keyword_matches[keyword].extend(positions)

        # Limit total positions to search
        if len(all_positions) > 1000:
            all_positions = set(list(all_positions)[:1000])

        # Score each position
        for pos in all_positions:
            char_pos = self._word_position_to_char_position(pos)
            context_text, _, _ = self._get_text_around_position(char_pos, context_chars)

            matched_keywords = []
            score = 0.0
            kw_positions = {}

            for keyword in keywords:
                nearby_positions = [p for p in keyword_matches[keyword] if abs(p - pos) <= context_chars // 10]
                if nearby_positions:
                    matched_keywords.append(keyword)
                    kw_positions[keyword] = nearby_positions
                    score += len(nearby_positions)

            if matched_keywords:
                results.append(
                    SearchResult(text=context_text, score=score, position=char_pos, matched_keywords=matched_keywords, keyword_positions=kw_positions)
                )

        # Sort by score and return top results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]


def format_search_results(results: list[SearchResult], keywords: list[str]) -> str:
    """Format search results for display."""
    if not results:
        return f"No matches found for keywords: {keywords}"

    output = [f"===== Search Results for {keywords} ====="]
    output.append(f"Found {len(results)} results:")

    for i, result in enumerate(results, 1):
        output.append(f"\n--- Result {i} (Score: {result.score:.2f}, Cursor: {result.cursor}) ---")
        output.append(f"Matched keywords: {', '.join(result.matched_keywords)}")
        output.append(f"Text: ...{result.text}...")

    return "\n".join(output)
