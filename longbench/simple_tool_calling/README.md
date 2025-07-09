# Advanced Search Engine for Long Context QA

This module implements an advanced search engine for long-context question answering tasks, featuring multiple classical search algorithms with sophisticated ranking mechanisms and cursor-based context exploration.

## Features

### 🔍 Multiple Search Algorithms

1. **TF-IDF Search** (`tf_idf_search`)
   - Uses Term Frequency-Inverse Document Frequency scoring
   - Best for general multi-keyword searches
   - Ranks results by relevance and keyword importance

2. **Boolean AND Search** (`boolean_search`)
   - Requires ALL keywords to appear within a proximity window
   - Scores based on keyword proximity and completeness
   - Ideal for finding contexts where all terms must be present

3. **Fuzzy Search** (`fuzzy_search`)
   - Uses partial string matching (OR logic)
   - Finds words that contain or are contained in keywords
   - Good for handling variations and typos

4. **Phrase Search** (`phrase_search`)
   - Searches for exact phrases first
   - Falls back to proximity search if no exact matches
   - Perfect for finding quotes or specific expressions

### 🎯 Cursor-Based Context Navigation

Each search result includes a **cursor position** (character index in the document) that enables:
- Precise location tracking of search matches
- Ability to expand context around interesting results
- Efficient deep-dive exploration of promising areas

### 🎛️ Configurable Parameters

Each search tool accepts two key parameters:

- **`max_results`**: Number of search results to return
  - Default: 10-15 depending on tool
  - Adjust based on information need

- **`context_chars`**: Characters to include around each match
  - Default: 300-4000 depending on tool
  - Use more for complex questions needing context
  - Use fewer for simple fact-finding

## Search Tools

### 1. `search_context` (Primary Tool)
```python
search_context(keywords: list[str], max_results: int = 15, context_chars: int = 300)
```
- **Algorithm**: TF-IDF with fallback to boolean and fuzzy search
- **Use case**: General multi-keyword searches
- **Returns**: Results with cursor positions for each match

### 2. `search_exact_phrase`
```python
search_exact_phrase(keywords: list[str], max_results: int = 10, context_chars: int = 4000)
```
- **Algorithm**: Exact phrase matching with proximity fallback
- **Use case**: Finding quotes, specific phrases, or exact expressions
- **Returns**: Results with cursor positions for each match

### 3. `search_boolean_and`
```python
search_boolean_and(keywords: list[str], max_results: int = 15, context_chars: int = 3000)
```
- **Algorithm**: Boolean AND with proximity scoring
- **Use case**: When ALL keywords must appear together
- **Returns**: Results with cursor positions for each match

### 4. `get_context_at_cursor` (Context Expansion Tool)
```python
get_context_at_cursor(cursor: int, chars_before: int = 1000, chars_after: int = 1000)
```
- **Purpose**: Expand context around a specific cursor position from search results
- **Use case**: Deep-dive into promising search results
- **Parameters**: 
  - `cursor`: Position from search results
  - `chars_before/after`: How much context to grab (be generous!)

## Search Workflow

### Two-Stage Search Pattern
1. **Discovery Stage**: Use search tools to find relevant passages
   - Each result includes a cursor position
   - Note interesting cursor positions
   
2. **Exploration Stage**: Use cursor tool to expand context
   - Take cursor positions from promising search results
   - Use `get_context_at_cursor` to read more
   - Adjust `chars_before/after` based on information flow

### Example Workflow
```python
# Stage 1: Search for relevant passages
results = search_context(["Great Wall", "construction"], max_results=10)
# Returns results with cursor positions, e.g., cursor: 1234

# Stage 2: Explore interesting result at cursor 1234
expanded_context = get_context_at_cursor(
    cursor=1234, 
    chars_before=2000,  # Get more context before
    chars_after=3000    # Get even more context after
)
```

## Search Result Format

Each search result includes:
- **Score**: Relevance score based on the algorithm
- **Cursor**: Character position in the document
- **Matched Keywords**: Which keywords were found
- **Text**: Context snippet around the match

Example output:
```
===== Search Results for ['Ming', 'Dynasty'] =====
Found 3 results:

--- Result 1 (Score: 12.45, Cursor: 1523) ---
Matched keywords: Ming, Dynasty
Text: ...built by the Ming Dynasty (1368-1644). The total length...

--- Result 2 (Score: 8.32, Cursor: 3847) ---
Matched keywords: Ming, Dynasty  
Text: ...during the Ming Dynasty. The wall served multiple purposes...
```

## Best Practices

### 1. **Use the Two-Stage Pattern**
- Start with search tools to discover relevant areas
- Note cursor positions from high-scoring results
- Use `get_context_at_cursor` to explore deeper

### 2. **Cursor Navigation Strategy**
- High-scoring results often have the most relevant cursors
- Explore multiple cursor positions for comprehensive coverage
- Adjust `chars_before/after` based on the information structure

### 3. **Parameter Tuning**
- **Initial search**: Use moderate `context_chars` (300-1000)
- **Cursor exploration**: Use generous `chars_before/after` (2000-5000)
- **Complex questions**: Increase both parameters significantly

### 4. **Search Strategy**
- Start with specific keywords for initial discovery
- Use cursor positions to explore context thoroughly
- Try different search algorithms if initial results are poor
- Combine multiple cursor explorations for full understanding

## Algorithm Details

### Cursor Position Calculation
- Each search match has a precise character position in the document
- Cursor positions are consistent across all search algorithms
- Positions refer to the approximate center of the matched context

### Context Expansion
- `get_context_at_cursor` provides asymmetric context windows
- Can grab more text before or after based on needs
- Handles document boundaries gracefully

## Performance Considerations

- **Cursor Storage**: Minimal overhead (single integer per result)
- **Context Retrieval**: O(1) time complexity for cursor lookup
- **Memory Efficient**: No duplicate storage of document content
- **Scalable**: Works well with very long documents

## File Structure

```
├── search_engine.py      # Core algorithms with cursor support
├── agent.py             # Agent with search and cursor tools
├── hooks.py             # Tool parameter logging utilities
└── README.md           # This documentation
```

## Example Use Cases

### Finding and Exploring a Quote
```python
# Find the quote
results = search_exact_phrase(["UNESCO", "World", "Heritage"])
# Get cursor from result, e.g., 5234

# Explore full context around the quote
context = get_context_at_cursor(5234, chars_before=500, chars_after=1500)
```

### Comprehensive Topic Research
```python
# Initial discovery
results = search_context(["construction", "methods"], max_results=20)

# Explore each promising cursor
for result in results[:5]:  # Top 5 results
    expanded = get_context_at_cursor(
        result.cursor, 
        chars_before=2000, 
        chars_after=2000
    )
```

### Verifying Information
```python
# Find all mentions
results = search_boolean_and(["length", "miles", "kilometers"])

# Deep dive into each mention
for result in results:
    full_context = get_context_at_cursor(
        result.cursor,
        chars_before=1000,
        chars_after=1000
    )
```

The cursor-based navigation system provides a powerful way to efficiently explore long documents, combining the precision of search algorithms with the flexibility of targeted context expansion.
