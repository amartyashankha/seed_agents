# LongBench Agent - Rolling Window

A clean implementation for solving LongBench tasks using a rolling window strategy with explicit context management.

## Overview

The agent uses a **rolling window strategy** that:
1. Processes the document in chunks
2. Maintains a rolling context that accumulates information
3. Compresses the context when it exceeds size limits
4. Uses the final context to answer the question

## Architecture

```
context_manager.py   - Explicit context window management
llm_ops.py          - LLM operations (process, compress, answer)
agent.py            - Main logic using rolling window
main.py             - Entry point
```

### Key Components

**ContextWindow** - Immutable context with size tracking:
- Tracks current size vs max size
- Provides `add()` and `compress()` operations
- Ensures context never exceeds limits

**Rolling Window Process**:
```python
for chunk in document:
    summary = process_chunk_with_context(context, chunk, question)
    if context.can_fit(summary):
        context = context.add(summary)
    else:
        context = context.compress(compress_fn)
        context = context.add(summary)
```

## Configuration

Environment variables:
- `LITELLM_MODEL` - LLM model to use (default: gpt-4o-mini)

Code constants:
- `CHUNK_SIZE` - Size of each chunk (default: 100K chars)
- `CONTEXT_SIZE` - Max maintained context (default: 50K chars)

## Usage

```bash
python main.py <task_id>
```

Input format (`inputs/<task_id>.json`):
```json
{
    "_id": "task_id",
    "question": "question text",
    "context": "long document text",
    "choices": ["A", "B", "C", "D"]
}
```

Output format (`outputs/<task_id>.json`):
```json
{
    "task_id": "task_id",
    "predicted_answer": "A|B|C|D",
    "choices": ["A", "B", "C", "D"]
}
```

## Design Principles

- **Explicit Context Management**: Always know context size and limits
- **Immutable Operations**: Context operations return new instances
- **Simple Flow**: One clear strategy, easy to understand
- **Extensible**: Easy to modify compression/processing functions

## Adding New Strategies

To add a new strategy:

1. Create a new function in `agent.py` following the pattern:
   ```python
   def my_strategy(context: str, question: str) -> str:
       # Process context and return final context
   ```

2. Add it to the strategy selection in `solve_longbench_task()`

3. Use the composable functions from `context_processor.py` and `llm_ops.py` 