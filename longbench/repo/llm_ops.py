"""
Simple LLM operations for context processing.
"""

import litellm

# Configuration
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.1


def llm_call(prompt: str, max_tokens: int = 1000) -> str:
    """Make a simple LLM call."""
    response = litellm.completion(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=TEMPERATURE, max_tokens=max_tokens)
    return response.choices[0].message.content


def process_chunk_with_context(context: str, chunk: str, question: str) -> str:
    """Process a new chunk given existing context and question."""
    prompt = f"""You are processing a long document to answer a question. 
You have a rolling context of previous information and a new chunk to process.

Question: {question}

Previous Context Summary:
{context if context else "No previous context yet."}

New Chunk to Process:
{chunk}

Provide a concise summary that:
1. Captures information relevant to answering the question
2. Integrates with the previous context
3. Maintains important details and facts

Summary:"""

    return llm_call(prompt, max_tokens=1000)


def compress_context(context: str, question: str) -> str:
    """Compress context while preserving question-relevant information."""
    prompt = f"""Compress the following context while preserving all information relevant to answering the question.
Remove redundant information but keep all important facts and details.

Question: {question}

Context to Compress:
{context}

Compressed Context:"""

    return llm_call(prompt, max_tokens=len(context) // 2)


def answer_question(question: str, context: str, choices: list[str]) -> str:
    """Answer a multiple choice question given context."""
    choices_text = "\n".join(f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices))

    prompt = f"""Based on the context, answer the question by selecting the best choice.

Context:
{context}

Question: {question}

Choices:
{choices_text}

Think step by step, then provide your answer as a single letter (A, B, C, or D).

Answer:"""

    response = llm_call(prompt, max_tokens=500)

    # Extract letter from response
    response = response.strip().upper()
    for letter in ["A", "B", "C", "D"]:
        if letter in response:
            return letter
    return "A"  # Default
