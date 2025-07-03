# GameOf24 Agent

A simple agent for solving Game of 24 puzzles.

## Overview

This agent uses GPT-4 to solve Game of 24 puzzles. The goal is to use four given numbers and basic arithmetic operations (+, -, ×, ÷) to create an expression that equals 24.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

The agent expects a JSON file with the problem text:

```bash
python main.py task.json
```

The JSON file should have this format:
```json
{
  "task_input": "Use the numbers 2, 3, 8, 8 to make 24"
}
```

## Output

The agent writes its solution to `output.json`:
```json
{
  "answer": "8 ÷ (3 - 8 ÷ 3)"
}
```

If the agent cannot find a valid solution, it will output:
```json
{
  "answer": "No solution"
}
``` 