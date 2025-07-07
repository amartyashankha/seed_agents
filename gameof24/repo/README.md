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

The agent now accepts the task string directly as a command-line argument:

```bash
python main.py "Use the numbers 2, 3, 8, 8 to make 24"
```

You can also specify a custom output file:
```bash
python main.py "Use the numbers 2, 3, 8, 8 to make 24" -o solution.json
```

## Output

The agent writes its solution to `output.json` (or the specified output file):
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