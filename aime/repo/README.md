# AIME Agent

A simple agent for solving AIME (American Invitational Mathematics Examination) problems.

## Overview

This agent uses GPT-4 to solve AIME mathematics problems. It expects problems that have integer answers between 0 and 999.

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
  "task_input": "Find the number of positive integers n less than 1000 such that n^2 + 1 is divisible by n + 1"
}
```

## Output

The agent writes its answer to `output.json`:
```json
{
  "answer": 3
}
```

If the agent cannot find a valid answer, it will output:
```json
{
  "answer": null
}
``` 