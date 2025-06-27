# Simple ARC Agent

A minimal agent for solving ARC-AGI (Abstraction and Reasoning Corpus) tasks from a file.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

The agent is designed to be run on a task file. The `example.py` script in the parent directory demonstrates how to generate a `task.json` and evaluate the agent's output.

1.  **Generate a task and run the agent:**
    ```bash
    # From the parent directory (e.g., .../arc/)
    python example.py
    ```
    This will:
    - Create `repo/task.json` with a sample task.
    - Run the agent, which generates `repo/output.json` with its predictions.
    - Evaluate the predictions against the ground truth.

2.  **Run the agent manually:**
    ```bash
    cd repo
    python main.py task.json
    ```

## Task File Format

The input `task.json` must have the following structure:

```json
{
  "train": [
    {
      "input": [[...]],
      "output": [[...]]
    }
  ],
  "test": [
    {
      "input": [[...]]
    }
  ]
}
```

-   `train`: A list of demonstration pairs with both "input" and "output" grids.
-   `test`: A list of test cases with only the "input" grid.

## Output Format

The agent will produce an `output.json` file containing a list of predicted grids:

```json
[
  [[...]],
  [[...]]
]
```

## Structure

-   `agent.py`: The core ARC solver agent.
-   `main.py`: The main entry point that runs the agent on a task file.
-   `task_utils.py`: Utilities for loading task files.
-   `formatting.py`: Utilities for formatting the task into a prompt for the agent.
-   `requirements.txt`: Python dependencies. 