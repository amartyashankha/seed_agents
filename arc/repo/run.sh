#!/bin/bash
# Run script for ARC agent
# Usage: ./run.sh [task_file.json]
# If no task file is provided, defaults to task.json if it exists

if [ "$#" -eq 0 ]; then
    if [ -f "task.json" ]; then
        echo "No task file provided, using default task.json (if you want to use a different task use --task-file <task_file.json>)"
        python main.py "task.json"
    else
        echo "Usage: $0 <task_file.json>"
        echo "No default task.json found"
        exit 1
    fi
elif [ "$#" -eq 1 ]; then
    python main.py "$1"
else
    echo "Usage: $0 [task_file.json]"
    exit 1
fi