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
