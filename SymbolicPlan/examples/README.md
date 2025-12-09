# Episode Logs

This directory contains example JSON logs of episode executions and evaluation summaries.

## File Naming Convention

### Episode Logs
`YYYYMMDD_HHMMSS_microseconds_scenario-name_status.json`

Example: `20251209_143022_123456_pick_and_place_box_episode_1_success.json`

### Evaluation Summaries
`YYYYMMDD_HHMMSS_eval_scenario-name.json`

Example: `20251209_143025_eval_pick_and_place_box.json`

## Episode Log Structure

```json
{
  "scenario": "Pick And Place Box Episode 1",
  "task_description": "Place the box from the table onto the shelf.",
  "initial_state": "Robot is at floor.\n...",
  "success": true,
  "total_steps": 4,
  "final_state": "Robot is at shelf.\n...",
  "steps": [
    {
      "step_number": 1,
      "action": "MOVE_TO(table)",
      "llm_raw_output": "ACTION: MOVE_TO(table)",
      "result": "Moved to table",
      "state_after": "Robot is at table.\n...",
      "step_count": 1
    }
  ],
  "timestamp": "2025-12-09T14:30:22.123456"
}
```

## Evaluation Summary Structure

```json
{
  "scenario": "Pick And Place Box",
  "timestamp": "2025-12-09T14:30:25.654321",
  "metrics": {
    "completion_rate": 0.8,
    "avg_steps": 4.2,
    "success_efficiency": 4.0,
    "successes": 4,
    "episodes": 5
  }
}
```

## Usage

Logs are automatically generated when running `python main.py` with `save_logs=True` (default).

To disable logging:
```python
results = evaluate_planner(planner, scenario_fn, n_episodes=5, save_logs=False)
```
