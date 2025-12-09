"""Logging utilities for saving episode traces."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def save_episode_log(episode_log: Any, log_dir: str = "logs") -> str:
    """Save an episode log to a JSON file.
    
    Args:
        episode_log: EpisodeLog object to save
        log_dir: Directory to save logs in
        
    Returns:
        Path to the saved log file
    """
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    scenario_name = episode_log.scenario.replace(" ", "_").lower()
    status = "success" if episode_log.success else "failed"
    filename = f"{timestamp}_{scenario_name}_{status}.json"
    filepath = os.path.join(log_dir, filename)
    
    # Convert to dict for JSON serialization
    log_dict = {
        "scenario": episode_log.scenario,
        "task_description": episode_log.task_description,
        "initial_state": episode_log.initial_state_text,
        "success": episode_log.success,
        "total_steps": episode_log.total_steps,
        "final_state": episode_log.final_state_text,
        "steps": episode_log.steps,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save to JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_dict, f, indent=2, ensure_ascii=False)
    
    return filepath


def save_evaluation_summary(
    results: Dict[str, Any],
    scenario_name: str,
    log_dir: str = "logs"
) -> str:
    """Save evaluation summary to a JSON file.
    
    Args:
        results: Dictionary of evaluation metrics
        scenario_name: Name of the scenario evaluated
        log_dir: Directory to save logs in
        
    Returns:
        Path to the saved summary file
    """
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_clean = scenario_name.replace(" ", "_").lower()
    filename = f"{timestamp}_eval_{scenario_clean}.json"
    filepath = os.path.join(log_dir, filename)
    
    # Add metadata
    summary = {
        "scenario": scenario_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": results,
    }
    
    # Save to JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return filepath
