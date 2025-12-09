"""Prompt building and parsing utilities for the LLM planner."""

import re
from typing import List, Tuple
from env.world import WorldState


def build_planner_prompt(state: WorldState) -> List[dict]:
    """Build the prompt messages for the LLM planner.
    
    Args:
        state: Current world state
        
    Returns:
        List of message dicts for the LLM client
    """
    system_msg = (
        "You are a robot task planner. "
        "You must output exactly one next action in a strict format.\n"
        "Valid action formats:\n"
        "  ACTION: MOVE_TO(location)\n"
        "  ACTION: PICK(object)\n"
        "  ACTION: PLACE(location)\n"
        "  ACTION: PUT_DOWN()\n"
        "  ACTION: INSPECT(object)\n"
        "  ACTION: OPEN(container)\n"
        "  ACTION: CLOSE(container)\n"
        "Or if the task is completed: DONE\n"
        "Do NOT explain. Do NOT output anything except the action line."
    )

    # Use the built-in to_natural_language method from WorldState
    state_text = state.to_natural_language()
    
    user_msg = (
        "Here is the current world state:\n"
        f"{state_text}\n\n"
        f"Valid locations: table, shelf, floor\n"
        f"Valid objects: box, book\n"
        "Choose the next action."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def parse_action_from_llm_output(text: str) -> Tuple[str, List[str]]:
    """Parse LLM output to extract action and arguments.
    
    Expected formats:
        'ACTION: MOVE_TO(table)'
        'ACTION: PICK(box)'
        'ACTION: PLACE(shelf)'
        'ACTION: PUT_DOWN()'
        'DONE'
    
    Args:
        text: Raw text output from LLM
        
    Returns:
        Tuple of (action_type, args_list)
        - action_type: Uppercase action name or "DONE" or "UNKNOWN"
        - args_list: List of string arguments (empty list if none)
    """
    text = text.strip().upper()

    if "DONE" in text:
        return "DONE", []

    # Parse action format: ACTION: FUNCTION(args)
    m = re.search(r"ACTION:\s*(\w+)\(([^)]*)\)", text, re.IGNORECASE)
    if not m:
        # fallback: treat as no-op or error
        return "UNKNOWN", []

    action_type = m.group(1).upper()
    arg_str = m.group(2).strip()
    args = [arg_str] if arg_str else []

    return action_type, args
