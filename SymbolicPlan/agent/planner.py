"""LLM-based planner that integrates language models with symbolic actions."""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
from env.world import WorldState
from env.actions import (
    move_to,
    pick,
    place,
    put_down,
    inspect,
    open_container,
    close_container,
)
from .llm_client import LLMClient
from .prompting import build_planner_prompt, parse_action_from_llm_output


@dataclass
class EpisodeLog:
    """Complete trace of an episode execution.
    
    Attributes:
        scenario: Name of the test scenario
        initial_state_text: Natural language description of initial state
        task_description: Task description from world state
        steps: List of step details (action, state, LLM response)
        success: Whether task was completed successfully
        total_steps: Number of steps taken
        final_state_text: Natural language description of final state
    """
    scenario: str
    initial_state_text: str
    task_description: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    total_steps: int = 0
    final_state_text: str = ""


class LLMPlanner:
    """Planner that uses an LLM to generate robot action sequences.
    
    Attributes:
        llm: LLM client for generating action decisions
        max_steps: Maximum steps allowed per episode
    """
    
    def __init__(self, llm: LLMClient, max_steps: int = 10):
        """Initialize the planner.
        
        Args:
            llm: LLM client instance
            max_steps: Maximum number of steps per episode
        """
        self.llm = llm
        self.max_steps = max_steps

    def step(self, state: WorldState) -> Tuple[WorldState, bool, str, str, str]:
        """Execute one planning and action step.
        
        Args:
            state: Current world state
            
        Returns:
            Tuple of (new_state, done, info_message, action_taken, llm_response)
            - new_state: Updated world state after action
            - done: True if task completed or failed
            - info_message: Human-readable description of what happened
            - action_taken: The action that was executed
            - llm_response: Raw LLM output
        """
        if state.is_task_completed():
            return state, True, "Task already completed", "NONE", ""

        # Build prompt and get LLM response
        messages = build_planner_prompt(state)
        raw_output = self.llm.generate(messages)
        action_type, args = parse_action_from_llm_output(raw_output)

        if action_type == "DONE":
            return state, True, "LLM claims task is done", "DONE", raw_output

        # Build action string for logging
        action_str = f"{action_type}({', '.join(args)})"
        
        # Dispatch to appropriate symbolic action
        if action_type == "MOVE_TO":
            new_state, success, msg = move_to(state, args[0])
        elif action_type == "PICK":
            new_state, success, msg = pick(state, args[0])
        elif action_type == "PLACE":
            new_state, success, msg = place(state, args[0])
        elif action_type == "PUT_DOWN":
            new_state, success, msg = put_down(state)
        elif action_type == "INSPECT":
            new_state, success, msg = inspect(state, args[0])
        elif action_type == "OPEN":
            new_state, success, msg = open_container(state, args[0])
        elif action_type == "CLOSE":
            new_state, success, msg = close_container(state, args[0])
        else:
            return state, False, f"Unknown action: {action_type}", action_str, raw_output

        if not success:
            # Action failed but state unchanged
            return state, False, f"Action failed: {msg}", action_str, raw_output

        done = new_state.is_task_completed() or (new_state.step_count >= new_state.max_steps)
        return new_state, done, msg, action_str, raw_output

    def run_episode(
        self,
        initial_state: WorldState,
        scenario_name: str = "unknown",
        verbose: bool = True
    ) -> Tuple[WorldState, bool, int, EpisodeLog]:
        """Run a complete planning episode until completion or max steps.
        
        Args:
            initial_state: Starting world state
            scenario_name: Name of the scenario for logging
            verbose: Whether to print step information
            
        Returns:
            Tuple of (final_state, success, steps_taken, episode_log)
            - final_state: World state after episode completion
            - success: True if task was completed successfully
            - steps_taken: Number of steps executed
            - episode_log: Complete trace of the episode
        """
        state = initial_state
        steps = 0
        
        # Initialize episode log
        episode_log = EpisodeLog(
            scenario=scenario_name,
            initial_state_text=initial_state.to_natural_language(),
            task_description=initial_state.task_description,
        )

        while True:
            state, done, info, action, llm_response = self.step(state)
            steps += 1
            
            # Log step details
            step_data = {
                "step_number": steps,
                "action": action,
                "llm_raw_output": llm_response,
                "result": info,
                "state_after": state.to_natural_language(),
                "step_count": state.step_count,
            }
            episode_log.steps.append(step_data)
            
            if verbose:
                print(f"Step {steps}: {info}")

            if done or steps >= self.max_steps:
                break

        success = state.is_task_completed()
        episode_log.success = success
        episode_log.total_steps = steps
        episode_log.final_state_text = state.to_natural_language()
        
        return state, success, steps, episode_log
