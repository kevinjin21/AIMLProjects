"""Evaluation metrics and test scenarios for the LLM planner."""

from typing import Dict
from env.world import WorldState, RobotState, ObjectState, Goal
from agent.planner import LLMPlanner
from .logging_utils import save_episode_log, save_evaluation_summary


def make_scenario_pick_and_place_box() -> WorldState:
    """Create a basic pick-and-place task scenario.
    
    Task: Move box from table to shelf while a book acts as a distractor.
    
    Returns:
        Initial world state for the scenario
    """
    objects = {
        "box": ObjectState(name="box", location="table"),
        "book": ObjectState(name="book", location="floor"),  # distractor
    }
    robot = RobotState(location="floor", holding=None)
    
    return WorldState(
        robot=robot,
        objects=objects,
        task_description="Place the box from the table onto the shelf.",
        goals=[
            Goal(type="object_at_location", params={"object": "box", "location": "shelf"})
        ],
        step_count=0,
        max_steps=10,
    )


def make_scenario_container_task() -> WorldState:
    """Create a scenario involving opening a container.
    
    Task: Open the box at the table.
    
    Returns:
        Initial world state for the scenario
    """
    objects = {
        "box": ObjectState(
            name="box",
            location="table",
            properties={"is_container": True, "is_open": False}
        ),
    }
    robot = RobotState(location="table", holding=None)
    
    return WorldState(
        robot=robot,
        objects=objects,
        task_description="Open the box at the table.",
        goals=[
            Goal(type="object_property", params={"object": "box", "property": "is_open", "value": "true"})
        ],
        step_count=0,
        max_steps=8,
    )


def make_scenario_multi_object() -> WorldState:
    """Create a scenario with multiple objects to organize.
    
    Task: Put book on shelf and box on floor.
    
    Returns:
        Initial world state for the scenario
    """
    objects = {
        "box": ObjectState(name="box", location="table"),
        "book": ObjectState(name="book", location="table"),
    }
    robot = RobotState(location="table", holding=None)
    
    return WorldState(
        robot=robot,
        objects=objects,
        task_description="Put the book on the shelf and the box on the floor.",
        goals=[
            Goal(type="object_at_location", params={"object": "book", "location": "shelf"}),
            Goal(type="object_at_location", params={"object": "box", "location": "floor"}),
        ],
        step_count=0,
        max_steps=15,
    )


def evaluate_planner(
    planner: LLMPlanner,
    scenario_fn=make_scenario_pick_and_place_box,
    n_episodes: int = 10,
    save_logs: bool = True,
    verbose: bool = True
) -> Dict[str, float]:
    """Evaluate planner performance on a specific scenario.
    
    Args:
        planner: The LLM planner to evaluate
        scenario_fn: Function that returns initial WorldState for testing
        n_episodes: Number of episodes to run
        save_logs: Whether to save episode logs to disk
        verbose: Whether to print step information during execution
        
    Returns:
        Dictionary containing evaluation metrics:
        - completion_rate: Fraction of successful episodes
        - avg_steps: Average steps taken per episode
        - success_efficiency: Avg steps for successful episodes only
        - episodes: Number of episodes run
    """
    successes = 0
    total_steps = 0
    successful_steps = 0
    
    # Get scenario name from function
    scenario_name = scenario_fn.__name__.replace("make_scenario_", "").replace("_", " ").title()

    for episode_num in range(n_episodes):
        init_state = scenario_fn()
        _, success, steps, episode_log = planner.run_episode(
            init_state,
            scenario_name=f"{scenario_name} Episode {episode_num + 1}",
            verbose=verbose
        )
        
        # Save episode log if requested
        if save_logs:
            log_path = save_episode_log(episode_log)
            if not verbose:
                status = "✓" if success else "✗"
                print(f"  [{status}] Episode {episode_num + 1}: {steps} steps (saved to {log_path})")
        
        if success:
            successes += 1
            successful_steps += steps
        total_steps += steps

    completion_rate = successes / n_episodes
    avg_steps = total_steps / n_episodes if n_episodes else 0.0
    success_efficiency = successful_steps / successes if successes > 0 else 0.0

    results = {
        "completion_rate": completion_rate,
        "avg_steps": avg_steps,
        "success_efficiency": success_efficiency,
        "successes": successes,
        "episodes": n_episodes,
    }
    
    # Save evaluation summary if logs are enabled
    if save_logs:
        save_evaluation_summary(results, scenario_name)
    
    return results


def print_evaluation_report(results: Dict[str, float]) -> None:
    """Print a formatted evaluation report.
    
    Args:
        results: Dictionary from evaluate_planner
    """
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"Episodes Run:       {results['episodes']}")
    print(f"Successes:          {results['successes']}")
    print(f"Completion Rate:    {results['completion_rate']:.1%}")
    print(f"Avg Steps (All):    {results['avg_steps']:.2f}")
    print(f"Avg Steps (Success):{results['success_efficiency']:.2f}")
    print("=" * 50 + "\n")
