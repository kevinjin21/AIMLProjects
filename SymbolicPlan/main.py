from dotenv import load_dotenv
from agent.llm_client import LLMClient
from agent.planner import LLMPlanner
from eval.metrics import (
    evaluate_planner,
    print_evaluation_report,
    make_scenario_pick_and_place_box,
    make_scenario_container_task,
    make_scenario_multi_object,
)

load_dotenv()

def main():
    print("="*60)
    print("SymbolicPlan - LLM Robot Task Planner")
    print("="*60)
    print("Episode logs will be saved to: logs/")
    print("="*60 + "\n")
    
    # Create planner
    llm = LLMClient()
    planner = LLMPlanner(llm=llm, max_steps=15)

    # Test 1: Pick and place box task
    print("\\n" + "="*60)
    print("Testing basic pick-and-place...")
    print("="*60)
    results1 = evaluate_planner(
        planner,
        make_scenario_pick_and_place_box,
        n_episodes=5,
        save_logs=True,
        verbose=True
    )
    print_evaluation_report(results1)

    # Test 2: Open container task
    print("\\n" + "="*60)
    print("Testing container task...")
    print("="*60)
    results2 = evaluate_planner(
        planner,
        make_scenario_container_task,
        n_episodes=5,
        save_logs=True,
        verbose=True
    )
    print_evaluation_report(results2)

    # Test 3: More complex task with multiple objects
    print("\\n" + "="*60)
    print("Testing multi-object task...")
    print("="*60)
    results3 = evaluate_planner(
        planner,
        make_scenario_multi_object,
        n_episodes=5,
        save_logs=True,
        verbose=True
    )
    print_evaluation_report(results3)
    
    print("\\n" + "="*60)
    print("Evaluation complete! Check logs/ directory for detailed traces.")
    print("="*60)

if __name__ == "__main__":
    main()