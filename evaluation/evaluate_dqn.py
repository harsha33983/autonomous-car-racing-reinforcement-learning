"""
Evaluate Discrete DQN Autonomous Racing Agent
=============================================
Evaluates the trained Discrete DQN CarRacing agent and saves metrics and detailed logs.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from evaluation.evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Evaluate Discrete DQN CarRacing Agent")
    parser.add_argument("--model", type=str, default="models/dqn/dqn_carracing_final.zip", help="Path to DQN model (.zip)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes")
    parser.add_argument("--seed", type=int, default=1000, help="Starting test seed")
    parser.add_argument("--output_dir", type=str, default="results/evaluation", help="Results directory")
    args = parser.parse_args()

    run_evaluation(
        algorithm="dqn",
        model_path=args.model,
        n_episodes=args.episodes,
        deterministic=True,
        output_dir=args.output_dir,
        seed_start=args.seed,
    )


if __name__ == "__main__":
    main()
