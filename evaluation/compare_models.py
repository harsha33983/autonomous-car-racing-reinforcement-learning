"""
Comparative Analysis: Continuous PPO vs Discrete DQN
====================================================
Alias wrapper around compare_agents for backward compatibility.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from evaluation.compare_agents import run_comparative_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Car Racing Multi-Agent Benchmark")
    parser.add_argument("--ppo_model", type=str, default="models/ppo/ppo_carracing_final.zip")
    parser.add_argument("--dqn_model", type=str, default="models/dqn/dqn_carracing_final.zip")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    args = parser.parse_args()

    run_comparative_benchmark(
        ppo_model_path=args.ppo_model,
        dqn_model_path=args.dqn_model,
        n_episodes=args.episodes,
        seed_start=args.seed,
        output_dir=args.output_dir,
    )
