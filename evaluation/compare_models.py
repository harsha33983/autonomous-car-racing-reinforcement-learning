"""
Comparative Analysis: DQN vs PPO
================================
Runs side-by-side benchmark comparison between DQN, PPO, and Random baseline agents
over identical test seeds, computing statistical significance and comparative metrics.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import argparse
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from environments.lunar_lander_env import make_lunar_lander_env
from agents.agent_factory import load_agent
from evaluation.evaluate import evaluate_policy
from evaluation.metrics import compute_rl_metrics, compare_two_distributions


class RandomAgent:
    """Baseline agent taking uniformly random actions."""
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=True):
        return int(self.action_space.sample()), None


def run_comparative_benchmark(
    dqn_model_path: str,
    ppo_model_path: str,
    n_episodes: int = 100,
    seed_start: int = 1000,
    output_dir: str = "results/evaluation",
) -> Dict[str, Any]:
    """
    Execute full comparative analysis between DQN, PPO, and Random baseline.
    """
    os.makedirs(output_dir, exist_ok=True)
    env = make_lunar_lander_env(seed=seed_start)

    print("\n=======================================================")
    print(f" Starting Multi-Agent Comparative Benchmark ({n_episodes} Episodes)")
    print("=======================================================")

    # 1. Random Agent Baseline
    print("\n[1/3] Benchmarking Uniform Random Agent Baseline...")
    random_agent = RandomAgent(env.action_space)
    rand_metrics = evaluate_policy(
        agent=random_agent,
        env=env,
        n_episodes=n_episodes,
        deterministic=False,
        seed_start=seed_start,
        verbose=False,
    )
    rand_records = rand_metrics.pop("detailed_records")

    # 2. DQN Agent
    print(f"\n[2/3] Benchmarking DQN Agent ({dqn_model_path})...")
    dqn_agent = load_agent("dqn", dqn_model_path, env=env)
    dqn_metrics = evaluate_policy(
        agent=dqn_agent,
        env=env,
        n_episodes=n_episodes,
        deterministic=True,
        seed_start=seed_start,
        verbose=False,
    )
    dqn_records = dqn_metrics.pop("detailed_records")

    # 3. PPO Agent
    print(f"\n[3/3] Benchmarking PPO Agent ({ppo_model_path})...")
    ppo_agent = load_agent("ppo", ppo_model_path, env=env)
    ppo_metrics = evaluate_policy(
        agent=ppo_agent,
        env=env,
        n_episodes=n_episodes,
        deterministic=True,
        seed_start=seed_start,
        verbose=False,
    )
    ppo_records = ppo_metrics.pop("detailed_records")

    env.close()

    # Statistical significance testing
    dqn_rewards = [r["reward"] for r in dqn_records]
    ppo_rewards = [r["reward"] for r in ppo_records]
    rand_rewards = [r["reward"] for r in rand_records]

    stat_test = compare_two_distributions(
        rewards_a=dqn_rewards,
        rewards_b=ppo_rewards,
        label_a="DQN",
        label_b="PPO",
    )

    # Compile comparative table
    comparison_table = [
        {
            "Algorithm": "Random Baseline",
            "Mean Reward": rand_metrics["mean_reward"],
            "Std Reward": rand_metrics["std_reward"],
            "Median Reward": rand_metrics["median_reward"],
            "Min Reward": rand_metrics["min_reward"],
            "Max Reward": rand_metrics["max_reward"],
            "Success Rate (%)": rand_metrics["success_rate_percent"],
            "Crash Rate (%)": rand_metrics["crash_rate_percent"],
            "Mean Length (steps)": rand_metrics["mean_episode_length"],
            "Improvement vs Random (%)": 0.0,
        },
        {
            "Algorithm": "Deep Q-Network (DQN)",
            "Mean Reward": dqn_metrics["mean_reward"],
            "Std Reward": dqn_metrics["std_reward"],
            "Median Reward": dqn_metrics["median_reward"],
            "Min Reward": dqn_metrics["min_reward"],
            "Max Reward": dqn_metrics["max_reward"],
            "Success Rate (%)": dqn_metrics["success_rate_percent"],
            "Crash Rate (%)": dqn_metrics["crash_rate_percent"],
            "Mean Length (steps)": dqn_metrics["mean_episode_length"],
            "Improvement vs Random (%)": dqn_metrics["improvement_over_random_percent"],
        },
        {
            "Algorithm": "Proximal Policy Opt (PPO)",
            "Mean Reward": ppo_metrics["mean_reward"],
            "Std Reward": ppo_metrics["std_reward"],
            "Median Reward": ppo_metrics["median_reward"],
            "Min Reward": ppo_metrics["min_reward"],
            "Max Reward": ppo_metrics["max_reward"],
            "Success Rate (%)": ppo_metrics["success_rate_percent"],
            "Crash Rate (%)": ppo_metrics["crash_rate_percent"],
            "Mean Length (steps)": ppo_metrics["mean_episode_length"],
            "Improvement vs Random (%)": ppo_metrics["improvement_over_random_percent"],
        },
    ]

    comp_df = pd.DataFrame(comparison_table)
    comp_csv_path = os.path.join(output_dir, "algorithm_comparison.csv")
    comp_df.to_csv(comp_csv_path, index=False)

    # Detailed merged dataset for plotting distributions
    combined_eval_records = []
    for r in rand_records:
        r["algorithm"] = "Random"
        combined_eval_records.append(r)
    for r in dqn_records:
        r["algorithm"] = "DQN"
        combined_eval_records.append(r)
    for r in ppo_records:
        r["algorithm"] = "PPO"
        combined_eval_records.append(r)

    combined_df = pd.DataFrame(combined_eval_records)
    combined_csv_path = os.path.join(output_dir, "combined_eval_episodes.csv")
    combined_df.to_csv(combined_csv_path, index=False)

    summary_payload = {
        "n_episodes": n_episodes,
        "seed_start": seed_start,
        "comparison_table": comparison_table,
        "statistical_test": stat_test,
        "dqn_summary": dqn_metrics,
        "ppo_summary": ppo_metrics,
        "random_summary": rand_metrics,
    }

    summary_json_path = os.path.join(output_dir, "algorithm_comparison.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n======================= BENCHMARK COMPARISON TABLE =======================")
    print(comp_df.to_string(index=False))
    print("\n---------------------- STATISTICAL HYPOTHESIS TEST ----------------------")
    print(f" Welch's t-test: t = {stat_test['welch_t_statistic']}, p = {stat_test['welch_p_value']:.4e}")
    print(f" Mann-Whitney U: U = {stat_test['mann_whitney_u']}, p = {stat_test['mann_whitney_p_value']:.4e}")
    print(f" Cohen's d:      d = {stat_test['cohens_d']} ({'Large' if abs(stat_test['cohens_d']) > 0.8 else 'Medium' if abs(stat_test['cohens_d']) > 0.5 else 'Small'})")
    print(f" Significant (p < 0.05): {stat_test['statistically_significant_005']}")
    print(f" Best Agent:     {stat_test['higher_mean_performer']}")
    print(f"\nSaved results to:\n  - {comp_csv_path}\n  - {summary_json_path}\n  - {combined_csv_path}")
    print("==========================================================================")

    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DQN and PPO on LunarLander-v3")
    parser.add_argument("--dqn_model", type=str, default="models/dqn/dqn_lunarlander_final.zip")
    parser.add_argument("--ppo_model", type=str, default="models/ppo/ppo_lunarlander_final.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    args = parser.parse_args()

    run_comparative_benchmark(
        dqn_model_path=args.dqn_model,
        ppo_model_path=args.ppo_model,
        n_episodes=args.episodes,
        seed_start=args.seed,
        output_dir=args.output_dir,
    )
