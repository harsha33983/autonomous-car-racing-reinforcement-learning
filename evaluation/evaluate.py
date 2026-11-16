"""
Agent Evaluation Pipeline
=========================
Evaluates trained reinforcement learning agents over independent episodes,
recording full trajectories, telemetry, and statistical performance.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import argparse
from typing import Any, Dict, List, Optional, Union
import gymnasium as gym
import pandas as pd
from tqdm import tqdm

from environments.lunar_lander_env import make_lunar_lander_env
from agents.agent_factory import load_agent
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from evaluation.metrics import compute_rl_metrics


def evaluate_policy(
    agent: Union[DQNAgent, PPOAgent],
    env: gym.Env,
    n_episodes: int = 100,
    deterministic: bool = True,
    seed_start: int = 1000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate an agent over n_episodes with fixed evaluation seeds.
    """
    episode_rewards = []
    episode_lengths = []
    detailed_records = []

    iterator = range(n_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Evaluating Agent")

    for i in iterator:
        current_seed = seed_start + i
        obs, info = env.reset(seed=current_seed)
        done = False
        truncated = False
        ep_reward = 0.0
        ep_steps = 0

        while not (done or truncated):
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)

        telemetry = info.get("telemetry", {})
        detailed_records.append({
            "episode": i + 1,
            "seed": current_seed,
            "reward": round(ep_reward, 3),
            "length": ep_steps,
            "is_successful": telemetry.get("is_successful_landing", ep_reward >= 200.0),
            "is_crashed": telemetry.get("is_crashed", ep_reward < -100.0),
            "main_engine_count": telemetry.get("main_engine_count", 0),
            "side_engine_count": telemetry.get("side_engine_count", 0),
        })

    metrics = compute_rl_metrics(
        rewards=episode_rewards,
        lengths=episode_lengths,
        success_threshold=200.0,
    )
    metrics["detailed_records"] = detailed_records

    return metrics


def run_evaluation(
    algorithm: str,
    model_path: str,
    n_episodes: int = 100,
    deterministic: bool = True,
    output_dir: str = "results/evaluation",
    seed_start: int = 1000,
) -> Dict[str, Any]:
    """
    Load a model from disk, evaluate on LunarLander-v3, and save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    env = make_lunar_lander_env(seed=seed_start)

    print(f"Loading {algorithm.upper()} model from: {model_path}")
    agent = load_agent(algorithm, model_path, env=env)

    results = evaluate_policy(
        agent=agent,
        env=env,
        n_episodes=n_episodes,
        deterministic=deterministic,
        seed_start=seed_start,
        verbose=True,
    )

    env.close()

    # Save detailed CSV
    detailed_df = pd.DataFrame(results.pop("detailed_records"))
    csv_filename = f"{algorithm.lower()}_eval_{n_episodes}eps.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    detailed_df.to_csv(csv_path, index=False)

    # Save summary JSON
    summary_filename = f"{algorithm.lower()}_eval_summary.json"
    summary_path = os.path.join(output_dir, summary_filename)
    results["model_path"] = os.path.abspath(model_path)
    results["csv_path"] = os.path.abspath(csv_path)
    results["algorithm"] = algorithm.upper()

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n==================== {algorithm.upper()} Evaluation Results ({n_episodes} Episodes) ====================")
    print(f" Mean Reward:        {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f" Median Reward:      {results['median_reward']:.2f} (IQR: {results['iqr_reward']:.2f})")
    print(f" Min / Max Reward:   {results['min_reward']:.2f} / {results['max_reward']:.2f}")
    print(f" 95% Confidence Int: [{results['ci_95_low']:.2f}, {results['ci_95_high']:.2f}]")
    print(f" Mean Episode Length:{results['mean_episode_length']:.2f} steps")
    print(f" Success Rate:       {results['success_rate_percent']:.1f}%")
    print(f" Crash Rate:         {results['crash_rate_percent']:.1f}%")
    print(f" Improvement v Rand: {results['improvement_over_random_percent']:.1f}%")
    print(f" Output saved to:    {summary_path}")
    print("==========================================================================")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL Agent on LunarLander-v3")
    parser.add_argument("--algo", type=str, required=True, choices=["dqn", "ppo"])
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.zip)")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=1000, help="Starting random seed")
    parser.add_argument("--output_dir", type=str, default="results/evaluation")
    args = parser.parse_args()

    run_evaluation(
        algorithm=args.algo,
        model_path=args.model,
        n_episodes=args.episodes,
        output_dir=args.output_dir,
        seed_start=args.seed,
    )
