"""
Car Racing Policy Evaluation Pipeline
=====================================
Evaluates trained autonomous racing agents over independent episodes,
recording full trajectories, telemetry, track progress, and statistical performance.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import argparse
from typing import Any, Dict, List, Optional, Union
import gymnasium as gym
import pandas as pd
import numpy as np
from tqdm import tqdm

from environments.car_racing_env import make_car_racing_env
from environments.discrete_car_racing_env import make_discrete_car_racing_env
from agents.agent_factory import load_racing_agent
from agents.ppo_agent import PPORacingAgent
from agents.dqn_agent import DQNRacingAgent
from evaluation.metrics import compute_racing_metrics


def evaluate_racing_policy(
    agent: Any,
    env: gym.Env,
    n_episodes: int = 10,
    deterministic: bool = True,
    seed_start: int = 1000,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate an autonomous racing agent over n_episodes with fixed evaluation seeds.
    Tracks rewards, episode lengths, tiles visited, track progress %, and off-track occurrences.
    """
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    track_progresses: List[float] = []
    detailed_records: List[Dict[str, Any]] = []

    iterator = range(n_episodes)
    if verbose:
        iterator = tqdm(iterator, desc="Evaluating Racing Agent")

    for i in iterator:
        current_seed = seed_start + i
        obs, info = env.reset(seed=current_seed)
        done = False
        truncated = False
        ep_reward = 0.0
        ep_steps = 0
        last_info = info

        while not (done or truncated):
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += float(reward)
            ep_steps += 1
            last_info = info

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)

        telemetry = last_info.get("telemetry", {})
        track_prog = float(telemetry.get("track_progress", 0.0))
        track_progresses.append(track_prog)

        detailed_records.append({
            "episode": i + 1,
            "seed": current_seed,
            "reward": round(ep_reward, 3),
            "length": ep_steps,
            "track_progress_percent": round(track_prog, 2),
            "tiles_visited": int(telemetry.get("tiles_visited", 0)),
            "total_tiles": int(telemetry.get("total_tiles", 0)),
            "lap_completed": bool(telemetry.get("lap_completed", False)),
            "off_track_count": int(telemetry.get("off_track_count", 0)),
            "off_track_fraction": round(float(telemetry.get("off_track_fraction", 0.0)), 3),
        })

    metrics = compute_racing_metrics(
        rewards=episode_rewards,
        lengths=episode_lengths,
        track_progresses=track_progresses,
    )
    metrics["detailed_records"] = detailed_records

    return metrics


def run_evaluation(
    algorithm: str,
    model_path: str,
    n_episodes: int = 10,
    deterministic: bool = True,
    output_dir: str = "results/evaluation",
    seed_start: int = 1000,
    grayscale: bool = True,
    frame_skip: int = 2,
) -> Dict[str, Any]:
    """
    Load a model from disk, evaluate on CarRacing-v3, and save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    algo_lower = algorithm.lower()

    if algo_lower == "dqn":
        env = make_discrete_car_racing_env(
            seed=seed_start,
            grayscale=grayscale,
            frame_skip=frame_skip,
        )
    else:
        env = make_car_racing_env(
            continuous=True,
            seed=seed_start,
            grayscale=grayscale,
            frame_skip=frame_skip,
        )

    print(f"Loading {algorithm.upper()} racing agent from: {model_path}")
    agent = load_racing_agent(algorithm, model_path, env=env)

    results = evaluate_racing_policy(
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
    csv_filename = f"{algo_lower}_carracing_eval_{n_episodes}eps.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    detailed_df.to_csv(csv_path, index=False)

    # Save summary JSON
    summary_filename = f"{algo_lower}_carracing_eval_summary.json"
    summary_path = os.path.join(output_dir, summary_filename)
    results["model_path"] = os.path.abspath(model_path)
    results["csv_path"] = os.path.abspath(csv_path)
    results["algorithm"] = algorithm.upper()

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n==================== {algorithm.upper()} Racing Evaluation Results ({n_episodes} Episodes) ====================")
    print(f" Mean Reward:        {results.get('mean_reward', 0.0):.2f} +/- {results.get('std_reward', 0.0):.2f}")
    print(f" Median Reward:      {results.get('median_reward', 0.0):.2f} (IQR: {results.get('iqr_reward', 0.0):.2f})")
    print(f" Min / Max Reward:   {results.get('min_reward', 0.0):.2f} / {results.get('max_reward', 0.0):.2f}")
    print(f" 95% Confidence Int: [{results.get('ci_95_low', 0.0):.2f}, {results.get('ci_95_high', 0.0):.2f}]")
    print(f" Mean Episode Length:{results.get('mean_episode_length', 0.0):.2f} steps")
    print(f" Mean Track Progress:{results.get('mean_track_progress_percent', 0.0):.1f}%")
    print(f" Max Track Progress: {results.get('max_track_progress_percent', 0.0):.1f}%")
    print(f" Lap Completion Rate:{results.get('track_completion_rate_percent', 0.0):.1f}%")
    print(f" Improvement v Rand: {results.get('improvement_over_random_percent', 0.0):.1f}%")
    print(f" Output saved to:    {summary_path}")
    print("==========================================================================")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL Agent on CarRacing-v3")
    parser.add_argument("--algo", type=str, required=True, choices=["dqn", "ppo"])
    parser.add_argument("--model", type=str, required=True, help="Path to saved model (.zip)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
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
