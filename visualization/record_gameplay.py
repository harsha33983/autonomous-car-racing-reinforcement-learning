"""
Batch Gameplay Recorder for Autonomous Racing Agents
===================================================
Generates high-definition GIF animations for both Continuous PPO and Discrete DQN agents.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
from visualization.gameplay import record_autonomous_racing


def record_all_gameplay(
    ppo_model: str = "models/ppo/ppo_carracing_final.zip",
    dqn_model: str = "models/dqn/dqn_carracing_final.zip",
    output_dir: str = "results/videos",
    seed: int = 1042,
    max_steps: int = 400,
):
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(ppo_model):
        ppo_gif = os.path.join(output_dir, "ppo_carracing.gif")
        print(f"\n[1/2] Recording Continuous PPO autonomous racing -> {ppo_gif}")
        try:
            record_autonomous_racing(
                algorithm="ppo",
                model_path=ppo_model,
                output_gif_path=ppo_gif,
                seed=seed,
                max_steps=max_steps,
            )
        except Exception as e:
            print(f"Error recording PPO gameplay: {e}")
    else:
        print(f"Notice: PPO model not found at {ppo_model}")

    if os.path.exists(dqn_model):
        dqn_gif = os.path.join(output_dir, "dqn_carracing.gif")
        print(f"\n[2/2] Recording Discrete DQN autonomous racing -> {dqn_gif}")
        try:
            record_autonomous_racing(
                algorithm="dqn",
                model_path=dqn_model,
                output_gif_path=dqn_gif,
                seed=seed,
                max_steps=max_steps,
            )
        except Exception as e:
            print(f"Error recording DQN gameplay: {e}")
    else:
        print(f"Notice: DQN model not found at {dqn_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record PPO & DQN Car Racing Videos")
    parser.add_argument("--ppo_model", type=str, default="models/ppo/ppo_carracing_final.zip")
    parser.add_argument("--dqn_model", type=str, default="models/dqn/dqn_carracing_final.zip")
    parser.add_argument("--output_dir", type=str, default="results/videos")
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--max_steps", type=int, default=300)
    args = parser.parse_args()

    record_all_gameplay(
        ppo_model=args.ppo_model,
        dqn_model=args.dqn_model,
        output_dir=args.output_dir,
        seed=args.seed,
        max_steps=args.max_steps,
    )
