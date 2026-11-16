"""
Orchestrator: Train Both PPO and DQN Racing Agents
==================================================
Runs training pipelines sequentially for PPO and DQN models.
"""

import argparse
from training.train_ppo import train_ppo
from training.train_dqn import train_dqn


def train_all(
    config_path: str = "config/config.yaml",
    timesteps: int = 50000,
    seed: int = 42,
    smoke_test: bool = False,
):
    print("=" * 70)
    print("      AUTONOMOUS CAR RACING MULTI-AGENT TRAINING ORCHESTRATOR")
    print("=" * 70)

    # 1. Train PPO
    print("\n>>> Phase 1/2: Training Continuous PPO Agent...")
    ppo_summary = train_ppo(
        config_path=config_path,
        total_timesteps=timesteps,
        seed=seed,
        smoke_test=smoke_test,
    )

    # 2. Train DQN
    print("\n>>> Phase 2/2: Training Discrete DQN Agent...")
    dqn_summary = train_dqn(
        config_path=config_path,
        total_timesteps=timesteps,
        seed=seed,
        smoke_test=smoke_test,
    )

    print("\n" + "=" * 70)
    print("  ALL RACING AGENT TRAINING PIPELINES COMPLETED SUCCESSFULLY!")
    print(f"  PPO Model: {ppo_summary['final_model']}")
    print(f"  DQN Model: {dqn_summary['final_model']}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO and DQN on CarRacing-v3")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    train_all(
        config_path=args.config,
        timesteps=args.timesteps,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )
