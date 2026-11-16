"""
Train Discrete DQN Agent on CarRacing-v3
========================================
Trains an autonomous racing car agent with discrete driving action choices.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import yaml
from typing import Any, Dict, Optional

from environments.discrete_car_racing_env import make_vectorized_discrete_car_racing
from agents.dqn_agent import DQNRacingAgent
from training.callbacks import RacingMetricsCallback, BestRacingModelCallback
from training.checkpoint_manager import RacingCheckpointManager


def train_dqn(
    config_path: str = "config/config.yaml",
    total_timesteps: Optional[int] = None,
    seed: int = 42,
    resume_from: Optional[str] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end DQN racing training pipeline."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    algo_cfg = config["algorithms"]["dqn"]
    timesteps = 500 if smoke_test else (total_timesteps or algo_cfg.get("total_timesteps", 50000))
    device = config["project"].get("device", "cpu")

    models_dir = config["paths"]["dqn_model_dir"]
    logs_dir = os.path.join(config["paths"]["logs_dir"], "dqn")
    tb_dir = os.path.join(config["paths"]["logs_dir"], "tensorboard")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    print("=" * 65)
    print("  STARTING DQN AUTONOMOUS RACING TRAINING (DiscreteCarRacing-v3)")
    print(f"  Target Timesteps : {timesteps:,} | Device: {device} | Seed: {seed}")
    print(f"  Model Output Dir : {models_dir}")
    print(f"  TensorBoard Logs : {tb_dir}")
    print("=" * 65)

    vec_env = make_vectorized_discrete_car_racing(
        n_envs=1,
        seed=seed,
        log_dir=logs_dir,
        frame_skip=config["environment"]["observation"].get("frame_skip", 2),
        grayscale=True,
    )

    metrics_cb = RacingMetricsCallback(
        log_dir=logs_dir,
        filename_prefix="dqn_racing_metrics",
        save_freq_steps=1000,
        verbose=1,
    )

    best_model_path = os.path.join(models_dir, "dqn_carracing_best")
    best_cb = BestRacingModelCallback(
        save_path=best_model_path,
        eval_freq=2500,
        verbose=1,
    )

    ckpt_mgr = RacingCheckpointManager(checkpoint_dir=models_dir, prefix="dqn")
    ckpt_cb = ckpt_mgr.get_sb3_callback(save_freq=5000, verbose=1)

    callbacks = [metrics_cb, best_cb, ckpt_cb]

    hyperparams = dict(algo_cfg)
    hyperparams.pop("policy", None)
    hyperparams.pop("total_timesteps", None)
    if smoke_test:
        hyperparams["learning_starts"] = 100
        hyperparams["buffer_size"] = 2000

    agent = DQNRacingAgent(
        env=vec_env,
        hyperparams=hyperparams,
        seed=seed,
        device=device,
        tensorboard_log=tb_dir,
    )

    if resume_from:
        print(f"Resuming DQN from checkpoint: {resume_from}")
        agent.load(resume_from, env=vec_env)

    agent.train(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name="DQN_CarRacing",
        reset_num_timesteps=(resume_from is None),
    )

    final_model_path = os.path.join(models_dir, "dqn_carracing_final.zip")
    agent.save(final_model_path)
    print(f"[+] Final DQN Racing model saved to: {final_model_path}")

    summary_info = {
        "algorithm": "DQN",
        "timesteps": timesteps,
        "seed": seed,
        "final_model": final_model_path,
        "best_model": f"{best_model_path}.zip",
        "hyperparameters": hyperparams,
    }
    ckpt_mgr.update_manifest(summary_info)

    vec_env.close()
    return summary_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Discrete DQN on CarRacing-v3")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    train_dqn(
        config_path=args.config,
        total_timesteps=args.timesteps,
        seed=args.seed,
        resume_from=args.resume,
        smoke_test=args.smoke_test,
    )
