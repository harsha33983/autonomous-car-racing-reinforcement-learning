"""
Train Deep Q-Network (DQN) Agent
================================
Trains a DQN agent on LunarLander-v3 (or specified environment),
monitors training with TensorBoard & CSV metrics, and saves model checkpoints.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import yaml
from typing import Any, Dict, Optional

from environments.lunar_lander_env import make_vectorized_lunar_lander
from agents.dqn_agent import DQNAgent
from training.callbacks import MetricsLoggingCallback, BestModelCheckpointCallback
from training.checkpoint_manager import CheckpointManager


def train_dqn(
    config_path: str = "config/config.yaml",
    experiment: str = "optimized",
    total_timesteps: Optional[int] = None,
    seed: int = 42,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute end-to-end DQN training pipeline.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    algo_cfg = config["algorithms"]["dqn"][experiment]
    timesteps = total_timesteps or algo_cfg.get("total_timesteps", 120000)
    device = config["project"].get("device", "cpu")

    models_dir = os.path.join(config["paths"]["dqn_model_dir"], experiment)
    logs_dir = os.path.join(config["paths"]["logs_dir"], "dqn", experiment)
    tb_dir = os.path.join(config["paths"]["logs_dir"], "tensorboard")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    print(f"============================================================")
    print(f" Starting DQN Training: Experiment = '{experiment}'")
    print(f" Target Timesteps: {timesteps:,} | Device: {device} | Seed: {seed}")
    print(f" Log Directory: {logs_dir}")
    print(f" Model Directory: {models_dir}")
    print(f"============================================================")

    # Vectorized training environment
    vec_env = make_vectorized_lunar_lander(
        n_envs=1,
        seed=seed,
        log_dir=logs_dir,
    )

    # Setup callbacks
    metrics_cb = MetricsLoggingCallback(
        log_dir=logs_dir,
        filename_prefix="dqn_training_metrics",
        save_freq_steps=2000,
        verbose=1,
    )

    best_model_path = os.path.join(models_dir, "dqn_lunarlander_best")
    best_cb = BestModelCheckpointCallback(
        save_path=best_model_path,
        eval_freq=5000,
        verbose=1,
    )

    ckpt_mgr = CheckpointManager(checkpoint_dir=models_dir, prefix="dqn")
    ckpt_cb = ckpt_mgr.get_sb3_callback(save_freq=25000, verbose=1)

    callbacks = [metrics_cb, best_cb, ckpt_cb]

    # Instantiate or load agent
    hyperparams = dict(algo_cfg)
    hyperparams.pop("total_timesteps", None)

    agent = DQNAgent(
        env=vec_env,
        hyperparams=hyperparams,
        seed=seed,
        device=device,
        tensorboard_log=tb_dir,
    )

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        agent.load(resume_from, env=vec_env)

    # Execute training
    agent.train(
        total_timesteps=timesteps,
        callback=callbacks,
        tb_log_name=f"DQN_{experiment}",
        reset_num_timesteps=(resume_from is None),
    )

    # Save final model
    final_model_path = os.path.join(models_dir, "dqn_lunarlander_final.zip")
    agent.save(final_model_path)
    print(f"[+] Final DQN model saved to: {final_model_path}")

    # Also save to main models/dqn root for easy discovery
    main_model_path = os.path.join(config["paths"]["dqn_model_dir"], "dqn_lunarlander_final.zip")
    agent.save(main_model_path)

    # Update manifest
    summary_info = {
        "experiment": experiment,
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
    parser = argparse.ArgumentParser(description="Train DQN Agent on LunarLander-v3")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--experiment", type=str, default="optimized", choices=["baseline", "optimized"])
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train_dqn(
        config_path=args.config,
        experiment=args.experiment,
        total_timesteps=args.timesteps,
        seed=args.seed,
        resume_from=args.resume,
    )
