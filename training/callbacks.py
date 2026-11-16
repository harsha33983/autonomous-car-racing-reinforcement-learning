"""
Training Callbacks
==================
Custom callbacks for metric logging, periodic evaluation, best-model checkpointing,
and CSV/JSON export for TensorBoard and Streamlit dashboards.
"""

import os
import json
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback


class MetricsLoggingCallback(BaseCallback):
    """
    Comprehensive callback that records episodic performance,
    loss values, and rolling metrics to disk for real-time tracking.
    """

    def __init__(
        self,
        log_dir: str,
        filename_prefix: str = "metrics",
        save_freq_steps: int = 2000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.filename_prefix = filename_prefix
        self.save_freq_steps = save_freq_steps

        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, f"{filename_prefix}.csv")
        self.json_path = os.path.join(self.log_dir, f"{filename_prefix}_summary.json")

        self.episode_records: List[Dict[str, Any]] = []
        self.step_counter = 0
        self.episode_counter = 0
        self.start_time = time.time()

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        self.step_counter = self.num_timesteps

        # Check for completed episodes in info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_counter += 1
                ep_reward = float(info["episode"]["r"])
                ep_length = int(info["episode"]["l"])
                elapsed_sec = time.time() - self.start_time

                # Extract loss and other training metrics if logged
                logger_vals = getattr(self.model, "logger", None)
                logged_dict = logger_vals.name_to_value if logger_vals else {}

                record = {
                    "episode": self.episode_counter,
                    "timestep": self.step_counter,
                    "reward": ep_reward,
                    "length": ep_length,
                    "elapsed_seconds": round(elapsed_sec, 2),
                    "policy_loss": logged_dict.get("train/loss", logged_dict.get("train/policy_gradient_loss", np.nan)),
                    "value_loss": logged_dict.get("train/value_loss", np.nan),
                    "entropy_loss": logged_dict.get("train/entropy_loss", np.nan),
                    "learning_rate": logged_dict.get("train/learning_rate", np.nan),
                    "exploration_rate": logged_dict.get("rollout/exploration_rate", np.nan),
                }
                self.episode_records.append(record)

        if self.step_counter % self.save_freq_steps == 0 and self.episode_records:
            self._save_to_disk()

        return True

    def _on_training_end(self) -> None:
        self._save_to_disk()

    def _save_to_disk(self) -> None:
        if not self.episode_records:
            return

        df = pd.DataFrame(self.episode_records)
        df["rolling_mean_10"] = df["reward"].rolling(window=10, min_periods=1).mean()
        df["rolling_mean_50"] = df["reward"].rolling(window=50, min_periods=1).mean()
        df["rolling_std_50"] = df["reward"].rolling(window=50, min_periods=1).std().fillna(0)

        df.to_csv(self.csv_path, index=False)

        # Summary JSON
        summary = {
            "total_timesteps": self.step_counter,
            "total_episodes": self.episode_counter,
            "final_reward_mean_50": float(df["rolling_mean_50"].iloc[-1]),
            "max_reward": float(df["reward"].max()),
            "min_reward": float(df["reward"].min()),
            "mean_reward": float(df["reward"].mean()),
            "std_reward": float(df["reward"].std()),
            "mean_episode_length": float(df["length"].mean()),
            "total_training_time_seconds": round(time.time() - self.start_time, 2),
        }
        with open(self.json_path, "w") as f:
            json.dump(summary, f, indent=2)


class BestModelCheckpointCallback(BaseCallback):
    """
    Saves the best model based on evaluation score during training.
    """

    def __init__(
        self,
        save_path: str,
        eval_freq: int = 10000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.best_mean_reward = -float("inf")
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Check SB3's internal eval callback best score or logger rollout mean
            logger_vals = getattr(self.model, "logger", None)
            if logger_vals and "rollout/ep_rew_mean" in logger_vals.name_to_value:
                current_mean = logger_vals.name_to_value["rollout/ep_rew_mean"]
                if current_mean > self.best_mean_reward:
                    self.best_mean_reward = current_mean
                    model_path = self.save_path if self.save_path.endswith(".zip") else f"{self.save_path}.zip"
                    self.model.save(model_path)
                    if self.verbose > 0:
                        print(f"[*] New best model saved ({current_mean:.2f} mean reward) -> {model_path}")
        return True
