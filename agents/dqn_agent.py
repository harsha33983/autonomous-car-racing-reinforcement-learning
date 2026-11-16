"""
Deep Q-Network (DQN) Agent Implementation
==========================================
Wraps Stable-Baselines3 DQN with architecture inspection,
hyperparameter configuration, serialization, and deterministic inference.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class DQNAgent:
    """
    Deep Q-Network agent adhering to MLOps and reinforcement learning standards.
    Features:
    - Experience replay buffer
    - Target network with Polyak (soft) or periodic target updates
    - Epsilon-greedy schedule
    - Huber / Smooth L1 loss optimization
    """

    def __init__(
        self,
        env: Optional[Union[gym.Env, VecEnv]] = None,
        hyperparams: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        device: str = "cpu",
        tensorboard_log: Optional[str] = None,
    ):
        self.env = env
        self.seed = seed
        self.device = device
        self.tensorboard_log = tensorboard_log
        self.hyperparams = hyperparams or self.default_hyperparameters()
        self.model: Optional[DQN] = None

        if self.env is not None:
            self.build_model()

    @staticmethod
    def default_hyperparameters() -> Dict[str, Any]:
        """Recommended hyperparameters for Box2D LunarLander."""
        return {
            "learning_rate": 6.3e-4,
            "buffer_size": 100000,
            "learning_starts": 2000,
            "batch_size": 128,
            "tau": 0.005,
            "gamma": 0.995,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 250,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.02,
            "exploration_fraction": 0.25,
            "max_grad_norm": 1.0,
            "policy_kwargs": {
                "net_arch": [256, 256],
            },
        }

    def build_model(self) -> DQN:
        """Instantiate Stable-Baselines3 DQN model."""
        if self.env is None:
            raise ValueError("Environment must be set before building the DQN model.")

        params = dict(self.hyperparams)
        policy_kwargs = params.pop("policy_kwargs", {"net_arch": [256, 256]})

        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            seed=self.seed,
            device=self.device,
            tensorboard_log=self.tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=1,
            **params,
        )
        return self.model

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Union[BaseCallback, list]] = None,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
    ) -> DQN:
        """Execute training loop with logging and checkpoints."""
        if self.model is None:
            self.build_model()

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        return self.model

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Optional[np.ndarray]]:
        """Select action given observation."""
        if self.model is None:
            raise ValueError("Model has not been initialized or loaded.")
        action, state = self.model.predict(observation, deterministic=deterministic)
        return int(action), state

    def save(self, path: str) -> str:
        """Save trained model weights to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Ensure path ends with .zip
        save_path = path if path.endswith(".zip") else f"{path}.zip"
        self.model.save(save_path)
        return save_path

    def load(self, path: str, env: Optional[Union[gym.Env, VecEnv]] = None) -> "DQNAgent":
        """Load trained model weights from disk."""
        load_path = path if path.endswith(".zip") else f"{path}.zip"
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found at {load_path}")

        target_env = env or self.env
        self.model = DQN.load(load_path, env=target_env, device=self.device)
        self.env = target_env
        return self

    def get_model_summary(self) -> Dict[str, Any]:
        """Extract architectural and parameter details."""
        if self.model is None:
            return {"status": "uninitialized"}

        q_net = self.model.q_net
        total_params = sum(p.numel() for p in q_net.parameters())
        trainable_params = sum(p.numel() for p in q_net.parameters() if p.requires_grad)

        layer_info = []
        for name, module in q_net.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_info.append({
                    "name": name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "bias": module.bias is not None,
                })

        return {
            "algorithm": "Deep Q-Network (DQN)",
            "policy_type": "MlpPolicy",
            "observation_space": str(self.model.observation_space),
            "action_space": str(self.model.action_space),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "layers": layer_info,
            "device": str(self.model.device),
            "hyperparameters": self.hyperparams,
        }
