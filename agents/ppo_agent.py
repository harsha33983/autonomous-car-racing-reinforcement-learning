"""
PPO Agent Implementation for Autonomous Car Racing
==================================================
Wraps Stable-Baselines3 PPO with a CNN policy for continuous steering,
throttle, and braking control from visual track observations.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class PPORacingAgent:
    """
    Proximal Policy Optimization agent for continuous CarRacing-v3.
    Features:
    - NatureCNN visual feature extraction network
    - Dual Actor-Critic heads for continuous action distribution [steering, gas, brake]
    - GAE-lambda advantage estimation
    - Clipped surrogate loss with entropy bonus
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
        self.model: Optional[PPO] = None

        if self.env is not None:
            self.build_model()

    @staticmethod
    def default_hyperparameters() -> Dict[str, Any]:
        """Recommended hyperparameters for CarRacing-v3 on CPU."""
        return {
            "learning_rate": 3e-4,
            "n_steps": 512,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
        }

    def build_model(self) -> PPO:
        """Instantiate Stable-Baselines3 PPO model with CnnPolicy."""
        if self.env is None:
            raise ValueError("Environment must be set before building the PPO model.")

        params = dict(self.hyperparams)
        self.model = PPO(
            policy="CnnPolicy",
            env=self.env,
            seed=self.seed,
            device=self.device,
            tensorboard_log=self.tensorboard_log,
            verbose=1,
            **params,
        )
        return self.model

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Union[BaseCallback, list]] = None,
        tb_log_name: str = "PPO_CarRacing",
        reset_num_timesteps: bool = True,
    ) -> PPO:
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
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict continuous action [steering, gas, brake]."""
        if self.model is None:
            raise ValueError("Model has not been initialized or loaded.")
        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state

    def save(self, path: str) -> str:
        """Save trained model weights to disk."""
        if self.model is None:
            raise ValueError("No model to save.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        save_path = path if path.endswith(".zip") else f"{path}.zip"
        self.model.save(save_path)
        return save_path

    def load(self, path: str, env: Optional[Union[gym.Env, VecEnv]] = None) -> "PPORacingAgent":
        """Load trained model weights from disk."""
        load_path = path if path.endswith(".zip") else f"{path}.zip"
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found at {load_path}")

        target_env = env or self.env
        self.model = PPO.load(load_path, env=target_env, device=self.device)
        self.env = target_env
        return self

    def get_model_summary(self) -> Dict[str, Any]:
        """Extract architectural and parameter details."""
        if self.model is None:
            return {"status": "uninitialized"}

        policy_net = self.model.policy
        total_params = sum(p.numel() for p in policy_net.parameters())
        trainable_params = sum(p.numel() for p in policy_net.parameters() if p.requires_grad)

        return {
            "algorithm": "Proximal Policy Optimization (PPO)",
            "policy_type": "CnnPolicy (Actor-Critic)",
            "action_space": str(self.model.action_space),
            "observation_space": str(self.model.observation_space),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.model.device),
            "hyperparameters": self.hyperparams,
        }
