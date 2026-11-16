"""
Proximal Policy Optimization (PPO) Agent Implementation
======================================================
Wraps Stable-Baselines3 PPO with actor-critic architecture inspection,
hyperparameter configuration, serialization, and deterministic/stochastic inference.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class PPOAgent:
    """
    Proximal Policy Optimization agent adhering to MLOps and reinforcement learning standards.
    Features:
    - Actor-Critic architecture with shared/separate feature extractors
    - Clipped surrogate policy objective L_CLIP
    - Generalized Advantage Estimation (GAE-lambda)
    - Value function squared error optimization
    - Entropy regularization bonus for exploratory stability
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
        """Recommended hyperparameters for Box2D LunarLander."""
        return {
            "learning_rate": 2.5e-4,
            "n_steps": 1024,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.999,
            "gae_lambda": 0.98,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch": {
                    "pi": [256, 256],
                    "vf": [256, 256],
                }
            },
        }

    def build_model(self) -> PPO:
        """Instantiate Stable-Baselines3 PPO model."""
        if self.env is None:
            raise ValueError("Environment must be set before building the PPO model.")

        params = dict(self.hyperparams)
        policy_kwargs = params.pop("policy_kwargs", {
            "net_arch": dict(pi=[256, 256], vf=[256, 256])
        })

        self.model = PPO(
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
        tb_log_name: str = "PPO",
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
        save_path = path if path.endswith(".zip") else f"{path}.zip"
        self.model.save(save_path)
        return save_path

    def load(self, path: str, env: Optional[Union[gym.Env, VecEnv]] = None) -> "PPOAgent":
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

        layer_info = []
        for name, module in policy_net.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_info.append({
                    "name": name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "bias": module.bias is not None,
                })

        return {
            "algorithm": "Proximal Policy Optimization (PPO)",
            "policy_type": "MlpPolicy (Actor-Critic)",
            "observation_space": str(self.model.observation_space),
            "action_space": str(self.model.action_space),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "layers": layer_info,
            "device": str(self.model.device),
            "hyperparameters": self.hyperparams,
        }
