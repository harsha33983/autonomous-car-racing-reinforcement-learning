"""
Agent Factory
=============
Dynamic instantiator and loader for RL agents.
"""

from typing import Any, Dict, Optional, Union
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent


def create_agent(
    algorithm: str,
    env: Optional[Union[gym.Env, VecEnv]] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    device: str = "cpu",
    tensorboard_log: Optional[str] = None,
) -> Union[DQNAgent, PPOAgent]:
    """
    Factory function to instantiate an agent by algorithm name.
    """
    algo_lower = algorithm.strip().lower()
    if algo_lower in ("dqn", "deep_q_network"):
        return DQNAgent(
            env=env,
            hyperparams=hyperparams,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
        )
    elif algo_lower in ("ppo", "proximal_policy_optimization"):
        return PPOAgent(
            env=env,
            hyperparams=hyperparams,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
        )
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Choose 'dqn' or 'ppo'.")


def load_agent(
    algorithm: str,
    model_path: str,
    env: Optional[Union[gym.Env, VecEnv]] = None,
    device: str = "cpu",
) -> Union[DQNAgent, PPOAgent]:
    """
    Load a pre-trained agent from a model file.
    """
    algo_lower = algorithm.strip().lower()
    if "dqn" in algo_lower:
        agent = DQNAgent(env=env, device=device)
        return agent.load(model_path, env=env)
    elif "ppo" in algo_lower:
        agent = PPOAgent(env=env, device=device)
        return agent.load(model_path, env=env)
    else:
        raise ValueError(f"Could not infer algorithm from '{algorithm}'. Choose 'dqn' or 'ppo'.")
