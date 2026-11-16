"""
Agent Factory for Autonomous Car Racing
======================================
Instantiates and loads continuous PPO and discrete DQN racing agents.
"""

from typing import Any, Dict, Optional, Union
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

from agents.ppo_agent import PPORacingAgent
from agents.dqn_agent import DQNRacingAgent


def create_racing_agent(
    algorithm: str,
    env: Optional[Union[gym.Env, VecEnv]] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    device: str = "cpu",
    tensorboard_log: Optional[str] = None,
) -> Union[PPORacingAgent, DQNRacingAgent]:
    """Factory to instantiate racing agent by algorithm name."""
    algo_lower = algorithm.strip().lower()
    if algo_lower in ("ppo", "proximal_policy_optimization"):
        return PPORacingAgent(
            env=env,
            hyperparams=hyperparams,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
        )
    elif algo_lower in ("dqn", "deep_q_network"):
        return DQNRacingAgent(
            env=env,
            hyperparams=hyperparams,
            seed=seed,
            device=device,
            tensorboard_log=tensorboard_log,
        )
    else:
        raise ValueError(f"Unsupported racing algorithm '{algorithm}'. Choose 'ppo' or 'dqn'.")


def load_racing_agent(
    algorithm: str,
    model_path: str,
    env: Optional[Union[gym.Env, VecEnv]] = None,
    device: str = "cpu",
) -> Union[PPORacingAgent, DQNRacingAgent]:
    """Load pre-trained racing agent from weights file."""
    algo_lower = algorithm.strip().lower()
    if "ppo" in algo_lower:
        agent = PPORacingAgent(env=env, device=device)
        return agent.load(model_path, env=env)
    elif "dqn" in algo_lower:
        agent = DQNRacingAgent(env=env, device=device)
        return agent.load(model_path, env=env)
    else:
        raise ValueError(f"Could not infer algorithm from '{algorithm}'. Choose 'ppo' or 'dqn'.")
