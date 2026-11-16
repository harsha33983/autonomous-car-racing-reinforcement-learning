"""
CartPole-v1 Environment Pipeline & Wrappers
===========================================
Provides environment factory, state space interpretation, and wrappers
for the CartPole-v1 balance benchmark.
"""

from typing import Any, Dict, Optional, Tuple, Callable
import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


ACTION_MEANINGS = {
    0: "Push Cart Left",
    1: "Push Cart Right",
}

OBSERVATION_FEATURES = [
    "cart_position",       # [-4.8, 4.8] (termination at |x| > 2.4)
    "cart_velocity",       # [-Inf, Inf]
    "pole_angle",          # [-0.418 rad, 0.418 rad] (termination at |theta| > 0.2095 rad / ~12 deg)
    "pole_angular_vel",    # [-Inf, Inf]
]


class CartPoleTelemetryWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that monitors CartPole telemetry,
    including maximum deviation, balance stability, and step survival.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_steps = 0
        self.cumulative_reward = 0.0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.episode_steps = 0
        self.cumulative_reward = 0.0
        obs, info = self.env.reset(**kwargs)
        info["telemetry"] = self._extract_telemetry(obs)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        self.cumulative_reward += reward

        telemetry = self._extract_telemetry(obs)
        telemetry["episode_steps"] = self.episode_steps
        telemetry["cumulative_reward"] = self.cumulative_reward
        if terminated or truncated:
            telemetry["is_successful"] = bool(self.cumulative_reward >= 475.0)

        info["telemetry"] = telemetry
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _extract_telemetry(obs: np.ndarray) -> Dict[str, Any]:
        return {
            "cart_position": float(obs[0]),
            "cart_velocity": float(obs[1]),
            "pole_angle": float(obs[2]),
            "pole_angular_velocity": float(obs[3]),
        }


def make_cartpole_env(
    env_id: str = "CartPole-v1",
    seed: Optional[int] = 42,
    render_mode: Optional[str] = None,
    log_file: Optional[str] = None,
) -> gym.Env:
    """Factory function for CartPole-v1."""
    env_kwargs = {}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **env_kwargs)
    env = CartPoleTelemetryWrapper(env)
    if log_file:
        env = Monitor(env, filename=log_file)
    else:
        env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


def make_vectorized_cartpole(
    n_envs: int = 1,
    seed: int = 42,
    render_mode: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> DummyVecEnv:
    """Create a DummyVecEnv for CartPole."""
    def _init(idx: int) -> Callable[[], gym.Env]:
        def _thunk() -> gym.Env:
            log_file = f"{log_dir}/cartpole_{idx}.monitor.csv" if log_dir else None
            return make_cartpole_env(
                env_id="CartPole-v1",
                seed=seed + idx,
                render_mode=render_mode,
                log_file=log_file,
            )
        return _thunk

    return DummyVecEnv([_init(i) for i in range(n_envs)])
