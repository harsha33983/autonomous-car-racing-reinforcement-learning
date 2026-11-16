"""
LunarLander-v3 Environment Pipeline & Wrappers
==============================================
Provides high-level environment creation, state space documentation,
telemetry tracking, and vectorized environment handling for LunarLander.
"""

from typing import Any, Dict, Optional, Tuple, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# Action semantics for LunarLander-v3
ACTION_MEANINGS = {
    0: "Do Nothing (Coasting)",
    1: "Fire Left Engine (Yaw Right)",
    2: "Fire Main Engine (Upward Thrust)",
    3: "Fire Right Engine (Yaw Left)",
}

# Observation feature names and units
OBSERVATION_FEATURES = [
    "x_position",          # [-1.5, 1.5] horizontal position (0 = landing pad center)
    "y_position",          # [-1.5, 1.5] vertical altitude
    "linear_velocity_x",   # [-5.0, 5.0] horizontal speed
    "linear_velocity_y",   # [-5.0, 5.0] vertical descent speed
    "angle",               # [-3.14, 3.14] tilt angle in radians (0 = vertical)
    "angular_velocity",    # [-5.0, 5.0] rotational velocity
    "left_leg_contact",    # 1.0 if left leg on ground, else 0.0
    "right_leg_contact",   # 1.0 if right leg on ground, else 0.0
]


class LunarLanderTelemetryWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that monitors LunarLander specific telemetry,
    including descent velocity, tilt stability, leg contacts, fuel usage (steps),
    and landing outcome classification.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_steps = 0
        self.cumulative_reward = 0.0
        self.main_engine_activations = 0
        self.side_engine_activations = 0

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.episode_steps = 0
        self.cumulative_reward = 0.0
        self.main_engine_activations = 0
        self.side_engine_activations = 0
        obs, info = self.env.reset(**kwargs)
        info["telemetry"] = self._extract_telemetry(obs)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1
        self.cumulative_reward += reward

        if action == 2:
            self.main_engine_activations += 1
        elif action in (1, 3):
            self.side_engine_activations += 1

        telemetry = self._extract_telemetry(obs)
        telemetry["episode_steps"] = self.episode_steps
        telemetry["cumulative_reward"] = self.cumulative_reward
        telemetry["main_engine_count"] = self.main_engine_activations
        telemetry["side_engine_count"] = self.side_engine_activations

        # Check landing status at termination
        if terminated or truncated:
            is_landed = (
                telemetry["left_leg_contact"] == 1.0
                and telemetry["right_leg_contact"] == 1.0
                and abs(telemetry["x_position"]) < 0.2
                and abs(telemetry["angle"]) < 0.2
            )
            is_crashed = terminated and (self.cumulative_reward < 0 or abs(telemetry["angle"]) > 1.0)
            telemetry["is_successful_landing"] = bool(is_landed or self.cumulative_reward >= 200.0)
            telemetry["is_crashed"] = bool(is_crashed and not telemetry["is_successful_landing"])

        info["telemetry"] = telemetry
        return obs, reward, terminated, truncated, info

    @staticmethod
    def _extract_telemetry(obs: np.ndarray) -> Dict[str, Any]:
        """Convert raw 8-dimensional observation vector to named telemetry dictionary."""
        return {
            "x_position": float(obs[0]),
            "y_position": float(obs[1]),
            "linear_velocity_x": float(obs[2]),
            "linear_velocity_y": float(obs[3]),
            "angle": float(obs[4]),
            "angular_velocity": float(obs[5]),
            "left_leg_contact": float(obs[6]),
            "right_leg_contact": float(obs[7]),
        }


def make_lunar_lander_env(
    env_id: str = "LunarLander-v3",
    seed: Optional[int] = 42,
    render_mode: Optional[str] = None,
    log_file: Optional[str] = None,
    gravity: float = -10.0,
    enable_wind: bool = False,
    wind_power: float = 15.0,
    turbulence_power: float = 1.5,
) -> gym.Env:
    """
    Factory function to initialize a properly instrumented LunarLander environment.

    Args:
        env_id: Gymnasium environment ID ('LunarLander-v3')
        seed: Random seed for deterministic reproducibility
        render_mode: Rendering mode (e.g. 'rgb_array' or 'human')
        log_file: Optional CSV path for Monitor wrapper episodic rewards
        gravity: Gravitational acceleration constant
        enable_wind: Whether to simulate atmospheric wind disturbances
        wind_power: Maximum wind force
        turbulence_power: Turbulence disturbance intensity

    Returns:
        Instrumented gym.Env instance
    """
    env_kwargs = {
        "gravity": gravity,
        "enable_wind": enable_wind,
        "wind_power": wind_power,
        "turbulence_power": turbulence_power,
    }
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **env_kwargs)

    # Wrap with telemetry and Monitor
    env = LunarLanderTelemetryWrapper(env)
    if log_file:
        env = Monitor(env, filename=log_file)
    else:
        env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


def make_vectorized_lunar_lander(
    n_envs: int = 1,
    seed: int = 42,
    render_mode: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> DummyVecEnv:
    """Create a DummyVecEnv for stable training with Stable-Baselines3."""
    def _init(idx: int) -> Callable[[], gym.Env]:
        def _thunk() -> gym.Env:
            log_file = f"{log_dir}/env_{idx}.monitor.csv" if log_dir else None
            return make_lunar_lander_env(
                env_id="LunarLander-v3",
                seed=seed + idx,
                render_mode=render_mode,
                log_file=log_file,
            )
        return _thunk

    return DummyVecEnv([_init(i) for i in range(n_envs)])
