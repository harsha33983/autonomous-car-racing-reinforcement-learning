"""
Continuous CarRacing-v3 Environment Pipeline & Wrappers (PPO Native)
===================================================================
Provides environment factory, state space inspection, and wrappers
for continuous autonomous racing control.
"""

from typing import Any, Dict, Optional, Tuple, Callable
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environments.wrappers import (
    CarRacingTelemetryWrapper,
    GrayscaleResizeObservation,
    FrameSkipWrapper,
)


ACTION_BOUNDS = {
    "steering": (-1.0, 1.0),
    "gas": (0.0, 1.0),
    "brake": (0.0, 1.0),
}


def make_car_racing_env(
    env_id: str = "CarRacing-v3",
    continuous: bool = True,
    seed: Optional[int] = 42,
    render_mode: Optional[str] = None,
    log_file: Optional[str] = None,
    frame_skip: int = 2,
    grayscale: bool = True,
    target_shape: Tuple[int, int] = (84, 84),
) -> gym.Env:
    """
    Factory function for continuous CarRacing-v3 environment.
    """
    env_kwargs = {"continuous": continuous}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **env_kwargs)

    # Frame skipping
    if frame_skip > 1:
        env = FrameSkipWrapper(env, skip=frame_skip)

    # Telemetry wrapper
    env = CarRacingTelemetryWrapper(env)

    # Visual downsampling to 84x84x1
    if grayscale:
        env = GrayscaleResizeObservation(env, shape=target_shape)

    # Monitor wrapper for episodic statistics
    if log_file:
        env = Monitor(env, filename=log_file)
    else:
        env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


def make_vectorized_car_racing(
    n_envs: int = 1,
    seed: int = 42,
    render_mode: Optional[str] = None,
    log_dir: Optional[str] = None,
    frame_skip: int = 2,
    grayscale: bool = True,
) -> DummyVecEnv:
    """Creates DummyVecEnv for stable PPO training."""
    def _init(idx: int) -> Callable[[], gym.Env]:
        def _thunk() -> gym.Env:
            log_file = f"{log_dir}/car_racing_{idx}.monitor.csv" if log_dir else None
            return make_car_racing_env(
                seed=seed + idx,
                render_mode=render_mode,
                log_file=log_file,
                frame_skip=frame_skip,
                grayscale=grayscale,
            )
        return _thunk

    return DummyVecEnv([_init(i) for i in range(n_envs)])

