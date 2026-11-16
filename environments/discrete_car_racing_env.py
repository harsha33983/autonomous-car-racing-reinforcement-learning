"""
Discrete Action Space Wrapper for CarRacing-v3 (DQN Compatible)
==============================================================
Maps discrete steering/throttle/brake selections to native continuous
CarRacing-v3 actions [steering, gas, brake].
"""

from typing import Any, Dict, List, Optional, Tuple, Callable
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environments.wrappers import (
    CarRacingTelemetryWrapper,
    GrayscaleResizeObservation,
    FrameSkipWrapper,
)


# 7 Discretized Driving Actions
DISCRETE_ACTIONS: Dict[int, List[float]] = {
    0: [0.0, 0.7, 0.0],    # Straight + Accelerate
    1: [-0.8, 0.4, 0.0],   # Steer Left + Accelerate
    2: [0.8, 0.4, 0.0],    # Steer Right + Accelerate
    3: [-0.8, 0.0, 0.6],   # Steer Left + Brake
    4: [0.8, 0.0, 0.6],    # Steer Right + Brake
    5: [0.0, 0.0, 0.6],    # Straight + Brake
    6: [0.0, 0.0, 0.0],    # Straight + Coast (Drifting/Idle)
}

DISCRETE_ACTION_NAMES: Dict[int, str] = {
    0: "Straight + Accelerate",
    1: "Steer Left + Accelerate",
    2: "Steer Right + Accelerate",
    3: "Steer Left + Brake",
    4: "Steer Right + Brake",
    5: "Straight + Brake",
    6: "Straight + Coast",
}
ACTION_NAMES = DISCRETE_ACTION_NAMES


class DiscreteCarRacingWrapper(gym.ActionWrapper):
    """
    ActionWrapper converting continuous Box(3,) action space
    into a discrete Discrete(7) action space for Deep Q-Networks.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

    def action(self, act: int) -> np.ndarray:
        cont_act = DISCRETE_ACTIONS.get(int(act), [0.0, 0.0, 0.0])
        return np.array(cont_act, dtype=np.float32)

    @staticmethod
    def discrete_to_continuous(act: int) -> List[float]:
        return DISCRETE_ACTIONS.get(int(act), [0.0, 0.0, 0.0])


def make_discrete_car_racing_env(
    env_id: str = "CarRacing-v3",
    seed: Optional[int] = 42,
    render_mode: Optional[str] = None,
    log_file: Optional[str] = None,
    frame_skip: int = 2,
    grayscale: bool = True,
    target_shape: Tuple[int, int] = (84, 84),
) -> gym.Env:
    """
    Factory function for a fully instrumented discrete CarRacing environment.
    """
    env_kwargs = {"continuous": True}
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    env = gym.make(env_id, **env_kwargs)

    # Wrap frame skip for control frequency
    if frame_skip > 1:
        env = FrameSkipWrapper(env, skip=frame_skip)

    # Wrap discrete action mapping
    env = DiscreteCarRacingWrapper(env)

    # Wrap telemetry
    env = CarRacingTelemetryWrapper(env)

    # Wrap visual preprocessing
    if grayscale:
        env = GrayscaleResizeObservation(env, shape=target_shape)

    # Monitor wrapper for episodic reward tracking
    if log_file:
        env = Monitor(env, filename=log_file)
    else:
        env = Monitor(env)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


def make_vectorized_discrete_car_racing(
    n_envs: int = 1,
    seed: int = 42,
    render_mode: Optional[str] = None,
    log_dir: Optional[str] = None,
    frame_skip: int = 2,
    grayscale: bool = True,
) -> DummyVecEnv:
    """Creates a vectorized DummyVecEnv for discrete DQN training."""
    def _init(idx: int) -> Callable[[], gym.Env]:
        def _thunk() -> gym.Env:
            log_file = f"{log_dir}/discrete_racing_{idx}.monitor.csv" if log_dir else None
            return make_discrete_car_racing_env(
                seed=seed + idx,
                render_mode=render_mode,
                log_file=log_file,
                frame_skip=frame_skip,
                grayscale=grayscale,
            )
        return _thunk

    return DummyVecEnv([_init(i) for i in range(n_envs)])
