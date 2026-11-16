"""
Environment Wrappers for Autonomous Car Racing (CarRacing-v3)
============================================================
Provides telemetry extraction, grayscale downsampling, frame skipping,
and action repetition wrappers.
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CarRacingTelemetryWrapper(gym.Wrapper):
    """
    Gymnasium wrapper that monitors CarRacing telemetry:
    - Speed & steering angle
    - Gas & brake activations
    - Consecutive off-track penalties
    - Estimated track tiles visited & lap completion
    """

    def __init__(self, env: gym.Env, off_track_limit: int = 50):
        super().__init__(env)
        self.off_track_limit = off_track_limit
        self.episode_steps = 0
        self.cumulative_reward = 0.0
        self.off_track_counter = 0
        self.positive_reward_tiles = 0
        self.last_action = [0.0, 0.0, 0.0]

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.episode_steps = 0
        self.cumulative_reward = 0.0
        self.off_track_counter = 0
        self.positive_reward_tiles = 0
        self.last_action = [0.0, 0.0, 0.0]

        obs, info = self.env.reset(**kwargs)
        info["telemetry"] = self._extract_telemetry(obs, reward=0.0)
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Handle continuous or converted discrete actions
        if isinstance(action, (int, np.integer)):
            raw_act = getattr(self.env, "discrete_to_continuous", lambda a: [0.0, 0.0, 0.0])(action)
        elif isinstance(action, (list, tuple, np.ndarray)):
            raw_act = [float(action[0]), float(action[1]), float(action[2])]
        else:
            raw_act = [0.0, 0.0, 0.0]

        self.last_action = raw_act
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.episode_steps += 1
        self.cumulative_reward += reward

        # Track tiles visited (each tile gives ~+1000/N reward)
        if reward > 0.1:
            self.positive_reward_tiles += 1
            self.off_track_counter = 0
        elif reward < 0:
            self.off_track_counter += 1

        # Check early termination if car gets stuck in the grass/off-track for too long
        is_off_track = self.off_track_counter >= self.off_track_limit
        if is_off_track and self.episode_steps > 100:
            truncated = True

        telemetry = self._extract_telemetry(obs, reward)
        telemetry["is_off_track"] = is_off_track
        telemetry["tiles_visited"] = self.positive_reward_tiles

        # Estimated track progress (CarRacing tracks typically consist of ~280-320 tiles)
        telemetry["track_progress_percent"] = min(100.0, round((self.positive_reward_tiles / 300.0) * 100.0, 1))
        telemetry["is_lap_completed"] = bool(self.cumulative_reward >= 800.0 or telemetry["track_progress_percent"] >= 95.0)

        info["telemetry"] = telemetry
        return obs, reward, terminated, truncated, info

    def _extract_telemetry(self, obs: np.ndarray, reward: float) -> Dict[str, Any]:
        prog = min(100.0, round((self.positive_reward_tiles / 300.0) * 100.0, 1))
        return {
            "episode_steps": self.episode_steps,
            "cumulative_reward": round(self.cumulative_reward, 2),
            "step_reward": round(reward, 3),
            "steering": round(self.last_action[0], 2),
            "gas": round(self.last_action[1], 2),
            "brake": round(self.last_action[2], 2),
            "off_track_steps": self.off_track_counter,
            "off_track_count": self.off_track_counter,
            "tiles_visited": self.positive_reward_tiles,
            "track_progress": prog,
            "track_progress_percent": prog,
            "is_off_track": self.off_track_counter >= self.off_track_limit,
            "is_lap_completed": bool(self.cumulative_reward >= 800.0 or prog >= 95.0),
        }


class GrayscaleResizeObservation(gym.ObservationWrapper):
    """
    Downsamples CarRacing visual observations to 84x84 grayscale.
    Drastically reduces memory footprint and CPU computational overhead while
    preserving high-contrast track boundaries and road edges.
    """

    def __init__(self, env: gym.Env, shape: Tuple[int, int] = (84, 84)):
        super().__init__(env)
        self.shape = shape
        # Observation space is uint8 with shape (84, 84, 1)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shape[0], shape[1], 1),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # Check if already 2D or 3D
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif len(obs.shape) == 2:
            gray = obs
        else:
            gray = obs[:, :, 0]

        resized = cv2.resize(gray, self.shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)


class FrameSkipWrapper(gym.Wrapper):
    """
    Repeats the agent's action for `skip` consecutive frames.
    Improves steering stability, accelerates exploration, and doubles training throughput.
    """

    def __init__(self, env: gym.Env, skip: int = 2):
        super().__init__(env)
        self._skip = max(1, skip)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info

