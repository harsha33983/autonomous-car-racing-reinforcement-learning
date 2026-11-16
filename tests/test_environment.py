"""
Unit Tests: Gymnasium CarRacing-v3 Environments & Telemetry Wrappers
===================================================================
"""

import pytest
import numpy as np
from environments.car_racing_env import make_car_racing_env
from environments.discrete_car_racing_env import make_discrete_car_racing_env, DISCRETE_ACTIONS, ACTION_NAMES


def test_continuous_car_racing_initialization():
    """Verify continuous CarRacing-v3 initializes with expected spaces."""
    env = make_car_racing_env(continuous=True, seed=42, grayscale=True, frame_skip=2)
    assert env.observation_space.shape == (84, 84, 1)
    assert env.action_space.shape == (3,)

    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (84, 84, 1)
    assert "telemetry" in info
    assert "track_progress" in info["telemetry"]
    env.close()


def test_discrete_car_racing_initialization():
    """Verify discrete CarRacing-v3 initializes with 7 action choices."""
    env = make_discrete_car_racing_env(seed=42, grayscale=True, frame_skip=2)
    assert env.observation_space.shape == (84, 84, 1)
    assert env.action_space.n == 7
    assert len(DISCRETE_ACTIONS) == 7
    assert len(ACTION_NAMES) == 7

    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (84, 84, 1)
    assert "telemetry" in info
    env.close()


def test_car_racing_stepping_and_telemetry():
    """Verify stepping produces valid rewards, terminations, and telemetry."""
    env = make_discrete_car_racing_env(seed=42, grayscale=True, frame_skip=2)
    obs, info = env.reset(seed=42)

    # Step through all 7 discrete actions
    for action in range(7):
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == (84, 84, 1)
        assert isinstance(reward, (float, np.floating, int))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert "telemetry" in info
        assert "track_progress" in info["telemetry"]
        assert "tiles_visited" in info["telemetry"]
        assert "off_track_count" in info["telemetry"]

    env.close()
