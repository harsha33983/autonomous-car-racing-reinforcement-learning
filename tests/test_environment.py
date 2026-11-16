"""
Unit Tests: Gymnasium Environments & Telemetry Wrappers
=======================================================
"""

import pytest
import numpy as np
from environments.lunar_lander_env import (
    make_lunar_lander_env,
    ACTION_MEANINGS as LUNAR_ACTIONS,
    OBSERVATION_FEATURES as LUNAR_OBS_FEATURES,
)
from environments.cartpole_env import (
    make_cartpole_env,
    ACTION_MEANINGS as CART_ACTIONS,
)


def test_lunar_lander_initialization():
    """Verify LunarLander-v3 initializes with expected spaces."""
    env = make_lunar_lander_env(seed=42)
    assert env.observation_space.shape == (8,)
    assert env.action_space.n == 4
    assert len(LUNAR_ACTIONS) == 4
    assert len(LUNAR_OBS_FEATURES) == 8

    obs, info = env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (8,)
    assert "telemetry" in info
    assert "x_position" in info["telemetry"]
    assert "y_position" in info["telemetry"]
    env.close()


def test_lunar_lander_stepping_and_telemetry():
    """Verify stepping produces valid rewards, terminations, and telemetry."""
    env = make_lunar_lander_env(seed=42)
    obs, info = env.reset(seed=42)

    # Step through all 4 discrete actions
    for action in range(4):
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(next_obs, np.ndarray)
        assert next_obs.shape == (8,)
        assert isinstance(reward, (float, np.floating, int))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "telemetry" in info
        assert "cumulative_reward" in info["telemetry"]
        assert info["telemetry"]["episode_steps"] == action + 1

    env.close()


def test_cartpole_initialization_and_stepping():
    """Verify CartPole-v1 baseline environment works correctly."""
    env = make_cartpole_env(seed=42)
    assert env.observation_space.shape == (4,)
    assert env.action_space.n == 2

    obs, info = env.reset(seed=42)
    assert obs.shape == (4,)
    assert "telemetry" in info

    next_obs, reward, terminated, truncated, info = env.step(1)
    assert next_obs.shape == (4,)
    assert reward == 1.0
    assert "cart_position" in info["telemetry"]
    env.close()
