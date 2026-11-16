"""
Unit Tests: Autonomous Racing Agents (PPO and DQN)
==================================================
"""

import os
import pytest
import numpy as np
from environments.car_racing_env import make_car_racing_env
from environments.discrete_car_racing_env import make_discrete_car_racing_env
from agents.agent_factory import create_racing_agent, load_racing_agent
from agents.ppo_agent import PPORacingAgent
from agents.dqn_agent import DQNRacingAgent


def test_ppo_agent_initialization_and_prediction():
    """Verify PPO racing agent builds with CNN policy and predicts continuous actions."""
    env = make_car_racing_env(continuous=True, seed=42)
    agent = create_racing_agent("ppo", env=env, seed=42)

    summary = agent.get_model_summary()
    assert "PPO" in summary["algorithm"]
    assert summary["total_parameters"] > 0
    assert summary["device"] == "cpu"

    obs, _ = env.reset(seed=42)
    action, _ = agent.predict(obs, deterministic=True)
    assert action.shape == (3,)
    assert -1.0 <= action[0] <= 1.0  # steering
    assert 0.0 <= action[1] <= 1.0   # gas
    assert 0.0 <= action[2] <= 1.0   # brake
    env.close()


def test_dqn_agent_initialization_and_prediction():
    """Verify DQN racing agent builds with CNN policy and predicts discrete actions."""
    env = make_discrete_car_racing_env(seed=42)
    agent = create_racing_agent("dqn", env=env, seed=42)

    summary = agent.get_model_summary()
    assert "DQN" in summary["algorithm"]
    assert summary["total_parameters"] > 0

    obs, _ = env.reset(seed=42)
    action, _ = agent.predict(obs, deterministic=True)
    assert 0 <= int(action) < 7
    env.close()
