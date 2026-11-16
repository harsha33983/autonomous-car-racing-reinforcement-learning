"""
Dashboard: Model Information & Checkpoint Registry - Car Racing AI
==================================================================
Deep architectural inspection, CNN feature extractor parameters,
hyperparameters, and checkpoint manifest browser.
"""

import os
import json
import glob
import streamlit as st
import yaml

from training.checkpoint_manager import RacingCheckpointManager


def render_model_info():
    st.header("🧠 Neural Network Architectures & Checkpoint Registry")

    cfg_path = "config/config.yaml"
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {"algorithms": {"ppo": {}, "dqn": {}}}

    tab_ppo, tab_dqn = st.tabs(["Continuous PPO (Primary)", "Discrete DQN (Comparative)"])

    with tab_ppo:
        st.subheader("PPO Continuous Actor-Critic CNN Architecture")
        ppo_cfg = config.get("algorithms", {}).get("ppo", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                **Perception & Policy Backbone**: Nature CNN
                - **Observation Input**: $(84, 84, 1)$ Grayscale birds-eye track camera
                - **Feature Extractor**:
                  - Conv 1: 32 filters, $8 \\times 8$ kernel, stride 4, ReLU
                  - Conv 2: 64 filters, $4 \\times 4$ kernel, stride 2, ReLU
                  - Conv 3: 64 filters, $3 \\times 3$ kernel, stride 1, ReLU
                  - Dense: 512 units, ReLU
                - **Policy Head (Actor $\\pi$)**: Linear(512, 3) $\\rightarrow$ Steering, Throttle, Brake
                - **Value Head (Critic $V$)**: Linear(512, 1) $\\rightarrow$ Baseline state value
                - **Action Distribution**: Diagonal Gaussian with learned log standard deviations
                - **Execution Runtime**: PyTorch 2.x on CPU
                """
            )
        with col2:
            st.markdown(
                f"""
                **PPO Hyperparameters**:
                - `learning_rate`: `{ppo_cfg.get('learning_rate', 0.0003)}`
                - `n_steps`: `{ppo_cfg.get('n_steps', 512)}`
                - `batch_size`: `{ppo_cfg.get('batch_size', 64)}`
                - `n_epochs`: `{ppo_cfg.get('n_epochs', 10)}`
                - `gamma`: `{ppo_cfg.get('gamma', 0.99)}`
                - `gae_lambda`: `{ppo_cfg.get('gae_lambda', 0.95)}`
                - `clip_range`: `{ppo_cfg.get('clip_range', 0.2)}`
                - `ent_coef`: `{ppo_cfg.get('ent_coef', 0.01)}`
                """
            )

        # Checkpoints
        st.markdown("---")
        st.subheader("PPO Checkpoints on Disk")
        mgr_ppo = RacingCheckpointManager(checkpoint_dir="models/ppo", prefix="ppo")
        ckpts = mgr_ppo.list_checkpoints()
        final_model = "models/ppo/ppo_carracing_final.zip"

        if os.path.exists(final_model):
            st.success(f"Final Model Weights: `{final_model}` ({os.path.getsize(final_model) / (1024 * 1024):.2f} MB)")
        if ckpts:
            st.write(f"Found {len(ckpts)} training checkpoints:")
            st.table([
                {
                    "Timestep": c["timestep"],
                    "Filename": c["filename"],
                    "Size (MB)": round(c["size_bytes"] / (1024 * 1024), 2),
                }
                for c in ckpts
            ])

    with tab_dqn:
        st.subheader("DQN Discrete Q-Network CNN Architecture")
        dqn_cfg = config.get("algorithms", {}).get("dqn", {})

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                **Q-Network Backbone**: Nature CNN
                - **Observation Input**: $(84, 84, 1)$ Grayscale birds-eye track camera
                - **Feature Extractor**:
                  - Conv 1: 32 filters, $8 \\times 8$ kernel, stride 4, ReLU
                  - Conv 2: 64 filters, $4 \\times 4$ kernel, stride 2, ReLU
                  - Conv 3: 64 filters, $3 \\times 3$ kernel, stride 1, ReLU
                  - Dense: 512 units, ReLU
                - **Q-Value Head**: Linear(512, 7) $\\rightarrow$ Discrete action values
                - **Replay Buffer**: Circular transition buffer storing up to 15,000 frames
                - **Target Network**: Periodic copy updated every 1,000 steps
                """
            )
        with col2:
            st.markdown(
                f"""
                **DQN Hyperparameters**:
                - `learning_rate`: `{dqn_cfg.get('learning_rate', 0.00025)}`
                - `batch_size`: `{dqn_cfg.get('batch_size', 64)}`
                - `buffer_size`: `{dqn_cfg.get('buffer_size', 15000):,}`
                - `gamma`: `{dqn_cfg.get('gamma', 0.99)}`
                - `train_freq`: `{dqn_cfg.get('train_freq', 4)}`
                - `exploration_initial_eps`: `{dqn_cfg.get('exploration_initial_eps', 1.0)}`
                - `exploration_final_eps`: `{dqn_cfg.get('exploration_final_eps', 0.05)}`
                """
            )

        # Checkpoints
        st.markdown("---")
        st.subheader("DQN Checkpoints on Disk")
        mgr_dqn = RacingCheckpointManager(checkpoint_dir="models/dqn", prefix="dqn")
        ckpts_dqn = mgr_dqn.list_checkpoints()
        final_dqn = "models/dqn/dqn_carracing_final.zip"

        if os.path.exists(final_dqn):
            st.success(f"Final Model Weights: `{final_dqn}` ({os.path.getsize(final_dqn) / (1024 * 1024):.2f} MB)")
        if ckpts_dqn:
            st.write(f"Found {len(ckpts_dqn)} training checkpoints:")
            st.table([
                {
                    "Timestep": c["timestep"],
                    "Filename": c["filename"],
                    "Size (MB)": round(c["size_bytes"] / (1024 * 1024), 2),
                }
                for c in ckpts_dqn
            ])
