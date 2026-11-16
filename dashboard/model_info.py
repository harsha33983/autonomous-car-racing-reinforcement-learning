"""
Dashboard: Model Information & Checkpoint Registry
==================================================
Deep architectural inspection, layer parameters, hyperparameters,
and checkpoint manifest browser.
"""

import os
import json
import glob
import streamlit as st
import yaml

from training.checkpoint_manager import CheckpointManager


def render_model_info():
    st.header("Model Architecture & Checkpoint Registry")

    cfg_path = "config/config.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    tab_dqn, tab_ppo = st.tabs(["Deep Q-Network (DQN)", "Proximal Policy Optimization (PPO)"])

    with tab_dqn:
        st.subheader("DQN Neural Network Architecture")
        dqn_cfg = config["algorithms"]["dqn"]["optimized"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                **Network Topology**: Multi-Layer Perceptron (MLP)
                - **Observation Space Input**: 8 continuous features
                - **Hidden Layers**: [256, 256] with ReLU non-linearities
                - **Action Space Output**: 4 discrete action Q-values
                - **Total Parameters**: ~68,356 weights and biases
                - **Device**: CPU Optimized
                """
            )
        with col2:
            st.markdown(
                f"""
                **Hyperparameters (Optimized Config)**:
                - `learning_rate`: `{dqn_cfg['learning_rate']}`
                - `batch_size`: `{dqn_cfg['batch_size']}`
                - `buffer_size`: `{dqn_cfg['buffer_size']:,}`
                - `gamma` (Discount factor): `{dqn_cfg['gamma']}`
                - `tau` (Soft target update): `{dqn_cfg['tau']}`
                - `target_update_interval`: `{dqn_cfg['target_update_interval']}` steps
                - `exploration_final_eps`: `{dqn_cfg['exploration_final_eps']}`
                - `exploration_fraction`: `{dqn_cfg['exploration_fraction']}`
                """
            )

        # Checkpoint list
        st.markdown("---")
        st.subheader("DQN Model Checkpoints on Disk")
        mgr_dqn = CheckpointManager(checkpoint_dir="models/dqn/optimized", prefix="dqn")
        ckpts = mgr_dqn.list_checkpoints()
        final_model = "models/dqn/dqn_lunarlander_final.zip"

        if os.path.exists(final_model):
            st.success(f"Final Model: `{final_model}` ({os.path.getsize(final_model) / 1024:.1f} KB)")
        if ckpts:
            st.write(f"Found {len(ckpts)} periodic training checkpoints:")
            st.table([
                {
                    "Timestep": c["timestep"],
                    "Filename": c["filename"],
                    "Size (KB)": round(c["size_bytes"] / 1024, 1),
                }
                for c in ckpts
            ])
        else:
            st.info("No periodic checkpoints found in `models/dqn/optimized`.")

    with tab_ppo:
        st.subheader("PPO Actor-Critic Architecture")
        ppo_cfg = config["algorithms"]["ppo"]["optimized"]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                **Network Topology**: Actor-Critic MLP
                - **Observation Space Input**: 8 continuous features
                - **Policy Network (Actor $\\pi$)**: [256, 256] -> 4 action logits
                - **Value Network (Critic $V$)**: [256, 256] -> 1 scalar baseline
                - **Total Parameters**: ~137,221 weights and biases
                - **Device**: CPU Optimized
                """
            )
        with col2:
            st.markdown(
                f"""
                **Hyperparameters (Optimized Config)**:
                - `learning_rate`: `{ppo_cfg['learning_rate']}`
                - `n_steps`: `{ppo_cfg['n_steps']}`
                - `batch_size`: `{ppo_cfg['batch_size']}`
                - `n_epochs`: `{ppo_cfg['n_epochs']}`
                - `gamma` (Discount factor): `{ppo_cfg['gamma']}`
                - `gae_lambda`: `{ppo_cfg['gae_lambda']}`
                - `clip_range`: `{ppo_cfg['clip_range']}`
                - `ent_coef` (Entropy bonus): `{ppo_cfg['ent_coef']}`
                - `vf_coef`: `{ppo_cfg['vf_coef']}`
                """
            )

        # Checkpoint list
        st.markdown("---")
        st.subheader("PPO Model Checkpoints on Disk")
        mgr_ppo = CheckpointManager(checkpoint_dir="models/ppo/optimized", prefix="ppo")
        ckpts_ppo = mgr_ppo.list_checkpoints()
        final_ppo = "models/ppo/ppo_lunarlander_final.zip"

        if os.path.exists(final_ppo):
            st.success(f"Final Model: `{final_ppo}` ({os.path.getsize(final_ppo) / 1024:.1f} KB)")
        if ckpts_ppo:
            st.write(f"Found {len(ckpts_ppo)} periodic training checkpoints:")
            st.table([
                {
                    "Timestep": c["timestep"],
                    "Filename": c["filename"],
                    "Size (KB)": round(c["size_bytes"] / 1024, 1),
                }
                for c in ckpts_ppo
            ])
        else:
            st.info("No periodic checkpoints found in `models/ppo/optimized`.")
