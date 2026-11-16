"""
Dashboard: Autonomous Racing Simulation & Live Telemetry
========================================================
Interactive simulation viewer allowing users to run autonomous racing agents,
inspect telemetry in real time, and view replay animations.
"""

import os
import glob
import time
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from environments.car_racing_env import make_car_racing_env
from environments.discrete_car_racing_env import make_discrete_car_racing_env, DISCRETE_ACTIONS
from agents.agent_factory import load_racing_agent
from visualization.gameplay import overlay_racing_telemetry


def render_simulation():
    st.header("🏎️ Autonomous Racing Simulation & Telemetry")

    st.markdown(
        """
        Execute trained neural network policies live inside the **Gymnasium CarRacing-v3** environment.
        The AI agent perceives top-down visual camera frames and autonomously controls steering, throttle,
        and braking in real time without human guidance or pre-programmed waypoints.
        """
    )

    # Simulation Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.5, 1])

    with ctrl_col1:
        algo_choice = st.selectbox(
            "Select Racing Agent:",
            ["PPO (Continuous Control)", "DQN (Discrete Maneuvers)", "Random Action Baseline"],
        )

    with ctrl_col2:
        algo_key = "ppo" if "PPO" in algo_choice else "dqn" if "DQN" in algo_choice else "random"
        model_paths = []
        if algo_key in ("ppo", "dqn"):
            model_paths = glob.glob(f"models/{algo_key}/**/*.zip", recursive=True) + glob.glob(f"models/{algo_key}/*.zip")
            model_paths = list(sorted(set(model_paths)))

        if model_paths:
            selected_model = st.selectbox("Select Model Weight Checkpoint:", model_paths)
        else:
            selected_model = None
            if algo_key != "random":
                st.warning(f"No trained {algo_key.upper()} models found on disk.")

    with ctrl_col3:
        sim_seed = st.number_input("Track Generation Seed:", value=1042, step=1)

    st.markdown("---")

    # Mode Selector
    sim_tab1, sim_tab2 = st.tabs(["High-Definition Recorded Replays (GIF)", "Run Interactive Live Simulation"])

    with sim_tab1:
        st.subheader("Autonomous Replay Animations")
        gif_candidates = {
            "Continuous PPO Autonomous Racing": "results/videos/ppo_carracing.gif",
            "Discrete DQN Autonomous Racing": "results/videos/dqn_carracing.gif",
        }
        avail_gifs = {k: v for k, v in gif_candidates.items() if os.path.exists(v)}

        if avail_gifs:
            selected_gif_title = st.selectbox("Select Recorded Replay:", list(avail_gifs.keys()))
            gif_path = avail_gifs[selected_gif_title]
            st.image(gif_path, caption=f"Autonomous Racing: {selected_gif_title}", use_container_width=True)
        else:
            st.info("Replay animations are being recorded or will appear here once generated.")

    with sim_tab2:
        st.subheader("Live Autonomous Track Execution")
        max_sim_steps = st.slider("Max Simulation Steps:", min_value=100, max_value=600, value=250, step=50)
        start_btn = st.button("Launch Autonomous Track Run", type="primary")

        if start_btn:
            if algo_key != "random" and (not selected_model or not os.path.exists(selected_model)):
                st.error("Please train a model or select a valid model checkpoint first.")
                return

            try:
                with st.spinner("Initializing CarRacing-v3 physics and loading vision neural network..."):
                    if algo_key == "dqn":
                        env = make_discrete_car_racing_env(seed=sim_seed, render_mode="rgb_array")
                    else:
                        env = make_car_racing_env(continuous=True, seed=sim_seed, render_mode="rgb_array")

                    if algo_key != "random":
                        agent = load_racing_agent(algo_key, selected_model, env=env)
                    else:
                        agent = None
            except Exception as e:
                st.error(f"Failed to initialize simulation environment or load model weights: {e}")
                return

            frame_placeholder = st.empty()
            telemetry_cols = st.columns(5)
            metric_score = telemetry_cols[0].empty()
            metric_step = telemetry_cols[1].empty()
            metric_prog = telemetry_cols[2].empty()
            metric_steer = telemetry_cols[3].empty()
            metric_status = telemetry_cols[4].empty()

            obs, info = env.reset(seed=sim_seed)
            done = False
            truncated = False
            total_reward = 0.0
            step = 0

            try:
                # Step loop
                while not (done or truncated) and step < max_sim_steps:
                    if agent is not None:
                        action, _ = agent.predict(obs, deterministic=True)
                    else:
                        action = env.action_space.sample()

                    obs, reward, done, truncated, info = env.step(action)
                    total_reward += float(reward)
                    step += 1

                    # Render frame every 2 steps
                    if step % 2 == 0 or done or truncated:
                        frame = env.render()
                        if frame is not None:
                            annotated = overlay_racing_telemetry(
                                frame, algo_key.upper(), step, action, total_reward, info.get("telemetry")
                            )
                            frame_placeholder.image(annotated, use_container_width=True)

                        telemetry = info.get("telemetry", {})
                        metric_score.metric("Score", f"{total_reward:+.1f}")
                        metric_step.metric("Step", f"{step}")
                        metric_prog.metric("Track Progress", f"{telemetry.get('track_progress', 0.0):.1f}%")

                        if isinstance(action, (int, np.integer)):
                            steer_val = DISCRETE_ACTIONS.get(int(action), [0, 0, 0])[0]
                        else:
                            steer_val = float(action[0])
                        metric_steer.metric("Steering", f"{steer_val:+.2f}")

                        off_track = telemetry.get("off_track_count", 0)
                        metric_status.metric("Track Status", "OFF-TRACK" if off_track > 0 else "ON-TRACK")
            finally:
                env.close()

            st.success(f"Autonomous episode completed! Final Score: {total_reward:+.2f} over {step} steps.")
