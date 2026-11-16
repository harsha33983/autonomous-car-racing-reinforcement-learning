"""
Dashboard: Autonomous Game Simulation
====================================
Interactive simulation viewer allowing users to run autonomous agents,
inspect telemetry in real-time, and view replay animations.
"""

import os
import glob
import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from environments.lunar_lander_env import make_lunar_lander_env, ACTION_MEANINGS
from agents.agent_factory import load_agent
from visualization.gameplay import overlay_telemetry


def render_simulation():
    st.header("Autonomous Game Simulation & Telemetry")

    st.markdown(
        """
        Execute trained neural network policies live inside the **Gymnasium LunarLander-v3** environment.
        The agent perceives the 8-dimensional continuous state vector and autonomously selects actions
        at 30 frames per second without any human control or intervention.
        """
    )

    # Simulation Controls
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1.5, 1])

    with ctrl_col1:
        algo_choice = st.selectbox("Select Agent Policy:", ["PPO (Proximal Policy Opt)", "DQN (Deep Q-Network)", "Random Baseline"])

    with ctrl_col2:
        algo_key = "ppo" if "PPO" in algo_choice else "dqn" if "DQN" in algo_choice else "random"
        model_paths = []
        if algo_key in ("ppo", "dqn"):
            model_paths = glob.glob(f"models/{algo_key}/**/*.zip", recursive=True)
            if not model_paths:
                model_paths = glob.glob(f"models/{algo_key}/*.zip")

        if model_paths:
            selected_model = st.selectbox("Select Model Weight Checkpoint:", model_paths)
        else:
            selected_model = None
            if algo_key != "random":
                st.warning(f"No trained {algo_key.upper()} models found on disk.")

    with ctrl_col3:
        sim_seed = st.number_input("Environment Seed:", value=1042, step=1)

    st.markdown("---")

    # Mode Selector
    sim_tab1, sim_tab2 = st.tabs(["High-Definition Recorded Gameplay (GIF)", "Run Interactive Live Episode"])

    with sim_tab1:
        st.subheader("Recorded Autonomous Missions")
        gif_candidates = {
            "PPO Flight Mission": "results/videos/ppo_gameplay.gif",
            "DQN Flight Mission": "results/videos/dqn_gameplay.gif",
        }
        avail_gifs = {k: v for k, v in gif_candidates.items() if os.path.exists(v)}

        if avail_gifs:
            selected_gif_title = st.selectbox("Select Recorded Mission Video:", list(avail_gifs.keys()))
            gif_path = avail_gifs[selected_gif_title]
            st.image(gif_path, caption=f"Autonomous Agent Flight: {selected_gif_title}", use_container_width=True)
        else:
            st.info("Gameplay animation videos not yet generated. They will appear here once recorded.")

    with sim_tab2:
        st.subheader("Live Autonomous Episode Execution")

        start_btn = st.button("Launch Autonomous Flight Mission", type="primary")

        if start_btn:
            if algo_key != "random" and (not selected_model or not os.path.exists(selected_model)):
                st.error("Please train a model or select a valid model checkpoint first.")
                return

            with st.spinner("Initializing physics engine and loading neural network policy..."):
                env = make_lunar_lander_env(seed=sim_seed, render_mode="rgb_array")
                if algo_key != "random":
                    agent = load_agent(algo_key, selected_model, env=env)
                else:
                    agent = None

            frame_placeholder = st.empty()
            telemetry_cols = st.columns(5)
            metric_score = telemetry_cols[0].empty()
            metric_step = telemetry_cols[1].empty()
            metric_vx = telemetry_cols[2].empty()
            metric_vy = telemetry_cols[3].empty()
            metric_legs = telemetry_cols[4].empty()

            progress_bar = st.progress(0)

            obs, info = env.reset(seed=sim_seed)
            done = False
            truncated = False
            step = 0
            total_reward = 0.0
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

            max_steps = 800
            rendered_frames = []

            while not (done or truncated) and step < max_steps:
                if agent is not None:
                    action, _ = agent.predict(obs, deterministic=True)
                else:
                    action = int(env.action_space.sample())

                action_counts[action] += 1
                obs, reward, done, truncated, info = env.step(action)
                step += 1
                total_reward += reward

                telemetry = info.get("telemetry", {})

                # Render frame every 2 steps for responsive streaming
                if step % 2 == 0 or done or truncated:
                    raw_frame = env.render()
                    if raw_frame is not None:
                        annotated = overlay_telemetry(
                            raw_frame, algo_key, 1, step, action, reward, total_reward, telemetry
                        )
                        frame_placeholder.image(annotated, use_container_width=True)

                    # Update telemetry cards
                    rew_delta = "Positive Return" if total_reward > 0 else "Descent Cost"
                    metric_score.metric("Total Score", f"{total_reward:+.1f}", delta=rew_delta)
                    metric_step.metric("Flight Step", f"{step} / {max_steps}")
                    metric_vx.metric("H-Speed (Vx)", f"{telemetry.get('linear_velocity_x', 0):+.2f} m/s")
                    metric_vy.metric("V-Speed (Vy)", f"{telemetry.get('linear_velocity_y', 0):+.2f} m/s")
                    legs_str = f"L:{int(telemetry.get('left_leg_contact', 0))} R:{int(telemetry.get('right_leg_contact', 0))}"
                    metric_legs.metric("Ground Legs", legs_str)

                    progress_bar.progress(min(step / max_steps, 1.0))
                    time.sleep(0.01)

            env.close()

            # Outcome message
            st.markdown("---")
            if total_reward >= 200.0:
                st.success(f"**MISSION ACCOMPLISHED!** Autonomous agent achieved a perfect soft landing with a score of **{total_reward:+.2f}** in {step} steps!")
            elif total_reward > 0:
                st.info(f"Flight terminated with positive score **{total_reward:+.2f}** in {step} steps.")
            else:
                st.warning(f"Flight terminated with landing penalty / crash (**{total_reward:+.2f}** in {step} steps).")

            # Action Distribution Chart
            st.subheader("Autonomous Thruster Activation Profile")
            act_df = pd.DataFrame([
                {"Thruster Action": ACTION_MEANINGS[a], "Activations": count, "Percentage": f"{count/max(step, 1)*100:.1f}%"}
                for a, count in action_counts.items()
            ])
            fig_act = px.bar(
                act_df,
                x="Thruster Action",
                y="Activations",
                color="Thruster Action",
                text="Percentage",
                color_discrete_sequence=["#94A3B8", "#F59E0B", "#EF4444", "#3B82F6"],
            )
            fig_act.update_layout(template="plotly_white", height=320, showlegend=False)
            st.plotly_chart(fig_act, use_container_width=True)
