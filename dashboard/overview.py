"""
Dashboard: Project Overview - Autonomous Car Racing AI
======================================================
Presents reinforcement learning theory, CarRacing-v3 dynamics,
mathematical formulations (MDP), and system architecture.
"""

import streamlit as st


def render_overview():
    st.header("🏎️ Project Overview & Autonomous Racing Architecture")

    st.markdown(
        """
        Welcome to the **Autonomous Car Racing AI** system.
        This project implements an end-to-end Reinforcement Learning engineering system capable of mastering
        high-speed track navigation in **Gymnasium CarRacing-v3** through trial-and-error environmental interaction
        without human supervision or hard-coded trajectories.
        """
    )

    # Key Metrics / Highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Primary Environment", value="CarRacing-v3", delta="Box2D Physics")
    with col2:
        st.metric(label="High-Performance Target", value=">= +700.0 pts", delta="Track Completion")
    with col3:
        st.metric(label="Dual RL Paradigms", value="PPO & DQN", delta="Continuous & Discrete")
    with col4:
        st.metric(label="Perception Pipeline", value="CNN Vision", delta="Grayscale 84x84")

    st.markdown("---")

    # Reinforcement Learning Formulation
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. Mathematical Formulation (MDP)")
        st.markdown(
            r"""
            The racing domain is formalized as a Markov Decision Process (MDP) defined by the 5-tuple:
            $$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

            - **State Space $\mathcal{S}$ (Visual Top-Down Camera)**:
              - Raw visual frame: $96 \times 96 \times 3$ RGB pixels.
              - Optimized input: $84 \times 84 \times 1$ uint8 grayscale frame.
              - The visual state captures track curvature, asphalt boundaries, red-white rumble strips, green off-track grass, and car orientation.

            - **Action Spaces $\mathcal{A}$**:
              - **Continuous Control (PPO)**: $\mathbf{a} = [s, g, b] \in [-1, 1] \times [0, 1] \times [0, 1]$
                - $s \in [-1.0, 1.0]$: Steering angle (Left $\rightarrow$ Right).
                - $g \in [0.0, 1.0]$: Gas / Throttle acceleration.
                - $b \in [0.0, 1.0]$: Friction Braking.
              - **Discrete Control (DQN)**: 7-action maneuver palette:
                1. `Straight + Accel` $[0.0, 0.7, 0.0]$
                2. `Left + Accel` $[-0.8, 0.4, 0.0]$
                3. `Right + Accel` $[0.8, 0.4, 0.0]$
                4. `Left + Brake` $[-0.8, 0.0, 0.6]$
                5. `Right + Brake` $[0.8, 0.0, 0.6]$
                6. `Straight + Brake` $[0.0, 0.0, 0.6]$
                7. `Straight + Coast` $[0.0, 0.0, 0.0]$

            - **Reward Function $\mathcal{R}(s, a, s')$**:
              - $+1000 / N$ for each unique track tile visited along the racing line (where $N \approx 250\text{--}350$).
              - $-0.1$ constant step time penalty per frame.
              - $-100$ truncation penalty when the vehicle veers completely off-track into the grass for $>50$ steps.
            """
        )

    with col_right:
        st.subheader("2. Algorithmic Architectures")
        tab_ppo, tab_dqn = st.tabs(["Continuous PPO (Primary)", "Discrete DQN (Comparative)"])

        with tab_ppo:
            st.markdown(
                r"""
                **Proximal Policy Optimization (Actor-Critic)**
                PPO optimizes a parameterized stochastic policy $\pi_\theta(\mathbf{a}|s)$ with continuous Gaussian output heads:
                $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
                where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio and $\hat{A}_t$ is the Generalized Advantage Estimator (GAE).

                **Key Continuous Features**:
                - **CNN Feature Extractor**: Nature CNN backbone extracting visual track features.
                - **Simultaneous Multimodal Output**: Produces continuous mean and log standard deviation for steering, throttle, and braking simultaneously.
                - **Entropy Regularization**: Prevents premature collapse to extreme steering angles during initial exploration.
                """
            )

        with tab_dqn:
            st.markdown(
                r"""
                **Deep Q-Network (Discrete Actions)**
                DQN approximates the optimal action-value function $Q^*(s, a)$ over the 7 discrete driving maneuvers:
                $$Q^*(s, a) = \mathbb{E} \left[ r_t + \gamma \max_{a'} Q^*(s_{t+1}, a') \mid s_t=s, a_t=a \right]$$

                **Key Stability Mechanisms**:
                - **Experience Replay Buffer $\mathcal{D}$**: 15,000-transition circular buffer mitigating temporal correlations on CPU.
                - **Target Network $\theta^-$**: Frozen periodic target network updated every 1,000 steps:
                  $$y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)(1 - d_t)$$
                - **$\epsilon$-Greedy Schedule**: Decays exploration from 1.0 to 0.05 over 25% of training.
                """
            )

    st.markdown("---")
    st.subheader("3. Pipeline Workflow")
    st.markdown(
        """
        ```mermaid
        flowchart LR
            Env["Gymnasium CarRacing-v3"] --> Pre["Grayscale & FrameSkip (2x)"]
            Pre --> CNN["Nature CNN Feature Extractor"]
            CNN --> PPO["Continuous PPO [Steer, Gas, Brake]"]
            CNN --> DQN["Discrete DQN (7 Actions)"]
            PPO --> Track["Track Navigation & Progress"]
            DQN --> Track
            Track --> Telemetry["Telemetry Wrapper (Tiles, Off-Track, Lap %)"]
            Telemetry --> Eval["Statistical Evaluation & Dashboard"]
        ```
        """
    )
