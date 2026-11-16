"""
Dashboard: Project Overview
===========================
Presents reinforcement learning theory, environment physics,
mathematical formulations, and system architecture.
"""

import streamlit as st


def render_overview():
    st.header("Project Overview & Theoretical Architecture")

    st.markdown(
        """
        Welcome to the **Autonomous Reinforcement Learning Game-Playing Bot** system.
        This project implements an end-to-end RL engineering pipeline capable of solving complex
        control tasks via trial-and-error environmental interaction without human supervision.
        """
    )

    # Key Metrics / Highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Primary Environment", value="LunarLander-v3", delta="Continuous Box2D")
    with col2:
        st.metric(label="Solved Reward Threshold", value=">= +200.0 pts", delta="Mission Success")
    with col3:
        st.metric(label="Algorithms Implemented", value="DQN & PPO", delta="Value & Policy Gradient")
    with col4:
        st.metric(label="Execution Device", value="CPU Optimized", delta="Vectorized SB3")

    st.markdown("---")

    # Reinforcement Learning Formulation
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. Mathematical Formulation (MDP)")
        st.markdown(
            r"""
            The environment is formalized as a Markov Decision Process (MDP) defined by the 5-tuple:
            $$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

            - **State Space $\mathcal{S} \subset \mathbb{R}^8$**:
              1. $x$-coordinate (horizontal displacement from pad center)
              2. $y$-coordinate (altitude above terrain)
              3. $v_x$ (linear horizontal velocity)
              4. $v_y$ (linear vertical descent velocity)
              5. $\theta$ (lander body angle in radians)
              6. $\omega$ (angular velocity)
              7. $c_{left} \in \{0, 1\}$ (left leg ground contact indicator)
              8. $c_{right} \in \{0, 1\}$ (right leg ground contact indicator)

            - **Action Space $\mathcal{A} = \{0, 1, 2, 3\}$**:
              - `0`: Idle / Coasting (no fuel consumed)
              - `1`: Fire Left Orientation Thruster (induces clockwise torque)
              - `2`: Fire Main Engine (applies upward thrust against gravity)
              - `3`: Fire Right Orientation Thruster (induces counter-clockwise torque)

            - **Reward Function $\mathcal{R}(s, a, s')$**:
              - $+100$ to $+140$ for moving closer to the center of the landing pad $(0, 0)$
              - Negative penalty for higher linear and angular speeds
              - $-0.3$ per frame when firing the main engine (fuel penalty)
              - $-0.03$ per frame when firing side engines
              - $+10$ points for each leg resting on ground
              - $+100$ bonus for a successful soft landing on the pad
              - $-100$ penalty for crashing or flipping over
            """
        )

    with col_right:
        st.subheader("2. Algorithmic Foundations")
        tab_dqn, tab_ppo = st.tabs(["Deep Q-Network (DQN)", "Proximal Policy Opt (PPO)"])

        with tab_dqn:
            st.markdown(
                r"""
                **Deep Q-Network (Value-Based)**
                DQN approximates the optimal action-value function $Q^*(s, a)$ using a deep neural network parameterized by $\theta$:
                $$Q^*(s, a) = \mathbb{E} \left[ r_t + \gamma \max_{a'} Q^*(s_{t+1}, a') \mid s_t=s, a_t=a \right]$$

                **Key Stability Mechanisms**:
                - **Experience Replay Buffer $\mathcal{D}$**: Stores transitions $(s, a, r, s', d)$ to break temporal correlations and stabilize gradient descent.
                - **Target Network $\theta^-$**: Frozen periodic or Polyak soft-updated network producing stable targets:
                  $$y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)(1 - d_t)$$
                - **Loss Optimization**: Smooth L1 (Huber) loss minimizing Bellman residual:
                  $$\mathcal{L}(\theta) = \frac{1}{|B|} \sum_{i \in B} \text{Huber}\left(y_i - Q(s_i, a_i; \theta)\right)$$
                - **$\epsilon$-Greedy Policy**: Exploration decay from 1.0 to 0.02.
                """
            )

        with tab_ppo:
            st.markdown(
                r"""
                **Proximal Policy Optimization (Actor-Critic)**
                PPO optimizes a parameterized stochastic policy $\pi_\theta$ using a clipped surrogate objective that prevents destructively large policy updates:
                $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
                where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.

                **Key Stability Mechanisms**:
                - **Generalized Advantage Estimation (GAE-$\lambda$)**:
                  $$\hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V, \quad \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$
                - **Joint Actor-Critic Loss**:
                  $$L^{PPO}(\theta) = \hat{\mathbb{E}}_t [ L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t) ]$$
                - **Entropy Regularization $S[\pi_\theta]$**: Encourages exploratory diversity and prevents premature policy collapse.
                """
            )

    st.markdown("---")
    st.subheader("3. System Architecture Diagram")
    st.markdown(
        """
        ```
        +-----------------------------------------------------------------------------------------+
        |                               MLOps Pipeline Architecture                               |
        +-----------------------------------------------------------------------------------------+
                                                     |
               +-------------------------------------+-----------------------------------+
               |                                                                         |
               v                                                                         v
        [ Gymnasium Env ]                                                         [ Agents Core ]
          - LunarLander-v3 (Box2D)                                                  - DQN (Q-Network)
          - CartPole-v1 (Baseline)                                                  - PPO (Actor-Critic)
          - Telemetry Wrapper                                                       - Factory & Serialization
               |                                                                         |
               +-------------------------------------+-----------------------------------+
                                                     |
                                                     v
                                          [ Training Pipeline ]
                                            - Vectorized Envs
                                            - TensorBoard Logging
                                            - MetricsLoggingCallback
                                            - Checkpoint Manager
                                                     |
                                                     v
                                          [ Evaluation Suite ]
                                            - 100 Test Episodes
                                            - Welch's t-test / Mann-Whitney
                                            - Success / Crash Rates
                                                     |
                                                     v
                                         [ Deployment & Dashboard ]
                                            - Interactive Streamlit UI
                                            - Live Episode Simulator
                                            - Dynamic Plotly Analytics
        ```
        """
    )
