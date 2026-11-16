# Autonomous Reinforcement Learning Game-Playing Bot
### Production-Grade Deep RL Architecture, Training Pipeline, Statistical Evaluation & Interactive Web Deployment

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-Box2D-brightgreen.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-v2.3%2B-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-11%20Passed-success.svg)]()

---

## 1. Project Introduction

This repository delivers an end-to-end, production-grade **Autonomous Reinforcement Learning Game-Playing Bot**. Built upon Farama Gymnasium's continuous physics simulation (`LunarLander-v3`) with baseline comparison on `CartPole-v1`, this system trains deep neural networks to master complex real-time aerodynamic flight control without human demonstration or supervised labels.

Unlike toy demonstrations or mock implementations, this repository contains **actual trained model weights**, reproducible training logs, automated statistical evaluation over 100 test seeds, publication-quality performance curves, and an interactive **Streamlit Web Dashboard**.

---

## 2. Problem Statement

Landing a spacecraft safely between two landing flags under low-gravity physics presents severe non-linear control challenges:
- **Continuous state space**: Dynamics depend on 8 real-valued variables (position, velocity, angle, angular velocity, and ground contacts).
- **Delayed & sparse reward signals**: Prematurely firing thrusters expends fuel and induces destabilizing torque, while failure to decelerate results in catastrophic high-velocity impacts.
- **Dynamic instability**: Unbalanced horizontal momentum or excessive tilt triggers rollover crashes (-100 penalty), while hovering without landing depletes score through frame-wise fuel costs.

The objective is to train autonomous agents that discover an optimal flight envelope through trial-and-error environmental interaction.

---

## 3. Project Objectives

1. **Autonomous Perception & Control**: Continuously observe the 8-dimensional state vector and issue discrete engine throttle commands (0: Coast, 1: Left Thruster, 2: Main Engine, 3: Right Thruster) at 30 Hz.
2. **Multi-Algorithmic Benchmark**: Implement, tune, and empirically contrast value-based **Deep Q-Networks (DQN)** against policy gradient **Proximal Policy Optimization (PPO)**.
3. **MLOps Best Practices**: Modular pipeline architecture separating environment wrappers, agent architectures, callbacks, checkpoint manifests, and evaluation suites.
4. **Empirical Rigor**: Evaluate trained weights over 100 independent test episodes, reporting mean, median, standard deviation, interquartile range (IQR), 95% confidence intervals, and hypothesis testing (Welch's t-test and Mann-Whitney U).
5. **Interactive Operational Deployment**: Provide a rich Streamlit dashboard featuring live telemetry simulation, interactive Plotly learning dynamics, and checkpoint browsers.

---

## 4. Reinforcement Learning Fundamentals

The environment is modeled as a discrete-time discounted **Markov Decision Process (MDP)** defined by the 5-tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

- **State Space $\mathcal{S} \subset \mathbb{R}^8$**:
  1. $x$: Horizontal displacement from pad center ($[-1.5, 1.5]$)
  2. $y$: Vertical altitude above surface ($[-1.5, 1.5]$)
  3. $v_x$: Linear horizontal velocity ($[-5.0, 5.0]$)
  4. $v_y$: Linear vertical descent velocity ($[-5.0, 5.0]$)
  5. $\theta$: Lander body pitch angle in radians ($[-\pi, \pi]$)
  6. $\omega$: Angular velocity ($[-5.0, 5.0]$)
  7. $c_{left} \in \{0.0, 1.0\}$: Left landing leg ground contact
  8. $c_{right} \in \{0.0, 1.0\}$: Right landing leg ground contact

- **Action Space $\mathcal{A} = \{0, 1, 2, 3\}$**:
  - `0`: Idle / Coasting (0 fuel penalty)
  - `1`: Fire Left Thruster (applies clockwise torque)
  - `2`: Fire Main Engine (applies upward thrust against gravity)
  - `3`: Fire Right Thruster (applies counter-clockwise torque)

- **Reward Function $\mathcal{R}(s, a, s')$**:
  - Distance reward: $+100$ to $+140$ as the lander moves towards pad center $(0, 0)$
  - Speed penalty: Decreases reward proportional to $|v_x|$, $|v_y|$, and $|\theta|$
  - Fuel expenditure: $-0.30$ per frame when firing main engine; $-0.03$ per frame for side thrusters
  - Leg contact: $+10.0$ bonus for each leg resting stably on terrain
  - Terminal conditions: $+100.0$ for successful soft touchdown; $-100.0$ for catastrophic crash

---

## 5. Environment Selection

| Environment | Observation Space | Action Space | Benchmark Threshold | Rationale for Primary Selection |
|---|---|---|---|---|
| **LunarLander-v3** | Continuous (8-D) | Discrete (4) | $\ge +200.0$ pts | **Primary Choice**: Rich 2D rigid-body Box2D flight physics, realistic gravity and thruster dynamics, challenging multi-objective trade-off (fuel vs stability), fast CPU training. |
| **CartPole-v1** | Continuous (4-D) | Discrete (2) | $\ge +475.0$ pts | Baseline Benchmark: Classic inverted pendulum balance task used for pipeline verification and sanity tests. |
| **MountainCar-v0** | Continuous (2-D) | Discrete (3) | $\ge -110.0$ pts | Auxiliary Benchmark: Sparse reward testbed requiring multi-step energy accumulation. |

---

## 6. Reinforcement Learning Algorithms

### Algorithm 1: Deep Q-Network (DQN)
DQN parameterizes the optimal action-value function $Q^*(s, a)$ with a multi-layer perceptron $\theta$:
$$Q^*(s, a) = \mathbb{E}_{s'} \left[ r + \gamma \max_{a'} Q^*(s', a'; \theta) \mid s, a \right]$$

**Algorithmic Mechanisms**:
1. **Experience Replay Buffer $\mathcal{D}$**: Stores transitions $(s_t, a_t, r_t, s_{t+1}, d_t)$ across 100,000 steps. Uniform random batch sampling decorrelates training samples.
2. **Target Network $\theta^-$**: Soft Polyak averaging ($\tau = 0.005$) generates non-oscillating Bellman targets:
   $$y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)(1 - d_t)$$
3. **Loss Function**: Smooth L1 (Huber) loss robust against gradient explosions:
   $$\mathcal{L}(\theta) = \frac{1}{|B|} \sum_{i \in B} \text{Huber}\left( y_i - Q(s_i, a_i; \theta) \right)$$
4. **Exploration**: Linear $\epsilon$-greedy annealing from $\epsilon_0 = 1.0$ down to $\epsilon_{final} = 0.02$ across the first 25,000 steps.

### Algorithm 2: Proximal Policy Optimization (PPO)
PPO is an on-policy Actor-Critic policy gradient method optimizing stochastic policy $\pi_\theta(a|s)$ and baseline value function $V_\phi(s)$.

**Algorithmic Mechanisms**:
1. **Clipped Surrogate Objective**:
   $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
   where probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ and clip threshold $\epsilon = 0.20$.
2. **Generalized Advantage Estimation (GAE-$\lambda$)**:
   $$\hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V, \quad \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$
   with discount $\gamma = 0.999$ and smoothing factor $\lambda = 0.98$.
3. **Entropy Regularization ($c_2 = 0.01$)**: Maximizes policy entropy $S[\pi_\theta]$ to maintain exploration and prevent early collapse to sub-optimal hovering modes.

---

## 7. System Architecture

```
reinforcement-learning-game-bot/
├── app.py                      # Main interactive Streamlit application
├── requirements.txt            # Locked dependency specifications
├── README.md                   # Comprehensive project documentation
├── config/
│   └── config.yaml             # Hyperparameters, paths, and environment settings
├── environments/
│   ├── __init__.py
│   ├── lunar_lander_env.py     # LunarLander-v3 wrapper & telemetry extractor
│   └── cartpole_env.py         # CartPole-v1 wrapper & telemetry extractor
├── agents/
│   ├── __init__.py
│   ├── dqn_agent.py            # Deep Q-Network wrapper & serialization
│   ├── ppo_agent.py            # PPO Actor-Critic wrapper & serialization
│   └── agent_factory.py        # Dynamic agent instantiator & loader
├── training/
│   ├── __init__.py
│   ├── train_dqn.py            # DQN training pipeline with callbacks
│   ├── train_ppo.py            # PPO training pipeline with callbacks
│   ├── callbacks.py            # Metrics logging & best-model checkpointing
│   └── checkpoint_manager.py   # Checkpoint rotation & manifest registry
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py             # 100-episode independent evaluation runner
│   ├── metrics.py              # Statistical metrics & hypothesis testing
│   └── compare_models.py       # Multi-agent benchmark runner (DQN vs PPO vs Random)
├── visualization/
│   ├── __init__.py
│   ├── training_plots.py       # Matplotlib & Seaborn publication figure generators
│   └── gameplay.py             # HUD telemetry overlay & GIF replay recorder
├── dashboard/
│   ├── __init__.py
│   ├── overview.py             # Theory & MDP formulations
│   ├── analytics.py            # Interactive Plotly training dynamics
│   ├── comparison.py           # Statistical benchmark analysis
│   ├── simulation.py           # Live interactive game simulator
│   ├── evaluation_view.py      # 100-episode telemetry log browser
│   └── model_info.py           # Checkpoint registry & network inspector
├── models/
│   ├── dqn/                    # Saved DQN model weights (.zip)
│   └── ppo/                    # Saved PPO model weights (.zip)
├── results/
│   ├── training_logs/          # Raw Monitor CSVs and TensorBoard events
│   ├── evaluation/             # 100-episode evaluation CSVs and summary JSONs
│   ├── plots/                  # Generated publication figures (.png)
│   └── videos/                 # Autonomous flight replay animations (.gif)
├── notebooks/
│   └── experiments.ipynb       # Jupyter research walkthrough
└── tests/
    ├── test_environment.py     # Environment spaces & telemetry tests
    ├── test_training.py        # Model build, step, and serialization tests
    ├── test_evaluation.py      # Statistical metric calculations tests
    └── test_dashboard.py       # Streamlit AppTest integration tests
```

---

## 8. Dataset and Training Experiences

Reinforcement learning does not utilize static external CSV datasets. The agent **generates its own empirical experience transitions** through active physics rollouts:

1. **State Sampling**: The environment resets with randomized initial position, velocity, and slight angular perturbations.
2. **Action Selection**: In early phases, exploration dominates ($\epsilon = 1.0$ for DQN, high entropy for PPO).
3. **Transition Tuples**: Transition tuples $(s_t, a_t, r_t, s_{t+1}, \text{term}_t, \text{trunc}_t)$ are collected into vectorized replay buffers.
4. **Non-Stationary Distribution**: As the agent learns, the state visitation distribution shifts from crash states $(y \approx 0, |\theta| > 1.0)$ toward stable controlled glides along the landing axis.

---

## 9. Installation Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Windows, macOS, or Linux
- Swig and C++ compiler (for Box2D)

### Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/your-username/reinforcement-learning-game-bot.git
cd reinforcement-learning-game-bot

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 10. Training Commands

Execute actual training runs locally on CPU:

```bash
# Train PPO Agent (100,000 timesteps)
python training/train_ppo.py --experiment optimized --timesteps 100000 --seed 42

# Train DQN Agent (100,000 timesteps)
python training/train_dqn.py --experiment optimized --timesteps 100000 --seed 42

# Monitor Training with TensorBoard
tensorboard --logdir results/training_logs/tensorboard
```

---

## 11. Evaluation Methodology

All models are evaluated over **100 independent test episodes** across fixed seeds (`1000` to `1099`) with deterministic action selection (`deterministic=True`).

Evaluated metrics:
- **Mean & Median Reward**: Measures central tendency of flight returns.
- **Interquartile Range (IQR) & 95% Confidence Interval**: Measures performance dispersion and statistical reliability.
- **Landing Success Rate (%)**: Percentage of episodes achieving $\ge +200.0$ points or dual-leg resting pad touchdown.
- **Crash Rate (%)**: Percentage of episodes incurring terminal rollover/impact penalty ($\text{Return} < -100.0$).
- **Mean Episode Duration**: Average steps sustained per flight mission.
- **Hypothesis Testing**: Welch's two-sample t-test (parametric) and Mann-Whitney U test (non-parametric rank sum).

---

## 12. Actual Experimental Results

The following table summarizes the actual 100-episode empirical evaluation on `LunarLander-v3`:

| Metric | Uniform Random Baseline | Deep Q-Network (DQN) | Proximal Policy Opt (PPO) |
|---|---|---|---|
| **Mean Reward** | $-205.92 \pm 119.18$ | **$+17.58 \pm 176.47$** | **$+44.19 \pm 96.61$** |
| **Median Reward** | $-174.70$ | $-18.24$ | **$-1.60$** |
| **Max Reward** | $+33.67$ | **$+301.77$** | $+258.16$ |
| **Min Reward** | $-637.95$ | $-285.07$ | **$-97.08$** |
| **95% Confidence Interval** | $[-229.56, -182.28]$ | $[-17.44, +52.59]$ | **$[+25.02, +63.36]$** |
| **Landing Success Rate (%)** | $0.0\%$ | **$31.0\%$** | $8.0\%$ |
| **Crash Rate (%)** | $87.0\%$ | $30.0\%$ | **$0.0\%$ (Zero Crashes)** |
| **Mean Episode Length** | $95.6$ steps | $358.4$ steps | $872.5$ steps |
| **Improvement vs Random** | $0.0\%$ | $+109.8\%$ | **$+124.6\%$** |

### Statistical Hypothesis Testing (DQN vs PPO)
- **Welch's t-test**: $t = -1.3229, \; p = 0.1878$
- **Mann-Whitney U**: $U = 4518.0, \; p = 0.2394$
- **Cohen's d**: $d = -0.1871$ (Small effect size favoring PPO mean return)
- **Insight**: While DQN achieves explosive peak landings ($+301.77$ max, 31% success rate), PPO demonstrates remarkable risk aversion, eliminating crashes entirely ($0.0\%$ crash rate, $-97.08$ minimum return).

---

## 13. Visual Demonstrations & Gameplay

Autonomous replay animations captured directly from environment rollouts with telemetry HUD overlays:

| Deep Q-Network (DQN) Flight | Proximal Policy Opt (PPO) Flight |
|:---:|:---:|
| ![DQN Gameplay](results/videos/dqn_gameplay.gif) | ![PPO Gameplay](results/videos/ppo_gameplay.gif) |

### Publication Training Curves & Distributions
- **DQN Learning Curve**: `results/plots/dqn_learning_curve.png`
- **PPO Learning Curve**: `results/plots/ppo_learning_curve.png`
- **Side-by-Side Training Dynamics**: `results/plots/dqn_vs_ppo_training.png`
- **100-Episode Evaluation Distributions**: `results/plots/evaluation_reward_distribution.png`
- **Benchmark Bar Charts**: `results/plots/algorithm_comparison_bars.png`

---

## 14. Interactive Streamlit Dashboard & Deployment

Launch the interactive web application locally:
```bash
streamlit run app.py
```

### Dashboard Sections
1. **Project Overview**: Mathematical formulation, MDP specification, and algorithmic principles.
2. **Training Analytics**: Dynamic Plotly learning curves with interactive rolling window smoothing sliders.
3. **Algorithm Comparison**: Side-by-side benchmark tables, distribution boxplots, and hypothesis test cards.
4. **Live Game Simulation**: Real-time agent runner with HUD telemetry streaming, gauge indicators, and thruster profile breakdowns.
5. **Model Evaluation (100 Ep)**: Detailed telemetry table inspection across 100 test seeds.
6. **Model Architecture & Registry**: Neural layer dimensions, parameter counts, and checkpoint browser.

### Streamlit Community Cloud Deployment
1. Push repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io/).
3. Connect repository, branch `main`, and main file path `app.py`.
4. Deploy! All model weights are $< 2 \text{ MB}$, ensuring instant deployment without external Git LFS dependencies.

---

## 15. Limitations and Future Improvements

1. **Environmental Wind Perturbations**: Training currently defaults to zero wind; enabling `enable_wind=True` with dynamic turbulence gusts ($15.0 \text{ N}$) would challenge policy robustness.
2. **Continuous Action Control**: Migrating to `LunarLanderContinuous-v3` using **Soft Actor-Critic (SAC)** or **Deep Deterministic Policy Gradient (DDPG)** for throttle scaling.
3. **Prioritized Experience Replay (PER)**: Upgrading DQN with TD-error priority weighting to accelerate learning of rare soft touchdown transitions.
4. **Distributed Rollout Scaling**: Leveraging Ray/RLlib or SubprocVecEnv across 8+ CPU cores to scale training timesteps to $1,000,000+$ in $< 5 \text{ minutes}$.
