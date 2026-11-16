"""
Evaluation Module for Autonomous Car Racing AI
==============================================
"""

from evaluation.metrics import compute_racing_metrics, compare_racing_distributions
from evaluation.evaluate import evaluate_racing_policy, run_evaluation
from evaluation.compare_agents import run_comparative_benchmark

__all__ = [
    "compute_racing_metrics",
    "compare_racing_distributions",
    "evaluate_racing_policy",
    "run_evaluation",
    "run_comparative_benchmark",
]
