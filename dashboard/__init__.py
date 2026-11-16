"""Dashboard views module."""
from dashboard.overview import render_overview
from dashboard.analytics import render_analytics
from dashboard.comparison import render_comparison
from dashboard.simulation import render_simulation
from dashboard.evaluation_view import render_evaluation_view
from dashboard.model_info import render_model_info

__all__ = [
    "render_overview",
    "render_analytics",
    "render_comparison",
    "render_simulation",
    "render_evaluation_view",
    "render_model_info",
]
