"""
Unit Tests: Streamlit Dashboard Execution
=========================================
"""

import os
import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_app_loads_and_executes():
    """Verify app.py initializes and executes without exception."""
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception, f"App execution failed with exception: {at.exception}"
    assert len(at.sidebar.radio) > 0
