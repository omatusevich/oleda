"""Shared test setup.

Matplotlib must be switched to a non-interactive backend before anything imports pyplot, and
every test that builds a report leaves figures open - pyplot holds them in a global registry,
so without an explicit close the suite accumulates them and matplotlib eventually warns.
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_figures():
    """Close every figure a test opened, pass or fail."""
    yield
    plt.close('all')
