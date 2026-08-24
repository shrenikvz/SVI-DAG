"""
Unit tests for generator functions defined in main.py.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import main as main_mod


class TestTwoNodeGeneratorNonlinear:
    """Tests for two_node_generator_nonlinear function."""

    def test_returns_three_values(self):
        """Generator should return (obs_data, true_adj, node_names)."""
        result = main_mod.two_node_generator_nonlinear()

        assert len(result) == 3

    def test_obs_data_shape(self):
        """Observational data should have shape (num_samples, 2)."""
        obs_data, _, _ = main_mod.two_node_generator_nonlinear()

        assert obs_data.shape[0] == 1000
        assert obs_data.shape[1] == 2

    def test_true_adj_shape(self):
        """True adjacency should be (2, 2)."""
        _, true_adj, _ = main_mod.two_node_generator_nonlinear()

        assert true_adj.shape == (2, 2)

    def test_true_adj_has_one_edge(self):
        """True adjacency should have exactly one edge (x1 -> x2)."""
        _, true_adj, _ = main_mod.two_node_generator_nonlinear()

        assert np.sum(true_adj) == 1.0
        assert true_adj[1, 0] == 1.0  # x1 -> x2 in j->i convention

    def test_node_names(self):
        """Should return correct node names."""
        _, _, node_names = main_mod.two_node_generator_nonlinear()

        assert node_names == ["x1", "x2"]

    def test_data_dtype(self):
        """Data should be float32."""
        obs_data, true_adj, _ = main_mod.two_node_generator_nonlinear()

        assert obs_data.dtype == np.float32
        assert true_adj.dtype == np.float32

    def test_x2_depends_on_x1(self):
        """x2 should have correlation with x1 (causal relationship)."""
        obs_data, _, _ = main_mod.two_node_generator_nonlinear()

        x1 = obs_data[:, 0]
        x2 = obs_data[:, 1]

        corr = np.corrcoef(x1, x2)[0, 1]
        assert abs(corr) > 0.1  # Non-negligible correlation
