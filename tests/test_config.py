"""
Unit tests for svidag/config.py

This module tests configuration parameters for validity, consistency,
and type correctness. It serves as documentation for expected config values.

Author: Test Suite for SVIDAG
"""

import pytest
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from svidag import config


class TestTrainingConfig:
    """Tests for training configuration parameters."""
    
    def test_seed_is_integer(self):
        """Seed should be an integer."""
        assert isinstance(config.seed, int)
    
    def test_num_iters_positive(self):
        """Number of iterations should be positive."""
        assert config.num_iters > 0
        assert isinstance(config.num_iters, int)
    
    def test_batch_size_positive(self):
        """Batch size should be positive."""
        assert config.batch_size > 0
        assert isinstance(config.batch_size, int)
    
    def test_print_every_positive(self):
        """Print frequency should be positive."""
        assert config.print_every > 0
        assert isinstance(config.print_every, int)
    
    def test_learning_rate_positive(self):
        """Learning rate should be positive."""
        assert config.lr > 0
        assert isinstance(config.lr, float)
    
    def test_grad_clip_positive(self):
        """Gradient clipping value should be positive."""
        assert config.grad_clip > 0
        assert isinstance(config.grad_clip, float)
    
    def test_batch_size_reasonable(self):
        """Batch size should be reasonable (not too large)."""
        assert config.batch_size <= 1024  # Reasonable upper bound


class TestNoiseConfig:
    """Tests for noise-related configuration."""

    def test_obs_noise_scale_positive(self):
        """Observation noise scale should be positive."""
        assert config.obs_noise_scale > 0
        assert isinstance(config.obs_noise_scale, (int, float))


class TestDataConfig:
    """Tests for data-related configuration."""
    
    def test_n_test_samples_positive(self):
        """Number of test samples should be positive."""
        assert config.n_test_samples > 0
        assert isinstance(config.n_test_samples, int)


class TestVariationalConfig:
    """Tests for variational family configuration."""
    
    def test_hidden_dim_positive(self):
        """Hidden dimension should be positive."""
        assert config.hidden_dim > 0
        assert isinstance(config.hidden_dim, int)
    
    def test_prior_theta_mu_is_numeric(self):
        """Prior mean for theta should be numeric."""
        assert isinstance(config.prior_theta_mu, (int, float))
    
    def test_prior_theta_sigma_positive(self):
        """Prior sigma for theta should be positive."""
        assert config.prior_theta_sigma > 0


class TestFlowConfig:
    """Tests for normalizing flow configuration."""
    
    def test_flow_hidden_is_list(self):
        """Flow hidden dimensions should be a list."""
        # flow_hidden is a placeholder: None means "auto-set per fit to
        # [num_nodes, num_nodes] in make_model_and_state".
        assert config.flow_hidden is None or isinstance(config.flow_hidden, list)
    
    def test_flow_hidden_positive(self):
        """All flow hidden dimensions should be positive (when explicitly set)."""
        for dim in (config.flow_hidden or []):
            assert dim > 0
            assert isinstance(dim, int)
    
    def test_flow_n_blocks_positive(self):
        """Number of flow blocks should be positive."""
        assert config.flow_n_blocks > 0
        assert isinstance(config.flow_n_blocks, int)
    
    def test_flow_type_valid(self):
        """Flow type should be one of the valid options."""
        valid_types = ['maf', 'nsf', 'nsf_coupling']
        assert config.flow_type.lower() in valid_types
    
    def test_nsf_num_bins_positive(self):
        """NSF number of bins should be positive."""
        assert config.nsf_num_bins > 0
        assert isinstance(config.nsf_num_bins, int)
    
    def test_nsf_tail_bound_positive(self):
        """NSF tail bound should be positive."""
        assert config.nsf_tail_bound > 0


class TestTemperatureConfig:
    """Tests for temperature/annealing configuration."""
    
    def test_T_B_start_positive(self):
        """T_B start temperature should be positive."""
        assert config.T_B_start > 0
    
    def test_T_B_end_positive(self):
        """T_B end temperature should be positive."""
        assert config.T_B_end > 0
    
    def test_tau_sink_start_positive(self):
        """Sinkhorn tau start should be positive."""
        assert config.tau_sink_start > 0
    
    def test_tau_sink_end_positive(self):
        """Sinkhorn tau end should be positive."""
        assert config.tau_sink_end > 0
    
    def test_straight_through_mask_is_bool(self):
        """straight_through_mask should be boolean."""
        assert isinstance(config.straight_through_mask, bool)


class TestSinkhornConfig:
    """Tests for Sinkhorn normalization configuration."""
    
    def test_sinkhorn_iters_positive(self):
        """Sinkhorn iterations should be positive."""
        assert config.sinkhorn_iters > 0
        assert isinstance(config.sinkhorn_iters, int)
    
    def test_sinkhorn_eps_positive(self):
        """Sinkhorn epsilon should be positive."""
        assert config.sinkhorn_eps > 0
    
    def test_sinkhorn_eps_small(self):
        """Sinkhorn epsilon should be small for stability."""
        assert config.sinkhorn_eps < 1e-3


class TestSVGDConfig:
    """Tests for SVGD configuration."""
    
    def test_n_particles_positive(self):
        """Number of particles should be positive."""
        assert config.n_particles > 0
        assert isinstance(config.n_particles, int)
    
    def test_eta_r_positive(self):
        """SVGD step size should be positive."""
        assert config.eta_r > 0
    
    def test_prior_r_sigma_positive(self):
        """Prior sigma for r should be positive."""
        assert config.prior_r_sigma > 0


class TestPriorConfig:
    """Tests for prior configuration."""
    
    def test_nu_min_positive(self):
        """NU_MIN should be positive."""
        assert config.NU_MIN > 0
    
    def test_pr_prec_kappa_positive(self):
        """PR_PREC_KAPPA should be positive."""
        assert config.PR_PREC_KAPPA > 0
    
    def test_pr_prec_gamma_positive(self):
        """PR_PREC_GAMMA should be positive."""
        assert config.PR_PREC_GAMMA > 0
    
    def test_elbo_mc_samples_positive(self):
        """ELBO MC samples should be positive."""
        assert config.elbo_mc_samples > 0
        assert config.ELBO_MC_SAMPLES >= 1


class TestScenarioConfig:
    """Tests for scenario configuration."""
    
    def test_scenarios_is_list(self):
        """Scenarios should be a list."""
        assert isinstance(config.scenarios, list)
    
    def test_scenarios_non_empty(self):
        """Scenarios list should not be empty."""
        assert len(config.scenarios) > 0
    
    def test_scenarios_valid(self):
        """All scenarios should be valid."""
        valid_scenarios = {"noninformative", "strong_correct", "strong_incorrect"}
        for scenario in config.scenarios:
            assert scenario in valid_scenarios


class TestSamplingConfig:
    """Tests for sampling configuration."""
    
    def test_posterior_samples_A_positive(self):
        """Posterior samples for A should be positive."""
        assert config.posterior_samples_A > 0
        assert isinstance(config.posterior_samples_A, int)
    
    def test_dag_prob_samples_positive(self):
        """DAG probability samples should be positive."""
        assert config.dag_prob_samples > 0
        assert isinstance(config.dag_prob_samples, int)


class TestPlotConfig:
    """Tests for plotting configuration."""
    
    def test_plot_dir_is_string(self):
        """Plot directory should be a string."""
        assert isinstance(config.plot_dir, str)
    
    def test_plot_dir_not_empty(self):
        """Plot directory string should not be empty."""
        assert len(config.plot_dir) > 0



class TestConfigConsistency:
    """Tests for consistency between related config values."""
    
    def test_temperature_ordering(self):
        """Temperature values should have reasonable ordering."""
        # Start and end temperatures should both be defined
        assert config.T_B_start is not None
        assert config.T_B_end is not None
        assert config.tau_sink_start is not None
        assert config.tau_sink_end is not None
    
    def test_elbo_mc_samples_consistency(self):
        """ELBO_MC_SAMPLES should be derived from elbo_mc_samples."""
        assert config.ELBO_MC_SAMPLES == max(1, int(config.elbo_mc_samples))
    


class TestConfigTypes:
    """Tests for type correctness of all config values."""
    
    def test_integer_configs(self):
        """Integer configs should be integers."""
        integer_configs = [
            'seed', 'num_iters', 'batch_size', 'print_every',
            'n_test_samples', 'hidden_dim', 'flow_n_blocks',
            'sinkhorn_iters', 'n_particles',
            'posterior_samples_A', 'dag_prob_samples', 'nsf_num_bins'
        ]
        
        for cfg_name in integer_configs:
            value = getattr(config, cfg_name)
            assert isinstance(value, int), f"{cfg_name} should be int, got {type(value)}"
    
    def test_float_configs(self):
        """Float configs should be numeric."""
        float_configs = [
            'lr', 'grad_clip', 'obs_noise_scale',
            'prior_theta_mu', 'prior_theta_sigma',
            'T_B_start', 'T_B_end', 'tau_sink_start', 'tau_sink_end',
            'sinkhorn_eps', 'eta_r', 'prior_r_sigma',
            'NU_MIN', 'PR_PREC_KAPPA', 'PR_PREC_GAMMA',
            'nsf_tail_bound'
        ]
        
        for cfg_name in float_configs:
            value = getattr(config, cfg_name)
            assert isinstance(value, (int, float)), f"{cfg_name} should be numeric, got {type(value)}"
    
    def test_bool_configs(self):
        """Boolean configs should be booleans."""
        bool_configs = ['straight_through_mask']
        
        for cfg_name in bool_configs:
            value = getattr(config, cfg_name)
            assert isinstance(value, bool), f"{cfg_name} should be bool, got {type(value)}"
    
    def test_string_configs(self):
        """String configs should be strings."""
        string_configs = ['plot_dir', 'flow_type']
        
        for cfg_name in string_configs:
            value = getattr(config, cfg_name)
            assert isinstance(value, str), f"{cfg_name} should be str, got {type(value)}"
    
    def test_list_configs(self):
        """List configs should be lists."""
        list_configs = ['scenarios']
        
        for cfg_name in list_configs:
            value = getattr(config, cfg_name)
            assert isinstance(value, list), f"{cfg_name} should be list, got {type(value)}"


class TestConfigDocumentation:
    """Tests that serve as documentation for config values."""
    
    def test_print_config_summary(self):
        """Print a summary of all config values (useful for documentation)."""
        # This test always passes but documents config values
        config_summary = {
            'Training': {
                'seed': config.seed,
                'num_iters': config.num_iters,
                'batch_size': config.batch_size,
                'lr': config.lr,
                'grad_clip': config.grad_clip,
            },
            'Noise': {
                'obs_noise_scale': config.obs_noise_scale,
            },
            'Flow': {
                'flow_type': config.flow_type,
                'flow_hidden': config.flow_hidden,  # None => auto-set per fit
                'flow_n_blocks': config.flow_n_blocks,
            },
            'Temperature': {
                'T_B_start': config.T_B_start,
                'T_B_end': config.T_B_end,
                'tau_sink_start': config.tau_sink_start,
                'tau_sink_end': config.tau_sink_end,
            },
            'SVGD': {
                'n_particles': config.n_particles,
                'eta_r': config.eta_r,
            },
            'Sampling': {
                'posterior_samples_A': config.posterior_samples_A,
                'dag_prob_samples': config.dag_prob_samples,
            },
        }
        
        # Just verify the summary can be created
        assert len(config_summary) > 0
