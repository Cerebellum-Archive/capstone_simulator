"""
Blue Water Macro Capstone Simulator

A quantitative trading simulation framework for backtesting strategies using Python,
designed for financial engineering education and research.

This package provides:
- Single-target and multi-target ETF prediction models
- Walk-forward backtesting with proper time-series validation
- xarray integration for multi-dimensional data handling
- Educational utilities and comprehensive documentation
- Professional-grade performance analysis and reporting

Author: Conrad Gann (Blue Water Macro Corp.)
License: Blue Water Macro Educational License (BWMEL)
"""

__version__ = "0.1.0"
__author__ = "Conrad Gann"
__email__ = "Conrad.Gann@BlueWaterMacro.com"
__license__ = "Blue Water Macro Educational License (BWMEL)"

# Import main modules for easy access
from . import utils_simulate
from . import single_target_simulator
from . import multi_target_simulator

# Import key utility functions for convenience
from .utils_simulate import (
    log_returns,
    EWMTransformer,
    create_results_xarray,
    calculate_performance_metrics,
    generate_train_predict_calendar,
    p_by_year,
    feature_profiles,
)

__all__ = [
    "utils_simulate",
    "single_target_simulator", 
    "multi_target_simulator",
    "log_returns",
    "EWMTransformer",
    "create_results_xarray",
    "calculate_performance_metrics",
    "generate_train_predict_calendar",
    "p_by_year",
    "feature_profiles",
]