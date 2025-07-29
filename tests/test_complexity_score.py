"""
Unit tests for complexity score functionality.

Tests cover model complexity scoring, complexity-adjusted metrics,
and integration with the simulation framework.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils_simulate import (
    get_complexity_score, 
    calculate_complexity_adjusted_metrics,
    unwrap_estimator,
    estimate_search_space_size
)

# Import sklearn models for testing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
try:
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
    HALVING_AVAILABLE = True
except ImportError:
    HALVING_AVAILABLE = False


class TestComplexityScoring:
    """Test complexity scoring for various sklearn models."""
    
    def test_linear_regression_complexity(self):
        """Test complexity score for LinearRegression (baseline)."""
        model = LinearRegression()
        complexity = get_complexity_score(model)
        assert complexity == 1.0, f"LinearRegression should have complexity 1.0, got {complexity}"
    
    def test_ridge_regression_complexity(self):
        """Test complexity score for Ridge regression with regularization."""
        # Strong regularization should reduce complexity
        model_strong = Ridge(alpha=10.0)
        complexity_strong = get_complexity_score(model_strong)
        
        # Weak regularization should have higher complexity
        model_weak = Ridge(alpha=0.1)
        complexity_weak = get_complexity_score(model_weak)
        
        assert complexity_strong < complexity_weak, "Stronger regularization should have lower complexity"
        assert complexity_strong < 1.0, "Ridge with alpha > 0 should have complexity < 1.0"
    
    def test_lasso_regression_complexity(self):
        """Test complexity score for Lasso regression."""
        model = Lasso(alpha=1.0)
        complexity = get_complexity_score(model)
        assert complexity < 1.0, "Lasso should have complexity < 1.0 due to sparsity"
    
    def test_random_forest_complexity(self):
        """Test complexity score for RandomForest."""
        # Default RandomForest should have higher complexity than linear models
        model = RandomForestRegressor(n_estimators=50, max_depth=5)
        complexity = get_complexity_score(model)
        assert complexity > 1.0, "RandomForest should have complexity > 1.0"
        
        # More trees and deeper trees should increase complexity
        model_complex = RandomForestRegressor(n_estimators=500, max_depth=30)
        complexity_complex = get_complexity_score(model_complex)
        assert complexity_complex > complexity, f"More complex RF should have higher score: {complexity_complex} > {complexity}"
    
    def test_decision_tree_complexity(self):
        """Test complexity score for DecisionTree."""
        model_shallow = DecisionTreeRegressor(max_depth=5)
        model_deep = DecisionTreeRegressor(max_depth=15)
        
        complexity_shallow = get_complexity_score(model_shallow)
        complexity_deep = get_complexity_score(model_deep)
        
        assert complexity_deep > complexity_shallow, "Deeper trees should have higher complexity"
    
    def test_svm_complexity(self):
        """Test complexity score for Support Vector Machines."""
        model_linear = SVR(kernel='linear')
        model_rbf = SVR(kernel='rbf', C=10.0, gamma=1.0)
        
        complexity_linear = get_complexity_score(model_linear)
        complexity_rbf = get_complexity_score(model_rbf)
        
        assert complexity_rbf > complexity_linear, "RBF kernel should have higher complexity than linear"
    
    def test_knn_complexity(self):
        """Test complexity score for K-Nearest Neighbors."""
        model_k1 = KNeighborsRegressor(n_neighbors=1)
        model_k10 = KNeighborsRegressor(n_neighbors=10)
        
        complexity_k1 = get_complexity_score(model_k1)
        complexity_k10 = get_complexity_score(model_k10)
        
        assert complexity_k1 > complexity_k10, "Fewer neighbors should have higher complexity (more overfitting prone)"
    
    def test_mlp_complexity(self):
        """Test complexity score for Multi-Layer Perceptron."""
        model_simple = MLPRegressor(hidden_layer_sizes=(50,))
        model_complex = MLPRegressor(hidden_layer_sizes=(100, 100, 50))
        
        complexity_simple = get_complexity_score(model_simple)
        complexity_complex = get_complexity_score(model_complex)
        
        assert complexity_complex > complexity_simple, "More complex MLP should have higher complexity"
    
    def test_unknown_estimator_default(self):
        """Test default complexity score for unknown estimators."""
        # Create a mock estimator
        class MockEstimator:
            def get_params(self):
                return {}
        
        model = MockEstimator()
        complexity = get_complexity_score(model)
        assert complexity == 2.0, "Unknown estimators should get default complexity of 2.0"


class TestGridSearchComplexity:
    """Test complexity scoring for hyperparameter search methods."""
    
    def test_grid_search_complexity_amplification(self):
        """Test that GridSearchCV amplifies base model complexity."""
        base_model = Ridge()
        base_complexity = get_complexity_score(base_model)
        
        # Create grid search with moderate search space
        param_grid = {'alpha': [0.1, 1.0, 10.0]}
        grid_model = GridSearchCV(base_model, param_grid)
        grid_complexity = get_complexity_score(grid_model)
        
        assert grid_complexity > base_complexity, "GridSearchCV should amplify base complexity"
    
    def test_randomized_search_complexity(self):
        """Test complexity scoring for RandomizedSearchCV."""
        base_model = RandomForestRegressor()
        param_distributions = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, 20]
        }
        
        random_search = RandomizedSearchCV(base_model, param_distributions, n_iter=5)
        complexity = get_complexity_score(random_search)
        
        base_complexity = get_complexity_score(base_model)
        assert complexity > base_complexity, "RandomizedSearchCV should amplify base complexity"
    
    @pytest.mark.skipif(not HALVING_AVAILABLE, reason="Halving search not available in this sklearn version")
    def test_halving_search_complexity(self):
        """Test complexity scoring for halving search methods."""
        base_model = Ridge()
        param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        
        halving_search = HalvingGridSearchCV(base_model, param_grid)
        complexity = get_complexity_score(halving_search)
        
        # Halving search should have lower complexity than full grid search
        full_search = GridSearchCV(base_model, param_grid)
        full_complexity = get_complexity_score(full_search)
        
        assert complexity < full_complexity, "Halving search should have lower complexity than full search"


class TestSearchSpaceEstimation:
    """Test search space size estimation."""
    
    def test_simple_grid_estimation(self):
        """Test search space estimation for simple parameter grids."""
        param_grid = {
            'alpha': [0.1, 1.0, 10.0],
            'fit_intercept': [True, False]
        }
        
        size = estimate_search_space_size(param_grid)
        assert size == 6, f"Expected 6 combinations, got {size}"
    
    def test_continuous_distribution_estimation(self):
        """Test search space estimation with continuous distributions."""
        from scipy.stats import uniform
        
        param_distributions = {
            'alpha': uniform(0.1, 10.0),
            'fit_intercept': [True, False]
        }
        
        size = estimate_search_space_size(param_distributions)
        assert size == 20, f"Expected 20 (10*2) for continuous distribution, got {size}"
    
    def test_empty_param_grid(self):
        """Test search space estimation with empty parameter grid."""
        param_grid = {}
        size = estimate_search_space_size(param_grid)
        assert size == 1, "Empty parameter grid should have size 1"


class TestComplexityAdjustedMetrics:
    """Test complexity-adjusted performance metrics."""
    
    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)  # For reproducible tests
        # Generate mock daily returns (252 trading days)
        self.returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        self.returns.index = pd.date_range('2023-01-01', periods=252, freq='B')
    
    def test_complexity_adjusted_metrics_calculation(self):
        """Test calculation of complexity-adjusted metrics."""
        complexity_score = 5.0
        
        metrics = calculate_complexity_adjusted_metrics(self.returns, complexity_score)
        
        # Check that all expected keys are present
        expected_keys = [
            'complexity_score', 'complexity_adjusted_return', 'complexity_adjusted_sharpe',
            'overfitting_penalty', 'complexity_efficiency', 'risk_adjusted_efficiency'
        ]
        
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"
        
        # Check that complexity score is stored correctly
        assert metrics['complexity_score'] == complexity_score
        
        # Check that adjusted metrics are lower than base metrics
        assert metrics['complexity_adjusted_return'] < metrics['Annual Return']
        assert metrics['complexity_adjusted_sharpe'] < metrics['Sharpe Ratio']
    
    def test_complexity_penalty_scaling(self):
        """Test that higher complexity scores lead to larger penalties."""
        simple_complexity = 1.0
        complex_complexity = 10.0
        
        simple_metrics = calculate_complexity_adjusted_metrics(self.returns, simple_complexity)
        complex_metrics = calculate_complexity_adjusted_metrics(self.returns, complex_complexity)
        
        # More complex models should have lower adjusted performance
        assert simple_metrics['complexity_adjusted_return'] > complex_metrics['complexity_adjusted_return']
        assert simple_metrics['complexity_adjusted_sharpe'] > complex_metrics['complexity_adjusted_sharpe']
        
        # Overfitting penalty should be higher for complex models
        assert simple_metrics['overfitting_penalty'] > complex_metrics['overfitting_penalty']
    
    def test_zero_complexity_handling(self):
        """Test handling of edge case with zero complexity."""
        # This shouldn't happen in practice, but test robustness
        metrics = calculate_complexity_adjusted_metrics(self.returns, 0.0)
        
        # Should handle division by zero gracefully
        assert not np.isnan(metrics['complexity_efficiency']), "Should handle zero complexity gracefully"
    
    def test_empty_returns_handling(self):
        """Test handling of empty returns series."""
        empty_returns = pd.Series([], dtype=float)
        metrics = calculate_complexity_adjusted_metrics(empty_returns, 1.0)
        
        # Should return metrics structure even with empty data
        assert 'complexity_score' in metrics
        assert metrics['complexity_score'] == 1.0


class TestUnwrappingFunctionality:
    """Test model unwrapping functionality."""
    
    def test_unwrap_grid_search(self):
        """Test unwrapping GridSearchCV to get base estimator."""
        base_model = Ridge(alpha=1.0)
        grid_model = GridSearchCV(base_model, {'alpha': [0.1, 1.0, 10.0]})
        
        unwrapped = unwrap_estimator(grid_model)
        assert isinstance(unwrapped, Ridge), "Should unwrap to Ridge estimator"
        assert unwrapped.alpha == 1.0, "Should preserve base model parameters"
    
    def test_unwrap_nested_estimators(self):
        """Test unwrapping multiply-wrapped estimators."""
        # Create nested wrapper structure (though this is artificial)
        base_model = LinearRegression()
        
        # Manually create nested structure
        class MockWrapper:
            def __init__(self, estimator):
                self.estimator = estimator
        
        nested_model = MockWrapper(MockWrapper(base_model))
        unwrapped = unwrap_estimator(nested_model)
        
        assert isinstance(unwrapped, LinearRegression), "Should unwrap to base LinearRegression"
    
    def test_unwrap_already_unwrapped(self):
        """Test unwrapping an estimator that doesn't need unwrapping."""
        model = Ridge()
        unwrapped = unwrap_estimator(model)
        
        assert unwrapped is model, "Should return same object if no unwrapping needed"


class TestIntegrationWithSimulation:
    """Test integration of complexity scoring with simulation framework."""
    
    def test_complexity_in_metadata(self):
        """Test that complexity scoring works with simulation metadata."""
        # This would test the integration, but requires more setup
        # For now, just test the basic concept
        
        pipe_steps = [
            ('scaler', 'StandardScaler'),  # Simplified for testing
            ('model', Ridge(alpha=1.0))
        ]
        
        # In real integration, this would come from metadata
        # For now, just test that we can extract the final estimator
        final_estimator = None
        for step_name, step_obj in pipe_steps:
            if hasattr(step_obj, 'predict') or isinstance(step_obj, Ridge):
                final_estimator = step_obj
        
        if final_estimator is not None:
            complexity = get_complexity_score(final_estimator)
            assert complexity == 0.5, f"Ridge(alpha=1.0) should have complexity 0.5, got {complexity}"


if __name__ == '__main__':
    pytest.main([__file__])