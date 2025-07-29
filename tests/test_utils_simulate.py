"""
Unit tests for utils_simulate.py module.

Tests cover core utility functions, data transformations, xarray operations,
and educational helper functions.
"""

import pytest
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import functions to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils_simulate import (
    simplify_teos, log_returns, p_by_slice, p_by_year, feature_profiles,
    generate_train_predict_calender, EWMTransformer, create_results_xarray,
    calculate_performance_metrics, plot_xarray_results, create_correlation_matrix,
    export_results_to_csv
)


class TestBasicUtilities:
    """Test basic utility functions."""
    
    def test_simplify_teos(self):
        """Test timezone simplification function."""
        # Create timezone-aware data
        dates = pd.date_range('2022-01-01', periods=5, freq='D', tz='UTC')
        df = pd.DataFrame({'price': [100, 101, 102, 103, 104]}, index=dates)
        
        # Apply function
        result = simplify_teos(df)
        
        # Check timezone was removed and dates normalized
        assert result.index.tz is None
        assert all(result.index.time == pd.Timestamp('00:00:00').time())
        assert len(result) == len(df)
    
    def test_log_returns_calculation(self, sample_price_data):
        """Test log returns calculation."""
        result = log_returns(sample_price_data)
        
        # Check basic properties
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_price_data)
        assert result.columns.equals(sample_price_data.columns)
        
        # First row should be NaN
        assert result.iloc[0].isna().all()
        
        # Check mathematical correctness
        for col in sample_price_data.columns:
            manual_calc = np.log(sample_price_data[col]) - np.log(sample_price_data[col].shift(1))
            pd.testing.assert_series_equal(result[col], manual_calc, check_names=False)
    
    def test_log_returns_with_zeros(self):
        """Test log returns with zero prices (should handle gracefully)."""
        df = pd.DataFrame({
            'normal': [100, 110, 120],
            'with_zero': [100, 0, 120]  # Zero price in middle
        }, index=pd.date_range('2022-01-01', periods=3))
        
        result = log_returns(df)
        
        # Should handle zeros (will produce -inf, but function shouldn't crash)
        assert not np.isfinite(result['with_zero'].iloc[1])
        assert np.isfinite(result['normal'].iloc[1])


class TestStatisticalAnalysis:
    """Test statistical analysis functions."""
    
    def test_p_by_slice(self, sample_features_targets):
        """Test feature significance analysis across time slices."""
        X, y_single, _ = sample_features_targets
        
        # Create time slices (first half, second half)
        n = len(X)
        t_list = [X.index[:n//2], X.index[n//2:]]
        t_list_labels = ['first_half', 'second_half']
        
        result = p_by_slice(X, y_single, t_list, t_list_labels)
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (len(X.columns), len(t_list_labels))
        assert list(result.columns) == t_list_labels
        assert list(result.index) == list(X.columns)
        
        # Values should be correlation coefficients (between -1 and 1)
        assert (result.abs() <= 1.0).all().all()
    
    def test_p_by_year(self, sample_features_targets):
        """Test annual feature analysis."""
        X, y_single, _ = sample_features_targets
        
        result = p_by_year(X, y_single, sort_by='p_value')
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X.columns)
        
        # Should have expected columns
        expected_cols = ['pearson', 'p_value', 'f_statistic', 'mutual_info']
        for col in expected_cols:
            assert col in result.columns
        
        # P-values should be between 0 and 1
        assert (result['p_value'] >= 0).all()
        assert (result['p_value'] <= 1).all()
    
    def test_feature_profiles(self, sample_features_targets):
        """Test comprehensive feature profiling."""
        X, y_single, _ = sample_features_targets
        
        result = feature_profiles(X, y_single, sort_by='pearson')
        
        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X.columns)
        
        # Should be sorted by pearson correlation (absolute values)
        pearson_abs = result['pearson'].abs()
        assert (pearson_abs.values == pearson_abs.sort_values(ascending=False).values).all()


class TestTrainingCalendar:
    """Test training calendar generation."""
    
    def test_generate_train_predict_calender_expanding(self, sample_features_targets):
        """Test expanding window calendar generation."""
        X, y_single, _ = sample_features_targets
        
        result = generate_train_predict_calender(
            X, window_type='expanding', window_size=100
        )
        
        # Check basic properties
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Each item should be a tuple of 3 dates
        for item in result:
            assert len(item) == 3
            train_start, train_end, predict_date = item
            
            # Dates should be in chronological order
            assert train_start <= train_end <= predict_date
            
            # All should be in the data index
            assert train_start in X.index
            assert train_end in X.index
            assert predict_date in X.index
    
    def test_generate_train_predict_calender_rolling(self, sample_features_targets):
        """Test rolling window calendar generation."""
        X, y_single, _ = sample_features_targets
        window_size = 50
        
        result = generate_train_predict_calender(
            X, window_type='rolling', window_size=window_size
        )
        
        # Check that rolling windows maintain size
        for item in result[10:]:  # Skip first few which might be shorter
            train_start, train_end, predict_date = item
            
            # Calculate actual window size
            train_data = X[train_start:train_end]
            actual_size = len(train_data)
            
            # Should be approximately the requested size (within a few days for weekends)
            assert abs(actual_size - window_size) <= 5


class TestEWMTransformer:
    """Test Exponentially Weighted Moving Average transformer."""
    
    def test_ewm_transformer_basic(self, sample_features_targets):
        """Test basic EWM transformer functionality."""
        X, _, _ = sample_features_targets
        
        transformer = EWMTransformer(halflife=5)
        
        # Test fit (should return self)
        fitted = transformer.fit(X)
        assert fitted is transformer
        
        # Test transform
        result = transformer.transform(X)
        
        # Check output properties
        assert isinstance(result, pd.DataFrame)
        assert result.shape == X.shape
        assert result.columns.equals(X.columns)
        assert result.index.equals(X.index)
        
        # EWM should smooth the data (reduce volatility)
        original_std = X.std()
        smoothed_std = result.std()
        assert (smoothed_std <= original_std).all()
    
    def test_ewm_transformer_different_halflives(self, sample_features_targets):
        """Test EWM with different halflife parameters."""
        X, _, _ = sample_features_targets
        
        # Test with different halflives
        short_hl = EWMTransformer(halflife=2)
        long_hl = EWMTransformer(halflife=20)
        
        short_result = short_hl.fit_transform(X)
        long_result = long_hl.fit_transform(X)
        
        # Longer halflife should produce smoother results
        short_std = short_result.std()
        long_std = long_result.std()
        assert (long_std <= short_std).all()
    
    def test_ewm_transformer_numpy_input(self, sample_features_targets):
        """Test EWM transformer with numpy array input."""
        X, _, _ = sample_features_targets
        
        transformer = EWMTransformer(halflife=5)
        
        # Convert to numpy and transform
        X_numpy = X.values
        result = transformer.transform(X_numpy)
        
        # Should return DataFrame even with numpy input
        assert isinstance(result, pd.DataFrame)
        assert result.shape == X_numpy.shape


class TestXArrayOperations:
    """Test xarray-based result handling."""
    
    def test_create_results_xarray(self):
        """Test xarray dataset creation from simulation results."""
        # Create sample results
        dates = pd.date_range('2022-01-01', periods=100, freq='B')
        strategies = ['strategy_1', 'strategy_2']
        
        results_dict = {
            'returns': pd.DataFrame(
                np.random.normal(0.001, 0.02, (len(dates), len(strategies))),
                index=dates, columns=strategies
            ),
            'predictions': pd.DataFrame(
                np.random.normal(0, 0.01, (len(dates), len(strategies))),
                index=dates, columns=strategies
            )
        }
        
        result = create_results_xarray(results_dict, time_coord=dates, strategy_coord=strategies)
        
        # Check output structure
        assert isinstance(result, xr.Dataset)
        assert 'time' in result.coords
        assert 'strategy' in result.coords
        assert 'returns' in result.data_vars
        assert 'predictions' in result.data_vars
        
        # Check dimensions
        assert result.returns.shape == (len(dates), len(strategies))
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        # Create sample returns with known properties
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # One year of daily returns
        
        metrics = calculate_performance_metrics(returns)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'calmar_ratio', 'skewness', 'kurtosis'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert np.isfinite(metrics[metric])
        
        # Check reasonable ranges
        assert -1 <= metrics['max_drawdown'] <= 0  # Drawdown should be negative
        assert metrics['volatility'] > 0  # Volatility should be positive
    
    def test_create_correlation_matrix(self):
        """Test correlation matrix creation from xarray dataset."""
        # Create sample xarray dataset
        dates = pd.date_range('2022-01-01', periods=100, freq='B')
        strategies = ['strategy_1', 'strategy_2', 'strategy_3']
        
        data = np.random.normal(0.001, 0.02, (len(dates), len(strategies)))
        # Add some correlation between strategies
        data[:, 1] = 0.7 * data[:, 0] + 0.3 * data[:, 1]
        
        ds = xr.Dataset({
            'returns': (['time', 'strategy'], data)
        }, coords={'time': dates, 'strategy': strategies})
        
        corr_matrix = create_correlation_matrix(ds, ['returns'])
        
        # Check output structure
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (len(strategies), len(strategies))
        
        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), 1.0)
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(corr_matrix.values, corr_matrix.T.values)


class TestEducationalHelpers:
    """Test educational helper functions."""
    
    @patch('builtins.print')
    def test_explain_log_returns(self, mock_print):
        """Test log returns explanation function."""
        from utils_simulate import explain_log_returns
        
        explain_log_returns()
        
        # Should have printed educational content
        assert mock_print.called
        
        # Check that key concepts were mentioned
        printed_text = ' '.join([call[0][0] for call in mock_print.call_args_list])
        assert 'log returns' in printed_text.lower()
        assert 'time-additive' in printed_text.lower()
    
    @patch('builtins.print')
    def test_explain_walk_forward_analysis(self, mock_print):
        """Test walk-forward analysis explanation function."""
        from utils_simulate import explain_walk_forward_analysis
        
        explain_walk_forward_analysis()
        
        # Should have printed educational content
        assert mock_print.called
        
        # Check that key concepts were mentioned
        printed_text = ' '.join([call[0][0] for call in mock_print.call_args_list])
        assert 'walk-forward' in printed_text.lower()
        assert 'look-ahead bias' in printed_text.lower()
    
    @patch('builtins.print')
    def test_get_educational_help(self, mock_print):
        """Test educational help function."""
        from utils_simulate import get_educational_help
        
        # Test with known topic
        get_educational_help('log_returns')
        assert mock_print.called
        
        # Test with unknown topic
        mock_print.reset_mock()
        get_educational_help('unknown_topic')
        assert mock_print.called


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_log_returns_empty_dataframe(self):
        """Test log returns with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = log_returns(empty_df)
        assert result.empty
    
    def test_ewm_transformer_single_column(self):
        """Test EWM transformer with single column."""
        single_col = pd.DataFrame({
            'test': np.random.normal(0, 1, 100)
        }, index=pd.date_range('2022-01-01', periods=100))
        
        transformer = EWMTransformer(halflife=5)
        result = transformer.fit_transform(single_col)
        
        assert result.shape == single_col.shape
        assert list(result.columns) == ['test']
    
    def test_performance_metrics_all_zeros(self):
        """Test performance metrics with zero returns."""
        zero_returns = pd.Series(np.zeros(100))
        metrics = calculate_performance_metrics(zero_returns)
        
        # Should handle gracefully
        assert metrics['total_return'] == 0
        assert metrics['volatility'] == 0
        assert np.isnan(metrics['sharpe_ratio'])  # 0/0 case


# Integration tests
class TestIntegration:
    """Integration tests combining multiple functions."""
    
    @pytest.mark.integration
    def test_full_pipeline(self, sample_features_targets, sample_pipeline_config):
        """Test complete pipeline from data to results."""
        X, y_single, _ = sample_features_targets
        pipe_steps, param_grid = sample_pipeline_config
        
        # 1. Apply EWM transformation
        transformer = EWMTransformer(halflife=5)
        X_transformed = transformer.fit_transform(X)
        
        # 2. Generate training calendar
        calendar = generate_train_predict_calender(
            X_transformed, window_type='expanding', window_size=50
        )
        
        # 3. Calculate feature profiles
        profiles = feature_profiles(X_transformed, y_single)
        
        # Basic checks that pipeline completed without errors
        assert len(X_transformed) == len(X)
        assert len(calendar) > 0
        assert len(profiles) == len(X.columns)
        
        # Verify transformations maintained data integrity
        assert X_transformed.index.equals(X.index)
        assert X_transformed.columns.equals(X.columns)