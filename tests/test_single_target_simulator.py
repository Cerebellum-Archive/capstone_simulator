"""
Unit tests for single_target_simulator.py module.

Tests cover single-target prediction, caching, metadata generation,
benchmarking, position sizing, and simulation engine functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import functions to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from single_target_simulator import (
    generate_simulation_metadata, generate_simulation_hash,
    save_simulation_results, load_simulation_results,
    calculate_information_ratio, Simulate, load_and_prepare_data,
    L_func_2, L_func_3, L_func_4
)


class TestMetadataAndCaching:
    """Test enhanced metadata and caching system."""
    
    def test_generate_simulation_metadata(self, sample_features_targets, sample_pipeline_config):
        """Test comprehensive metadata generation."""
        X, y_single, _ = sample_features_targets
        pipe_steps, param_grid = sample_pipeline_config
        
        metadata = generate_simulation_metadata(
            X=X, y=y_single, window_size=200, window_type='expanding',
            pipe_steps=pipe_steps, param_grid=param_grid, tag='test_sim',
            etf_symbols=['XLK', 'XLF', 'XLV'], target_etf='SPY',
            start_date='2022-01-01', random_seed=42
        )
        
        # Check top-level structure
        expected_keys = ['data_source', 'training_params', 'model_config', 
                        'preprocessing', 'simulation_info']
        for key in expected_keys:
            assert key in metadata
        
        # Check data source information
        data_source = metadata['data_source']
        assert data_source['etf_symbols'] == ['XLK', 'XLF', 'XLV']
        assert data_source['target_etf'] == 'SPY'
        assert data_source['start_date'] == '2022-01-01'
        assert data_source['data_shapes']['X_shape'] == X.shape
        assert data_source['data_shapes']['y_shape'] == y_single.shape
        
        # Check training parameters
        training = metadata['training_params']
        assert training['window_size'] == 200
        assert training['window_type'] == 'expanding'
        assert training['random_seed'] == 42
        
        # Check simulation info
        sim_info = metadata['simulation_info']
        assert sim_info['tag'] == 'test_sim'
        assert sim_info['framework_version'] == '0.1.0'
        assert sim_info['simulation_type'] == 'single_target'
        assert 'creation_timestamp' in sim_info
    
    def test_generate_simulation_hash(self, sample_features_targets, sample_pipeline_config):
        """Test simulation hash generation with metadata."""
        X, y_single, _ = sample_features_targets
        pipe_steps, param_grid = sample_pipeline_config
        
        hash_id, metadata = generate_simulation_hash(
            X=X, y=y_single, window_size=200, window_type='expanding',
            pipe_steps=pipe_steps, param_grid=param_grid, tag='test_sim'
        )
        
        # Check hash properties
        assert isinstance(hash_id, str)
        assert len(hash_id) == 32  # MD5 hash length
        
        # Check metadata is returned
        assert isinstance(metadata, dict)
        assert 'data_source' in metadata
        
        # Test reproducibility - same inputs should produce same hash
        hash_id2, _ = generate_simulation_hash(
            X=X, y=y_single, window_size=200, window_type='expanding',
            pipe_steps=pipe_steps, param_grid=param_grid, tag='test_sim'
        )
        assert hash_id == hash_id2
        
        # Different inputs should produce different hash
        hash_id3, _ = generate_simulation_hash(
            X=X, y=y_single, window_size=300, window_type='expanding',  # Different window_size
            pipe_steps=pipe_steps, param_grid=param_grid, tag='test_sim'
        )
        assert hash_id != hash_id3
    
    def test_save_load_simulation_results(self, temp_cache_dir, sample_metadata):
        """Test saving and loading simulation results with metadata."""
        # Create sample results
        dates = pd.date_range('2022-01-01', periods=100, freq='B')
        results_df = pd.DataFrame({
            'prediction': np.random.normal(0.001, 0.01, len(dates))
        }, index=dates)
        
        hash_id = 'test_hash_123'
        tag = 'test_simulation'
        
        # Change to temp directory for testing
        original_cwd = os.getcwd()
        os.chdir(temp_cache_dir.parent)
        
        try:
            # Test saving
            cache_path = save_simulation_results(results_df, hash_id, tag, sample_metadata)
            assert os.path.exists(cache_path)
            
            # Test loading
            loaded_results, loaded_metadata = load_simulation_results(hash_id, tag)
            
            # Check results
            pd.testing.assert_frame_equal(loaded_results, results_df)
            
            # Check metadata
            assert loaded_metadata is not None
            assert loaded_metadata['data_source']['target_etf'] == sample_metadata['data_source']['target_etf']
            
        finally:
            os.chdir(original_cwd)
    
    def test_load_nonexistent_simulation(self):
        """Test loading non-existent simulation returns None."""
        result_df, metadata = load_simulation_results('nonexistent_hash', 'nonexistent_tag')
        assert result_df is None
        assert metadata is None


class TestBenchmarkingFunctions:
    """Test benchmarking and performance calculation functions."""
    
    def test_calculate_information_ratio(self):
        """Test Information Ratio calculation."""
        # Create sample returns with known relationship
        np.random.seed(42)
        benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 252))
        
        # Strategy with slight outperformance
        strategy_returns = benchmark_returns + np.random.normal(0.0002, 0.005, 252)
        
        ir = calculate_information_ratio(strategy_returns, benchmark_returns)
        
        # Should be a finite number
        assert np.isfinite(ir)
        
        # Should be positive (strategy outperforms on average)
        assert ir > 0
        
        # Test with identical returns (should be ~0)
        ir_identical = calculate_information_ratio(benchmark_returns, benchmark_returns)
        assert abs(ir_identical) < 0.1  # Should be close to zero
    
    def test_calculate_information_ratio_edge_cases(self):
        """Test Information Ratio with edge cases."""
        dates = pd.date_range('2022-01-01', periods=100, freq='D')
        
        # Test with zero tracking error (identical returns)
        identical_returns = pd.Series(np.ones(100) * 0.001, index=dates)
        ir = calculate_information_ratio(identical_returns, identical_returns)
        assert ir == 0.0
        
        # Test with misaligned indices
        strategy = pd.Series([0.01, 0.02, 0.03], index=[1, 2, 3])
        benchmark = pd.Series([0.005, 0.015, 0.025], index=[2, 3, 4])
        ir = calculate_information_ratio(strategy, benchmark)
        assert np.isfinite(ir)  # Should handle alignment gracefully
    
    def test_calculate_information_ratio_error_handling(self):
        """Test Information Ratio error handling."""
        # Test with empty series
        empty_series = pd.Series(dtype=float)
        ir = calculate_information_ratio(empty_series, empty_series)
        assert ir == 0.0


class TestPositionSizingFunctions:
    """Test position sizing functions."""
    
    def test_L_func_2_basic(self):
        """Test basic position sizing function L_func_2."""
        # Create sample prediction data
        dates = pd.date_range('2022-01-01', periods=50, freq='B')
        df = pd.DataFrame({
            'predicted': np.random.normal(0.001, 0.01, len(dates))
        }, index=dates)
        
        result = L_func_2(df, pred_col='predicted', params=[])
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert 'leverage' in result.columns
        
        # Check leverage values are reasonable
        assert result['leverage'].abs().max() <= 2.0  # Reasonable leverage limits
    
    def test_L_func_3_index_predictions(self):
        """Test position sizing with index predictions."""
        dates = pd.date_range('2022-01-01', periods=50, freq='B')
        df = pd.DataFrame({
            'preds_index': np.random.normal(0.001, 0.01, len(dates)).cumsum()  # Cumulative for index
        }, index=dates)
        
        result = L_func_3(df, pred_col='preds_index', params=[])
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert 'leverage' in result.columns
    
    def test_L_func_4_dataset_input(self):
        """Test position sizing with xarray Dataset input."""
        import xarray as xr
        
        dates = pd.date_range('2022-01-01', periods=50, freq='B')
        data = np.random.normal(0.001, 0.01, len(dates))
        
        ds = xr.Dataset({
            'predictions': (['time'], data)
        }, coords={'time': dates})
        
        result = L_func_4(ds, params=[])
        
        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(dates)


class TestSimulationEngine:
    """Test the core simulation engine."""
    
    @pytest.mark.slow
    def test_simulate_basic_functionality(self, sample_features_targets, sample_pipeline_config):
        """Test basic Simulate function functionality."""
        X, y_single, _ = sample_features_targets
        pipe_steps, param_grid = sample_pipeline_config
        
        # Use smaller window for faster testing
        result_df, fit_list = Simulate(
            X=X.iloc[:100], y=y_single.iloc[:100],  # Smaller dataset
            window_size=30, window_type='expanding',
            pipe_steps=pipe_steps, param_grid=param_grid,
            tag='test_simulation'
        )
        
        # Check output structure
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(fit_list, dict)
        
        # Check that predictions were generated
        if len(result_df) > 0:  # If we have enough data
            assert 'prediction' in result_df.columns
            assert not result_df['prediction'].isna().all()
    
    def test_simulate_insufficient_data(self, sample_features_targets, sample_pipeline_config):
        """Test Simulate with insufficient data."""
        X, y_single, _ = sample_features_targets
        pipe_steps, param_grid = sample_pipeline_config
        
        # Try with window size larger than data
        result_df, fit_list = Simulate(
            X=X.iloc[:10], y=y_single.iloc[:10],  # Very small dataset
            window_size=50, window_type='expanding',  # Window larger than data
            pipe_steps=pipe_steps, param_grid=param_grid,
            tag='insufficient_data_test'
        )
        
        # Should return empty results gracefully
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0
    
    def test_simulate_rolling_window(self, sample_features_targets, sample_pipeline_config):
        """Test Simulate with rolling window."""
        X, y_single, _ = sample_features_targets
        pipe_steps, param_grid = sample_pipeline_config
        
        result_df, fit_list = Simulate(
            X=X.iloc[:100], y=y_single.iloc[:100],
            window_size=30, window_type='rolling',  # Rolling instead of expanding
            pipe_steps=pipe_steps, param_grid=param_grid,
            tag='rolling_test'
        )
        
        # Should work with rolling windows
        assert isinstance(result_df, pd.DataFrame)


class TestDataLoading:
    """Test data loading and preparation functions."""
    
    @patch('yfinance.download')
    def test_load_and_prepare_data(self, mock_download):
        """Test data loading and preparation."""
        # Mock yfinance download
        dates = pd.date_range('2022-01-01', periods=100, freq='B')
        mock_price_data = pd.DataFrame({
            'SPY': np.random.uniform(400, 450, len(dates)),
            'XLK': np.random.uniform(140, 160, len(dates)),
            'XLF': np.random.uniform(30, 40, len(dates))
        }, index=dates)
        
        mock_download.return_value = {'Close': mock_price_data}
        
        # Test the function
        etf_list = ['SPY', 'XLK', 'XLF']
        target_etf = 'SPY'
        
        X, y, all_returns = load_and_prepare_data(etf_list, target_etf, start_date='2022-01-01')
        
        # Check output structure
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(all_returns, pd.DataFrame)
        
        # Check data alignment - target should be shifted
        assert len(X) == len(y)
        
        # Y should be next-day returns of target
        assert y.name == target_etf
        
        # Verify yfinance was called correctly
        mock_download.assert_called_once_with(etf_list, start='2022-01-01', auto_adjust=True)
    
    def test_load_and_prepare_data_feature_target_separation(self, mock_yfinance_download):
        """Test that features and targets are properly separated."""
        etf_list = ['SPY', 'QQQ', 'XLK', 'XLF']
        target_etf = 'SPY'
        
        X, y, all_returns = load_and_prepare_data(etf_list, target_etf)
        
        # Features should not include the target
        assert target_etf not in X.columns
        
        # All non-target ETFs should be features
        expected_features = [etf for etf in etf_list if etf != target_etf]
        assert set(X.columns) == set(expected_features)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_simulate_with_nan_data(self, sample_pipeline_config):
        """Test Simulate handles NaN data gracefully."""
        # Create data with NaN values
        dates = pd.date_range('2022-01-01', periods=50, freq='B')
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 0.02, len(dates)),
            'feature2': np.random.normal(0, 0.02, len(dates))
        }, index=dates)
        
        # Introduce some NaN values
        X.iloc[10:15, 0] = np.nan
        
        y = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
        y[20:25] = np.nan
        
        pipe_steps, param_grid = sample_pipeline_config
        
        # Should handle NaN data without crashing
        result_df, fit_list = Simulate(
            X=X, y=y, window_size=20, window_type='expanding',
            pipe_steps=pipe_steps, param_grid=param_grid,
            tag='nan_test'
        )
        
        # Should return some result (even if limited due to NaN)
        assert isinstance(result_df, pd.DataFrame)
    
    def test_information_ratio_with_extreme_values(self):
        """Test Information Ratio with extreme values."""
        # Very high volatility
        high_vol_returns = pd.Series(np.random.normal(0, 0.1, 100))  # 100% annual vol
        normal_returns = pd.Series(np.random.normal(0, 0.02, 100))   # 20% annual vol
        
        ir = calculate_information_ratio(high_vol_returns, normal_returns)
        assert np.isfinite(ir)
        
        # Very low volatility (near zero)
        low_vol_returns = pd.Series(np.random.normal(0.001, 0.0001, 100))
        ir_low = calculate_information_ratio(low_vol_returns, normal_returns)
        assert np.isfinite(ir_low)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_single_target_workflow(self, mock_yfinance_download, temp_cache_dir):
        """Test complete single-target simulation workflow."""
        # Setup
        etf_list = ['SPY', 'XLK', 'XLF', 'XLV']
        target_etf = 'SPY'
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_cache_dir.parent)
        
        try:
            # 1. Load and prepare data
            X, y, all_returns = load_and_prepare_data(etf_list, target_etf, start_date='2022-01-01')
            
            # 2. Setup pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import Ridge
            
            pipe_steps = [('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]
            param_grid = {'model__alpha': 1.0}
            
            # 3. Generate metadata and hash
            hash_id, metadata = generate_simulation_hash(
                X=X.iloc[:100], y=y.iloc[:100],  # Smaller dataset for speed
                window_size=30, window_type='expanding',
                pipe_steps=pipe_steps, param_grid=param_grid,
                tag='integration_test',
                etf_symbols=[e for e in etf_list if e != target_etf],
                target_etf=target_etf,
                start_date='2022-01-01'
            )
            
            # 4. Run simulation
            results_df, fit_list = Simulate(
                X=X.iloc[:100], y=y.iloc[:100],
                window_size=30, window_type='expanding',
                pipe_steps=pipe_steps, param_grid=param_grid,
                tag='integration_test'
            )
            
            # 5. Save results
            if len(results_df) > 0:
                cache_path = save_simulation_results(results_df, hash_id, 'integration_test', metadata)
                
                # 6. Load results back
                loaded_results, loaded_metadata = load_simulation_results(hash_id, 'integration_test')
                
                # Verify round-trip
                pd.testing.assert_frame_equal(loaded_results, results_df)
                assert loaded_metadata['data_source']['target_etf'] == target_etf
            
            # Basic integration checks
            assert len(X) > 0
            assert len(y) > 0
            assert isinstance(hash_id, str)
            assert isinstance(metadata, dict)
            
        finally:
            os.chdir(original_cwd)


# Performance tests
class TestPerformance:
    """Performance-related tests."""
    
    @pytest.mark.slow
    def test_simulate_performance_reasonable(self, sample_features_targets, sample_pipeline_config):
        """Test that simulation completes in reasonable time."""
        import time
        
        X, y_single, _ = sample_features_targets
        pipe_steps, param_grid = sample_pipeline_config
        
        start_time = time.time()
        
        # Run on moderate-sized dataset
        result_df, fit_list = Simulate(
            X=X.iloc[:200], y=y_single.iloc[:200],
            window_size=50, window_type='expanding',
            pipe_steps=pipe_steps, param_grid=param_grid,
            tag='performance_test'
        )
        
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert elapsed < 30.0  # 30 seconds max for 200 data points
        
        # Should produce some output
        assert isinstance(result_df, pd.DataFrame)