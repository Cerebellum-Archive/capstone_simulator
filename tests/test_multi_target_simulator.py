"""
Unit tests for multi_target_simulator.py module.

Tests cover multi-target prediction, portfolio construction, position sizing strategies,
enhanced metadata system, and comprehensive benchmarking functionality.
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

from multi_target_simulator import (
    generate_simulation_metadata, generate_simulation_hash,
    save_simulation_results, load_simulation_results,
    L_func_multi_target_equal_weight, L_func_multi_target_confidence_weighted,
    L_func_multi_target_long_short, EqualWeight, ConfidenceWeighted, LongShort,
    calculate_individual_position_weights, _determine_strategy_type
)


class TestMultiTargetMetadata:
    """Test enhanced metadata system for multi-target simulations."""
    
    def test_generate_simulation_metadata_multi_target(self, sample_features_targets, sample_pipeline_config):
        """Test comprehensive metadata generation for multi-target."""
        X, _, y_multi = sample_features_targets
        pipe_steps, param_grid = sample_pipeline_config
        
        metadata = generate_simulation_metadata(
            X=X, y_multi=y_multi, window_size=200, window_type='expanding',
            pipe_steps=pipe_steps, param_grid=param_grid, tag='test_multi_sim',
            train_frequency=30, etf_symbols=['XLK', 'XLF', 'XLV'], 
            target_etfs=['SPY', 'QQQ', 'IWM'], start_date='2022-01-01', 
            random_seed=42
        )
        
        # Check top-level structure
        expected_keys = ['data_source', 'training_params', 'model_config', 
                        'preprocessing', 'simulation_info']
        for key in expected_keys:
            assert key in metadata
        
        # Check data source information specific to multi-target
        data_source = metadata['data_source']
        assert data_source['etf_symbols'] == ['XLK', 'XLF', 'XLV']
        assert data_source['target_etfs'] == ['SPY', 'QQQ', 'IWM']
        assert data_source['data_shapes']['y_multi_shape'] == y_multi.shape
        assert set(data_source['data_shapes']['target_columns']) == set(y_multi.columns)
        
        # Check training parameters include frequency
        training = metadata['training_params']
        assert training['train_frequency'] == 30
        
        # Check data fingerprints exist
        fingerprint = data_source['data_fingerprint']
        assert 'y_head_hash' in fingerprint
        assert 'y_tail_hash' in fingerprint
        assert len(fingerprint['y_head_hash']) == 32  # MD5 length
    
    def test_generate_simulation_hash_multi_target(self, sample_features_targets, sample_pipeline_config):
        """Test simulation hash generation for multi-target with position function."""
        X, _, y_multi = sample_features_targets
        pipe_steps, param_grid = sample_pipeline_config
        
        hash_id, metadata = generate_simulation_hash(
            X=X, y_multi=y_multi, window_size=200, window_type='expanding',
            pipe_steps=pipe_steps, param_grid=param_grid, tag='test_sim',
            position_func=L_func_multi_target_equal_weight, position_params=[1.0],
            train_frequency=30
        )
        
        # Check hash and metadata structure
        assert isinstance(hash_id, str)
        assert len(hash_id) == 32
        assert isinstance(metadata, dict)
        
        # Check position strategy metadata
        pos_strategy = metadata['position_strategy']
        assert pos_strategy['function_name'] == 'L_func_multi_target_equal_weight'
        assert pos_strategy['parameters'] == [1.0]
        assert pos_strategy['strategy_type'] is not None
    
    def test_save_load_multi_target_results(self, temp_cache_dir):
        """Test saving and loading multi-target results with enhanced metadata."""
        # Create sample multi-target results
        dates = pd.date_range('2022-01-01', periods=100, freq='B')
        targets = ['SPY', 'QQQ', 'IWM']
        
        results_df = pd.DataFrame({
            f'{target}_prediction': np.random.normal(0.001, 0.01, len(dates))
            for target in targets
        }, index=dates)
        
        # Add portfolio columns
        results_df['portfolio_return'] = np.random.normal(0.001, 0.015, len(dates))
        results_df['leverage'] = np.random.uniform(0.5, 2.0, len(dates))
        
        # Sample metadata
        metadata = {
            'data_source': {
                'target_etfs': targets,
                'start_date': '2022-01-01',
                'end_date': '2023-12-31'
            },
            'position_strategy': {
                'function_name': 'L_func_multi_target_equal_weight',
                'strategy_type': 'EqualWeight'
            },
            'simulation_info': {
                'tag': 'multi_target_test',
                'creation_timestamp': datetime.now().isoformat()
            }
        }
        
        hash_id = 'multi_test_hash_123'
        tag = 'multi_target_test'
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_cache_dir.parent)
        
        try:
            # Test saving
            cache_path = save_simulation_results(results_df, hash_id, tag, metadata)
            assert os.path.exists(cache_path)
            
            # Test loading
            loaded_results, loaded_metadata = load_simulation_results(hash_id, tag)
            
            # Verify results
            pd.testing.assert_frame_equal(loaded_results, results_df)
            assert loaded_metadata['data_source']['target_etfs'] == targets
            
        finally:
            os.chdir(original_cwd)


class TestPositionSizingStrategies:
    """Test multi-target position sizing strategies."""
    
    def test_equal_weight_strategy(self):
        """Test equal weight position sizing strategy."""
        # Create sample predictions for multiple targets
        dates = pd.date_range('2022-01-01', periods=50, freq='B')
        predictions_df = pd.DataFrame({
            'SPY': np.random.normal(0.002, 0.01, len(dates)),   # Positive bias
            'QQQ': np.random.normal(-0.001, 0.01, len(dates)),  # Slight negative bias
            'IWM': np.random.normal(0.001, 0.01, len(dates))    # Neutral
        }, index=dates)
        
        # Test with default parameters
        leverage = L_func_multi_target_equal_weight(predictions_df, params=[])
        
        # Check output structure
        assert isinstance(leverage, pd.Series)
        assert len(leverage) == len(dates)
        assert leverage.index.equals(predictions_df.index)
        
        # Check leverage values are reasonable
        assert leverage.abs().max() <= 2.0
        
        # Test with custom base leverage
        leverage_custom = L_func_multi_target_equal_weight(predictions_df, params=[1.5])
        assert leverage_custom.abs().max() <= 1.5
    
    def test_confidence_weighted_strategy(self):
        """Test confidence-weighted position sizing strategy."""
        dates = pd.date_range('2022-01-01', periods=50, freq='B')
        
        # Create predictions with varying confidence (magnitude)
        predictions_df = pd.DataFrame({
            'SPY': np.concatenate([
                np.random.normal(0.005, 0.002, 25),  # High confidence period
                np.random.normal(0.001, 0.001, 25)   # Low confidence period
            ]),
            'QQQ': np.random.normal(0.002, 0.01, len(dates)),
            'IWM': np.random.normal(-0.001, 0.008, len(dates))
        }, index=dates)
        
        leverage = L_func_multi_target_confidence_weighted(predictions_df, params=[2.0])
        
        # Check output structure
        assert isinstance(leverage, pd.Series)
        assert len(leverage) == len(dates)
        
        # Leverage should vary with confidence
        assert leverage.std() > 0  # Should have variation
        assert leverage.abs().max() <= 2.0  # Should respect max leverage
    
    def test_long_short_strategy(self):
        """Test long-short position sizing strategy."""
        dates = pd.date_range('2022-01-01', periods=50, freq='B')
        
        # Create predictions with clear long/short signals
        predictions_df = pd.DataFrame({
            'SPY': np.concatenate([
                np.random.normal(0.01, 0.002, 25),   # Strong positive
                np.random.normal(-0.01, 0.002, 25)   # Strong negative
            ]),
            'QQQ': np.random.normal(0.005, 0.005, len(dates)),  # Moderate positive
            'IWM': np.random.normal(-0.003, 0.005, len(dates))  # Moderate negative
        }, index=dates)
        
        leverage = L_func_multi_target_long_short(predictions_df, params=[1.5])
        
        # Check output structure
        assert isinstance(leverage, pd.Series)
        assert len(leverage) == len(dates)
        
        # Should have both positive and negative values (long and short)
        assert leverage.min() < 0
        assert leverage.max() > 0
        
        # Should respect max leverage
        assert leverage.abs().max() <= 1.5
    
    def test_position_sizing_edge_cases(self):
        """Test position sizing with edge cases."""
        dates = pd.date_range('2022-01-01', periods=10, freq='B')
        
        # Test with all zero predictions
        zero_predictions = pd.DataFrame({
            'SPY': np.zeros(len(dates)),
            'QQQ': np.zeros(len(dates)),
            'IWM': np.zeros(len(dates))
        }, index=dates)
        
        leverage_eq = L_func_multi_target_equal_weight(zero_predictions)
        leverage_conf = L_func_multi_target_confidence_weighted(zero_predictions)
        leverage_ls = L_func_multi_target_long_short(zero_predictions)
        
        # Should handle zeros gracefully
        assert isinstance(leverage_eq, pd.Series)
        assert isinstance(leverage_conf, pd.Series)
        assert isinstance(leverage_ls, pd.Series)
        
        # Test with single column
        single_col = pd.DataFrame({
            'SPY': np.random.normal(0.001, 0.01, len(dates))
        }, index=dates)
        
        leverage_single = L_func_multi_target_equal_weight(single_col)
        assert len(leverage_single) == len(dates)


class TestPositionSizerClasses:
    """Test object-oriented position sizing classes."""
    
    def test_equal_weight_class(self):
        """Test EqualWeight position sizer class."""
        sizer = EqualWeight(base_leverage=1.2)
        
        # Test with sample predictions
        predictions = pd.Series([0.01, -0.005, 0.008, -0.002])
        weights = sizer.calculate_weights(predictions)
        
        # Check output
        assert isinstance(weights, pd.Series)
        assert len(weights) == len(predictions)
        
        # Check that it respects base leverage
        assert weights.abs().max() <= 1.2
    
    def test_confidence_weighted_class(self):
        """Test ConfidenceWeighted position sizer class."""
        sizer = ConfidenceWeighted(max_leverage=2.0, confidence_threshold=0.01)
        
        # Test with predictions of varying magnitude
        predictions = pd.Series([0.02, 0.001, -0.015, 0.0005])  # High and low confidence
        weights = sizer.calculate_weights(predictions)
        
        # Check output structure
        assert isinstance(weights, pd.Series)
        assert len(weights) == len(predictions)
        
        # Higher magnitude predictions should get higher weights
        high_pred_idx = predictions.abs().idxmax()
        low_pred_idx = predictions.abs().idxmin()
        assert weights.abs()[high_pred_idx] >= weights.abs()[low_pred_idx]
    
    def test_long_short_class(self):
        """Test LongShort position sizer class."""
        sizer = LongShort(max_leverage=1.8, confidence_threshold=0.005)
        
        # Test with mixed positive/negative predictions
        predictions = pd.Series([0.015, -0.012, 0.002, -0.008])
        weights = sizer.calculate_weights(predictions)
        
        # Check output structure
        assert isinstance(weights, pd.Series)
        assert len(weights) == len(predictions)
        
        # Should have both long and short positions
        assert weights.min() < 0
        assert weights.max() > 0
        
        # Should respect max leverage
        assert weights.abs().max() <= 1.8
    
    def test_position_sizer_with_insufficient_confidence(self):
        """Test position sizers when predictions don't meet confidence threshold."""
        sizer = ConfidenceWeighted(max_leverage=2.0, confidence_threshold=0.05)
        
        # Very low magnitude predictions (below threshold)
        low_predictions = pd.Series([0.001, -0.002, 0.0015, -0.0008])
        weights = sizer.calculate_weights(low_predictions)
        
        # Should result in low or zero weights
        assert weights.abs().max() < 0.1


class TestUtilityFunctions:
    """Test utility functions for multi-target simulation."""
    
    def test_calculate_individual_position_weights(self):
        """Test individual position weight calculation."""
        # Create sample prediction row
        predictions_row = [0.01, -0.005, 0.008]
        
        # Test with equal weight function
        weights = calculate_individual_position_weights(
            predictions_row, L_func_multi_target_equal_weight, [1.0]
        )
        
        # Should return a pandas Series
        assert isinstance(weights, pd.Series)
        assert len(weights) == len(predictions_row)
    
    def test_determine_strategy_type(self):
        """Test strategy type determination from function."""
        # Test with different position functions
        eq_type = _determine_strategy_type(L_func_multi_target_equal_weight)
        conf_type = _determine_strategy_type(L_func_multi_target_confidence_weighted)
        ls_type = _determine_strategy_type(L_func_multi_target_long_short)
        
        # Should return reasonable strategy types
        assert eq_type == 'EqualWeight'
        assert conf_type == 'ConfidenceWeighted'
        assert ls_type == 'LongShort'
        
        # Test with None
        none_type = _determine_strategy_type(None)
        assert none_type is None


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_position_sizing_with_nan_predictions(self):
        """Test position sizing functions handle NaN predictions."""
        dates = pd.date_range('2022-01-01', periods=20, freq='B')
        predictions_df = pd.DataFrame({
            'SPY': np.random.normal(0.001, 0.01, len(dates)),
            'QQQ': np.random.normal(0.001, 0.01, len(dates)),
            'IWM': np.random.normal(0.001, 0.01, len(dates))
        }, index=dates)
        
        # Introduce NaN values
        predictions_df.iloc[5:8] = np.nan
        
        # Should handle NaN gracefully
        leverage_eq = L_func_multi_target_equal_weight(predictions_df)
        leverage_conf = L_func_multi_target_confidence_weighted(predictions_df)
        leverage_ls = L_func_multi_target_long_short(predictions_df)
        
        # Check that functions don't crash
        assert isinstance(leverage_eq, pd.Series)
        assert isinstance(leverage_conf, pd.Series)
        assert isinstance(leverage_ls, pd.Series)
        
        # NaN positions should result in zero leverage or be handled appropriately
        assert len(leverage_eq) == len(dates)
    
    def test_position_sizing_with_extreme_predictions(self):
        """Test position sizing with extreme prediction values."""
        dates = pd.date_range('2022-01-01', periods=10, freq='B')
        
        # Create extreme predictions
        extreme_predictions = pd.DataFrame({
            'SPY': [0.5, -0.3, 0.1, -0.8],  # Very large predictions
            'QQQ': [1e-8, -1e-8, 0, 1e-10],  # Very small predictions
            'IWM': [np.inf, -np.inf, 0, 1.0]  # Infinite values
        }, index=dates[:4])
        
        # Should handle extreme values
        try:
            leverage_eq = L_func_multi_target_equal_weight(extreme_predictions)
            leverage_conf = L_func_multi_target_confidence_weighted(extreme_predictions)
            leverage_ls = L_func_multi_target_long_short(extreme_predictions)
            
            # Check that results are finite where possible
            assert np.isfinite(leverage_eq).any()
            assert np.isfinite(leverage_conf).any()
            assert np.isfinite(leverage_ls).any()
            
        except Exception as e:
            # If functions can't handle extreme values, they should fail gracefully
            assert "overflow" in str(e).lower() or "invalid" in str(e).lower()
    
    def test_empty_predictions_dataframe(self):
        """Test position sizing with empty predictions."""
        empty_df = pd.DataFrame()
        
        # Should handle empty input gracefully
        try:
            leverage_eq = L_func_multi_target_equal_weight(empty_df)
            assert len(leverage_eq) == 0
        except (ValueError, IndexError):
            # Acceptable to raise error for empty input
            pass


class TestBenchmarkIntegration:
    """Test integration with benchmarking systems."""
    
    def test_metadata_includes_benchmark_info(self, sample_features_targets, sample_pipeline_config):
        """Test that metadata can include benchmark configuration."""
        X, _, y_multi = sample_features_targets
        pipe_steps, param_grid = sample_pipeline_config
        
        # Include position function in metadata generation
        hash_id, metadata = generate_simulation_hash(
            X=X, y_multi=y_multi, window_size=100, window_type='expanding',
            pipe_steps=pipe_steps, param_grid=param_grid, tag='benchmark_test',
            position_func=L_func_multi_target_equal_weight, position_params=[1.0],
            train_frequency=30
        )
        
        # Check that position strategy info is captured
        pos_info = metadata['position_strategy']
        assert pos_info['function_name'] == 'L_func_multi_target_equal_weight'
        assert pos_info['parameters'] == [1.0]
        assert pos_info['strategy_type'] == 'EqualWeight'


class TestIntegration:
    """Integration tests for multi-target simulation."""
    
    @pytest.mark.integration
    def test_position_sizing_pipeline(self):
        """Test complete position sizing pipeline."""
        # Create realistic multi-target predictions
        dates = pd.date_range('2022-01-01', periods=100, freq='B')
        
        # Simulate predictions with different characteristics
        predictions_df = pd.DataFrame({
            'SPY': np.random.normal(0.001, 0.01, len(dates)),    # Market-like
            'QQQ': np.random.normal(0.002, 0.015, len(dates)),   # Tech-like (higher vol)
            'IWM': np.random.normal(0.0005, 0.02, len(dates))    # Small-cap-like (highest vol)
        }, index=dates)
        
        # Test all position sizing strategies
        strategies = {
            'equal_weight': L_func_multi_target_equal_weight,
            'confidence_weighted': L_func_multi_target_confidence_weighted,
            'long_short': L_func_multi_target_long_short
        }
        
        results = {}
        for name, func in strategies.items():
            leverage = func(predictions_df, params=[1.0])
            results[name] = leverage
            
            # Basic validation
            assert isinstance(leverage, pd.Series)
            assert len(leverage) == len(dates)
            assert leverage.index.equals(predictions_df.index)
        
        # Compare strategies
        eq_vol = results['equal_weight'].std()
        conf_vol = results['confidence_weighted'].std()
        ls_vol = results['long_short'].std()
        
        # Confidence-weighted should be more dynamic than equal weight
        assert conf_vol >= eq_vol
        
        # Long-short should have different characteristics
        assert results['long_short'].min() < 0  # Should have short positions
        assert results['long_short'].max() > 0  # Should have long positions
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_metadata_workflow(self, sample_features_targets, temp_cache_dir):
        """Test complete workflow with metadata persistence."""
        X, _, y_multi = sample_features_targets
        
        # Setup pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        pipe_steps = [('scaler', StandardScaler()), ('model', Ridge())]
        param_grid = {'model__alpha': 1.0}
        
        original_cwd = os.getcwd()
        os.chdir(temp_cache_dir.parent)
        
        try:
            # 1. Generate hash and metadata
            hash_id, metadata = generate_simulation_hash(
                X=X.iloc[:50], y_multi=y_multi.iloc[:50],
                window_size=20, window_type='expanding',
                pipe_steps=pipe_steps, param_grid=param_grid,
                tag='complete_test', train_frequency=10,
                position_func=L_func_multi_target_confidence_weighted,
                position_params=[1.5],
                etf_symbols=['XLK', 'XLF', 'XLV'],
                target_etfs=['SPY', 'QQQ', 'IWM'],
                start_date='2022-01-01', random_seed=42
            )
            
            # 2. Create mock results
            dates = X.iloc[:50].index
            mock_results = pd.DataFrame({
                'SPY_prediction': np.random.normal(0.001, 0.01, len(dates)),
                'QQQ_prediction': np.random.normal(0.002, 0.012, len(dates)),
                'IWM_prediction': np.random.normal(0.0005, 0.015, len(dates)),
                'portfolio_return': np.random.normal(0.0015, 0.01, len(dates))
            }, index=dates)
            
            # 3. Save with metadata
            cache_path = save_simulation_results(mock_results, hash_id, 'complete_test', metadata)
            
            # 4. Load and verify
            loaded_results, loaded_metadata = load_simulation_results(hash_id, 'complete_test')
            
            # Verify complete round trip
            pd.testing.assert_frame_equal(loaded_results, mock_results)
            
            # Verify all metadata components
            assert loaded_metadata['data_source']['etf_symbols'] == ['XLK', 'XLF', 'XLV']
            assert loaded_metadata['data_source']['target_etfs'] == ['SPY', 'QQQ', 'IWM']
            assert loaded_metadata['training_params']['window_size'] == 20
            assert loaded_metadata['position_strategy']['function_name'] == 'L_func_multi_target_confidence_weighted'
            assert loaded_metadata['position_strategy']['parameters'] == [1.5]
            
        finally:
            os.chdir(original_cwd)


class TestPerformance:
    """Performance tests for multi-target functions."""
    
    @pytest.mark.slow
    def test_position_sizing_performance(self):
        """Test that position sizing functions perform reasonably with large data."""
        import time
        
        # Create large dataset
        dates = pd.date_range('2020-01-01', periods=1000, freq='B')  # ~4 years of data
        predictions_df = pd.DataFrame({
            'SPY': np.random.normal(0.001, 0.01, len(dates)),
            'QQQ': np.random.normal(0.002, 0.012, len(dates)),
            'IWM': np.random.normal(0.0005, 0.015, len(dates)),
            'EFA': np.random.normal(0.0008, 0.018, len(dates)),
            'EEM': np.random.normal(0.001, 0.025, len(dates))
        }, index=dates)
        
        # Test performance of each strategy
        strategies = [
            L_func_multi_target_equal_weight,
            L_func_multi_target_confidence_weighted,
            L_func_multi_target_long_short
        ]
        
        for strategy in strategies:
            start_time = time.time()
            result = strategy(predictions_df, params=[1.0])
            elapsed = time.time() - start_time
            
            # Should complete quickly even with large dataset
            assert elapsed < 5.0  # 5 seconds max
            assert len(result) == len(dates)
            assert isinstance(result, pd.Series)