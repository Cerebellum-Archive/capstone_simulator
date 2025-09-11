"""
Shared test fixtures and configuration for the test suite.

This module provides common test data, mock objects, and configuration
that can be reused across multiple test files.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import yfinance as yf


@pytest.fixture
def sample_price_data():
    """
    Generate sample price data for testing.
    
    Returns realistic-looking ETF price data with proper datetime index
    and typical price movements for testing purposes.
    """
    # Create 500 trading days of data (roughly 2 years)
    dates = pd.date_range(start='2022-01-01', periods=500, freq='B')  # Business days
    
    # Generate realistic price movements
    np.random.seed(42)  # For reproducible tests
    
    # Starting prices for different ETFs
    start_prices = {
        'SPY': 400.0,
        'QQQ': 350.0,
        'IWM': 200.0,
        'XLK': 150.0,
        'XLF': 35.0,
        'XLV': 125.0
    }
    
    price_data = {}
    for symbol, start_price in start_prices.items():
        # Generate log returns with realistic volatility
        daily_returns = np.random.normal(0.0008, 0.02, len(dates))  # ~20% annual vol
        log_prices = np.log(start_price) + np.cumsum(daily_returns)
        prices = np.exp(log_prices)
        price_data[symbol] = prices
    
    df = pd.DataFrame(price_data, index=dates)
    return df


@pytest.fixture
def sample_returns_data(sample_price_data):
    """Generate log returns from sample price data."""
    log_prices = np.log(sample_price_data)
    returns = log_prices - log_prices.shift(1)
    return returns.dropna()


@pytest.fixture
def sample_features_targets(sample_returns_data):
    """Split returns data into features and targets for testing."""
    feature_cols = ['XLK', 'XLF', 'XLV']  # Sector ETFs as features
    target_cols = ['SPY', 'QQQ', 'IWM']   # Major ETFs as targets
    
    X = sample_returns_data[feature_cols].copy()
    y_single = sample_returns_data['SPY'].copy()  # Single target
    y_multi = sample_returns_data[target_cols].copy()  # Multi-target
    
    return X, y_single, y_multi


@pytest.fixture
def mock_yfinance_download():
    """Mock yfinance download to avoid external API calls during testing."""
    def _mock_download(symbols, start=None, end=None, auto_adjust=True, group_by='column', progress=True, **kwargs):
        # Generate fake price data based on symbols
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Create realistic date range
        start_date = pd.Timestamp(start) if start else pd.Timestamp('2020-01-01')
        end_date = pd.Timestamp(end) if end else pd.Timestamp.now()
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Generate prices for each symbol
        np.random.seed(42)
        data = {}
        for symbol in symbols:
            base_price = 100 + hash(symbol) % 400  # Deterministic but varied start price
            returns = np.random.normal(0.0005, 0.015, len(dates))
            prices = base_price * np.exp(np.cumsum(returns))
            data[symbol] = prices
        
        df = pd.DataFrame(data, index=dates)
        
        # Return in yfinance format (nested dict structure)
        return {'Close': df}
    
    with patch('yfinance.download', side_effect=_mock_download):
        yield _mock_download


@pytest.fixture
def sample_pipeline_config():
    """Standard pipeline configuration for testing."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    
    pipe_steps = [
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ]
    
    param_grid = {
        'model__alpha': 1.0
    }
    
    return pipe_steps, param_grid


@pytest.fixture
def sample_metadata():
    """Sample simulation metadata for testing caching functions."""
    return {
        'data_source': {
            'etf_symbols': ['XLK', 'XLF', 'XLV'],
            'target_etf': 'SPY',
            'start_date': '2022-01-01',
            'end_date': '2023-12-31',
            'data_shapes': {
                'X_shape': (500, 3),
                'y_shape': (500,),
                'feature_columns': ['XLK', 'XLF', 'XLV'],
                'target_name': 'SPY'
            }
        },
        'training_params': {
            'window_size': 200,
            'window_type': 'expanding',
            'random_seed': 42
        },
        'model_config': {
            'pipe_steps': [('scaler', 'StandardScaler'), ('model', 'Ridge')],
            'param_grid': {'model__alpha': 1.0}
        },
        'simulation_info': {
            'tag': 'test_simulation',
            'creation_timestamp': '2024-01-15T10:30:00',
            'framework_version': '0.1.0'
        }
    }


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create temporary cache directory for testing file operations."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take >1 second)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "data_dependent: marks tests that require external data"
    )


# Suppress warnings for cleaner test output
@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress common warnings during testing."""
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=".*yfinance.*")