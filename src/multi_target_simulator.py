# -*- coding: utf-8 -*-
"""
Multi-Target Quantitative Trading Simulation Framework

Purpose:
This script provides a framework for backtesting quantitative trading strategies
using multi-target regression. It leverages sklearn's multi-target capabilities
to predict returns for multiple ETFs simultaneously, enabling more sophisticated
portfolio construction and risk management.

Key Enhancements over Single-Target:
- Multi-Target Prediction: Predicts returns for multiple ETFs in a single model
- Portfolio Construction: Combines predictions across targets for position sizing
- Cross-Asset Analytics: Analyzes performance and correlations across multiple targets
- Enhanced Risk Management: Diversification benefits from multi-target approach

Core Components:
- Data Loading: Fetches multiple ETF price data from Yahoo Finance
- Multi-Target Feature/Target Engineering: Sets up prediction for multiple ETFs
- Multi-Target Simulation Engine: Trains models to predict multiple targets
- Portfolio Construction: Combines multi-target predictions into positions
- Multi-Target Performance Analytics: Comprehensive performance analysis

How to Use:
1. Configure target ETFs and features in the main() function
2. Choose multi-target compatible estimators (most sklearn regressors work)
3. Run the script to get portfolio strategies based on multi-target predictions
"""

import os
import warnings
import sys
from datetime import datetime
import time
import pytz
import numpy as np
import pandas as pd
import xarray as xr
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
from IPython.display import display
import pickle
import hashlib
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252

def format_benchmark_name(benchmark_col: str) -> str:
    """Format benchmark column name for display with abbreviations."""
    name = benchmark_col.replace('benchmark_', '').replace('_', ' ').title()
    # Apply abbreviations
    if name == 'Equal Weight Targets':
        return 'EQ Weight'
    return name
QUARTERLY_WINDOW_DAYS = 63
MONTHLY_RETRAIN_DAYS = 21
WEEKLY_RETRAIN_DAYS = 5
DAILY_RETRAIN_DAYS = 1
BASE_LEVERAGE_DEFAULT = 1.0
MAX_LEVERAGE_DEFAULT = 2.0
LONG_SHORT_LEVERAGE_DEFAULT = 1.0

@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    train_frequency: str = 'monthly'
    window_size: int = 400
    window_type: str = 'expanding'
    start_date: str = '2010-01-01'
    use_cache: bool = True
    force_retrain: bool = False
    csv_output_dir: str = '/Volumes/ext_2t/ERM3_Data/stock_data/csv'
    
    def __post_init__(self):
        if self.train_frequency not in ['daily', 'weekly', 'monthly']:
            raise ValueError(f"Invalid train_frequency: {self.train_frequency}")
        if self.window_type not in ['expanding', 'rolling']:
            raise ValueError(f"Invalid window_type: {self.window_type}")

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark calculations."""
    include_transaction_costs: bool = True
    rebalancing_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly'
    benchmark_types: List[str] = None
    volatility_window: int = 63  # Days for volatility calculation
    
    def __post_init__(self):
        if self.benchmark_types is None:
            # Don't filter benchmarks by default - let strategy type determine appropriate ones
            self.benchmark_types = None
        if self.rebalancing_frequency not in ['daily', 'weekly', 'monthly']:
            raise ValueError(f"Invalid rebalancing_frequency: {self.rebalancing_frequency}")

# Scikit-learn and statsmodels
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as rmse, mean_absolute_error as mae, r2_score
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Utility functions from our custom library  
from utils_simulate import (
    simplify_teos, log_returns, generate_train_predict_calender,
    StatsModelsWrapper_with_OLS, p_by_year, EWMTransformer,
    create_results_xarray, plot_xarray_results, calculate_performance_metrics
)



# --- Caching and Performance Utilities ---

def generate_simulation_metadata(X, y_multi, window_size, window_type, pipe_steps, param_grid, 
                               tag, position_func, position_params, train_frequency,
                               etf_symbols=None, target_etfs=None, start_date=None, 
                               random_seed=None, feature_engineering_steps=None):
    """
    Generate comprehensive metadata for full simulation reconstruction.
    
    This enhanced version stores all information needed to perfectly reproduce
    a simulation, enabling full reproducibility and audit trails.
    
    Educational Note:
        Reproducibility is crucial in quantitative finance for:
        1. Regulatory compliance and audit requirements
        2. Model validation and backtesting verification  
        3. Research collaboration and peer review
        4. Production deployment confidence
    
    Args:
        X (pd.DataFrame): Feature matrix
        y_multi (pd.DataFrame): Multi-target variables  
        window_size (int): Training window size
        window_type (str): 'expanding' or 'rolling'
        pipe_steps (list): sklearn Pipeline steps
        param_grid (dict): Model hyperparameters
        tag (str): Simulation identifier
        position_func (callable): Position sizing function
        position_params (list): Position sizing parameters
        train_frequency (int): Retraining frequency
        etf_symbols (list, optional): ETF symbols used as features
        target_etfs (list, optional): ETF symbols used as targets
        start_date (str, optional): Data start date
        random_seed (int, optional): Random seed for reproducibility
        feature_engineering_steps (dict, optional): Feature preprocessing metadata
        
    Returns:
        dict: Complete simulation metadata for reconstruction
    """
    metadata = {
        # Data source information for reconstruction
        'data_source': {
            'etf_symbols': etf_symbols,
            'target_etfs': target_etfs,
            'start_date': start_date,
            'end_date': X.index[-1].strftime('%Y-%m-%d') if hasattr(X.index[-1], 'strftime') else str(X.index[-1]),
            'data_shapes': {
                'X_shape': X.shape, 
                'y_multi_shape': y_multi.shape,
                'feature_columns': list(X.columns),
                'target_columns': list(y_multi.columns)
            },
            'data_fingerprint': {
                'X_head_hash': hashlib.md5(str(X.head().values).encode()).hexdigest(),
                'X_tail_hash': hashlib.md5(str(X.tail().values).encode()).hexdigest(),
                'y_head_hash': hashlib.md5(str(y_multi.head().values).encode()).hexdigest(),
                'y_tail_hash': hashlib.md5(str(y_multi.tail().values).encode()).hexdigest()
            }
        },
        
        # Training configuration
        'training_params': {
            'window_size': window_size,
            'window_type': window_type,
            'train_frequency': train_frequency,
            'random_seed': random_seed
        },
        
        # Model configuration with full reproducibility
        'model_config': {
            'pipe_steps': pipe_steps,  # Store actual pipeline configuration
            'param_grid': param_grid,
            'pipeline_string': str(pipe_steps)  # Human-readable backup
        },
        
        # Position sizing strategy
        'position_strategy': {
            'function_name': position_func.__name__ if position_func else None,
            'parameters': position_params,
            'strategy_type': _determine_strategy_type(position_func) if position_func else None
        },
        
        # Feature engineering and preprocessing
        'preprocessing': {
            'feature_engineering_steps': feature_engineering_steps or {},
            'data_transformations': {
                'log_returns_applied': True,  # Assumption based on framework
                'timezone_normalization': True,
                'missing_data_handling': 'dropna'
            }
        },
        
        # Simulation metadata
        'simulation_info': {
            'tag': tag,
            'creation_timestamp': datetime.now().isoformat(),
            'framework_version': '0.1.0',  # From package version
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
    }
    
    return metadata

def generate_simulation_hash(X, y_multi, window_size, window_type, pipe_steps, param_grid, tag, 
                           position_func, position_params, train_frequency,
                           etf_symbols=None, target_etfs=None, start_date=None, 
                           random_seed=None, feature_engineering_steps=None):
    """
    Generate a unique hash for simulation parameters to enable caching.
    
    This function now uses the enhanced metadata system while maintaining
    backward compatibility for caching purposes.
    """
    # Generate full metadata
    metadata = generate_simulation_metadata(
        X, y_multi, window_size, window_type, pipe_steps, param_grid, tag,
        position_func, position_params, train_frequency, etf_symbols, target_etfs,
        start_date, random_seed, feature_engineering_steps
    )
    
    # Create hash from key parameters (excluding timestamps)
    hash_components = [
        str(metadata['data_source']['data_shapes']),
        str(metadata['training_params']),
        str(metadata['model_config']['pipeline_string']),
        str(metadata['model_config']['param_grid']),
        str(metadata['position_strategy']),
        metadata['data_source']['data_fingerprint']['X_head_hash'],
        metadata['data_source']['data_fingerprint']['X_tail_hash'],
        metadata['data_source']['data_fingerprint']['y_head_hash'],
        metadata['data_source']['data_fingerprint']['y_tail_hash']
    ]
    
    hash_string = '_'.join(hash_components)
    return hashlib.md5(hash_string.encode()).hexdigest(), metadata

def save_simulation_results(regout_df, simulation_hash, tag, metadata=None):
    """
    Save simulation results and metadata to disk for future reuse and full reconstruction.
    
    Educational Note:
        Saving comprehensive metadata enables complete reproducibility,
        which is essential for regulatory compliance, peer review, and 
        production deployment in quantitative finance.
    
    Args:
        regout_df (pd.DataFrame): Simulation results
        simulation_hash (str): Unique simulation identifier
        tag (str): Human-readable simulation tag
        metadata (dict, optional): Complete simulation metadata for reconstruction
        
    Returns:
        str: Path to saved cache file
    """
    os.makedirs('cache', exist_ok=True)
    cache_filename = f'cache/simulation_{simulation_hash}_{tag}.pkl'
    
    # Package results with metadata
    cache_data = {
        'results': regout_df,
        'metadata': metadata,
        'cache_version': '2.0',  # Version for backward compatibility
        'save_timestamp': datetime.now().isoformat()
    }
    
    with open(cache_filename, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Saved simulation results with metadata: {cache_filename}")
    if metadata:
        print(f"  - Simulation recreatable from: {metadata['data_source']['start_date']} to {metadata['data_source']['end_date']}")
        print(f"  - Features: {len(metadata['data_source']['data_shapes']['feature_columns'])} columns")
        print(f"  - Targets: {len(metadata['data_source']['data_shapes']['target_columns'])} columns")
    
    return cache_filename

def load_simulation_results(simulation_hash, tag):
    """
    Load cached simulation results and metadata if they exist.
    
    Returns:
        tuple: (results_df, metadata_dict) or (None, None) if not found
    """
    cache_filename = f'cache/simulation_{simulation_hash}_{tag}.pkl'
    
    if os.path.exists(cache_filename):
        print(f"Loading cached results: {cache_filename}")
        with open(cache_filename, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Handle both old and new cache formats
        if isinstance(cache_data, dict) and 'cache_version' in cache_data:
            # New format with metadata
            print(f"  - Loaded enhanced cache (version {cache_data['cache_version']})")
            if cache_data.get('metadata'):
                metadata = cache_data['metadata']
                print(f"  - Original simulation: {metadata['simulation_info']['creation_timestamp']}")
                print(f"  - Data period: {metadata['data_source']['start_date']} to {metadata['data_source']['end_date']}")
            return cache_data['results'], cache_data.get('metadata')
        else:
            # Legacy format - just results
            print("  - Loaded legacy cache (no metadata)")
            return cache_data, None
    
    return None, None

def generate_train_predict_calendar_with_frequency(X, train_frequency, window_type, window_size):
    """
    Enhanced calendar generation with configurable training frequency.
    
    Args:
        X: Feature DataFrame
        train_frequency: 'daily', 'weekly', 'monthly' - how often to retrain (required)
        window_type: 'expanding' or 'rolling' (required)
        window_size: Size of training window (required)
    
    Returns:
        List of (train_start, train_end, predict_date) tuples
    """
    dates = X.index
    date_ranges = []
    
    # Determine retraining frequency
    frequency_map = {
        'weekly': WEEKLY_RETRAIN_DAYS,
        'monthly': MONTHLY_RETRAIN_DAYS,
        'daily': DAILY_RETRAIN_DAYS
    }
    retrain_step = frequency_map.get(train_frequency, DAILY_RETRAIN_DAYS)
    
    # Generate prediction dates based on frequency
    prediction_indices = list(range(window_size, len(dates), retrain_step))
    
    # Add final date if not included
    if prediction_indices[-1] < len(dates) - 1:
        prediction_indices.append(len(dates) - 1)
    
    for pred_idx in prediction_indices:
        prediction_date = dates[pred_idx]
        
        if window_type == 'expanding':
            start_training = dates[0]
            end_training = dates[pred_idx - 1]
        else:  # rolling
            start_idx = max(0, pred_idx - window_size)
            start_training = dates[start_idx]
            end_training = dates[pred_idx - 1]
        
        date_ranges.append((start_training, end_training, prediction_date))
    
    return date_ranges


# --- Benchmarking Framework ---

class BenchmarkCalculator(ABC):
    """Abstract base class for benchmark calculations."""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
    
    @abstractmethod
    def calculate_returns(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate benchmark returns for given dates."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return description of this benchmark."""
        pass
    
    def get_name(self) -> str:
        """Return short name for this benchmark."""
        return self.__class__.__name__.replace('Benchmark', '').lower()

class EqualWeightBenchmark(BenchmarkCalculator):
    """Equal-weight benchmark of specified ETFs."""
    
    def __init__(self, etfs: List[str], config: BenchmarkConfig = None):
        super().__init__(config)
        self.etfs = etfs
    
    def calculate_returns(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
        if not self.etfs or not any(etf in data.columns for etf in self.etfs):
            logger.warning(f"ETFs {self.etfs} not found in data columns")
            return pd.Series(0.0, index=dates)
        
        # Filter to available ETFs
        available_etfs = [etf for etf in self.etfs if etf in data.columns]
        
        equal_weights = 1.0 / len(available_etfs)
        weighted_returns = data[available_etfs].mul(equal_weights, axis=1).sum(axis=1)
        return weighted_returns.reindex(dates).fillna(0)
    
    def get_description(self) -> str:
        return f"Equal-weight portfolio of {len(self.etfs)} ETFs: {', '.join(self.etfs)}"

class SingleETFBenchmark(BenchmarkCalculator):
    """Single ETF buy-and-hold benchmark."""
    
    def __init__(self, etf_symbol: str, config: BenchmarkConfig = None):
        super().__init__(config)
        self.etf_symbol = etf_symbol
    
    def calculate_returns(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
        if self.etf_symbol not in data.columns:
            logger.warning(f"ETF {self.etf_symbol} not found in data, using zeros")
            return pd.Series(0.0, index=dates)
        
        return data[self.etf_symbol].reindex(dates).fillna(0)
    
    def get_description(self) -> str:
        return f"Buy-and-hold {self.etf_symbol}"

class RiskParityBenchmark(BenchmarkCalculator):
    """Risk parity weighted benchmark (inverse volatility weighting)."""
    
    def __init__(self, etfs: List[str], config: BenchmarkConfig = None):
        super().__init__(config)
        self.etfs = etfs
    
    def calculate_returns(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
        if not self.etfs or not any(etf in data.columns for etf in self.etfs):
            logger.warning(f"ETFs {self.etfs} not found in data columns")
            return pd.Series(0.0, index=dates)
        
        # Filter to available ETFs
        available_etfs = [etf for etf in self.etfs if etf in data.columns]
        etf_data = data[available_etfs]
        
        # Calculate rolling volatilities
        volatilities = etf_data.rolling(window=self.config.volatility_window).std()
        
        # Inverse volatility weights (higher vol gets lower weight)
        inv_vol_weights = (1 / volatilities).div((1 / volatilities).sum(axis=1), axis=0)
        
        # Calculate weighted returns (use lagged weights to avoid lookahead bias)
        weighted_returns = (etf_data * inv_vol_weights.shift(1)).sum(axis=1)
        
        return weighted_returns.reindex(dates).fillna(0)
    
    def get_description(self) -> str:
        return f"Risk parity portfolio of {len(self.etfs)} ETFs (inverse volatility weighted)"

class LongShortRandomBenchmark(BenchmarkCalculator):
    """Random long-short benchmark for comparison with long-short strategies."""
    
    def __init__(self, etfs: List[str], n_long: int = 1, n_short: int = 1, 
                 leverage: float = 1.0, config: BenchmarkConfig = None):
        super().__init__(config)
        self.etfs = etfs
        self.n_long = n_long
        self.n_short = n_short
        self.leverage = leverage
        self.random_seed = 42  # For reproducibility
    
    def calculate_returns(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
        if not self.etfs or not any(etf in data.columns for etf in self.etfs):
            logger.warning(f"ETFs {self.etfs} not found in data columns")
            return pd.Series(0.0, index=dates)
        
        # Filter to available ETFs
        available_etfs = [etf for etf in self.etfs if etf in data.columns]
        
        np.random.seed(self.random_seed)
        returns = []
        
        # Determine rebalancing frequency
        rebal_freq = {'daily': 1, 'weekly': 5, 'monthly': 21}[self.config.rebalancing_frequency]
        
        current_long_etfs = None
        current_short_etfs = None
        
        for i, date in enumerate(dates):
            if date in data.index:
                # Rebalance periodically
                if i % rebal_freq == 0 or current_long_etfs is None:
                    # Randomly select long and short positions
                    shuffled_etfs = available_etfs.copy()
                    np.random.shuffle(shuffled_etfs)
                    
                    current_long_etfs = shuffled_etfs[:self.n_long]
                    current_short_etfs = shuffled_etfs[-self.n_short:]
                
                # Calculate position weights
                long_weight = self.leverage / (2 * self.n_long) if self.n_long > 0 else 0
                short_weight = -self.leverage / (2 * self.n_short) if self.n_short > 0 else 0
                
                # Calculate daily return
                day_return = 0.0
                if current_long_etfs:
                    day_return += data.loc[date, current_long_etfs].sum() * long_weight
                if current_short_etfs:
                    day_return += data.loc[date, current_short_etfs].sum() * short_weight
                
                returns.append(day_return)
            else:
                returns.append(0.0)
        
        return pd.Series(returns, index=dates)
    
    def get_description(self) -> str:
        return f"Random long-short ({self.n_long}L/{self.n_short}S, {self.leverage:.1f}x leverage)"

class ZeroReturnBenchmark(BenchmarkCalculator):
    """Zero return benchmark for dollar-neutral strategies."""
    
    def calculate_returns(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(0.0, index=dates)
    
    def get_description(self) -> str:
        return "Zero return (cash equivalent)"

class BenchmarkManager:
    """Manages benchmark selection and calculation for different strategy types."""
    
    def __init__(self, strategy_type: str, target_etfs: List[str], 
                 feature_etfs: List[str] = None, config: BenchmarkConfig = None):
        self.strategy_type = strategy_type
        self.target_etfs = target_etfs
        self.feature_etfs = feature_etfs or []
        self.all_etfs = target_etfs + self.feature_etfs
        self.config = config or BenchmarkConfig()
        self.benchmarks = self._create_benchmarks()
    
    def _create_benchmarks(self) -> Dict[str, BenchmarkCalculator]:
        """Create appropriate benchmarks based on strategy type."""
        benchmarks = {}
        
        # Common benchmarks for all strategy types
        benchmarks['equal_weight_targets'] = EqualWeightBenchmark(self.target_etfs, self.config)
        benchmarks['spy_only'] = SingleETFBenchmark('SPY', self.config)
        
        if self.strategy_type == 'equal_weight':
            benchmarks['equal_weight_all'] = EqualWeightBenchmark(self.all_etfs, self.config)
            benchmarks['vti_market'] = SingleETFBenchmark('VTI', self.config)
            
        elif self.strategy_type == 'confidence_weighted':
            benchmarks['risk_parity'] = RiskParityBenchmark(self.target_etfs, self.config)
            if self.all_etfs:
                benchmarks['risk_parity_all'] = RiskParityBenchmark(self.all_etfs, self.config)
            
        elif self.strategy_type == 'long_short':
            benchmarks['zero_return'] = ZeroReturnBenchmark(self.config)
            benchmarks['random_long_short'] = LongShortRandomBenchmark(
                self.target_etfs, n_long=1, n_short=1, leverage=1.0, config=self.config
            )
            if len(self.target_etfs) >= 2:
                benchmarks['random_long_short_2v2'] = LongShortRandomBenchmark(
                    self.target_etfs, n_long=2, n_short=2, leverage=1.0, config=self.config
                )
        
        # Filter benchmarks based on config (only if explicitly set)
        if self.config.benchmark_types is not None:
            benchmarks = {k: v for k, v in benchmarks.items() 
                         if k in self.config.benchmark_types}
        
        return benchmarks
    
    def calculate_all_benchmarks(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Calculate returns for all benchmarks."""
        benchmark_returns = {}
        
        for name, calculator in self.benchmarks.items():
            try:
                returns = calculator.calculate_returns(data, dates)
                benchmark_returns[f'benchmark_{name}'] = returns
                logger.info(f"Calculated benchmark: {name} - {calculator.get_description()}")
            except Exception as e:
                logger.error(f"Failed to calculate benchmark {name}: {str(e)}")
                benchmark_returns[f'benchmark_{name}'] = pd.Series(0.0, index=dates)
        
        return pd.DataFrame(benchmark_returns, index=dates)
    
    def get_benchmark_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all benchmarks."""
        return {name: calc.get_description() for name, calc in self.benchmarks.items()}

def _determine_strategy_type(position_func) -> str:
    """Determine strategy type from position function."""
    if position_func is None:
        return 'equal_weight'
    
    func_name = position_func.__name__
    if 'long_short' in func_name:
        return 'long_short'
    elif 'confidence' in func_name:
        return 'confidence_weighted'
    else:
        return 'equal_weight'

# --- Position Sizing Strategy Pattern ---

class PositionSizer(ABC):
    """Abstract base class for position sizing strategies."""
    
    def __init__(self, params: List[float] = None):
        self.params = params or []
    
    @abstractmethod
    def calculate_weights(self, predictions: pd.Series) -> np.ndarray:
        """Calculate individual position weights for each asset.
        
        Args:
            predictions: Series with predictions for each target
            
        Returns:
            numpy array with individual weights for each asset
        """
        pass
    
    @abstractmethod
    def calculate_portfolio_leverage(self, predictions_df: pd.DataFrame) -> pd.Series:
        """Calculate portfolio-level leverage for each date.
        
        Args:
            predictions_df: DataFrame with predictions for each target
            
        Returns:
            Series with portfolio leverage for each date
        """
        pass

class EqualWeightSizer(PositionSizer):
    """Equal-weight position sizing across all predicted targets."""
    
    def calculate_weights(self, predictions: pd.Series) -> np.ndarray:
        if predictions.isna().all():
            return np.zeros(len(predictions))
        
        predictions = predictions.fillna(0)
        base_leverage = self.params[0] if self.params else BASE_LEVERAGE_DEFAULT
        avg_prediction = predictions.mean()
        n_assets = len(predictions)
        
        if avg_prediction > 0:
            weights = np.full(n_assets, base_leverage / n_assets)
        else:
            weights = np.full(n_assets, -base_leverage / n_assets)
            
        return weights
    
    def calculate_portfolio_leverage(self, predictions_df: pd.DataFrame) -> pd.Series:
        base_leverage = self.params[0] if self.params else BASE_LEVERAGE_DEFAULT
        avg_prediction = predictions_df.mean(axis=1)
        leverage = np.where(avg_prediction > 0, base_leverage, -base_leverage)
        return pd.Series(leverage, index=predictions_df.index)

class ConfidenceWeightedSizer(PositionSizer):
    """Confidence-weighted position sizing based on prediction magnitude."""
    
    def calculate_weights(self, predictions: pd.Series) -> np.ndarray:
        if predictions.isna().all():
            return np.zeros(len(predictions))
        
        predictions = predictions.fillna(0)
        base_leverage = self.params[0] if self.params else MAX_LEVERAGE_DEFAULT
        confidence = predictions.abs()
        avg_prediction = predictions.mean()
        
        if confidence.sum() > 0:
            confidence_normalized = confidence / confidence.sum()
            weights = np.sign(avg_prediction) * confidence_normalized * base_leverage
        else:
            weights = np.zeros(len(predictions))
            
        return weights
    
    def calculate_portfolio_leverage(self, predictions_df: pd.DataFrame) -> pd.Series:
        max_leverage = self.params[0] if self.params else MAX_LEVERAGE_DEFAULT
        confidence = predictions_df.abs().mean(axis=1)
        avg_prediction = predictions_df.mean(axis=1)
        confidence_normalized = confidence.rank(pct=True)
        leverage = np.sign(avg_prediction) * confidence_normalized * max_leverage
        return pd.Series(leverage, index=predictions_df.index)

class LongShortSizer(PositionSizer):
    """Long-short position sizing: long best predictions, short worst."""
    
    def calculate_weights(self, predictions: pd.Series) -> np.ndarray:
        if predictions.isna().all():
            return np.zeros(len(predictions))
        
        predictions = predictions.fillna(0)
        base_leverage = self.params[0] if self.params else LONG_SHORT_LEVERAGE_DEFAULT
        
        ranked = predictions.rank(ascending=False)
        n_assets = len(predictions)
        
        if n_assets == 1:
            weights = np.array([base_leverage if predictions.iloc[0] > 0 else -base_leverage])
        elif n_assets == 2:
            long_mask = ranked == 1
            weights = np.where(long_mask, base_leverage/2, -base_leverage/2)
        else:
            long_threshold = n_assets / 3
            short_threshold = 2 * n_assets / 3
            
            long_mask = ranked <= long_threshold
            short_mask = ranked >= short_threshold
            
            n_long = long_mask.sum()
            n_short = short_mask.sum()
            
            if n_long > 0 and n_short > 0:
                long_weight = base_leverage / (2 * n_long)
                short_weight = (n_long / n_short) * long_weight
                weights = np.where(long_mask, long_weight,
                                 np.where(short_mask, -short_weight, 0))
            elif n_long > 0:
                long_weight = base_leverage / n_long
                weights = np.where(long_mask, long_weight, 0)
            elif n_short > 0:
                short_weight = base_leverage / n_short
                weights = np.where(short_mask, -short_weight, 0)
            else:
                weights = np.zeros(n_assets)
                
        return weights
    
    def calculate_portfolio_leverage(self, predictions_df: pd.DataFrame) -> pd.Series:
        base_leverage = self.params[0] if self.params else LONG_SHORT_LEVERAGE_DEFAULT
        portfolio_returns = []
        
        for _, row in predictions_df.iterrows():
            weights = self.calculate_weights(row)
            portfolio_returns.append(weights.sum())
        
        return pd.Series(portfolio_returns, index=predictions_df.index)

# --- Legacy Position Sizing Functions (for backward compatibility) ---

def calculate_individual_position_weights(predictions_row, position_func, position_params):
    """Legacy wrapper for backward compatibility."""
    # Convert to new strategy pattern
    if position_func.__name__ == 'L_func_multi_target_long_short':
        sizer = LongShortSizer(position_params)
    elif position_func.__name__ == 'L_func_multi_target_confidence_weighted':
        sizer = ConfidenceWeightedSizer(position_params)
    elif position_func.__name__ == 'L_func_multi_target_equal_weight':
        sizer = EqualWeightSizer(position_params)
    else:
        sizer = EqualWeightSizer([1.0])
    
    if predictions_row is None or len(predictions_row) == 0:
        logger.warning("Empty predictions_row in calculate_individual_position_weights")
        return np.array([])
    
    if not isinstance(predictions_row, pd.Series):
        predictions_row = pd.Series(predictions_row)
    
    return sizer.calculate_weights(predictions_row)

def L_func_multi_target_equal_weight(predictions_df, params=[]):
    """
    Equal-weight position sizing strategy for multi-asset portfolios.
    
    This strategy implements a democratic approach to portfolio construction where
    each asset gets equal influence regardless of prediction confidence or volatility.
    It's the simplest multi-target strategy and serves as a baseline for comparison.
    
    Educational Note:
        Equal weighting is often used as a benchmark in portfolio management because:
        1. It avoids concentration risk from any single asset
        2. It's transparent and easy to explain to stakeholders  
        3. It often outperforms cap-weighted benchmarks due to rebalancing effects
        4. It requires minimal complexity in implementation
    
    Args:
        predictions_df (pd.DataFrame): Predictions for each target asset (rows=dates, cols=assets)
        params (list): [base_leverage] - base leverage multiplier (default: 1.0)
    
    Returns:
        pd.Series: Portfolio leverage for each date
        
    Strategy Logic:
        1. Average predictions across all target assets
        2. Take long position if average > 0, short if average < 0
        3. Apply base leverage uniformly across all positions
        
    Example:
        >>> predictions = pd.DataFrame({'SPY': [0.01, -0.02], 'QQQ': [0.03, 0.01]})
        >>> leverage = L_func_multi_target_equal_weight(predictions, params=[1.5])
        >>> # Result: [1.5, -1.5] based on average predictions [0.02, -0.005]
    """
    base_leverage = params[0] if params else 1.0
    n_targets = len(predictions_df.columns)
    
    # Simple equal weight: average prediction across targets
    avg_prediction = predictions_df.mean(axis=1)
    
    # Binary position: long if avg > 0, short if avg < 0
    leverage = np.where(avg_prediction > 0, base_leverage, -base_leverage)
    
    return pd.Series(leverage, index=predictions_df.index)

def L_func_multi_target_confidence_weighted(predictions_df, params=[]):
    """
    Confidence-weighted position sizing based on prediction magnitude.
    
    Args:
        predictions_df: DataFrame with predictions for each target
        params: [max_leverage] - maximum leverage to apply
    
    Returns:
        Series with portfolio leverage for each date
    """
    max_leverage = params[0] if params else 2.0
    
    # Calculate confidence as average absolute prediction
    confidence = predictions_df.abs().mean(axis=1)
    avg_prediction = predictions_df.mean(axis=1)
    
    # Normalize confidence to [0, 1] using historical quantiles
    confidence_normalized = confidence.rank(pct=True)
    
    # Apply leverage based on confidence and direction
    leverage = np.sign(avg_prediction) * confidence_normalized * max_leverage
    
    return pd.Series(leverage, index=predictions_df.index)

def L_func_multi_target_long_short(predictions_df, params=[]):
    """
    Long-short position sizing: long best predictions, short worst.
    Creates a dollar-neutral strategy where sum of longs = absolute sum of shorts.
    
    Args:
        predictions_df: DataFrame with predictions for each target
        params: [base_leverage] - base leverage for the strategy
    
    Returns:
        Series with portfolio leverage for each date
    """
    base_leverage = params[0] if params else 1.0
    
    portfolio_returns = []
    
    for date, row in predictions_df.iterrows():
        # Rank predictions: highest gets long, lowest gets short
        ranked = row.rank(ascending=False)
        n_assets = len(row)
        
        # Determine long and short positions based on ranking
        if n_assets == 1:
            # Single asset: just use sign of prediction
            positions = np.array([base_leverage if row.iloc[0] > 0 else -base_leverage])
        elif n_assets == 2:
            # Two assets: long the better one, short the worse one with equal weights
            long_mask = ranked == 1
            short_mask = ranked == 2
            positions = np.where(long_mask, base_leverage, -base_leverage)
        else:
            # Multiple assets: long top 1/3, short bottom 1/3, neutral middle
            long_threshold = n_assets / 3
            short_threshold = 2 * n_assets / 3
            
            # Identify long and short positions
            long_mask = ranked <= long_threshold
            short_mask = ranked >= short_threshold
            neutral_mask = ~(long_mask | short_mask)
            
            # Count positions
            n_long = long_mask.sum()
            n_short = short_mask.sum()
            
            # Calculate weights to ensure dollar neutrality
            if n_long > 0 and n_short > 0:
                # Make sum of longs = absolute sum of shorts
                # If we have n_long positions and n_short positions:
                # n_long * long_weight = n_short * short_weight
                # Total exposure = n_long * long_weight + n_short * short_weight = 2 * n_long * long_weight
                # We want total exposure = base_leverage, so:
                # 2 * n_long * long_weight = base_leverage
                # long_weight = base_leverage / (2 * n_long)
                # short_weight = (n_long / n_short) * long_weight
                
                long_weight = base_leverage / (2 * n_long)
                short_weight = (n_long / n_short) * long_weight
                
                positions = np.where(long_mask, long_weight,
                                   np.where(short_mask, -short_weight, 0))
            elif n_long > 0:
                # Only long positions
                long_weight = base_leverage / n_long
                positions = np.where(long_mask, long_weight, 0)
            elif n_short > 0:
                # Only short positions  
                short_weight = base_leverage / n_short
                positions = np.where(short_mask, -short_weight, 0)
            else:
                # No positions
                positions = np.zeros(n_assets)
        
        # Portfolio return is sum of individual positions
        portfolio_returns.append(positions.sum())
    
    return pd.Series(portfolio_returns, index=predictions_df.index)


# --- Multi-Target Analytics Functions ---

def plot_individual_target_performance(regout_list, sweep_tags, target_etfs, config):
    """
    Creates individual performance plots for each target ETF with benchmarking.
    Saves plots to reports/ subdirectory.
    """
    os.makedirs('reports', exist_ok=True)
    
    for target in target_etfs:
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # --- Subplot 1: Cumulative Returns Comparison ---
        ax1.set_title(f'{target} - Strategy vs Buy & Hold Comparison', fontsize=14, fontweight='bold')
        
        # Plot strategy returns for this target
        for i, (regout_df, tag) in enumerate(zip(regout_list, sweep_tags)):
            if f'actual_{target}' in regout_df.columns:
                # Strategy return: leverage * actual target return
                strategy_ret = regout_df['leverage'] * regout_df[f'actual_{target}']
                cumulative_strategy = strategy_ret.cumsum()
                ax1.plot(cumulative_strategy.index, cumulative_strategy.values, 
                        label=f'{tag} Strategy', alpha=0.7, linewidth=2)
        
        # Buy & Hold benchmark (1.0 leverage)
        if regout_list and f'actual_{target}' in regout_list[0].columns:
            buy_hold_ret = regout_list[0][f'actual_{target}']
            buy_hold_cumulative = buy_hold_ret.cumsum()
            ax1.plot(buy_hold_cumulative.index, buy_hold_cumulative.values, 
                    label=f'{target} Buy & Hold (1.0x)', 
                    linestyle='--', linewidth=3, color='black')
        
        ax1.set_ylabel('Cumulative Log-Return')
        ax1.set_xlabel('Date')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add axis lines
        ax1.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        ax1.axvline(x=ax1.get_xlim()[0], color='black', linewidth=0.5, alpha=0.5)
        
        # --- Subplot 2: Rolling Sharpe Ratio (252-day window) ---
        ax2.set_title(f'{target} - Rolling Sharpe Ratio (252-day window)', fontsize=12)
        
        for i, (regout_df, tag) in enumerate(zip(regout_list, sweep_tags)):
            if f'actual_{target}' in regout_df.columns:
                strategy_ret = regout_df['leverage'] * regout_df[f'actual_{target}']
                rolling_sharpe = strategy_ret.rolling(252).mean() / strategy_ret.rolling(252).std() * np.sqrt(252)
                ax2.plot(rolling_sharpe.index, rolling_sharpe.values, 
                        label=f'{tag} Strategy', alpha=0.7, linewidth=2)
        
        # Buy & Hold rolling Sharpe
        if regout_list and f'actual_{target}' in regout_list[0].columns:
            buy_hold_ret = regout_list[0][f'actual_{target}']
            bh_rolling_sharpe = buy_hold_ret.rolling(252).mean() / buy_hold_ret.rolling(252).std() * np.sqrt(252)
            ax2.plot(bh_rolling_sharpe.index, bh_rolling_sharpe.values, 
                    label=f'{target} Buy & Hold', 
                    linestyle='--', linewidth=2, color='black')
        
        ax2.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        ax2.set_ylabel('Rolling Sharpe Ratio')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # --- Subplot 3: Drawdown Analysis ---
        ax3.set_title(f'{target} - Drawdown Analysis', fontsize=12)
        
        for i, (regout_df, tag) in enumerate(zip(regout_list, sweep_tags)):
            if f'actual_{target}' in regout_df.columns:
                strategy_ret = regout_df['leverage'] * regout_df[f'actual_{target}']
                cum_ret = strategy_ret.cumsum()
                running_max = cum_ret.expanding().max()
                drawdown = cum_ret - running_max
                ax3.plot(drawdown.index, drawdown.values, 
                        label=f'{tag} Strategy', alpha=0.7, linewidth=2)
        
        # Buy & Hold drawdown
        if regout_list and f'actual_{target}' in regout_list[0].columns:
            buy_hold_ret = regout_list[0][f'actual_{target}']
            bh_cum_ret = buy_hold_ret.cumsum()
            bh_running_max = bh_cum_ret.expanding().max()
            bh_drawdown = bh_cum_ret - bh_running_max
            ax3.plot(bh_drawdown.index, bh_drawdown.values, 
                    label=f'{target} Buy & Hold', 
                    linestyle='--', linewidth=2, color='black')
        
        ax3.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        ax3.set_ylabel('Drawdown (Log-Return)')
        ax3.set_xlabel('Date')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f'reports/{target}_individual_performance.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved individual performance plot: {plot_filename}")
        plt.close()


def plot_portfolio_summary(regout_list, sweep_tags, target_etfs, config):
    """
    Creates comprehensive portfolio summary plots and saves them to reports/.
    """
    os.makedirs('reports', exist_ok=True)
    
    # --- Main Portfolio Performance Plot ---
    plt.figure(figsize=(20, 15))
    
    # Portfolio returns comparison
    plt.subplot(3, 2, 1)
    portfolio_rets = pd.concat([
        df['portfolio_ret'].rename(tag) 
        for df, tag in zip(regout_list, sweep_tags)
    ], axis=1)
    portfolio_rets.cumsum().plot(title="Multi-Target Portfolio Cumulative Returns")
    plt.ylabel("Cumulative Log-Return")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Individual target buy & hold performance
    plt.subplot(3, 2, 2)
    if regout_list:
        benchmark_ret = regout_list[0]['benchmark_ret']
        individual_targets = pd.DataFrame()
        for target in target_etfs:
            if f'actual_{target}' in regout_list[0].columns:
                individual_targets[target] = regout_list[0][f'actual_{target}']
        
        if not individual_targets.empty:
            individual_targets.cumsum().plot(title="Individual Target Returns (Buy & Hold)")
            benchmark_ret.cumsum().plot(label='Equal-Weight Benchmark', linestyle='--', linewidth=2)
            plt.ylabel("Cumulative Log-Return")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
    
    # Rolling volatility comparison
    plt.subplot(3, 2, 3)
    portfolio_vol = portfolio_rets.rolling(63).std() * np.sqrt(252)  # Quarterly rolling vol
    portfolio_vol.plot(title="Rolling Volatility (63-day window, annualized)")
    plt.ylabel("Annualized Volatility")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Portfolio leverage over time
    plt.subplot(3, 2, 4)
    leverage_df = pd.concat([
        df['leverage'].rename(tag) 
        for df, tag in zip(regout_list, sweep_tags)
    ], axis=1)
    leverage_df.plot(title="Portfolio Leverage Over Time")
    plt.ylabel("Leverage")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Return distribution comparison
    plt.subplot(3, 2, 5)
    for tag in portfolio_rets.columns[:5]:  # Limit to first 5 strategies for readability
        portfolio_rets[tag].hist(alpha=0.6, bins=50, label=tag, density=True)
    plt.title("Return Distribution Comparison (Top 5 Strategies)")
    plt.xlabel("Daily Returns")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Correlation heatmap of strategies
    plt.subplot(3, 2, 6)
    corr_matrix = portfolio_rets.corr()
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.title("Strategy Correlation Matrix")
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
    
    # Add correlation values to the heatmap
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                    ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    plot_filename = 'reports/portfolio_comprehensive_analysis.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive portfolio analysis: {plot_filename}")
    plt.close()


def calculate_performance_metrics(returns: pd.Series) -> Dict[str, float]:
    """Calculate comprehensive performance metrics for a return series."""
    if returns.empty or returns.isna().all():
        return {'annual_return': 0, 'annual_vol': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
    
    annual_return = returns.mean() * TRADING_DAYS_PER_YEAR
    annual_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = returns.cumsum()
    running_max = cumulative_returns.expanding().max()
    drawdown = cumulative_returns - running_max
    max_drawdown = drawdown.min()
    
    return {
        'annual_return': annual_return,
        'annual_vol': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def calculate_information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate information ratio (excess return / tracking error)."""
    if strategy_returns.empty or benchmark_returns.empty:
        return 0.0
    
    # Align series
    aligned_strategy, aligned_benchmark = strategy_returns.align(benchmark_returns, join='inner')
    
    if aligned_strategy.empty:
        return 0.0
    
    excess_returns = aligned_strategy - aligned_benchmark
    excess_mean = excess_returns.mean() * TRADING_DAYS_PER_YEAR
    tracking_error = excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return excess_mean / tracking_error if tracking_error != 0 else 0.0

def create_benchmark_comparison_table(regout_df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """Create comprehensive benchmark comparison table for a single strategy."""
    benchmark_cols = [col for col in regout_df.columns if col.startswith('benchmark_')]
    
    if not benchmark_cols:
        logger.warning(f"No benchmark columns found for strategy {strategy_name}")
        return pd.DataFrame()
    
    results = []
    strategy_returns = regout_df['portfolio_ret']
    
    for benchmark_col in benchmark_cols:
        benchmark_returns = regout_df[benchmark_col]
        benchmark_name = format_benchmark_name(benchmark_col)
        
        # Calculate metrics
        strategy_metrics = calculate_performance_metrics(strategy_returns)
        benchmark_metrics = calculate_performance_metrics(benchmark_returns)
        info_ratio = calculate_information_ratio(strategy_returns, benchmark_returns)
        
        results.append({
            'Benchmark': benchmark_name,
            'Strategy_Return': strategy_metrics['annual_return'],
            'Strategy_Sharpe': strategy_metrics['sharpe_ratio'],
            'Strategy_MaxDD': strategy_metrics['max_drawdown'],
            'Benchmark_Return': benchmark_metrics['annual_return'],
            'Benchmark_Sharpe': benchmark_metrics['sharpe_ratio'],
            'Benchmark_MaxDD': benchmark_metrics['max_drawdown'],
            'Excess_Return': strategy_metrics['annual_return'] - benchmark_metrics['annual_return'],
            'Sharpe_Diff': strategy_metrics['sharpe_ratio'] - benchmark_metrics['sharpe_ratio'],
            'Information_Ratio': info_ratio
        })
    
    return pd.DataFrame(results)

def create_performance_summary_table(regout_list, sweep_tags, target_etfs):
    """
    Creates enhanced performance summary table with multiple benchmark comparisons.
    """
    summary_data = []
    
    for regout_df, tag in zip(regout_list, sweep_tags):
        # Portfolio-level analysis
        portfolio_returns = regout_df['portfolio_ret']
        portfolio_metrics = calculate_performance_metrics(portfolio_returns)
        
        # Get benchmark comparison data
        benchmark_comparison = create_benchmark_comparison_table(regout_df, tag)
        
        # Add portfolio summary
        base_summary = {
            'Strategy': tag,
            'Portfolio_Annual_Return': portfolio_metrics['annual_return'],
            'Portfolio_Sharpe': portfolio_metrics['sharpe_ratio'],
            'Portfolio_MaxDD': portfolio_metrics['max_drawdown'],
            'Portfolio_Vol': portfolio_metrics['annual_vol']
        }
        
        # Add best benchmark comparison
        if not benchmark_comparison.empty:
            # Find benchmark with highest Sharpe ratio for comparison
            best_benchmark_idx = benchmark_comparison['Benchmark_Sharpe'].idxmax()
            best_benchmark = benchmark_comparison.loc[best_benchmark_idx]
            
            base_summary.update({
                'Best_Benchmark': best_benchmark['Benchmark'],
                'Best_Benchmark_Return': best_benchmark['Benchmark_Return'],
                'Best_Benchmark_Sharpe': best_benchmark['Benchmark_Sharpe'],
                'Excess_vs_Best': best_benchmark['Excess_Return'],
                'Info_Ratio_vs_Best': best_benchmark['Information_Ratio']
            })
        
        summary_data.append(base_summary)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    os.makedirs('reports', exist_ok=True)
    summary_df.to_csv('reports/enhanced_performance_summary.csv', index=False)
    logger.info("Saved enhanced performance summary: reports/enhanced_performance_summary.csv")
    
    return summary_df

def sim_stats_multi_target(regout_list, sweep_tags, target_etfs, author='CG', trange=None):
    """
    Calculates comprehensive statistics for multi-target strategies.
    """
    # Build results dictionary first to avoid dtype issues
    results_dict = {}
    
    # Handle case where trange is None
    if trange is None and regout_list:
        # Use the full date range from the first result
        trange = slice(regout_list[0].index[0], regout_list[0].index[-1])
        print('MULTI-TARGET SIMULATION RANGE:', 'from', trange.start, 'to', trange.stop)
    elif trange is not None:
        print('MULTI-TARGET SIMULATION RANGE:', 'from', trange.start, 'to', trange.stop)
    else:
        print('MULTI-TARGET SIMULATION RANGE: No data available')
        return pd.DataFrame()

    for n, testlabel in enumerate(sweep_tags):
        reg_out = regout_list[n].loc[trange, :]

        # Portfolio-level metrics
        mean_ret = TRADING_DAYS_PER_YEAR * reg_out.portfolio_ret.mean()
        std_ret = (np.sqrt(TRADING_DAYS_PER_YEAR)) * reg_out.portfolio_ret.std()
        sharpe = mean_ret / std_ret if std_ret != 0 else np.nan

        # Store results in dictionary
        results_dict[testlabel] = {
            'portfolio_return': mean_ret,
            'portfolio_stdev': std_ret,
            'portfolio_sharpe': sharpe,
            'avg_leverage': reg_out.leverage.mean()
        }

        # Multi-target prediction metrics
        prediction_cols = [col for col in reg_out.columns if col.startswith('pred_')]
        actual_cols = [col for col in reg_out.columns if col.startswith('actual_')]
        
        if prediction_cols and actual_cols:
            # Average RMSE across all targets
            rmse_scores = []
            mae_scores = []
            r2_scores = []
            
            for pred_col, actual_col in zip(prediction_cols, actual_cols):
                if pred_col in reg_out.columns and actual_col in reg_out.columns:
                    pred_data = reg_out[pred_col].dropna()
                    actual_data = reg_out[actual_col].reindex(pred_data.index).dropna()
                    
                    if len(pred_data) > 0 and len(actual_data) > 0:
                        rmse_scores.append(np.sqrt(rmse(actual_data, pred_data)))
                        mae_scores.append(mae(actual_data, pred_data))
                        r2_scores.append(r2_score(actual_data, pred_data))
            
            results_dict[testlabel].update({
                'avg_rmse': np.mean(rmse_scores) if rmse_scores else np.nan,
                'avg_mae': np.mean(mae_scores) if mae_scores else np.nan,
                'avg_r2': np.mean(r2_scores) if r2_scores else np.nan
            })

        # Enhanced benchmark analysis
        benchmark_cols = [col for col in reg_out.columns if col.startswith('benchmark_')]
        if benchmark_cols:
            # Calculate metrics for each benchmark
            for benchmark_col in benchmark_cols:
                benchmark_name = benchmark_col.replace('benchmark_', '')
                benchmark_ret = reg_out[benchmark_col]
                
                bench_annual_ret = TRADING_DAYS_PER_YEAR * benchmark_ret.mean()
                bench_annual_std = (np.sqrt(TRADING_DAYS_PER_YEAR)) * benchmark_ret.std()
                bench_sharpe = bench_annual_ret / bench_annual_std if bench_annual_std != 0 else np.nan
                
                # Information ratio vs this benchmark
                info_ratio = calculate_information_ratio(reg_out.portfolio_ret, benchmark_ret)
                
                # Excess return vs this benchmark
                excess_return = mean_ret - bench_annual_ret
                
                # Add to results dictionary
                results_dict[testlabel].update({
                    f'{benchmark_name}_return': bench_annual_ret,
                    f'{benchmark_name}_std': bench_annual_std,
                    f'{benchmark_name}_sharpe': bench_sharpe,
                    f'info_ratio_vs_{benchmark_name}': info_ratio,
                    f'excess_return_vs_{benchmark_name}': excess_return
                })
        
        # Legacy benchmark for backward compatibility
        if 'benchmark_ret' in reg_out.columns:
            bench_ret = TRADING_DAYS_PER_YEAR * reg_out.benchmark_ret.mean()
            bench_std = (np.sqrt(TRADING_DAYS_PER_YEAR)) * reg_out.benchmark_ret.std()
            bench_sharpe = bench_ret / bench_std if bench_std != 0 else np.nan
            
            results_dict[testlabel].update({
                'benchmark_return': bench_ret,
                'benchmark_std': bench_std,
                'benchmark_sharpe': bench_sharpe
            })

        # Add metadata
        results_dict[testlabel].update({
            'beg_pred': str(reg_out.index.min().date()),
            'end_pred': str(reg_out.index.max().date()),
            'author': str(author),
            'n_targets': len(target_etfs)
        })


    # Convert results dictionary to DataFrame
    df = pd.DataFrame(results_dict)
    df.index.name = 'metric'
    
    return df


def _prepare_training_data(X: pd.DataFrame, y_multi: pd.DataFrame, 
                          start_training: pd.Timestamp, end_training: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare and validate training data."""
    fit_X = X[start_training:end_training]
    fit_y = y_multi[start_training:end_training]
    
    # Validate training data
    if fit_X.isna().any().any():
        logger.warning(f"NaN values found in training features, filling with zeros")
        fit_X = fit_X.fillna(0)
    
    if fit_y.isna().any().any():
        logger.warning(f"NaN values found in training targets, filling with zeros")
        fit_y = fit_y.fillna(0)
    
    return fit_X, fit_y

def _train_model(fit_obj, fit_X: pd.DataFrame, fit_y: pd.DataFrame, 
                prediction_date: pd.Timestamp) -> Any:
    """Train the model with error handling."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            fit_obj.fit(fit_X, fit_y)
        
        # Validate model training
        test_pred = fit_obj.predict(fit_X.iloc[:1])
        if np.isnan(test_pred).any():
            logger.warning(f"Model producing NaN predictions after training for date {prediction_date.date()}")
            # Try refitting with different approach
            from sklearn.pipeline import Pipeline
            fit_obj = Pipeline(fit_obj.steps)
            fit_obj.fit(fit_X.fillna(0), fit_y.fillna(0))
        
        return fit_obj
        
    except Exception as e:
        logger.error(f"Error training model for date {prediction_date.date()}: {str(e)}")
        # Fallback to simple model
        from sklearn.linear_model import LinearRegression
        fallback_model = LinearRegression()
        fallback_model.fit(fit_X.fillna(0), fit_y.fillna(0))
        return fallback_model

def _make_predictions(model, pred_X: pd.DataFrame, target_cols: List[str], 
                     prediction_date: pd.Timestamp) -> np.ndarray:
    """Make predictions with error handling."""
    # Validate prediction input
    if pred_X.isna().any().any():
        logger.warning(f"NaN values in prediction input for date {prediction_date.date()}")
        pred_X = pred_X.fillna(0)
    
    try:
        predictions = model.predict(pred_X)
    except Exception as e:
        logger.error(f"Error making prediction for date {prediction_date.date()}: {str(e)}")
        # Use zero predictions as fallback
        predictions = np.zeros((1, len(target_cols)))
    
    # Handle different prediction formats
    if hasattr(predictions, 'values'):
        pred_values = predictions.values[0]
    else:
        pred_values = predictions[0] if predictions.ndim > 1 else predictions

    # Ensure pred_values is a 1D array with correct length
    if isinstance(pred_values, (list, tuple)):
        pred_values = np.array(pred_values)
    
    # Check for NaN predictions and replace with zeros
    if np.isnan(pred_values).any():
        logger.debug(f"NaN predictions detected for date {prediction_date.date()}, replacing with zeros")
        pred_values = np.nan_to_num(pred_values, nan=0.0)
    
    # Debug: check prediction shape
    if len(pred_values) != len(target_cols):
        logger.warning(f"Prediction length {len(pred_values)} != target count {len(target_cols)}")
        # Try to handle this gracefully
        if len(pred_values) > len(target_cols):
            pred_values = pred_values[:len(target_cols)]
        elif len(pred_values) < len(target_cols):
            # Pad with zeros if too short
            pred_values = np.pad(pred_values, (0, len(target_cols) - len(pred_values)), 'constant')

    return pred_values

def _calculate_portfolio_returns(regout: pd.DataFrame, target_cols: List[str], 
                               position_func, position_params: List[float]) -> pd.DataFrame:
    """Calculate portfolio returns using individual asset weights."""
    logger.info("Calculating portfolio returns using individual asset contributions...")
    
    # Create predictions DataFrame for position sizing
    pred_cols = [f'pred_{target}' for target in target_cols]
    predictions_df = regout[pred_cols].copy()
    predictions_df.columns = target_cols  # Remove 'pred_' prefix for position function

    # Create actual returns DataFrame
    actual_cols = [f'actual_{target}' for target in target_cols]
    actual_returns = regout[actual_cols].copy()
    actual_returns.columns = target_cols  # Remove 'actual_' prefix

    portfolio_returns = []
    for date in regout.index:
        if date in predictions_df.index:
            # Get predictions for this date
            pred_row = predictions_df.loc[date]
            
            # Calculate individual position weights
            if position_func:
                weights = calculate_individual_position_weights(pred_row, position_func, position_params)
            else:
                # Default equal weight
                avg_prediction = pred_row.mean()
                n_assets = len(pred_row)
                if avg_prediction > 0:
                    weights = np.full(n_assets, 1.0 / n_assets)
                else:
                    weights = np.full(n_assets, -1.0 / n_assets)
            
            # Ensure weights and asset_returns have same length
            asset_returns = actual_returns.loc[date].values
            if len(weights) != len(asset_returns):
                logger.warning(f"Weight length {len(weights)} != asset returns length {len(asset_returns)}")
                # Pad or truncate to match
                min_len = min(len(weights), len(asset_returns))
                weights = weights[:min_len]
                asset_returns = asset_returns[:min_len]
            
            # Calculate weighted return for this date
            portfolio_return = np.sum(weights * asset_returns)
            portfolio_returns.append(portfolio_return)
        else:
            portfolio_returns.append(0.0)
    
    regout['portfolio_ret'] = portfolio_returns
    
    # Legacy leverage column for backward compatibility (sum of weights)
    if position_func:
        regout['leverage'] = position_func(predictions_df, position_params)
    else:
        regout['leverage'] = L_func_multi_target_equal_weight(predictions_df, [1.0])
    
    # Store individual asset weights for analysis (optional)
    if position_func and position_func.__name__ == 'L_func_multi_target_long_short':
        # For long-short strategies, also store individual weights for debugging
        for i, target in enumerate(target_cols):
            weight_series = []
            for date in regout.index:
                if date in predictions_df.index:
                    pred_row = predictions_df.loc[date]
                    weights = calculate_individual_position_weights(pred_row, position_func, position_params)
                    weight_series.append(weights[i])
                else:
                    weight_series.append(0.0)
            regout[f'weight_{target}'] = weight_series
    
    return regout

def Simulate_MultiTarget(X, y_multi, train_frequency, window_size, window_type, 
                        pipe_steps={}, param_grid={}, tag=None, position_func=None, position_params=[], 
                        use_cache=None, benchmark_config: BenchmarkConfig = None):
    """
    Multi-target walk-forward simulation engine with caching and configurable training frequency.
    
    Args:
        X: Feature DataFrame
        y_multi: Multi-target DataFrame (multiple ETF returns)
        train_frequency: 'daily', 'weekly', 'monthly' - how often to retrain model (required)
        window_size: Training window size (required)
        window_type: 'expanding' or 'rolling' (required)
        pipe_steps: Pipeline steps for the model
        param_grid: Parameters for the model
        tag: Label for this simulation run
        position_func: Function to convert predictions to positions
        position_params: Parameters for position function
        use_cache: Whether to use cached results if available
    
    Returns:
        DataFrame with predictions, actuals, and portfolio performance
    """
    # Generate unique hash for this simulation configuration
    simulation_hash = generate_simulation_hash(
        X, y_multi, window_size, window_type, pipe_steps, param_grid, tag, 
        position_func, position_params, train_frequency
    )
    
    # Try to load cached results first
    if use_cache:
        cached_result = load_simulation_results(simulation_hash, tag)
        if cached_result is not None:
            logger.info(f"Using cached results for {tag}")
            return cached_result
    
    # Setup benchmarking
    if benchmark_config is None:
        benchmark_config = BenchmarkConfig()
    
    strategy_type = _determine_strategy_type(position_func)
    
    # We'll need the original data for benchmarking, extract ETF lists
    target_etfs = y_multi.columns.tolist()
    feature_etfs = [col for col in X.columns if col not in target_etfs]
    
    benchmark_manager = BenchmarkManager(
        strategy_type=strategy_type,
        target_etfs=target_etfs,
        feature_etfs=feature_etfs,
        config=benchmark_config
    )
    
    # Initialize results DataFrame
    regout = pd.DataFrame(index=y_multi.index)
    target_cols = y_multi.columns.tolist()
    
    # Use enhanced calendar generation with training frequency
    date_ranges = generate_train_predict_calendar_with_frequency(
        X, train_frequency, window_type, window_size
    )

    if not date_ranges:
        logger.warning(f"Not enough data for multi-target simulation '{tag}' with window size {window_size}")
        logger.warning(f"Required data points: >{window_size}, Data points available: {len(X)}")
        return pd.DataFrame()

    # Set up multi-target pipeline
    fit_obj = Pipeline(steps=pipe_steps)
    fit_obj.set_params(**param_grid)

    logger.info(f"Starting multi-target simulation for tag: {tag}")
    logger.info(f"Predicting targets: {target_cols}")
    logger.info(f"Training frequency: {train_frequency}")
    logger.info(f"Total training iterations: {len(date_ranges)} (vs {len(X)-window_size} for daily)")
    
    last_trained_model = None
    last_training_end = None
    
    for n, dates in enumerate(date_ranges):
        start_training, end_training, prediction_date = dates[0], dates[1], dates[2]

        # Check if we need to retrain the model
        need_retrain = (last_trained_model is None or 
                       last_training_end != end_training)
        
        if need_retrain:
            fit_X, fit_y = _prepare_training_data(X, y_multi, start_training, end_training)
            
            if n % max(1, len(date_ranges)//10) == 0:  # Progress update
                logger.info(f"Training model for date {prediction_date.date()} ({n+1}/{len(date_ranges)})")

            last_trained_model = _train_model(fit_obj, fit_X, fit_y, prediction_date)
            last_training_end = end_training
        
        # Make prediction using current model
        pred_X = X[prediction_date:prediction_date]
        pred_values = _make_predictions(last_trained_model, pred_X, target_cols, prediction_date)

        # Store predictions for each target
        for i, target in enumerate(target_cols):
            if i < len(pred_values):
                regout.loc[prediction_date, f'pred_{target}'] = np.round(pred_values[i], 5)
            else:
                regout.loc[prediction_date, f'pred_{target}'] = 0.0

    # Fill in predictions for dates between training points using forward fill
    print(f"  ... filling predictions for all dates using forward fill...")
    
    # Forward fill predictions for all dates in the original index
    for target in target_cols:
        pred_col = f'pred_{target}'
        regout[pred_col] = regout[pred_col].ffill()

    # Add actual values for each target
    for target in target_cols:
        regout[f'actual_{target}'] = y_multi[target].reindex(regout.index)

    # Create predictions DataFrame for position sizing
    pred_cols = [f'pred_{target}' for target in target_cols]
    predictions_df = regout[pred_cols].copy()
    predictions_df.columns = target_cols  # Remove 'pred_' prefix for position function

    # Calculate portfolio performance using new helper function
    regout = _calculate_portfolio_returns(regout, target_cols, position_func, position_params)

    # Legacy benchmark for backward compatibility
    actual_cols = [f'actual_{target}' for target in target_cols]
    actual_returns_benchmark = regout[actual_cols].copy()
    actual_returns_benchmark.columns = target_cols
    regout['benchmark_ret'] = actual_returns_benchmark.mean(axis=1)
    
    # Calculate comprehensive benchmarks
    try:
        # Reconstruct the original returns data for benchmarking
        all_returns_data = pd.concat([X, y_multi.shift(1)], axis=1).dropna()
        benchmark_returns = benchmark_manager.calculate_all_benchmarks(all_returns_data, regout.index)
        
        # Add benchmark columns to results
        for col in benchmark_returns.columns:
            regout[col] = benchmark_returns[col]
        
        # Log benchmark descriptions
        descriptions = benchmark_manager.get_benchmark_descriptions()
        logger.info(f"Added {len(descriptions)} benchmarks for {strategy_type} strategy:")
        for name, desc in descriptions.items():
            logger.info(f"  - {name}: {desc}")
            
    except Exception as e:
        logger.error(f"Failed to calculate comprehensive benchmarks: {str(e)}")
        logger.info("Continuing with legacy benchmark only")

    # Remove rows with NaN values
    regout_clean = regout.dropna()
    
    # Save results to cache
    if use_cache:
        save_simulation_results(regout_clean, simulation_hash, tag)

    logger.info(f"Multi-target simulation for {tag} complete.")
    return regout_clean


def load_and_prepare_multi_target_data(etf_list: List[str], target_etfs: List[str], 
                                      start_date: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Downloads and prepares data for multi-target regression with error handling.
    
    Args:
        etf_list: List of all ETFs to download (features + targets)
        target_etfs: List of ETFs to predict (subset of etf_list)
        start_date: Start date for data download
    
    Returns:
        X: Feature DataFrame (all ETFs except targets)
        y_multi: Multi-target DataFrame (target ETFs only)
    
    Raises:
        ValueError: If data download fails or insufficient data
    """
    logger.info(f"Downloading multi-target data from {start_date}")
    logger.info(f"Feature ETFs: {[etf for etf in etf_list if etf not in target_etfs]}")
    logger.info(f"Target ETFs: {target_etfs}")
    
    try:
        all_etf_closing_prices_df = yf.download(etf_list, start=start_date)['Close']
        if all_etf_closing_prices_df.empty:
            raise ValueError("No data downloaded from Yahoo Finance")
        
        etf_log_returns_df = log_returns(all_etf_closing_prices_df).dropna()
        
        if etf_log_returns_df.empty:
            raise ValueError("No valid returns data after processing")
            
    except Exception as e:
        logger.error(f"Failed to download data: {str(e)}")
        raise ValueError(f"Data download failed: {str(e)}")

    # Set timezone and align timestamps
    # Handle case where index doesn't have timezone info
    if etf_log_returns_df.index.tz is None:
        etf_log_returns_df.index = etf_log_returns_df.index.tz_localize('America/New_York')
    
    etf_log_returns_df.index = etf_log_returns_df.index.map(
        lambda x: x.replace(hour=16, minute=00)
    ).tz_convert('UTC')
    etf_log_returns_df.index.name = 'teo'
    
    # Create features (day t) and targets (day t+1)
    etf_features_df = etf_log_returns_df
    etf_targets_df = etf_features_df.shift(-1)

    # Align DataFrames
    etf_features_df = etf_features_df.loc[etf_targets_df.dropna().index]
    etf_targets_df = etf_targets_df.dropna()

    # Simplify timestamps
    etf_features_df = simplify_teos(etf_features_df)
    etf_targets_df = simplify_teos(etf_targets_df)
    
    # Create feature matrix (exclude target ETFs from features)
    feature_etfs = [etf for etf in etf_list if etf not in target_etfs]
    X = etf_features_df[feature_etfs]
    
    # Create multi-target matrix
    y_multi = etf_targets_df[target_etfs]
    
    # Validate data quality
    print("Validating data quality...")
    print(f"    Feature NaN count: {X.isna().sum().sum()}")
    print(f"    Target NaN count: {y_multi.isna().sum().sum()}")
    print(f"    Feature infinite count: {np.isinf(X.values).sum()}")
    print(f"    Target infinite count: {np.isinf(y_multi.values).sum()}")
    
    # Clean any remaining NaN or infinite values
    X = X.fillna(0)
    y_multi = y_multi.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    y_multi = y_multi.replace([np.inf, -np.inf], 0)
    
    logger.info("Multi-target data preparation complete.")
    logger.info(f"Feature shape: {X.shape}")
    logger.info(f"Target shape: {y_multi.shape}")
    logger.info(f"Feature range: [{X.min().min():.6f}, {X.max().max():.6f}]")
    logger.info(f"Target range: [{y_multi.min().min():.6f}, {y_multi.max().max():.6f}]")
    
    return X, y_multi, etf_log_returns_df


def create_csv_metadata_file(csv_output_dir, strategy_tags, timestamp):
    """
    Create a metadata CSV file describing all generated CSV files.
    
    Args:
        csv_output_dir: Directory where CSV files are saved
        strategy_tags: List of strategy tags that were run
        timestamp: Timestamp string used in filenames
    """
    metadata = []
    
    # Individual strategy files
    for tag in strategy_tags:
        metadata.append({
            'filename': f'{timestamp}_{tag}_results.csv',
            'description': f'Individual simulation results for strategy: {tag}',
            'type': 'strategy_results',
            'columns': 'date, predictions, actuals, portfolio_ret, leverage, benchmark_ret, individual_weights',
            'strategy': tag,
            'timestamp': timestamp
        })
    
    # Summary files
    metadata.extend([
        {
            'filename': f'{timestamp}_performance_summary.csv',
            'description': 'Performance comparison table across all strategies and targets',
            'type': 'summary',
            'columns': 'Strategy, Target, Strategy_Return, Benchmark_Return, Excess_Return, Strategy_Sharpe, Benchmark_Sharpe, Strategy_MaxDD, Benchmark_MaxDD',
            'strategy': 'all',
            'timestamp': timestamp
        },
        {
            'filename': f'{timestamp}_detailed_statistics.csv', 
            'description': 'Detailed statistical metrics for each strategy (transposed format)',
            'type': 'statistics',
            'columns': 'portfolio_mean, portfolio_std, portfolio_sharpe, portfolio_maxdd, etc.',
            'strategy': 'all',
            'timestamp': timestamp
        },
        {
            'filename': f'{timestamp}_all_strategies_combined.csv',
            'description': 'Combined results from all strategies in one large dataset',
            'type': 'combined',
            'columns': 'date, predictions, actuals, portfolio_ret, leverage, benchmark_ret, strategy, strategy_index',
            'strategy': 'all',
            'timestamp': timestamp
        },
        {
            'filename': f'{timestamp}_csv_metadata.csv',
            'description': 'This metadata file describing all generated CSV files',
            'type': 'metadata',
            'columns': 'filename, description, type, columns, strategy, timestamp',
            'strategy': 'metadata',
            'timestamp': timestamp
        }
    ])
    
    # Create metadata DataFrame and save
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(csv_output_dir, f'{timestamp}_csv_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    
    return metadata_path


def plot_cumulative_returns_xarray(regout_list, sweep_tags, config):
    """
    Create simplified tear sheet with cumulative returns and performance statistics.
    Portrait orientation optimized for PDF tear sheet.
    """
    os.makedirs('reports', exist_ok=True)
    
    # Prepare data for all strategies
    cumulative_returns_data = []
    
    for regout_df, tag in zip(regout_list, sweep_tags):
        if 'portfolio_ret' in regout_df.columns:
            returns = regout_df['portfolio_ret'].dropna()
            if len(returns) > 0:
                cumulative_returns = returns.cumsum()
                cumulative_returns_data.append({
                    'strategy': tag,
                    'cumulative_returns': cumulative_returns,
                    'returns': returns
                })
    
    if not cumulative_returns_data:
        print("No valid cumulative returns data found")
        return
    
    # Create portrait-oriented figure for PDF tear sheet
    plt.close('all')  # Close all existing plots
    fig = plt.figure(figsize=(12, 18))  # Increased height for better spacing
    
    # Create 4x1 subplot layout: title, plot, legend, table
    gs = fig.add_gridspec(4, 1, hspace=0.4, height_ratios=[0.2, 2, 0.4, 1])
    
    # Use a better color palette optimized for printing
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    # Create strategy names for legend
    strategy_names = [data['strategy'].replace('mt_', '').replace('_', ' ') for data in cumulative_returns_data]
    
    # 0. Title Section
    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'STRATEGY PERFORMANCE TEAR SHEET', 
                 fontsize=20, fontweight='bold', ha='center', va='center',
                 transform=ax_title.transAxes)
    
    # 1. Main Cumulative Returns Plot
    ax1 = fig.add_subplot(gs[1])
    
    for i, data in enumerate(cumulative_returns_data):
        ax1.plot(data['cumulative_returns'].index, data['cumulative_returns'].values, 
                linewidth=2.5, color=colors[i], alpha=0.8, label=None)
    
    ax1.set_title('ALL STRATEGIES - Cumulative Returns Comparison', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Cumulative Log-Return', fontsize=12)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax1.grid(True, alpha=0.2)
    
    # Explicitly set legend to None to prevent any automatic legend
    ax1.legend_ = None
    
    # Add axis lines AFTER plotting data - use actual data limits
    ax1.axhline(y=0, color='black', linewidth=1.0, alpha=0.7)
    # Get the actual x-axis limits after plotting
    x_min = min([data['cumulative_returns'].index.min() for data in cumulative_returns_data])
    ax1.axvline(x=x_min, color='black', linewidth=1.0, alpha=0.7)
    
    # 2. Legend (middle)
    ax_legend = fig.add_subplot(gs[2])
    ax_legend.axis('off')
    
    # Create legend in the middle subplot with better formatting
    legend_elements = [plt.Line2D([0], [0], color=colors[i], linewidth=3, label=strategy_names[i]) 
                      for i in range(len(cumulative_returns_data))]
    legend = ax_legend.legend(handles=legend_elements, loc='center', ncol=3, fontsize=11, 
                             framealpha=0.9, fancybox=True, shadow=True)
    legend.set_bbox_to_anchor((0.5, 0.5))  # Ensure legend is centered in its subplot
    
    # 3. Performance Summary Table (bottom)
    ax2 = fig.add_subplot(gs[3])
    ax2.axis('off')
    
    # Calculate performance metrics (same as view_cached_results)
    performance_data = []
    for data in cumulative_returns_data:
        returns = data['returns']
        annual_return = 252 * returns.mean()
        annual_vol = np.sqrt(252) * returns.std()
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
        
        performance_data.append({
            'Strategy': data['strategy'].replace('mt_', '').replace('_', ' '),
            'Annual Return (%)': f"{annual_return:.2%}",
            'Annual Vol (%)': f"{annual_vol:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown (%)': f"{max_drawdown:.2%}"
        })
    
    # Create table
    df = pd.DataFrame(performance_data)
    df_sorted = df.sort_values('Sharpe Ratio', key=lambda x: pd.to_numeric(x.str.rstrip('%'), errors='coerce'), ascending=False)
    
    # Add table title
    ax2.text(0.5, 0.95, 'PERFORMANCE SUMMARY', fontsize=16, fontweight='bold', 
             ha='center', va='top', transform=ax2.transAxes)
    
    table = ax2.table(cellText=df_sorted.values, 
                     colLabels=df_sorted.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0.05, 0.05, 0.9, 0.85])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2.0)
    
    # Set column widths - make Strategy column wider
    col_widths = [0.35, 0.15, 0.15, 0.15, 0.20]  # Strategy gets 35% width
    for i, width in enumerate(col_widths):
        for j in range(len(df_sorted) + 1):  # +1 for header row
            table[(j, i)].set_width(width)
    
    # Style the table with better colors
    for i in range(len(df_sorted.columns)):
        table[(0, i)].set_facecolor('#2E8B57')  # Sea green header
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code rows based on Sharpe ratio
    for i in range(1, len(df_sorted) + 1):
        sharpe_val = float(df_sorted.iloc[i-1]['Sharpe Ratio'])
        if sharpe_val > 0.5:
            row_color = '#E8F5E8'  # Light green for good performers
        elif sharpe_val > 0:
            row_color = '#FFF8DC'  # Light yellow for positive
        else:
            row_color = '#FFE6E6'  # Light red for negative
        
        for j in range(len(df_sorted.columns)):
            table[(i, j)].set_facecolor(row_color)
    
    ax2.set_title('PERFORMANCE SUMMARY TABLE', fontsize=14, fontweight='bold', pad=20)
    
    # Create unified legend for all plots (positioned at the top)
    legend_elements = [plt.Line2D([0], [0], color=colors[i], linewidth=3, label=strategy_names[i]) 
                      for i in range(len(cumulative_returns_data))]
    
    # Add unified legend at the top center
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
              fontsize=11, framealpha=0.9, ncol=3, fancybox=True, shadow=True)
    
    # Add overall title
    plt.suptitle(f'STRATEGY ANALYSIS TEAR SHEET - {len(cumulative_returns_data)} Strategies', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Save plot
    plot_filename = f'reports/strategy_tear_sheet_{config["run_timestamp"]}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Tear sheet saved: {plot_filename}")
    print(f"   📁 Full path: {os.path.abspath(plot_filename)}")
    print(f"   🖼️  Open with: open {plot_filename}")
    
    # Also save as PDF for professional tear sheet
    pdf_filename = f'reports/strategy_tear_sheet_{config["run_timestamp"]}.pdf'
    plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', format='pdf')
    print(f"📄 PDF tear sheet saved: {pdf_filename}")
    print(f"   📁 Full path: {os.path.abspath(pdf_filename)}")
    print(f"   📖 Open with: open {pdf_filename}")
    
    # Display plot inline in VS Code/Cursor
    plt.show()
    plt.close()
    
    # Save summary data as CSV
    try:
        summary_df = pd.DataFrame(performance_data)
        csv_filename = f'reports/strategy_summary_{config["run_timestamp"]}.csv'
        summary_df.to_csv(csv_filename, index=False)
        print(f"Saved strategy summary CSV: {csv_filename}")
    except Exception as e:
        print(f"Warning: Could not save CSV file: {e}")
    
    return cumulative_returns_data


def main():
    """
    Main execution function for multi-target ETF prediction simulation.
    Runs comprehensive strategy sweep and generates performance reports.
    """
    # Configuration
    config = {
        'train_frequency': 'monthly',  # 'daily', 'weekly', 'monthly'
        'window_size': 400,  # Training window size
        'window_type': 'expanding',  # 'expanding' or 'rolling'
        'start_date': '2010-01-01',  # Data start date
        'use_cache': True,
        'force_retrain': False,
        'csv_output_dir': '/Volumes/ext_2t/ERM3_Data/stock_data/csv'  # External drive CSV directory
    }
    
    # Force cache usage for quick results
    config['use_cache'] = True
    config['force_retrain'] = False
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['run_timestamp'] = timestamp
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Create CSV output directory if it doesn't exist
    os.makedirs(config['csv_output_dir'], exist_ok=True)
    print(f"\nCSV outputs will be saved to: {config['csv_output_dir']}")
    print(f"Run timestamp: {timestamp}")
    
    # ETF configuration
    feature_etfs = ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU']
    target_etfs = ['SPY', 'QQQ', 'IWM']
    all_etfs = feature_etfs + target_etfs
    
    # Load and prepare data
    X, y_multi, _ = load_and_prepare_multi_target_data(
        etf_list=all_etfs, 
        target_etfs=target_etfs,
        start_date=config['start_date']
    )
    
    # Strategy sweep configuration
    models = {
        'linear': {'regressor': LinearRegression()},
        'huber': {'regressor': MultiOutputRegressor(HuberRegressor(epsilon=1.35))},
        'elasticnet': {'regressor': MultiOutputRegressor(ElasticNet(alpha=0.01, l1_ratio=0.5))}
    }
    
    scalers = {'std': StandardScaler()}  # Only StandardScaler
    
    position_strategies = {
        'EqualWeight': (L_func_multi_target_equal_weight, [1.0]),
        'ConfidenceWeighted': (L_func_multi_target_confidence_weighted, [2.0]),
        'LongShort': (L_func_multi_target_long_short, [1.0])
    }
    
    # Generate strategy combinations
    strategy_combinations = []
    for model_name, model_config in models.items():
        for scaler_name, scaler in scalers.items():
            for pos_name, (pos_func, pos_params) in position_strategies.items():
                # Create pipeline steps
                pipe_steps = []
                if scaler is not None:
                    pipe_steps.append(('scaler', scaler))
                pipe_steps.append(('regressor', model_config['regressor']))
                
                strategy_combinations.append({
                    'tag': f'mt_{model_name}_{scaler_name}_{pos_name}',
                    'pipe_steps': pipe_steps,
                    'param_grid': {},
                    'position_func': pos_func,
                    'position_params': pos_params
                })
    
    print(f"\nStarting simulation sweep: {len(strategy_combinations)} total strategies\n")
    
    # Run simulations
    regout_list = []
    sweep_tags = []
    
    for i, strategy in enumerate(strategy_combinations):
        print(f"--- Strategy {i+1}/{len(strategy_combinations)}: {strategy['tag']} ---")
        
        try:
            # Create benchmark config for this strategy
            benchmark_config = BenchmarkConfig(
                include_transaction_costs=False,  # Start simple
                rebalancing_frequency='monthly'
            )
            
            regout_df = Simulate_MultiTarget(
                X, y_multi, config['train_frequency'],
                window_size=config['window_size'],
                window_type=config['window_type'],
                pipe_steps=strategy['pipe_steps'],
                param_grid=strategy['param_grid'],
                tag=strategy['tag'],
                position_func=strategy['position_func'],
                position_params=strategy['position_params'],
                use_cache=config['use_cache'],
                benchmark_config=benchmark_config
            )
            
            if not regout_df.empty:
                regout_list.append(regout_df)
                sweep_tags.append(strategy['tag'])
                
                # Save individual strategy results to CSV
                csv_filename = f"{timestamp}_{strategy['tag']}_results.csv"
                csv_path = os.path.join(config['csv_output_dir'], csv_filename)
                regout_df.to_csv(csv_path)
                print(f"    Saved results to: {csv_path}")
            else:
                print(f"    Warning: No results for {strategy['tag']}")
                
        except Exception as e:
            print(f"    Error in strategy {strategy['tag']}: {str(e)}")
            continue
    
    print(f"\nCompleted {len(regout_list)} successful simulations out of {len(strategy_combinations)} total strategies.")
    
    if regout_list:
        # Generate performance reports and save to CSV
        print("\nGenerating performance analysis...")
        
        # Create performance summary table and save to CSV
        summary_df = create_performance_summary_table(regout_list, sweep_tags, target_etfs)
        summary_csv_path = os.path.join(config['csv_output_dir'], f"{timestamp}_performance_summary.csv")
        summary_df.to_csv(summary_csv_path)
        print(f"Performance summary saved to: {summary_csv_path}")
        
        # Save detailed statistics to CSV
        stats_results = sim_stats_multi_target(regout_list, sweep_tags, target_etfs, author='CG')
        if stats_results is not None and not stats_results.empty:
            stats_df = pd.DataFrame(stats_results).T  # Transpose for better CSV format
            stats_csv_path = os.path.join(config['csv_output_dir'], f"{timestamp}_detailed_statistics.csv")
            stats_df.to_csv(stats_csv_path)
            print(f"Detailed statistics saved to: {stats_csv_path}")
        
        # Save combined results to one large CSV
        combined_results = []
        for i, (regout_df, tag) in enumerate(zip(regout_list, sweep_tags)):
            df_copy = regout_df.copy()
            df_copy['strategy'] = tag
            df_copy['strategy_index'] = i
            combined_results.append(df_copy)
        
        if combined_results:
            combined_df = pd.concat(combined_results, ignore_index=False)
            combined_csv_path = os.path.join(config['csv_output_dir'], f"{timestamp}_all_strategies_combined.csv")
            combined_df.to_csv(combined_csv_path)
            print(f"Combined results saved to: {combined_csv_path}")
        
        # Create professional tear sheet and benchmark analysis
        logger.info("Generating professional tear sheet and benchmark analysis...")
        try:
            from plotting_utils import (create_tear_sheet, create_simple_comparison_plot,
                                       plot_strategy_vs_benchmarks, create_benchmark_comparison_heatmap)
            
            # Generate professional tear sheet
            pdf_path = create_tear_sheet(regout_list, sweep_tags, config)
            if pdf_path:
                logger.info(f"✅ Professional tear sheet created: {pdf_path}")
            
            # Create benchmark comparison heatmap
            heatmap_path = create_benchmark_comparison_heatmap(regout_list, sweep_tags, config)
            if heatmap_path:
                logger.info(f"📊 Benchmark comparison heatmap created: {heatmap_path}")
            
            # Create individual strategy vs benchmark plots
            for regout_df, tag in zip(regout_list, sweep_tags):
                benchmark_cols = [col for col in regout_df.columns if col.startswith('benchmark_')]
                if benchmark_cols:
                    plot_path = plot_strategy_vs_benchmarks(regout_df, tag, config)
                    if plot_path:
                        logger.info(f"📈 Strategy benchmark plot created: {plot_path}")
            
            # Also create simple comparison plot
            simple_plot = create_simple_comparison_plot(regout_list, sweep_tags, config)
            if simple_plot:
                logger.info(f"📈 Simple comparison plot created: {simple_plot}")
            
            # Display enhanced performance summary table
            display_performance_summary_table(regout_list, sweep_tags)
            
        except Exception as e:
            logger.error(f"Warning: Could not generate tear sheet: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Create CSV metadata file
        metadata_path = create_csv_metadata_file(config['csv_output_dir'], sweep_tags, timestamp)
        
        # Final output summary with all file links
        print(f"\n🎉 All simulation results and analysis saved to: {config['csv_output_dir']}")
        print(f"\nRun ID: {timestamp}")
        print("\n📁 Generated Files:")
        print(f"  📊 Individual strategy results: {timestamp}_[strategy_name]_results.csv")
        print(f"  📈 Performance summary: {timestamp}_performance_summary.csv")
        print(f"  📋 Detailed statistics: {timestamp}_detailed_statistics.csv") 
        print(f"  🔗 Combined results: {timestamp}_all_strategies_combined.csv")
        print(f"  📄 Metadata file: {timestamp}_csv_metadata.csv")
        print(f"  📄 Metadata saved to: {metadata_path}")
        
        # Show tear sheet links at the end with full paths
        reports_dir = os.path.abspath('reports/')
        print(f"\n📊 Visualization Files:")
        print(f"  📄 Simulation tear sheet: {reports_dir}/sim_tear_sheet_{timestamp}.pdf")
        print(f"  🖼️  Simple comparison: {reports_dir}/simple_comparison_{timestamp}.png")
        print(f"  📁 Reports directory: {reports_dir}")
        
    else:
        print("No successful simulations completed.")

def display_performance_summary_table(regout_list, sweep_tags):
    """
    Display a performance summary table similar to view_cached_results.
    """
    print("\n" + "="*100)
    print("SIMULATION RESULTS SUMMARY")
    print("="*100)
    
    results_summary = []
    
    for regout_df, tag in zip(regout_list, sweep_tags):
        if 'portfolio_ret' in regout_df.columns:
            portfolio_returns = regout_df['portfolio_ret'].dropna()
            
            if len(portfolio_returns) > 0:
                annual_return = 252 * portfolio_returns.mean()
                annual_vol = np.sqrt(252) * portfolio_returns.std()
                sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                max_drawdown = (portfolio_returns.cumsum() - portfolio_returns.cumsum().expanding().max()).min()
                
                # Count available benchmarks
                benchmark_cols = [col for col in regout_df.columns if col.startswith('benchmark_')]
                benchmark_count = len(benchmark_cols)
                
                # Find best benchmark for comparison (prioritize strategy-appropriate ones)
                best_benchmark_info = "N/A"
                if benchmark_cols:
                    # Determine strategy type for better benchmark selection
                    strategy_type = 'equal_weight'
                    if 'longshort' in tag.lower() or 'long_short' in tag.lower():
                        strategy_type = 'long_short'
                    elif 'confidenceweighted' in tag.lower() or 'confidence_weighted' in tag.lower() or 'confidence' in tag.lower():
                        strategy_type = 'confidence_weighted'
                    
                    # Priority order for different strategy types
                    if strategy_type == 'long_short':
                        priority_benchmarks = ['benchmark_zero_return', 'benchmark_random_long_short', 'benchmark_equal_weight_targets']
                    elif strategy_type == 'confidence_weighted':
                        priority_benchmarks = ['benchmark_risk_parity', 'benchmark_spy_only', 'benchmark_equal_weight_targets']
                    else:  # equal_weight
                        priority_benchmarks = ['benchmark_spy_only', 'benchmark_equal_weight_targets', 'benchmark_vti_market']
                    
                    best_info_ratio = -np.inf
                    
                    # First, try priority benchmarks
                    for priority_benchmark in priority_benchmarks:
                        if priority_benchmark in regout_df.columns:
                            benchmark_ret = regout_df[priority_benchmark].dropna()
                            if len(benchmark_ret) > 0:
                                info_ratio = calculate_information_ratio(portfolio_returns, benchmark_ret)
                                if info_ratio > best_info_ratio:
                                    best_info_ratio = info_ratio
                                    benchmark_name = format_benchmark_name(priority_benchmark)
                                    best_benchmark_info = f"{benchmark_name} (IR: {info_ratio:.2f})"
                    
                    # If no priority benchmarks found, use any available benchmark
                    if best_benchmark_info == "N/A":
                        for benchmark_col in benchmark_cols:
                            benchmark_ret = regout_df[benchmark_col].dropna()
                            if len(benchmark_ret) > 0:
                                info_ratio = calculate_information_ratio(portfolio_returns, benchmark_ret)
                                if info_ratio > best_info_ratio:
                                    best_info_ratio = info_ratio
                                    benchmark_name = format_benchmark_name(benchmark_col)
                                    best_benchmark_info = f"{benchmark_name} (IR: {info_ratio:.2f})"
                
                results_summary.append({
                    'Strategy': tag,
                    'Annual Return (%)': annual_return,
                    'Annual Vol (%)': annual_vol,
                    'Sharpe Ratio': sharpe_ratio,
                    'Max Drawdown (%)': max_drawdown,
                    'Benchmarks': benchmark_count,
                    'Best vs Benchmark': best_benchmark_info,
                    'Data Points': len(portfolio_returns),
                    'Date Range': f"{portfolio_returns.index.min().date()} to {portfolio_returns.index.max().date()}"
                })
    
    if results_summary:
        # Convert to DataFrame and sort by Sharpe ratio
        df = pd.DataFrame(results_summary)
        df_sorted = df.sort_values('Sharpe Ratio', ascending=False)
        
        # Format for display
        display_df = df_sorted.copy()
        display_df['Annual Return (%)'] = display_df['Annual Return (%)'].apply(lambda x: f"{x:.2%}")
        display_df['Annual Vol (%)'] = display_df['Annual Vol (%)'].apply(lambda x: f"{x:.2%}")
        display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        display_df['Max Drawdown (%)'] = display_df['Max Drawdown (%)'].apply(lambda x: f"{x:.2%}")
        
        print("📊 STRATEGY PERFORMANCE WITH BENCHMARKING:")
        print(display_df[['Strategy', 'Annual Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Benchmarks', 'Best vs Benchmark']].to_string(index=False))
        
        print(f"\n📊 Completed {len(results_summary)} successful simulations")
        
        # Show top performers
        if len(results_summary) > 0:
            print(f"\n🏆 Top 3 by Sharpe Ratio:")
            for i, row in df_sorted.head(3).iterrows():
                print(f"   {i+1}. {row['Strategy']}: {row['Sharpe Ratio']:.2f} Sharpe, {row['Annual Return (%)']:.2%} return")
            
            print(f"\n📈 Top 3 by Annual Return:")
            for i, row in df_sorted.sort_values('Annual Return (%)', ascending=False).head(3).iterrows():
                print(f"   {i+1}. {row['Strategy']}: {row['Annual Return (%)']:.2%} return, {row['Sharpe Ratio']:.2f} Sharpe")
    else:
        print("No valid results to display")
    
    print("="*100)

def demonstrate_portfolio_return_calculation():
    """
    Demonstrate the difference between old and new portfolio return calculations.
    
    This function shows why the new method is correct for long-short strategies.
    """
    print("\n" + "="*80)
    print("DEMONSTRATION: Portfolio Return Calculation Methods")
    print("="*80)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    
    # Sample predictions and actual returns
    predictions = pd.DataFrame({
        'SPY': [0.02, -0.01, 0.03, 0.01, -0.02],
        'QQQ': [-0.01, 0.02, -0.01, 0.03, 0.01],
        'IWM': [0.01, 0.03, -0.02, -0.01, 0.02]
    }, index=dates)
    
    actual_returns = pd.DataFrame({
        'SPY': [0.015, -0.005, 0.025, 0.008, -0.015],
        'QQQ': [-0.008, 0.018, -0.012, 0.028, 0.009],
        'IWM': [0.012, 0.025, -0.018, -0.008, 0.018]
    }, index=dates)
    
    print("\nSample Data:")
    print("\nPredictions:")
    print(predictions.round(4))
    print("\nActual Returns:")
    print(actual_returns.round(4))
    
    # Calculate long-short positions
    print("\n" + "-"*50)
    print("LONG-SHORT STRATEGY ANALYSIS")
    print("-"*50)
    
    for date in dates:
        pred_row = predictions.loc[date]
        actual_row = actual_returns.loc[date]
        
        # Calculate individual weights using our new method
        weights = calculate_individual_position_weights(
            pred_row, 
            L_func_multi_target_long_short, 
            [2.0]  # base leverage = 2.0
        )
        
        # Old method (WRONG for long-short)
        leverage_sum = weights.sum()  # This will be ~0 for long-short
        equal_weight_return = actual_row.mean()
        old_portfolio_return = leverage_sum * equal_weight_return
        
        # New method (CORRECT)
        new_portfolio_return = np.sum(weights * actual_row.values)
        
        print(f"\nDate: {date.strftime('%Y-%m-%d')}")
        print(f"Individual Weights: {dict(zip(pred_row.index, weights.round(4)))}")
        print(f"Sum of Weights: {leverage_sum:.4f}")
        print(f"Equal-Weight Return: {equal_weight_return:.4f}")
        print(f"OLD Method Return: {old_portfolio_return:.4f} (= {leverage_sum:.4f} × {equal_weight_return:.4f})")
        print(f"NEW Method Return: {new_portfolio_return:.4f} (= weighted sum of individual returns)")
        print(f"Difference: {abs(new_portfolio_return - old_portfolio_return):.4f}")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("- OLD method gives 0% return for long-short strategies (WRONG!)")
    print("- NEW method properly calculates weighted returns (CORRECT!)")
    print("- The difference is critical for long-short strategy evaluation")
    print("="*80)


def test_benchmarking_framework():
    """Basic test of the benchmarking framework."""
    try:
        # Test benchmark calculator creation
        config = BenchmarkConfig()
        
        # Test equal weight benchmark
        eq_benchmark = EqualWeightBenchmark(['SPY', 'QQQ'], config)
        assert eq_benchmark.get_description() is not None
        logger.info(f"✅ EqualWeightBenchmark test passed: {eq_benchmark.get_description()}")
        
        # Test benchmark manager
        manager = BenchmarkManager('equal_weight', ['SPY', 'QQQ'], ['XLK', 'XLF'], config)
        assert len(manager.benchmarks) > 0
        logger.info(f"✅ BenchmarkManager test passed: {len(manager.benchmarks)} benchmarks created")
        
        # Test benchmark descriptions
        descriptions = manager.get_benchmark_descriptions()
        for name, desc in descriptions.items():
            logger.info(f"  - {name}: {desc}")
        
        logger.info("✅ All benchmarking framework tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Benchmarking framework test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Test benchmarking framework
    logger.info("Testing benchmarking framework...")
    test_benchmarking_framework()
    
    # Run demonstration
    demonstrate_portfolio_return_calculation()
    
    # Run main simulation
    main()
