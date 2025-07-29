# -*- coding: utf-8 -*-
"""
Quantitative Trading Simulation Framework

Purpose:
This script provides a framework for backtesting quantitative trading strategies
using a time series approach. It is designed for educational purposes to demonstrate
concepts in financial engineering, such as rolling-window backtesting, feature
engineering with sklearn pipelines, and performance analytics.

Core Components:
- Data Loading: Fetches ETF price data from Yahoo Finance.
- Feature/Target Engineering: Calculates log returns and sets up the prediction
  problem to avoid look-ahead bias.
- Simulation Engine (`Simulate`): A walk-forward simulator that trains a model on a
  rolling or expanding window of historical data and makes predictions for the next
  period.
- Position Sizing (`L_func_*`): A set of functions to translate model predictions
  into portfolio leverage.
- Performance Analytics (`sim_stats`): Calculates key performance indicators (KPIs)
  like Sharpe ratio, annualized returns, and volatility.

How to Use:
1. Configure the simulation parameters in the `if __name__ == '__main__':` block.
   This includes defining the ETFs, the machine learning pipeline, and parameter
   sweeps.
2. Run the script. The simulation results will be stored in an xarray Dataset
   and key performance metrics will be printed.
3. A detailed performance report will be generated using professional tear sheets.
"""

import os
import warnings
import logging
import sys
import pickle
import hashlib
from datetime import datetime
import time
import pytz
import numpy as np
import pandas as pd
import xarray as xr
import yfinance as yf
# matplotlib and scipy imports moved to where needed
from IPython.display import display
from abc import ABC, abstractmethod
from typing import Dict, List
from dataclasses import dataclass

# Scikit-learn and statsmodels
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as rmse, mean_absolute_error as mae, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TRADING_DAYS_PER_YEAR = 252
QUARTERLY_WINDOW_DAYS = 63
MONTHLY_RETRAIN_DAYS = 21
WEEKLY_RETRAIN_DAYS = 5
DAILY_RETRAIN_DAYS = 1

def format_benchmark_name(benchmark_col: str) -> str:
    """Format benchmark column name for display with abbreviations."""
    name = benchmark_col.replace('benchmark_', '').replace('_', ' ').title()
    # Apply abbreviations
    if name == 'Equal Weight Targets':
        return 'EQ Weight'
    return name

# Utility functions from our custom library
try:
    from .utils_simulate import (
        simplify_teos, log_returns, generate_train_predict_calender,
        StatsModelsWrapper_with_OLS, get_complexity_score, 
        calculate_complexity_adjusted_metrics
    )
except ImportError:
    # Fallback for direct execution or testing
    from utils_simulate import (
        simplify_teos, log_returns, generate_train_predict_calender,
        StatsModelsWrapper_with_OLS, get_complexity_score, 
        calculate_complexity_adjusted_metrics
    )

# Professional plotting utilities
try:
    from .plotting_utils import create_tear_sheet, create_simple_comparison_plot
except ImportError:
    # Fallback if plotting_utils is not available
    def create_tear_sheet(*args, **kwargs):
        print("Warning: Plotting utilities not available")
        return None
    def create_simple_comparison_plot(*args, **kwargs):
        print("Warning: Plotting utilities not available")
        return None


# --- Enhanced Metadata and Caching System ---

def generate_simulation_metadata(X, y, window_size, window_type, pipe_steps, param_grid, 
                               tag, etf_symbols=None, target_etf=None, start_date=None, 
                               random_seed=None, feature_engineering_steps=None):
    """
    Generate comprehensive metadata for full single-target simulation reconstruction.
    
    This enhanced version stores all information needed to perfectly reproduce
    a single-target simulation, enabling full reproducibility and audit trails.
    
    Educational Note:
        Reproducibility is crucial in quantitative finance for:
        1. Regulatory compliance and audit requirements
        2. Model validation and backtesting verification  
        3. Research collaboration and peer review
        4. Production deployment confidence
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable  
        window_size (int): Training window size
        window_type (str): 'expanding' or 'rolling'
        pipe_steps (list): sklearn Pipeline steps
        param_grid (dict): Model hyperparameters
        tag (str): Simulation identifier
        etf_symbols (list, optional): ETF symbols used as features
        target_etf (str, optional): ETF symbol used as target
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
            'target_etf': target_etf,
            'start_date': start_date,
            'end_date': X.index[-1].strftime('%Y-%m-%d') if hasattr(X.index[-1], 'strftime') else str(X.index[-1]),
            'data_shapes': {
                'X_shape': X.shape, 
                'y_shape': y.shape,
                'feature_columns': list(X.columns),
                'target_name': y.name if hasattr(y, 'name') else 'target'
            },
            'data_fingerprint': {
                'X_head_hash': hashlib.md5(str(X.head().values).encode()).hexdigest(),
                'X_tail_hash': hashlib.md5(str(X.tail().values).encode()).hexdigest(),
                'y_head_hash': hashlib.md5(str(y.head().values).encode()).hexdigest(),
                'y_tail_hash': hashlib.md5(str(y.tail().values).encode()).hexdigest()
            }
        },
        
        # Training configuration
        'training_params': {
            'window_size': window_size,
            'window_type': window_type,
            'random_seed': random_seed
        },
        
        # Model configuration with full reproducibility
        'model_config': {
            'pipe_steps': pipe_steps,  # Store actual pipeline configuration
            'param_grid': param_grid,
            'pipeline_string': str(pipe_steps)  # Human-readable backup
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
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'simulation_type': 'single_target'
        }
    }
    
    return metadata

def generate_simulation_hash(X, y, window_size, window_type, pipe_steps, param_grid, tag,
                           etf_symbols=None, target_etf=None, start_date=None, 
                           random_seed=None, feature_engineering_steps=None):
    """
    Generate a unique hash for single-target simulation parameters to enable caching.
    
    This function now uses the enhanced metadata system while maintaining
    backward compatibility for caching purposes.
    """
    # Generate full metadata
    metadata = generate_simulation_metadata(
        X, y, window_size, window_type, pipe_steps, param_grid, tag,
        etf_symbols, target_etf, start_date, random_seed, feature_engineering_steps
    )
    
    # Create hash from key parameters (excluding timestamps)
    hash_components = [
        str(metadata['data_source']['data_shapes']),
        str(metadata['training_params']),
        str(metadata['model_config']['pipeline_string']),
        str(metadata['model_config']['param_grid']),
        metadata['data_source']['data_fingerprint']['X_head_hash'],
        metadata['data_source']['data_fingerprint']['X_tail_hash'],
        metadata['data_source']['data_fingerprint']['y_head_hash'],
        metadata['data_source']['data_fingerprint']['y_tail_hash']
    ]
    
    hash_string = '_'.join(hash_components)
    return hashlib.md5(hash_string.encode()).hexdigest(), metadata

def save_simulation_results(regout_df, simulation_hash, tag, metadata=None):
    """
    Save single-target simulation results and metadata to disk for future reuse and full reconstruction.
    
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
    
    print(f"Saved single-target simulation results with metadata: {cache_filename}")
    if metadata:
        print(f"  - Target: {metadata['data_source']['target_etf']}")
        print(f"  - Features: {len(metadata['data_source']['data_shapes']['feature_columns'])} columns")
        print(f"  - Data period: {metadata['data_source']['start_date']} to {metadata['data_source']['end_date']}")
    
    return cache_filename

def load_simulation_results(simulation_hash, tag):
    """
    Load cached single-target simulation results and metadata if they exist.
    
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
                print(f"  - Target: {metadata['data_source']['target_etf']}")
            return cache_data['results'], cache_data.get('metadata')
        else:
            # Legacy format - just results
            print("  - Loaded legacy cache (no metadata)")
            return cache_data, None
    
    return None, None


# --- Benchmarking Framework for Single-Target ---

@dataclass
class SingleTargetBenchmarkConfig:
    """Configuration for single-target benchmark calculations."""
    include_transaction_costs: bool = True
    rebalancing_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    benchmark_types: List[str] = None
    volatility_window: int = 63  # Days for volatility calculation
    
    def __post_init__(self):
        if self.benchmark_types is None:
            # Don't filter benchmarks by default - let strategy type determine appropriate ones
            self.benchmark_types = None
        if self.rebalancing_frequency not in ['daily', 'weekly', 'monthly']:
            raise ValueError(f"Invalid rebalancing_frequency: {self.rebalancing_frequency}")

class SingleTargetBenchmarkCalculator(ABC):
    """Abstract base class for single-target benchmark calculations."""
    
    def __init__(self, config: SingleTargetBenchmarkConfig = None):
        self.config = config or SingleTargetBenchmarkConfig()
    
    @abstractmethod
    def calculate_returns(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
        """Calculate benchmark returns for given dates."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a description of this benchmark."""
        pass

class BuyAndHoldBenchmark(SingleTargetBenchmarkCalculator):
    """Buy and hold the target ETF."""
    
    def __init__(self, target_etf: str, config: SingleTargetBenchmarkConfig = None):
        super().__init__(config)
        self.target_etf = target_etf
    
    def calculate_returns(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
        if self.target_etf in data.columns:
            return data[self.target_etf].reindex(dates).fillna(0)
        return pd.Series(0.0, index=dates)
    
    def get_description(self) -> str:
        return f"Buy-and-hold {self.target_etf}"

class ZeroReturnBenchmark(SingleTargetBenchmarkCalculator):
    """Zero return benchmark (cash equivalent)."""
    
    def calculate_returns(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
        return pd.Series(0.0, index=dates)
    
    def get_description(self) -> str:
        return "Zero return (cash equivalent)"

class MarketIndexBenchmark(SingleTargetBenchmarkCalculator):
    """Benchmark against a market index."""
    
    def __init__(self, index_etf: str, config: SingleTargetBenchmarkConfig = None):
        super().__init__(config)
        self.index_etf = index_etf
    
    def calculate_returns(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
        if self.index_etf in data.columns:
            return data[self.index_etf].reindex(dates).fillna(0)
        logger.warning(f"ETF {self.index_etf} not found in data, using zeros")
        return pd.Series(0.0, index=dates)
    
    def get_description(self) -> str:
        return f"Buy-and-hold {self.index_etf}"

class SingleTargetBenchmarkManager:
    """Manages benchmark selection for single-target strategies."""
    
    def __init__(self, target_etf: str, feature_etfs: List[str] = None, 
                 config: SingleTargetBenchmarkConfig = None):
        self.target_etf = target_etf
        self.feature_etfs = feature_etfs or []
        self.all_etfs = [target_etf] + self.feature_etfs
        self.config = config or SingleTargetBenchmarkConfig()
        self.benchmarks = self._create_benchmarks()
    
    def _create_benchmarks(self) -> Dict[str, SingleTargetBenchmarkCalculator]:
        """Create appropriate benchmarks for single-target strategies."""
        benchmarks = {}
        
        # Core benchmarks
        benchmarks['buy_and_hold'] = BuyAndHoldBenchmark(self.target_etf, self.config)
        benchmarks['zero_return'] = ZeroReturnBenchmark(self.config)
        
        # Market benchmarks
        if 'SPY' not in [self.target_etf] and 'SPY' in self.all_etfs:
            benchmarks['spy_market'] = MarketIndexBenchmark('SPY', self.config)
        if 'VTI' in self.all_etfs:
            benchmarks['vti_market'] = MarketIndexBenchmark('VTI', self.config)
        
        # Filter benchmarks based on config (only if explicitly set)
        if self.config.benchmark_types is not None:
            benchmarks = {k: v for k, v in benchmarks.items() 
                         if k in self.config.benchmark_types}
        
        return benchmarks
    
    def calculate_all_benchmarks(self, data: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Calculate returns for all benchmarks."""
        benchmark_returns = {}
        
        for name, benchmark in self.benchmarks.items():
            try:
                returns = benchmark.calculate_returns(data, dates)
                benchmark_returns[f'benchmark_{name}'] = returns
                logger.info(f"Calculated benchmark: {name} - {benchmark.get_description()}")
            except Exception as e:
                logger.error(f"Failed to calculate benchmark {name}: {e}")
                benchmark_returns[f'benchmark_{name}'] = pd.Series(0.0, index=dates)
        
        return pd.DataFrame(benchmark_returns, index=dates)

def calculate_information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate the Information Ratio between strategy and benchmark returns.
    
    The Information Ratio measures risk-adjusted performance relative to a benchmark.
    It's widely used in institutional portfolio management for performance evaluation.
    
    Educational Note:
        Information Ratio = (Strategy Return - Benchmark Return) / Tracking Error
        - Higher values indicate better risk-adjusted outperformance
        - Values > 0.5 are considered good, > 1.0 are excellent
        - It's the active return per unit of active risk
    
    Args:
        strategy_returns (pd.Series): Strategy daily returns
        benchmark_returns (pd.Series): Benchmark daily returns
        
    Returns:
        float: Annualized Information Ratio
        
    Formula:
        IR = (E[R_strategy - R_benchmark]) / σ(R_strategy - R_benchmark)
        where E[] is expected value and σ is standard deviation
    """
    try:
        # Align the series to ensure matching dates
        aligned_strategy, aligned_benchmark = strategy_returns.align(benchmark_returns, join='inner')
        
        if len(aligned_strategy) == 0:
            return 0.0
        
        # Calculate excess returns
        excess_returns = aligned_strategy - aligned_benchmark
        
        # Annualize
        excess_mean = excess_returns.mean() * TRADING_DAYS_PER_YEAR
        tracking_error = excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Calculate Information Ratio
        if tracking_error == 0 or np.isnan(tracking_error):
            return 0.0
        
        return excess_mean / tracking_error
    except Exception as e:
        logger.error(f"Error calculating information ratio: {e}")
        return 0.0

# --- Position Sizing Strategy Pattern ---

class PositionSizer(ABC):
    """Abstract base class for position sizing strategies."""
    
    @abstractmethod
    def calculate_position(self, predictions: pd.Series, **kwargs) -> pd.Series:
        """Calculate position sizes based on predictions."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this position sizing strategy."""
        pass

class BinaryPositionSizer(PositionSizer):
    """Binary position sizing: long if prediction > 0, else short."""
    
    def __init__(self, short_position: float = -1.0, long_position: float = 1.0):
        self.short_position = short_position
        self.long_position = long_position
    
    def calculate_position(self, predictions: pd.Series) -> pd.Series:
        conditions = [predictions <= 0, predictions > 0]
        positions = [self.short_position, self.long_position]
        return pd.Series(np.select(conditions, positions, default=np.nan), index=predictions.index)
    
    def get_name(self) -> str:
        return f"Binary({self.short_position:.1f},{self.long_position:.1f})"

class QuartilePositionSizer(PositionSizer):
    """Quartile-based position sizing based on prediction confidence."""
    
    def __init__(self, positions: List[float] = None):
        self.positions = positions or [0, 0.5, 1.5, 2.0]
        if len(self.positions) != 4:
            raise ValueError("QuartilePositionSizer requires exactly 4 position values")
    
    def calculate_position(self, predictions: pd.Series) -> pd.Series:
        # Convert predictions to normalized quantiles
        from scipy.stats import norm
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        normalized_preds = scaler.fit_transform(predictions.values.reshape(-1, 1)).flatten()
        quantile_preds = norm.cdf(normalized_preds)
        
        conditions = [
            (quantile_preds >= 0.00) & (quantile_preds < 0.25),
            (quantile_preds >= 0.25) & (quantile_preds < 0.50),
            (quantile_preds >= 0.50) & (quantile_preds < 0.75),
            (quantile_preds >= 0.75) & (quantile_preds <= 1.00)
        ]
        
        return pd.Series(np.select(conditions, self.positions, default=np.nan), index=predictions.index)
    
    def get_name(self) -> str:
        return f"Quartile({','.join(map(str, self.positions))})"

class ProportionalPositionSizer(PositionSizer):
    """Position sizing proportional to prediction strength."""
    
    def __init__(self, max_position: float = 2.0, min_position: float = 0.0):
        self.max_position = max_position
        self.min_position = min_position
    
    def calculate_position(self, predictions: pd.Series) -> pd.Series:
        # Normalize predictions to [0,1] range
        pred_min, pred_max = predictions.min(), predictions.max()
        if pred_max == pred_min:
            return pd.Series(self.min_position, index=predictions.index)
        
        normalized = (predictions - pred_min) / (pred_max - pred_min)
        positions = self.min_position + normalized * (self.max_position - self.min_position)
        
        return positions
    
    def get_name(self) -> str:
        return f"Proportional({self.min_position:.1f}-{self.max_position:.1f})"

# Legacy functions for backwards compatibility
def L_func_2(df, pred_col='predicted', params=[]):
    """Binary position sizing: long if prediction > 0, else short."""
    sizer = BinaryPositionSizer(params[0] if len(params) > 0 else -1.0, 
                                params[1] if len(params) > 1 else 1.0)
    return sizer.calculate_position(df[pred_col])

def L_func_3(df, pred_col='preds_index', params=[]):
    """Quartile-based position sizing based on prediction confidence."""
    sizer = QuartilePositionSizer(params if params else [0, 0.5, 1.5, 2])
    return sizer.calculate_position(df[pred_col])

def L_func_4(ds, params=[]):
    """Alternative quartile position sizing (operates on a Series)."""
    sizer = QuartilePositionSizer(params if params else [0, 0.5, 1.5, 2])
    return sizer.calculate_position(ds)

# --- Core Simulation Functions ---

def sim_stats_single_target(regout_list, sweep_tags, author='CG', trange=None, target_etf='SPY', 
                           feature_etfs=None, benchmark_manager=None, config=None, metadata_list=None):
    """
    Enhanced simulation statistics with benchmarking for single-target strategies.
    Calculates and prints comprehensive simulation statistics including benchmark comparisons.
    """
    results = {}
    df = pd.DataFrame(dtype=object)
    df.index.name = 'metric'
    
    print('SIMULATION RANGE:', 'from', trange.start, 'to', trange.stop)
    logger.info(f"Calculating statistics for {len(regout_list)} strategies")

    for n, testlabel in enumerate(sweep_tags):
        try:
            reg_out = regout_list[n].loc[trange, :]
            
            if reg_out.empty:
                logger.warning(f"No data for strategy {testlabel} in specified range")
                continue

            # --- Core Performance Metrics ---
            mean_return = TRADING_DAYS_PER_YEAR * reg_out.perf_ret.mean()
            volatility = np.sqrt(TRADING_DAYS_PER_YEAR) * reg_out.perf_ret.std()
            sharpe_ratio = mean_return / volatility if volatility != 0 else np.nan
            
            # Maximum drawdown calculation
            cumulative_returns = reg_out.perf_ret.cumsum()
            running_max = cumulative_returns.expanding().max()
            drawdown = cumulative_returns - running_max
            max_drawdown = drawdown.min()
            
            df.loc['return', testlabel] = mean_return
            df.loc['stdev', testlabel] = volatility
            df.loc['sharpe', testlabel] = sharpe_ratio
            df.loc['max_drawdown', testlabel] = max_drawdown
            # Safe leverage column access
            if 'leverage' in reg_out.columns:
                df.loc['avg_leverage', testlabel] = reg_out.leverage.mean()
                df.loc['leverage_1_return', testlabel] = mean_return / reg_out.leverage.mean() if reg_out.leverage.mean() != 0 else np.nan
            else:
                df.loc['avg_leverage', testlabel] = np.nan
                df.loc['leverage_1_return', testlabel] = np.nan
                logger.warning(f"Leverage column missing for strategy {testlabel}")
            
            # Prediction accuracy metrics
            df.loc['pos_prediction_ratio', testlabel] = (
                np.sum(np.isfinite(reg_out['prediction']) & (reg_out['prediction'] > 0)) /
                np.sum(np.isfinite(reg_out['prediction'])) if np.sum(np.isfinite(reg_out['prediction'])) > 0 else np.nan
            )
            
            # Model complexity analysis from metadata/hash
            if metadata_list and n < len(metadata_list) and metadata_list[n]:
                try:
                    metadata = metadata_list[n]
                    
                    # Extract model configuration from metadata
                    if 'model_config' in metadata and 'pipe_steps' in metadata['model_config']:
                        pipe_steps = metadata['model_config']['pipe_steps']
                        param_grid = metadata['model_config'].get('param_grid', {})
                        
                        # Reconstruct the model from metadata to calculate complexity
                        from sklearn.pipeline import Pipeline
                        from sklearn.preprocessing import StandardScaler
                        
                        # Create a mock estimator from the pipeline configuration
                        try:
                            # Find the final estimator in the pipeline
                            final_estimator = None
                            for step_name, step_obj in pipe_steps:
                                if hasattr(step_obj, 'predict'):  # This is likely the final estimator
                                    final_estimator = step_obj
                            
                            if final_estimator is not None:
                                # Apply any parameters from param_grid
                                if param_grid:
                                    final_estimator.set_params(**param_grid)
                                
                                complexity_score = get_complexity_score(final_estimator)
                                df.loc['complexity_score', testlabel] = complexity_score
                                
                                # Calculate complexity-adjusted metrics
                                if not reg_out.perf_ret.empty:
                                    complexity_metrics = calculate_complexity_adjusted_metrics(reg_out.perf_ret, complexity_score)
                                    df.loc['complexity_adj_return', testlabel] = complexity_metrics['complexity_adjusted_return']
                                    df.loc['complexity_adj_sharpe', testlabel] = complexity_metrics['complexity_adjusted_sharpe']
                                    df.loc['complexity_efficiency', testlabel] = complexity_metrics['complexity_efficiency']
                                    df.loc['overfitting_penalty', testlabel] = complexity_metrics['overfitting_penalty']
                                else:
                                    df.loc['complexity_adj_return', testlabel] = np.nan
                                    df.loc['complexity_adj_sharpe', testlabel] = np.nan
                                    df.loc['complexity_efficiency', testlabel] = np.nan
                                    df.loc['overfitting_penalty', testlabel] = np.nan
                            else:
                                # Default complexity score if no final estimator found
                                df.loc['complexity_score', testlabel] = 2.0  # Default for unknown
                                df.loc['complexity_adj_return', testlabel] = np.nan
                                df.loc['complexity_adj_sharpe', testlabel] = np.nan
                                df.loc['complexity_efficiency', testlabel] = np.nan
                                df.loc['overfitting_penalty', testlabel] = np.nan
                                
                        except Exception as inner_e:
                            logger.warning(f"Could not reconstruct model from metadata for {testlabel}: {inner_e}")
                            df.loc['complexity_score', testlabel] = 2.0  # Default for unknown
                            df.loc['complexity_adj_return', testlabel] = np.nan
                            df.loc['complexity_adj_sharpe', testlabel] = np.nan
                            df.loc['complexity_efficiency', testlabel] = np.nan
                            df.loc['overfitting_penalty', testlabel] = np.nan
                    else:
                        # No model config in metadata
                        df.loc['complexity_score', testlabel] = np.nan
                        df.loc['complexity_adj_return', testlabel] = np.nan
                        df.loc['complexity_adj_sharpe', testlabel] = np.nan
                        df.loc['complexity_efficiency', testlabel] = np.nan
                        df.loc['overfitting_penalty', testlabel] = np.nan
                        
                except Exception as e:
                    logger.warning(f"Could not calculate complexity score from metadata for {testlabel}: {e}")
                    df.loc['complexity_score', testlabel] = np.nan
                    df.loc['complexity_adj_return', testlabel] = np.nan
                    df.loc['complexity_adj_sharpe', testlabel] = np.nan
                    df.loc['complexity_efficiency', testlabel] = np.nan
                    df.loc['overfitting_penalty', testlabel] = np.nan
            else:
                df.loc['complexity_score', testlabel] = np.nan
                df.loc['complexity_adj_return', testlabel] = np.nan
                df.loc['complexity_adj_sharpe', testlabel] = np.nan
                df.loc['complexity_efficiency', testlabel] = np.nan
                df.loc['overfitting_penalty', testlabel] = np.nan
            df.loc['rmse', testlabel] = np.sqrt(rmse(reg_out.prediction, reg_out.actual))
            df.loc['mae', testlabel] = mae(reg_out.prediction, reg_out.actual)
            df.loc['r2', testlabel] = r2_score(reg_out.actual, reg_out.prediction)

            # Basic target benchmark (buy-and-hold target ETF)
            target_return = TRADING_DAYS_PER_YEAR * reg_out.actual.mean()
            target_volatility = np.sqrt(TRADING_DAYS_PER_YEAR) * reg_out.actual.std()
            df.loc['target_return', testlabel] = target_return
            df.loc['target_volatility', testlabel] = target_volatility
            df.loc['target_sharpe', testlabel] = target_return / target_volatility if target_volatility != 0 else np.nan
            
            # Enhanced benchmarking using multi-target approach
            if benchmark_manager:
                try:
                    # Get all ETF returns for the period
                    all_etfs = [target_etf] + (feature_etfs or [])
                    start_date = config.get('start_date', '2010-01-01') if config else '2010-01-01'
                    X, y = load_and_prepare_data(all_etfs, target_etf, start_date=start_date)
                    all_returns_data = pd.concat([X, y.to_frame(target_etf)], axis=1)
                    
                    # Calculate benchmark returns for the simulation period
                    benchmark_returns_df = benchmark_manager.calculate_all_benchmarks(
                        all_returns_data, reg_out.index
                    )
                    
                    # Add benchmark results to reg_out for later use
                    for benchmark_col in benchmark_returns_df.columns:
                        reg_out = reg_out.copy()  # Avoid SettingWithCopyWarning
                        reg_out[benchmark_col] = benchmark_returns_df[benchmark_col]
                    
                    # Enhanced benchmark analysis (using multi-target approach)
                    benchmark_cols = [col for col in benchmark_returns_df.columns if col.startswith('benchmark_')]
                    if benchmark_cols:
                        best_info_ratio = -np.inf
                        best_benchmark_name = "N/A"
                        best_excess_return = 0
                        
                        # Calculate metrics for each benchmark
                        for benchmark_col in benchmark_cols:
                            benchmark_name = benchmark_col.replace('benchmark_', '')
                            benchmark_ret = benchmark_returns_df[benchmark_col]
                            
                            # Calculate information ratio vs this benchmark
                            info_ratio = calculate_information_ratio(reg_out.perf_ret, benchmark_ret)
                            
                            # Calculate excess return vs this benchmark
                            excess_return = reg_out.perf_ret.mean() * TRADING_DAYS_PER_YEAR - benchmark_ret.mean() * TRADING_DAYS_PER_YEAR
                            
                            # Track the best benchmark
                            if info_ratio > best_info_ratio:
                                best_info_ratio = info_ratio
                                best_benchmark_name = benchmark_name.replace('_', ' ').title()
                                best_excess_return = excess_return
                    
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FutureWarning)
                            df.loc['best_benchmark', testlabel] = str(best_benchmark_name)
                        df.loc['best_info_ratio', testlabel] = best_info_ratio
                        df.loc['best_excess_return', testlabel] = best_excess_return
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FutureWarning)
                            df.loc['best_benchmark', testlabel] = str("No benchmarks")
                        df.loc['best_info_ratio', testlabel] = np.nan
                        df.loc['best_excess_return', testlabel] = np.nan
                    
                except Exception as e:
                    logger.error(f"Benchmark calculation failed for {testlabel}: {e}")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        df.loc['best_benchmark', testlabel] = str("Error")
                    df.loc['best_info_ratio', testlabel] = np.nan
                    df.loc['best_excess_return', testlabel] = np.nan

                df.loc['start_date', testlabel] = str(min(reg_out.prediction.index).date())
                df.loc['end_date', testlabel] = str(max(reg_out.prediction.index).date())
                df.loc['author', testlabel] = author
            
            # Store enhanced results for plotting
            results[testlabel] = reg_out.copy()


        except Exception as e:
            logger.error(f"Error calculating statistics for {testlabel}: {e}")
            continue

    return df, results


def Simulate(X, y, window_size=400, window_type='expanding', pipe_steps={}, param_grid={}, tag=None, 
            etf_symbols=None, target_etf=None, start_date=None):
    """
    Walk-forward simulation engine for time-series backtesting.
    
    This function implements a rigorous walk-forward analysis methodology that prevents
    look-ahead bias by training models only on historical data available at each 
    prediction point. This is essential for realistic backtesting in quantitative finance.
    
    Educational Note:
        Walk-forward analysis is the gold standard for time-series model validation.
        It simulates how a model would perform in real-time trading by strictly
        enforcing temporal ordering of training and prediction data.
    
    Args:
        X (pd.DataFrame): Feature matrix with datetime index
        y (pd.Series): Target variable (typically log returns) with datetime index  
        window_size (int): Number of periods for training window (default: 400 ~ 1.5 years)
        window_type (str): 'expanding' (growing window) or 'rolling' (fixed window)
        pipe_steps (dict): sklearn Pipeline steps for preprocessing and modeling
        param_grid (dict): Model hyperparameters
        tag (str): Identifier for caching and logging purposes
        
    Returns:
        tuple: (prediction_results_df, metadata_dict)
            - prediction_results_df: DataFrame with predictions and metadata
            - metadata_dict: Simulation metadata for complexity analysis and reproducibility
            
    Example:
        >>> results, models = Simulate(
        ...     X=features, y=target_returns,
        ...     window_size=500, window_type='expanding',
        ...     pipe_steps=[('scaler', StandardScaler()), ('model', Ridge())],
        ...     param_grid={'model__alpha': 1.0}
        ... )
    """
    regout = pd.DataFrame(index=y.index)
    fit_list = []

    # Generate simulation metadata
    metadata = generate_simulation_metadata(
        X, y, window_size, window_type, pipe_steps, param_grid, tag,
        etf_symbols, target_etf, start_date
    )

    date_ranges = generate_train_predict_calender(X, window_type=window_type, window_size=window_size)

    if not date_ranges:
        print(f"\nWarning: Not enough data to run simulation for tag '{tag}' with window size {window_size}.")
        print(f"    Required data points: >{window_size}, Data points available: {len(X)}")
        return pd.DataFrame(), metadata

    fit_obj = Pipeline(steps=pipe_steps).set_output(transform="pandas")
    fit_obj.set_params(**param_grid)

    print(f"Starting simulation for tag: {tag}...")
    for n, dates in enumerate(date_ranges):
        start_training, end_training, prediction_date = dates[0], dates[1], dates[2]

        fit_X = X[start_training:end_training]
        fit_y = y[start_training:end_training]
        pred_X = X[prediction_date:prediction_date]

        if n % 252 == 0: # Print progress once a year
             print(f"  ... processing date {prediction_date.date()} ({n}/{len(date_ranges)})")

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')
            fit_obj.fit(fit_X, fit_y)

        if hasattr(fit_obj.predict(pred_X), 'values'):
            prediction = np.round(fit_obj.predict(pred_X).values[0], 5)
        else:
            prediction = np.round(fit_obj.predict(pred_X)[0], 5)

        regout.loc[prediction_date, 'prediction'] = prediction

    print(f"Simulation for {tag} complete.")
    return regout.dropna(), metadata


def load_and_prepare_data(etf_list, target_etf, start_date=None):
    """
    Download and prepare ETF data for quantitative trading simulation.
    
    This function handles the complete data preparation pipeline for single-target
    prediction models. It ensures proper temporal alignment between features and
    targets to prevent look-ahead bias.
    
    Educational Note:
        The target variable represents tomorrow's return using today's features,
        simulating realistic prediction scenarios where you predict future returns
        based on current market conditions.
    
    Args:
        etf_list (list): List of ETF symbols for features (e.g., ['XLK', 'XLF', 'XLV'])
        target_etf (str): ETF symbol to predict (e.g., 'SPY')
        start_date (str, optional): Start date for data download (YYYY-MM-DD format)
        
    Returns:
        tuple: (X_features, y_target, all_returns_df)
            - X_features: Feature matrix (t-day features)
            - y_target: Target returns (t+1 day target returns)
            - all_returns_df: Complete log returns DataFrame for analysis
            
    Data Processing Steps:
        1. Download adjusted closing prices from Yahoo Finance
        2. Calculate log returns for all ETFs
        3. Create feature matrix X using all ETF returns
        4. Create target y using next-day target ETF returns
        5. Align data temporally to prevent look-ahead bias
    """
    print(f"Downloading and preparing data from {start_date}...")
    all_etf_closing_prices_df = yf.download(etf_list, start=start_date, auto_adjust=True)['Close']
    etf_log_returns_df = log_returns(all_etf_closing_prices_df).dropna()

    # Set timezone and align timestamps
    etf_log_returns_df.index = etf_log_returns_df.index.tz_localize('America/New_York').map(lambda x: x.replace(hour=16, minute=00)).tz_convert('UTC')
    etf_log_returns_df.index.name = 'teo'
    
    # --- Correctly create features and targets for predicting next-day return ---
    # Features are the returns on day `t`
    etf_features_df = etf_log_returns_df

    # Targets are the returns on day `t+1`, so we shift the data up by one row.
    etf_targets_df = etf_features_df.shift(-1)

    # Drop the last row where the target is now NaN and align the two DataFrames
    etf_features_df = etf_features_df.loc[etf_targets_df.dropna().index]
    etf_targets_df = etf_targets_df.dropna()

    # Simplify to date format (this removes timezone info and normalizes to midnight)
    etf_features_df = simplify_teos(etf_features_df)
    etf_targets_df = simplify_teos(etf_targets_df)
    
    X = etf_features_df.drop([target_etf], axis=1)
    y = etf_targets_df[target_etf]
    
    print("Data preparation complete.")
    return X, y


def main():
    """
    Enhanced main function with benchmarking and professional reporting.
    """
    start_time = time.time()
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # --- Enhanced Simulation Configuration ---
    config = {
        "target_etf": "SPY",
        "feature_etfs": ['SPY','XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU'],
        "start_date": "2010-01-01",
        "window_size": 400,
        "window_type": "expanding",
        "author": "CG",
        "run_timestamp": run_timestamp
    }
    
    logger.info(f"Starting single-target simulation: {run_timestamp}")
    print(f"🚀 Single-Target Simulation Started - ID: {run_timestamp}")

    try:
        # --- Data Loading ---
        logger.info("Loading and preparing data...")
        X, y = load_and_prepare_data(
            config["feature_etfs"], 
            config["target_etf"], 
            config["start_date"]
        )
        
        # --- Initialize Benchmarking ---
        benchmark_config = SingleTargetBenchmarkConfig()
        benchmark_manager = SingleTargetBenchmarkManager(
            target_etf=config["target_etf"],
            feature_etfs=config["feature_etfs"],
            config=benchmark_config
        )
        logger.info(f"Initialized benchmarks: {list(benchmark_manager.benchmarks.keys())}")
        
        # --- Parameter Sweep Setup ---
        position_strategies = [
            ('Binary', BinaryPositionSizer(-1.0, 1.0)),
            ('Quartile', QuartilePositionSizer([0, 0.5, 1.5, 2.0])),
            ('Proportional', ProportionalPositionSizer(2.0, 0.0))
        ]
        
        n_ewa_lags_list = [2, 4, 8]
        sweep_combinations = []
        
        # Create combinations of EWA parameters and position sizing strategies
        for ewa_lag in n_ewa_lags_list:
            for pos_name, pos_sizer in position_strategies:
                sweep_combinations.append({
                    'ewa_lag': ewa_lag,
                    'pos_name': pos_name,
                    'pos_sizer': pos_sizer,
                    'tag': f'st_ewa{ewa_lag}_{pos_name.lower()}'
                })
        
        X_processed = {}
        y_processed = {}
        for ewa_lag in n_ewa_lags_list:
            X_ewa = X.ewm(halflife=ewa_lag, min_periods=ewa_lag).mean().dropna()
            X_processed[ewa_lag] = X_ewa
            y_processed[ewa_lag] = y.reindex(X_ewa.index)

        regout_list = []
        sweep_tags = []
        metadata_list = []

        # --- Run Enhanced Simulation Sweep ---
        logger.info(f"Running {len(sweep_combinations)} strategy combinations...")
        for combo in sweep_combinations:
            ewa_lag = combo['ewa_lag']
            pos_sizer = combo['pos_sizer']
            tag = combo['tag']
            
            logger.info(f"Processing strategy: {tag}")
            
            # Prepare pipeline with StatsModels OLS (preserved as requested)
            pipe_steps = [
                ('scaler', StandardScaler()),
                ('final_estimator', StatsModelsWrapper_with_OLS())
            ]

            regout_df, metadata = Simulate(
                X=X_processed[ewa_lag],
                y=y_processed[ewa_lag],
                window_size=config["window_size"],
                window_type=config["window_type"],
                pipe_steps=pipe_steps,
                tag=tag,
                etf_symbols=config["feature_etfs"],
                target_etf=config["target_etf"],
                start_date=config["start_date"]
            )

            # If simulation produced no results, skip to the next sweep
            if regout_df.empty:
                logger.warning(f"No results for strategy {tag}")
                continue

            # Enhanced results processing with new position sizing
            try:
                regout_df['actual'] = y.loc[regout_df.index].dropna()
                
                # Use new position sizer instead of legacy functions
                regout_df['leverage'] = pos_sizer.calculate_position(regout_df['prediction'])
                regout_df['perf_ret'] = regout_df['leverage'] * regout_df['actual']
                
                # Rename for consistency with multi-target plotting
                regout_df['portfolio_ret'] = regout_df['perf_ret']
                
                regout_list.append(regout_df)
                sweep_tags.append(tag)
                metadata_list.append(metadata)
                
                logger.info(f"✅ Strategy {tag} completed successfully")
                
            except Exception as e:
                logger.error(f"Error processing results for {tag}: {e}")
                continue

        if not regout_list:
            logger.error("No successful simulations completed")
            return

        # --- Enhanced Analysis and Reporting ---
        trange = slice(regout_list[-1].index[0], regout_list[-1].index[-1])
        logger.info("Calculating comprehensive statistics...")
        
        stats_df, enhanced_results = sim_stats_single_target(
            regout_list, sweep_tags, 
            author=config["author"], 
            trange=trange,
            target_etf=config["target_etf"],
            feature_etfs=config["feature_etfs"],
            benchmark_manager=benchmark_manager,
            config=config,
            metadata_list=metadata_list
        )
        
        print("\n📊 ENHANCED PERFORMANCE SUMMARY")
        print("=" * 50)
        try:
            display(stats_df.round(4))
        except NameError:
            print(stats_df.round(4))
        
        # --- Visualization ---
        logger.info("Generating tear sheet...")
        tear_sheet_path = create_tear_sheet(
            list(enhanced_results.values()), sweep_tags, config
        )
        
        # Simple comparison plot as backup
        simple_plot_path = create_simple_comparison_plot(
            list(enhanced_results.values()), sweep_tags, config
        )
        
        # --- Performance Summary ---
        elapsed_time = time.time() - start_time
        print(f"\n🎯 SINGLE-TARGET SIMULATION COMPLETE")
        print(f"   ⏱️  Runtime: {elapsed_time:.1f} seconds")
        print(f"   📈 Strategies: {len(regout_list)}")
        print(f"   📊 Reports generated in ./reports/")
        
        if tear_sheet_path:
            print(f"   📄 Tear Sheet: {tear_sheet_path}")
        if simple_plot_path:
            print(f"   📈 Comparison: {simple_plot_path}")
            
        logger.info(f"Simulation completed successfully in {elapsed_time:.1f}s")
        
        return stats_df, enhanced_results
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        print(f"❌ Simulation failed: {e}")
        raise

if __name__ == "__main__":
    main()
