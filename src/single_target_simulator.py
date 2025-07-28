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
from utils_simulate import (
    simplify_teos, log_returns, generate_train_predict_calender,
    StatsModelsWrapper_with_OLS
)

# Professional plotting utilities
from plotting_utils import create_professional_tear_sheet, create_simple_comparison_plot



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
    """Calculate the Information Ratio between strategy and benchmark returns."""
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
                           feature_etfs=None, benchmark_manager=None, config=None):
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

            df.loc['start_date', testlabel] = min(reg_out.prediction.index).date()
            df.loc['end_date', testlabel] = max(reg_out.prediction.index).date()
            df.loc['author', testlabel] = author
            
            # Store enhanced results for plotting
            results[testlabel] = reg_out.copy()


        except Exception as e:
            logger.error(f"Error calculating statistics for {testlabel}: {e}")
            continue

    return df, results


def Simulate(X, y, window_size=400, window_type='expanding', pipe_steps={}, param_grid={}, tag=None):
    """
    Walk-forward simulation engine.
    Trains a model on historical data and predicts the next period.
    """
    regout = pd.DataFrame(index=y.index)
    fit_list = []

    date_ranges = generate_train_predict_calender(X, window_type=window_type, window_size=window_size)

    if not date_ranges:
        print(f"\nWarning: Not enough data to run simulation for tag '{tag}' with window size {window_size}.")
        print(f"    Required data points: >{window_size}, Data points available: {len(X)}")
        return pd.DataFrame(), []

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
    return regout.dropna(), fit_list


def load_and_prepare_data(etf_list, target_etf, start_date=None):
    """
    Downloads, processes, and prepares feature (X) and target (y) data.
    The target `y` for a given day `t` is the return of the `target_etf` on day `t+1`.
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

            regout_df, _ = Simulate(
                X=X_processed[ewa_lag],
                y=y_processed[ewa_lag],
                window_size=config["window_size"],
                window_type=config["window_type"],
                pipe_steps=pipe_steps,
                tag=tag
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
            config=config
        )
        
        print("\n📊 ENHANCED PERFORMANCE SUMMARY")
        print("=" * 50)
        try:
            display(stats_df.round(4))
        except NameError:
            print(stats_df.round(4))
        
        # --- Professional Visualization ---
        logger.info("Generating professional tear sheet...")
        tear_sheet_path = create_professional_tear_sheet(
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
