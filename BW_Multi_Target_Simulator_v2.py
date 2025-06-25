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

# Scikit-learn and statsmodels
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as rmse, mean_absolute_error as mae, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Utility functions from our custom library
from utils_simulate import (
    simplify_teos, log_returns, generate_train_predict_calender,
    StatsModelsWrapper_with_OLS, p_by_year, EWMTransformer
)

# Automated performance analytics
try:
    import quantstats as qs
except ImportError:
    print("quantstats not found. Please install it: pip install quantstats")
    qs = None


# --- Caching and Performance Utilities ---

def generate_simulation_hash(X, y_multi, window_size, window_type, pipe_steps, param_grid, tag, 
                           position_func, position_params, train_frequency):
    """
    Generate a unique hash for simulation parameters to enable caching.
    """
    # Create a string representation of all parameters
    param_str = f"{X.shape}_{y_multi.shape}_{window_size}_{window_type}_{tag}_{train_frequency}"
    param_str += f"_{str(pipe_steps)}_{str(param_grid)}"
    param_str += f"_{position_func.__name__ if position_func else 'None'}_{str(position_params)}"
    
    # Add data hash (sample of first/last rows to detect data changes)
    data_sample = str(X.iloc[:5].values.tolist() + X.iloc[-5:].values.tolist())
    data_sample += str(y_multi.iloc[:5].values.tolist() + y_multi.iloc[-5:].values.tolist())
    param_str += data_sample
    
    return hashlib.md5(param_str.encode()).hexdigest()

def save_simulation_results(regout_df, simulation_hash, tag):
    """
    Save simulation results to disk for future reuse.
    """
    os.makedirs('cache', exist_ok=True)
    cache_filename = f'cache/simulation_{simulation_hash}_{tag}.pkl'
    
    with open(cache_filename, 'wb') as f:
        pickle.dump(regout_df, f)
    
    print(f"Saved simulation results: {cache_filename}")
    return cache_filename

def load_simulation_results(simulation_hash, tag):
    """
    Load previously saved simulation results.
    """
    cache_filename = f'cache/simulation_{simulation_hash}_{tag}.pkl'
    
    if os.path.exists(cache_filename):
        with open(cache_filename, 'rb') as f:
            regout_df = pickle.load(f)
        print(f"Loaded cached simulation results: {cache_filename}")
        return regout_df
    
    return None

def generate_train_predict_calendar_with_frequency(X, window_type='expanding', window_size=400, train_frequency='daily'):
    """
    Enhanced calendar generation with configurable training frequency.
    
    Args:
        X: Feature DataFrame
        window_type: 'expanding' or 'rolling'
        window_size: Size of training window
        train_frequency: 'daily', 'weekly', 'monthly' - how often to retrain
    
    Returns:
        List of (train_start, train_end, predict_date) tuples
    """
    dates = X.index
    date_ranges = []
    
    # Determine retraining frequency
    if train_frequency == 'weekly':
        retrain_step = 5  # Retrain every 5 business days
    elif train_frequency == 'monthly':
        retrain_step = 21  # Retrain every ~21 business days (monthly)
    else:  # 'daily'
        retrain_step = 1
    
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


# --- Multi-Target Position Sizing Functions ---

def calculate_individual_position_weights(predictions_row, position_func, position_params):
    """
    Calculate individual position weights for each asset given predictions.
    
    Args:
        predictions_row: Series with predictions for each target
        position_func: Position sizing function
        position_params: Parameters for position function
    
    Returns:
        numpy array with individual weights for each asset
    """
    if position_func.__name__ == 'L_func_multi_target_long_short':
        # Replicate the long-short logic to get individual weights
        ranked = predictions_row.rank(ascending=False)
        n_assets = len(predictions_row)
        base_leverage = position_params[0] if position_params else 1.0
        
        if n_assets == 1:
            weights = np.array([base_leverage if predictions_row.iloc[0] > 0 else -base_leverage])
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
    
    elif position_func.__name__ == 'L_func_multi_target_confidence_weighted':
        # For confidence weighted: distribute leverage across all assets based on confidence
        base_leverage = position_params[0] if position_params else 2.0
        confidence = predictions_row.abs()
        avg_prediction = predictions_row.mean()
        
        if confidence.sum() > 0:
            confidence_normalized = confidence / confidence.sum()
            weights = np.sign(avg_prediction) * confidence_normalized * base_leverage
        else:
            weights = np.zeros(len(predictions_row))
            
        return weights
    
    elif position_func.__name__ == 'L_func_multi_target_equal_weight':
        # For equal weight: same weight for all assets based on average prediction
        base_leverage = position_params[0] if position_params else 1.0
        avg_prediction = predictions_row.mean()
        n_assets = len(predictions_row)
        
        if avg_prediction > 0:
            weights = np.full(n_assets, base_leverage / n_assets)
        else:
            weights = np.full(n_assets, -base_leverage / n_assets)
            
        return weights
    
    else:
        # Fallback: equal weights
        n_assets = len(predictions_row)
        return np.full(n_assets, 1.0 / n_assets)

def L_func_multi_target_equal_weight(predictions_df, params=[]):
    """
    Equal-weight position sizing across all predicted targets.
    
    Args:
        predictions_df: DataFrame with predictions for each target
        params: [base_leverage] - base leverage to apply
    
    Returns:
        Series with portfolio leverage for each date
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
        plt.figure(figsize=(15, 12))
        
        # Create 3 subplots for each target
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # --- Subplot 1: Cumulative Returns Comparison ---
        ax1.set_title(f'{target} - Strategy vs Buy & Hold Comparison', fontsize=14, fontweight='bold')
        
        # Plot strategy returns for this target
        for i, (regout_df, tag) in enumerate(zip(regout_list, sweep_tags)):
            if f'actual_{target}' in regout_df.columns:
                # Strategy return: leverage * actual target return
                strategy_ret = regout_df['leverage'] * regout_df[f'actual_{target}']
                strategy_ret.cumsum().plot(ax=ax1, label=f'{tag} Strategy', alpha=0.7)
        
        # Buy & Hold benchmark (1.0 leverage)
        if regout_list and f'actual_{target}' in regout_list[0].columns:
            buy_hold_ret = regout_list[0][f'actual_{target}']
            buy_hold_ret.cumsum().plot(ax=ax1, label=f'{target} Buy & Hold (1.0x)', 
                                     linestyle='--', linewidth=3, color='black')
        
        ax1.set_ylabel('Cumulative Log-Return')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # --- Subplot 2: Rolling Sharpe Ratio (252-day window) ---
        ax2.set_title(f'{target} - Rolling Sharpe Ratio (252-day window)', fontsize=12)
        
        for i, (regout_df, tag) in enumerate(zip(regout_list, sweep_tags)):
            if f'actual_{target}' in regout_df.columns:
                strategy_ret = regout_df['leverage'] * regout_df[f'actual_{target}']
                rolling_sharpe = strategy_ret.rolling(252).mean() / strategy_ret.rolling(252).std() * np.sqrt(252)
                rolling_sharpe.plot(ax=ax2, label=f'{tag} Strategy', alpha=0.7)
        
        # Buy & Hold rolling Sharpe
        if regout_list and f'actual_{target}' in regout_list[0].columns:
            buy_hold_ret = regout_list[0][f'actual_{target}']
            bh_rolling_sharpe = buy_hold_ret.rolling(252).mean() / buy_hold_ret.rolling(252).std() * np.sqrt(252)
            bh_rolling_sharpe.plot(ax=ax2, label=f'{target} Buy & Hold', 
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
                drawdown.plot(ax=ax3, label=f'{tag} Strategy', alpha=0.7)
        
        # Buy & Hold drawdown
        if regout_list and f'actual_{target}' in regout_list[0].columns:
            buy_hold_ret = regout_list[0][f'actual_{target}']
            bh_cum_ret = buy_hold_ret.cumsum()
            bh_running_max = bh_cum_ret.expanding().max()
            bh_drawdown = bh_cum_ret - bh_running_max
            bh_drawdown.plot(ax=ax3, label=f'{target} Buy & Hold', 
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


def create_performance_summary_table(regout_list, sweep_tags, target_etfs):
    """
    Creates a detailed performance summary table for each target and strategy.
    """
    summary_data = []
    
    for i, (regout_df, tag) in enumerate(zip(regout_list, sweep_tags)):
        for target in target_etfs:
            if f'actual_{target}' in regout_df.columns:
                # Strategy metrics
                strategy_ret = regout_df['leverage'] * regout_df[f'actual_{target}']
                buy_hold_ret = regout_df[f'actual_{target}']
                
                # Calculate metrics
                strategy_annual_ret = strategy_ret.mean() * 252
                strategy_annual_vol = strategy_ret.std() * np.sqrt(252)
                strategy_sharpe = strategy_annual_ret / strategy_annual_vol if strategy_annual_vol != 0 else np.nan
                
                buy_hold_annual_ret = buy_hold_ret.mean() * 252
                buy_hold_annual_vol = buy_hold_ret.std() * np.sqrt(252)
                buy_hold_sharpe = buy_hold_annual_ret / buy_hold_annual_vol if buy_hold_annual_vol != 0 else np.nan
                
                # Maximum drawdown
                strategy_cum = strategy_ret.cumsum()
                strategy_dd = (strategy_cum - strategy_cum.expanding().max()).min()
                
                buy_hold_cum = buy_hold_ret.cumsum()
                buy_hold_dd = (buy_hold_cum - buy_hold_cum.expanding().max()).min()
                
                summary_data.append({
                    'Strategy': tag,
                    'Target': target,
                    'Strategy_Annual_Return': strategy_annual_ret,
                    'Strategy_Annual_Vol': strategy_annual_vol,
                    'Strategy_Sharpe': strategy_sharpe,
                    'Strategy_Max_DD': strategy_dd,
                    'BuyHold_Annual_Return': buy_hold_annual_ret,
                    'BuyHold_Annual_Vol': buy_hold_annual_vol,
                    'BuyHold_Sharpe': buy_hold_sharpe,
                    'BuyHold_Max_DD': buy_hold_dd,
                    'Excess_Return': strategy_annual_ret - buy_hold_annual_ret,
                    'Sharpe_Improvement': strategy_sharpe - buy_hold_sharpe
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    os.makedirs('reports', exist_ok=True)
    summary_df.to_csv('reports/individual_target_performance_summary.csv', index=False)
    print("Saved individual target performance summary: reports/individual_target_performance_summary.csv")
    
    return summary_df

def sim_stats_multi_target(regout_list, sweep_tags, target_etfs, author='CG', trange=None):
    """
    Calculates comprehensive statistics for multi-target strategies.
    """
    df = pd.DataFrame(dtype=object)
    df.index.name = 'metric'
    print('MULTI-TARGET SIMULATION RANGE:', 'from', trange.start, 'to', trange.stop)

    for n, testlabel in enumerate(sweep_tags):
        reg_out = regout_list[n].loc[trange, :]

        # Portfolio-level metrics
        mean_ret = 252 * reg_out.portfolio_ret.mean()
        std_ret = (np.sqrt(252)) * reg_out.portfolio_ret.std()
        sharpe = mean_ret / std_ret if std_ret != 0 else np.nan

        df.loc['portfolio_return', testlabel] = mean_ret
        df.loc['portfolio_stdev', testlabel] = std_ret
        df.loc['portfolio_sharpe', testlabel] = sharpe
        df.loc['avg_leverage', testlabel] = reg_out.leverage.mean()

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
            
            df.loc['avg_rmse', testlabel] = np.mean(rmse_scores) if rmse_scores else np.nan
            df.loc['avg_mae', testlabel] = np.mean(mae_scores) if mae_scores else np.nan
            df.loc['avg_r2', testlabel] = np.mean(r2_scores) if r2_scores else np.nan

        # Benchmark comparison (equal-weight portfolio of all targets)
        if 'benchmark_ret' in reg_out.columns:
            bench_ret = 252 * reg_out.benchmark_ret.mean()
            bench_std = (np.sqrt(252)) * reg_out.benchmark_ret.std()
            df.loc['benchmark_return', testlabel] = bench_ret
            df.loc['benchmark_std', testlabel] = bench_std
            df.loc['benchmark_sharpe', testlabel] = bench_ret / bench_std if bench_std != 0 else np.nan

        df.loc['beg_pred', testlabel] = reg_out.index.min().date()
        df.loc['end_pred', testlabel] = reg_out.index.max().date()
        df.loc['author', testlabel] = author
        df.loc['n_targets', testlabel] = len(target_etfs)

        # Generate QuantStats report for portfolio
        if qs:
            os.makedirs('reports', exist_ok=True)
            report_filename = f'reports/{testlabel}_multi_target_report.html'
            print(f"\n--- Generating Multi-Target QuantStats report for: {testlabel} ---")
            print(f"    To view, open the file: {report_filename}")
            try:
                returns = reg_out['portfolio_ret']
                benchmark = reg_out['benchmark_ret'] if 'benchmark_ret' in reg_out.columns else None

                full_date_range = pd.date_range(start=returns.index.min(), end=returns.index.max(), freq='D')
                daily_returns = returns.reindex(full_date_range).fillna(0)
                
                if benchmark is not None:
                    daily_benchmark = benchmark.reindex(full_date_range).fillna(0)
                    qs.reports.html(daily_returns, benchmark=daily_benchmark,
                                          output=report_filename, title=f'{testlabel} Multi-Target Performance')
                else:
                    qs.reports.html(daily_returns, output=report_filename, 
                                          title=f'{testlabel} Multi-Target Performance')
            except Exception as e:
                print(f"Could not generate QuantStats report for {testlabel}: {e}")

    return df


def Simulate_MultiTarget(X, y_multi, window_size=400, window_type='expanding', 
                        pipe_steps={}, param_grid={}, tag=None, position_func=None, position_params=[], 
                        train_frequency='daily', use_cache=True):
    """
    Multi-target walk-forward simulation engine with caching and configurable training frequency.
    
    Args:
        X: Feature DataFrame
        y_multi: Multi-target DataFrame (multiple ETF returns)
        window_size: Training window size
        window_type: 'expanding' or 'rolling'
        pipe_steps: Pipeline steps for the model
        param_grid: Parameters for the model
        tag: Label for this simulation run
        position_func: Function to convert predictions to positions
        position_params: Parameters for position function
        train_frequency: 'daily', 'weekly', 'monthly' - how often to retrain model
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
            print(f"Using cached results for {tag}")
            return cached_result
    
    # Initialize results DataFrame
    regout = pd.DataFrame(index=y_multi.index)
    target_cols = y_multi.columns.tolist()
    
    # Use enhanced calendar generation with training frequency
    date_ranges = generate_train_predict_calendar_with_frequency(
        X, window_type=window_type, window_size=window_size, train_frequency=train_frequency
    )

    if not date_ranges:
        print(f"\nWarning: Not enough data for multi-target simulation '{tag}' with window size {window_size}.")
        print(f"    Required data points: >{window_size}, Data points available: {len(X)}")
        return pd.DataFrame()

    # Set up multi-target pipeline
    fit_obj = Pipeline(steps=pipe_steps)
    fit_obj.set_params(**param_grid)

    print(f"Starting multi-target simulation for tag: {tag}...")
    print(f"    Predicting targets: {target_cols}")
    print(f"    Training frequency: {train_frequency}")
    print(f"    Total training iterations: {len(date_ranges)} (vs {len(X)-window_size} for daily)")
    
    last_trained_model = None
    last_training_end = None
    
    for n, dates in enumerate(date_ranges):
        start_training, end_training, prediction_date = dates[0], dates[1], dates[2]

        # Check if we need to retrain the model
        need_retrain = (last_trained_model is None or 
                       last_training_end != end_training)
        
        if need_retrain:
            fit_X = X[start_training:end_training]
            fit_y = y_multi[start_training:end_training]
            
            if n % max(1, len(date_ranges)//10) == 0:  # Progress update
                print(f"  ... training model for date {prediction_date.date()} ({n+1}/{len(date_ranges)})")

            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore')
                fit_obj.fit(fit_X, fit_y)
            
            last_trained_model = fit_obj
            last_training_end = end_training
        
        # Make prediction using current model
        pred_X = X[prediction_date:prediction_date]
        predictions = last_trained_model.predict(pred_X)
        
        # Handle different prediction formats
        if hasattr(predictions, 'values'):
            pred_values = predictions.values[0]
        else:
            pred_values = predictions[0] if predictions.ndim > 1 else predictions

        # Store predictions for each target
        for i, target in enumerate(target_cols):
            regout.loc[prediction_date, f'pred_{target}'] = np.round(pred_values[i], 5)

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

    # Calculate portfolio performance - ENHANCED FOR ALL STRATEGY TYPES
    actual_cols = [f'actual_{target}' for target in target_cols]
    actual_returns = regout[actual_cols].copy()
    actual_returns.columns = target_cols  # Remove 'actual_' prefix

    # Calculate portfolio returns using individual asset weights
    print(f"  ... calculating portfolio returns using individual asset contributions...")
    
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
            
            # Calculate weighted return for this date
            asset_returns = actual_returns.loc[date].values
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

    # Benchmark: equal-weight buy-and-hold of all targets
    regout['benchmark_ret'] = actual_returns.mean(axis=1)

    # Remove rows with NaN values
    regout_clean = regout.dropna()
    
    # Save results to cache
    if use_cache:
        save_simulation_results(regout_clean, simulation_hash, tag)

    print(f"Multi-target simulation for {tag} complete.")
    return regout_clean


def load_and_prepare_multi_target_data(etf_list, target_etfs, start_date=None):
    """
    Downloads and prepares data for multi-target regression.
    
    Args:
        etf_list: List of all ETFs to download (features + targets)
        target_etfs: List of ETFs to predict (subset of etf_list)
        start_date: Start date for data download
    
    Returns:
        X: Feature DataFrame (all ETFs except targets)
        y_multi: Multi-target DataFrame (target ETFs only)
    """
    print(f"Downloading multi-target data from {start_date}...")
    print(f"    Feature ETFs: {[etf for etf in etf_list if etf not in target_etfs]}")
    print(f"    Target ETFs: {target_etfs}")
    
    all_etf_closing_prices_df = yf.download(etf_list, start=start_date)['Close']
    etf_log_returns_df = log_returns(all_etf_closing_prices_df).dropna()

    # Set timezone and align timestamps
    etf_log_returns_df.index = etf_log_returns_df.index.tz_localize('America/New_York').map(
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
    
    print("Multi-target data preparation complete.")
    print(f"    Feature shape: {X.shape}")
    print(f"    Target shape: {y_multi.shape}")
    
    return X, y_multi


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


def main():
    """
    Main execution function for multi-target ETF prediction simulation.
    Runs comprehensive strategy sweep and generates performance reports.
    """
    # Configuration
    config = {
        'train_frequency': 'weekly',  # 'daily', 'weekly', 'monthly'
        'use_cache': True,
        'force_retrain': False,
        'csv_output_dir': '/Volumes/ext_2t/ERM3_Data/stock_data/csv'  # External drive CSV directory
    }
    
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
    X, y_multi = load_and_prepare_multi_target_data(
        etf_list=all_etfs, 
        target_etfs=target_etfs,
        start_date='2015-01-01'
    )
    
    # Strategy sweep configuration
    models = {
        'linear': {'regressor': LinearRegression()},
        'ridge': {'regressor': Ridge(alpha=1.0)},
        'random_forest': {'regressor': MultiOutputRegressor(RandomForestRegressor(n_estimators=50, random_state=42))}
    }
    
    scalers = {'std': StandardScaler(), 'none': None}
    
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
            regout_df = Simulate_MultiTarget(
                X, y_multi,
                window_size=400,
                window_type='expanding',
                pipe_steps=strategy['pipe_steps'],
                param_grid=strategy['param_grid'],
                tag=strategy['tag'],
                position_func=strategy['position_func'],
                position_params=strategy['position_params'],
                train_frequency=config['train_frequency'],
                use_cache=config['use_cache']
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
        if stats_results:
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
        
        # Create plots and save to reports directory (for visualization)
        print("\nGenerating visualization reports...")
        try:
            plot_individual_target_performance(regout_list, sweep_tags, target_etfs, config)
            plot_portfolio_summary(regout_list, sweep_tags, target_etfs, config)
            print("Visualization reports saved to reports/ directory")
        except Exception as e:
            print(f"Warning: Could not generate plots: {str(e)}")
        
        print(f"\n🎉 All simulation results and analysis saved to: {config['csv_output_dir']}")
        print(f"\nRun ID: {timestamp}")
        print("\nCSV Files Created:")
        print(f"  - Individual strategy results: {timestamp}_[strategy_name]_results.csv")
        print(f"  - Performance summary: {timestamp}_performance_summary.csv")
        print(f"  - Detailed statistics: {timestamp}_detailed_statistics.csv") 
        print(f"  - Combined results: {timestamp}_all_strategies_combined.csv")
        print(f"  - Metadata file: {timestamp}_csv_metadata.csv")
        
        # Create CSV metadata file
        metadata_path = create_csv_metadata_file(config['csv_output_dir'], sweep_tags, timestamp)
        print(f"  - Metadata saved to: {metadata_path}")
        
    else:
        print("No successful simulations completed.")

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


if __name__ == "__main__":
    # Run demonstration
    demonstrate_portfolio_return_calculation()
    
    # Run main simulation
    main()
