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
3. A detailed performance report will be generated using the `quantstats` library.
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

# Scikit-learn and statsmodels
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as rmse, mean_absolute_error as mae, r2_score

# Utility functions from our custom library
from .utils_simulate import (
    simplify_teos, log_returns, generate_train_predict_calender,
    StatsModelsWrapper_with_OLS, p_by_year, create_results_xarray,
    plot_xarray_results, calculate_performance_metrics
)

# Automated performance analytics
try:
    import quantstats as qs
except ImportError:
    print("quantstats not found. Please install it: pip install quantstats")
    qs = None


# --- Core Simulation Functions ---

def L_func_2(df, pred_col='predicted', params=[]):
    """Binary position sizing: long if prediction > 0, else short."""
    t_conditions = [df[pred_col] <= 0, df[pred_col] > 0]
    t_positions = [params[0], params[1]]
    return np.select(t_conditions, t_positions, default=np.nan)

def L_func_3(df, pred_col='preds_index', params=[]):
    """Quartile-based position sizing based on prediction confidence."""
    t_conditions = [
        (df[pred_col].between(0.00, 0.25)),
        (df[pred_col].between(0.25, 0.50)),
        (df[pred_col].between(0.50, 0.75)),
        (df[pred_col].between(0.75, 1.00))
    ]
    t_positions = params
    return np.select(t_conditions, t_positions, default=np.nan)

def L_func_4(ds, params=[]):
    """Alternative quartile position sizing (operates on a Series)."""
    t_conditions = [
        (ds.between(0.00, 0.25)),
        (ds.between(0.25, 0.50)),
        (ds.between(0.50, 0.75)),
        (ds.between(0.75, 1.00))
    ]
    t_positions = params
    return np.select(t_conditions, t_positions, default=np.nan)

def sim_stats(regout_list, sweep_tags, author='CG', trange=None):
    """
    Calculates and prints comprehensive simulation statistics.
    It also uses quantstats for a detailed HTML report if available.
    """
    df = pd.DataFrame(dtype=object)
    df.index.name = 'metric'
    print('SIMULATION RANGE:', 'from', trange.start, 'to', trange.stop)

    for n, testlabel in enumerate(sweep_tags):
        reg_out = regout_list[n].loc[trange, :]

        # --- Metric Calculation Confirmation ---
        # Annualized Return: (Daily Mean Return) * 252
        mean = 252 * reg_out.perf_ret.mean()
        # Annualized Volatility: (Daily StDev) * sqrt(252)
        std = (np.sqrt(252)) * reg_out.perf_ret.std()
        # Sharpe Ratio: (Annualized Return) / (Annualized Volatility)
        # Note: Assumes a risk-free rate of 0.
        sharpe = mean / std if std != 0 else np.nan

        df.loc['return', testlabel] = mean
        df.loc['stdev', testlabel] = std
        df.loc['sharpe', testlabel] = sharpe
        df.loc['avg_beta', testlabel] = reg_out.leverage.mean()
        df.loc['beta_1_return', testlabel] = df.loc['return', testlabel] / reg_out.leverage.mean() if reg_out.leverage.mean() != 0 else np.nan
        df.loc['pos_bet_ratio', testlabel] = (
            np.sum(np.isfinite(reg_out['prediction']) & (reg_out['prediction'] > 0)) /
            np.sum(np.isfinite(reg_out['prediction'])) if np.sum(np.isfinite(reg_out['prediction'])) > 0 else np.nan
        )
        df.loc['rmse', testlabel] = np.sqrt(rmse(reg_out.prediction, reg_out.actual))
        df.loc['mae', testlabel] = mae(reg_out.prediction, reg_out.actual)
        df.loc['r2', testlabel] = r2_score(reg_out.actual, reg_out.prediction)

        # Benchmark calculations
        bench_ret = 252 * reg_out.actual.mean()
        bench_std = (np.sqrt(252)) * reg_out.actual.std()
        df.loc['benchmark return', testlabel] = bench_ret
        df.loc['benchmark std', testlabel] = bench_std
        df.loc['benchmark sharpe', testlabel] = bench_ret / bench_std if bench_std != 0 else np.nan

        df.loc['beg_pred', testlabel] = min(reg_out.prediction.index).date()
        df.loc['end_pred', testlabel] = max(reg_out.prediction.index).date()
        df.loc['author', testlabel] = author

        # Automated reporting with quantstats
        if qs:
            # Ensure the reports directory exists
            os.makedirs('reports', exist_ok=True)
            report_filename = f'reports/{testlabel}_report.html'
            print(f"\n--- Generating QuantStats HTML report for: {testlabel} ---")
            print(f"    To view, open the file: {report_filename}")
            try:
                # Manually create a complete daily index to fix reporting issues.
                # quantstats requires a series with a value for every single day.
                returns = reg_out['perf_ret']
                benchmark = reg_out['actual']

                # Create a full daily date range from the first to the last date.
                full_date_range = pd.date_range(start=returns.index.min(), end=returns.index.max(), freq='D')

                # Reindex the series to the full daily range, filling non-trading days with 0.
                daily_returns = returns.reindex(full_date_range).fillna(0)
                daily_benchmark = benchmark.reindex(full_date_range).fillna(0)

                # Generate a full HTML report
                qs.reports.html(daily_returns, benchmark=daily_benchmark,
                                        output=report_filename, title=f'{testlabel} Performance')
            except Exception as e:
                print(f"Could not generate QuantStats report for {testlabel}: {e}")

    return df


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
    all_etf_closing_prices_df = yf.download(etf_list, start=start_date)['Close']
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
    Main function to configure and run the simulation.
    """
    # --- Simulation Configuration ---
    config = {
        "target_etf": "SPY",
        "feature_etfs": ['SPY','XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU'],
        "start_date": "2022-01-01",
        "window_size": 400,
        "window_type": "expanding",
        "author": "CG"
    }

    # --- Data Loading ---
    X, y = load_and_prepare_data(
        config["feature_etfs"], 
        config["target_etf"], 
        config["start_date"]
    )
    
    # --- Parameter Sweep Setup ---
    n_ewa_lags_list = [2, 4, 8]
    sweep_tags = [f'ewa_halflife_n={x}' for x in n_ewa_lags_list]
    
    X_list = [X.ewm(halflife=n, min_periods=n).mean().dropna() for n in n_ewa_lags_list]
    y_list = [y.reindex(X_list[n].index) for n in range(len(X_list))]

    regout_list = []
    zscaler = StandardScaler().set_output(transform='pandas')

    # --- Run Simulation Sweep ---
    for n, tag in enumerate(sweep_tags):
        pipe_steps = [
            ('scaler', StandardScaler()),
            ('final_estimator', StatsModelsWrapper_with_OLS(X_list[n], y_list[n]))
        ]

        regout_df, _ = Simulate(
            X=X_list[n],
            y=y_list[n],
            window_size=config["window_size"],
            window_type=config["window_type"],
            pipe_steps=pipe_steps,
            tag=tag
        )

        # If simulation produced no results, skip to the next sweep
        if regout_df.empty:
            continue

        # Process results
        regout_df['preds_index'] = norm.cdf(zscaler.fit_transform(regout_df[['prediction']]))
        regout_df['actual'] = y.loc[regout_df.index].dropna()
        regout_df['leverage'] = L_func_3(regout_df, pred_col='preds_index', params=[0, 0.5, 1.5, 2])
        regout_df['perf_ret'] = regout_df['leverage'] * regout_df['actual']
        regout_list.append(regout_df)

    # --- Analyze and Report Results ---
    trange = slice(regout_list[-1].index[0], regout_list[-1].index[-1])
    stats_df = sim_stats(regout_list, sweep_tags, author=config["author"], trange=trange)
    
    print("\n--- Summary Statistics ---")
    try:
        display(stats_df)
    except NameError:
        print(stats_df)
    
    # Plot cumulative returns
    all_perf_df = pd.concat([df['perf_ret'].rename(tag) for df, tag in zip(regout_list, sweep_tags)], axis=1)
    all_perf_df.cumsum().plot(figsize=(15,8), title="Cumulative Strategy Returns")
    plt.ylabel("Cumulative Log-Return")
    plt.show()

if __name__ == "__main__":
    main()
