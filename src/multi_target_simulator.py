# -*- coding: utf-8 -*-
"""
Multi-Target Quantitative Trading Simulation Framework

Purpose:
This script provides a framework for backtesting quantitative trading strategies
using multi-target regression. It leverages sklearn's multi-target capabilities
to predict returns for multiple ETFs simultaneously, enabling more sophisticated
portfolio construction and risk management.

NEW: Optionally enhanced with riskmodels.net integration for advanced risk analysis.

Core Simulation Flow:
1. Data Loading and Preparation
2. Multi-Target Model Training
3. Walk-Forward Prediction
4. Portfolio Construction
5. Performance Analysis

How to Use:
1. Configure target ETFs and features in the main() function
2. Choose multi-target compatible estimators (most sklearn regressors work)
3. Run the script to get portfolio strategies based on multi-target predictions
4. OPTIONAL: Configure riskmodels.net API key for enhanced risk analysis

Note: Most utility functions have been moved to multi_target_utils.py for better organization.
"""

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from typing import List, Tuple, Any, Dict, Optional
import logging

# Scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as rmse, mean_absolute_error as mae, r2_score
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Import utilities from our organized utils module
try:
    from .multi_target_utils import (
    # Configuration and logging
    setup_logging, SimulationConfig, BenchmarkConfig, format_benchmark_name,
    
    # Data caching and storage
    download_etf_data_with_cache, clean_yfinance_cache, list_yfinance_cache,
    
    # Simulation metadata and hashing
    generate_simulation_metadata, generate_simulation_hash,
    save_simulation_results, load_simulation_results,
    get_born_on_date_from_zarr, reconstruct_pipeline_from_metadata,
    
    # Risk model integration (riskmodels.net)
    load_riskmodels_data, risk_adjusted_portfolio_optimization,
    
    # Position sizing functions
    L_func_multi_target_equal_weight_long_only, L_func_multi_target_equal_weight, 
    L_func_multi_target_confidence_weighted, L_func_multi_target_long_short,
    # Position sizer classes and helpers
    EqualWeight, ConfidenceWeighted, LongShort,
    calculate_individual_position_weights, _determine_strategy_type,
    
    # Performance metrics
    calculate_performance_metrics, calculate_information_ratio,
    
    # Constants
    TRADING_DAYS_PER_YEAR
)
except ImportError:
    from multi_target_utils import (
        # Configuration and logging
        setup_logging, SimulationConfig, BenchmarkConfig, format_benchmark_name,
        
        # Data caching and storage
        download_etf_data_with_cache, clean_yfinance_cache, list_yfinance_cache,
        
        # Simulation metadata and hashing
        generate_simulation_metadata, generate_simulation_hash,
        save_simulation_results, load_simulation_results,
        get_born_on_date_from_zarr, reconstruct_pipeline_from_metadata,
        
        # Risk model integration (riskmodels.net)
        load_riskmodels_data, risk_adjusted_portfolio_optimization,
        
        # Position sizing functions
        L_func_multi_target_equal_weight_long_only, L_func_multi_target_equal_weight, 
        L_func_multi_target_confidence_weighted, L_func_multi_target_long_short,
        # Position sizer classes and helpers
        EqualWeight, ConfidenceWeighted, LongShort,
        calculate_individual_position_weights, _determine_strategy_type,
        
        # Performance metrics
        calculate_performance_metrics, calculate_information_ratio,
        
        # Constants
        TRADING_DAYS_PER_YEAR
    )

# Import existing utilities from utils_simulate
try:
    from .utils_simulate import (
        simplify_teos, log_returns, generate_train_predict_calender,
        StatsModelsWrapper_with_OLS, p_by_year, EWMTransformer,
        create_results_xarray, plot_xarray_results
    )
except ImportError:
    from utils_simulate import (
        simplify_teos, log_returns, generate_train_predict_calender,
        StatsModelsWrapper_with_OLS, p_by_year, EWMTransformer,
        create_results_xarray, plot_xarray_results
    )

# Set up logging
logger = setup_logging()

def generate_train_predict_calendar_with_frequency(X, train_frequency, window_type, window_size):
    """
    Generate training calendar based on specified frequency.
    
    This function creates the schedule for when to retrain models based on
    the specified frequency (daily, weekly, monthly).
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature data with DatetimeIndex
    train_frequency : str
        'daily', 'weekly', or 'monthly'
    window_type : str
        'expanding' or 'rolling'
    window_size : int
        Size of training window for rolling type
        
    Returns:
    --------
    list
        List of (train_start, train_end, predict_date) tuples
    """
    
    logger.info(f"Generating training calendar: {train_frequency} frequency, {window_type} window")
    
    dates = X.index
    calendar = []
    
    # Determine retraining frequency
    if train_frequency == 'daily':
        retrain_interval = 1
    elif train_frequency == 'weekly':
        retrain_interval = 5  # Business days
    elif train_frequency == 'monthly':
        retrain_interval = 21  # Business days
    else:
        raise ValueError(f"Invalid train_frequency: {train_frequency}")
    
    # Generate calendar
    for i in range(window_size, len(dates), retrain_interval):
        predict_date = dates[i]
        
        if window_type == 'expanding':
            train_start = dates[0]
            train_end = dates[i-1]
        else:  # rolling
            train_start = dates[max(0, i-window_size)]
            train_end = dates[i-1]
        
        calendar.append((train_start, train_end, predict_date))
    
    logger.info(f"Generated {len(calendar)} training periods")
    return calendar

def _prepare_training_data(X: pd.DataFrame, y_multi: pd.DataFrame, 
                          train_start: pd.Timestamp, train_end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare training data for the specified date range."""
    
    train_mask = (X.index >= train_start) & (X.index <= train_end)
    fit_X = X.loc[train_mask].copy()
    fit_y = y_multi.loc[train_mask].copy()
    
    # Remove any remaining NaN values
    clean_mask = fit_X.notna().all(axis=1) & fit_y.notna().all(axis=1)
    fit_X = fit_X.loc[clean_mask]
    fit_y = fit_y.loc[clean_mask]
    
    return fit_X, fit_y

def _train_model(fit_obj, fit_X: pd.DataFrame, fit_y: pd.DataFrame, 
                target_cols: List[str]) -> Any:
    """Train the multi-target model."""
    
    logger.debug(f"Training model with {fit_X.shape[0]} samples, {fit_X.shape[1]} features, {len(target_cols)} targets")
    
    try:
        # Fit the model
        fitted_model = fit_obj.fit(fit_X, fit_y)
        return fitted_model
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        # Return a simple fallback model
        from sklearn.linear_model import LinearRegression
        fallback_model = MultiOutputRegressor(LinearRegression())
        return fallback_model.fit(fit_X, fit_y)

def _make_predictions(model, pred_X: pd.DataFrame, target_cols: List[str], 
                     predict_date: pd.Timestamp) -> Dict[str, float]:
    """Make predictions for a single date."""
    
    try:
        # Get prediction data for this date
        pred_data = pred_X.loc[[predict_date]]
        
        if pred_data.isna().any().any():
            logger.warning(f"NaN values in prediction data for {predict_date}")
            return {col: 0.0 for col in target_cols}
        
        # Make prediction
        pred_values = model.predict(pred_data)[0]  # Get single prediction
        
        # Create prediction dictionary
        predictions = {target_cols[i]: pred_values[i] for i in range(len(target_cols))}
        
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction failed for {predict_date}: {e}")
        return {col: 0.0 for col in target_cols}

def _calculate_portfolio_returns(regout: pd.DataFrame, target_cols: List[str], 
                               position_func, position_params: List[Any]) -> pd.DataFrame:
    """
    Calculate portfolio returns using the specified position sizing function.
    
    This function applies the position sizing logic to convert predictions
    into portfolio positions and then calculates the resulting returns.
    """
    
    logger.info(f"Calculating portfolio returns using {getattr(position_func, '__name__', 'unknown')} position sizing")
    
    try:
        # Extract predictions and actual returns
        prediction_cols = [f'{col}_pred' for col in target_cols]
        actual_cols = [f'{col}_actual' for col in target_cols]
        
        # Check if all required columns exist
        missing_pred_cols = [col for col in prediction_cols if col not in regout.columns]
        missing_actual_cols = [col for col in actual_cols if col not in regout.columns]
        
        if missing_pred_cols or missing_actual_cols:
            logger.error(f"Missing columns - Predictions: {missing_pred_cols}, Actuals: {missing_actual_cols}")
            # Return zero returns as fallback
            regout['portfolio_return'] = 0.0
            return regout
        
        predictions_df = regout[prediction_cols].copy()
        predictions_df.columns = target_cols  # Remove '_pred' suffix
        
        actuals_df = regout[actual_cols].copy()
        actuals_df.columns = target_cols  # Remove '_actual' suffix
        
        # Apply position sizing function
        positions_df = position_func(predictions_df, position_params)
        
        # Calculate portfolio returns (positions are applied with 1-day lag)
        portfolio_returns = []
        
        for i in range(len(regout)):
            if i == 0:
                # No position on first day
                portfolio_returns.append(0.0)
            else:
                # Use previous day's positions with current day's returns
                prev_positions = positions_df.iloc[i-1]
                current_returns = actuals_df.iloc[i]
                
                # Portfolio return = sum(position * return)
                portfolio_return = (prev_positions * current_returns).sum()
                portfolio_returns.append(portfolio_return)
        
        # Add portfolio returns to regout
        regout['portfolio_return'] = portfolio_returns
        
        # Also store position details for analysis
        for j, col in enumerate(target_cols):
            regout[f'{col}_position'] = positions_df.iloc[:, j]
        
        logger.info(f"Portfolio returns calculated: {len(portfolio_returns)} periods")
        
        return regout
        
    except Exception as e:
        logger.error(f"Failed to calculate portfolio returns: {e}")
        # Return zero returns as fallback
        regout['portfolio_return'] = 0.0
        return regout

def Simulate_MultiTarget(X, y_multi, train_frequency, window_size, window_type, 
                        fit_obj, position_func, tag='multi_target',
                        position_params=None, use_cache=True, 
                        save_results=True, **kwargs):
    """
    Core multi-target simulation engine.
    
    This function implements walk-forward analysis for multi-target prediction,
    training models to predict returns for multiple ETFs simultaneously and
    constructing portfolios based on these predictions.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature data (sector ETFs, technical indicators, etc.)
    y_multi : pd.DataFrame
        Target data (multiple ETF returns to predict)
    train_frequency : str
        How often to retrain: 'daily', 'weekly', 'monthly'
    window_size : int
        Training window size (only used for rolling window)
    window_type : str
        'expanding' or 'rolling' training window
    fit_obj : sklearn estimator
        Model to use (must support multi-target prediction)
    position_func : callable
        Function to convert predictions to portfolio positions
    tag : str
        Identifier for this simulation
    position_params : list
        Parameters to pass to position sizing function
    use_cache : bool
        Whether to use cached results if available
    save_results : bool
        Whether to save results to cache
        
    Returns:
    --------
    pd.DataFrame
        Complete simulation results with predictions, actuals, and portfolio returns
    """
    
    logger.info(f"Starting multi-target simulation: {tag}")
    logger.info(f"Data shape: {X.shape} features, {y_multi.shape} targets")
    logger.info(f"Training: {train_frequency} frequency, {window_type} window of {window_size}")
    
    if position_params is None:
        position_params = []
    
    target_cols = list(y_multi.columns)
    
    # Default position function to equal-weight long-only if not provided
    if position_func is None:
        position_func = L_func_multi_target_equal_weight_long_only
    if position_params is None:
        position_params = []

    # Generate unique simulation hash for caching
    simulation_hash, _metadata_for_hash = generate_simulation_hash(
        X, y_multi, window_size, window_type, [], {}, tag,
        position_func, position_params, train_frequency
    )
    
    logger.info(f"Simulation hash: {simulation_hash}")
    
    # Check cache if enabled
    if use_cache:
        cached_results, cached_metadata = load_simulation_results(simulation_hash, tag)
        if cached_results is not None:
            logger.info("Using cached simulation results")
            return cached_results
    
    # Generate training calendar
    calendar = generate_train_predict_calendar_with_frequency(
        X, train_frequency, window_type, window_size
    )
    
    # Initialize results storage
    results = []
    
    logger.info(f"Running simulation with {len(calendar)} training periods...")
    
    # Walk-forward simulation
    for i, (train_start, train_end, predict_date) in enumerate(calendar):
        if i % 50 == 0:
            logger.info(f"Processing period {i+1}/{len(calendar)}: {predict_date}")
        
        try:
            # 1. Prepare training data
            fit_X, fit_y = _prepare_training_data(X, y_multi, train_start, train_end)
            
            if len(fit_X) < 20:  # Minimum training samples
                logger.warning(f"Insufficient training data for {predict_date}: {len(fit_X)} samples")
                continue
            
            # 2. Train model
            trained_model = _train_model(fit_obj, fit_X, fit_y, target_cols)
            
            # 3. Make predictions
            predictions = _make_predictions(trained_model, X, target_cols, predict_date)
            
            # 4. Get actual returns for this date
            if predict_date in y_multi.index:
                actuals = y_multi.loc[predict_date].to_dict()
            else:
                logger.warning(f"No actual data for {predict_date}")
                actuals = {col: 0.0 for col in target_cols}
            
            # 5. Store results
            result_row = {'date': predict_date}
            
            # Add predictions and actuals
            for col in target_cols:
                result_row[f'{col}_pred'] = predictions[col]
                result_row[f'{col}_actual'] = actuals[col]
            
            results.append(result_row)
            
        except Exception as e:
            logger.error(f"Error processing {predict_date}: {e}")
            continue
    
    # Convert results to DataFrame
    regout = pd.DataFrame(results)
    regout.set_index('date', inplace=True)
    regout.sort_index(inplace=True)
    
    logger.info(f"Simulation completed: {len(regout)} predictions generated")
    
    # Calculate portfolio returns
    regout = _calculate_portfolio_returns(regout, target_cols, position_func, position_params)
    
    # Generate metadata for reproducibility
    metadata = generate_simulation_metadata(
        X, y_multi, window_size, window_type, [], {},
        tag, position_func, position_params, train_frequency,
        etf_symbols=list(X.columns), target_etfs=target_cols
    )
    
    # Save results if requested
    if save_results:
        save_simulation_results(regout, simulation_hash, tag, metadata)
    
    logger.info(f"Multi-target simulation completed successfully: {tag}")
    
    return regout

def load_and_prepare_multi_target_data(etf_list: List[str], target_etfs: List[str], 
                                      start_date: str = '2010-01-01', 
                                      end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare data for multi-target simulation.
    
    This function downloads ETF data, calculates returns, and splits into
    features (predictors) and targets (assets to predict).
    
    Parameters:
    -----------
    etf_list : List[str]
        Complete list of ETFs (features + targets)
    target_etfs : List[str]
        Subset of ETFs to use as prediction targets
    start_date : str
        Start date for data download
    end_date : str, optional
        End date for data download
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        (X_features, y_targets) ready for simulation
    """
    
    logger.info(f"Loading multi-target data: {len(etf_list)} ETFs, {len(target_etfs)} targets")
    
    # Download data with caching
    price_data = download_etf_data_with_cache(
        etf_list, 
        start_date=start_date, 
        end_date=end_date
    )
    
    if price_data is None:
        raise ValueError("Failed to load ETF data")
    
    # Extract closing prices and calculate returns
    if len(etf_list) == 1:
        # Single ETF case
        close_prices = price_data['Close'].to_frame()
        close_prices.columns = etf_list
    else:
        # Multiple ETFs
        close_prices = price_data.xs('Close', level=1, axis=1)
    
    # Calculate log returns
    returns = log_returns(close_prices).dropna()
    
    logger.info(f"Calculated returns: {returns.shape}")
    
    # Split into features and targets
    feature_etfs = [etf for etf in etf_list if etf not in target_etfs]
    
    if not feature_etfs:
        # If no separate features, use lagged targets as features
        logger.info("No separate features specified, using lagged target returns")
        X_features = returns[target_etfs].shift(1).dropna()
        y_targets = returns[target_etfs].loc[X_features.index]
    else:
        # Use separate feature ETFs
        X_features = returns[feature_etfs]
        y_targets = returns[target_etfs]
        
        # Align indices
        common_idx = X_features.index.intersection(y_targets.index)
        X_features = X_features.loc[common_idx]
        y_targets = y_targets.loc[common_idx]
    
    logger.info(f"Features: {X_features.shape}, Targets: {y_targets.shape}")
    
    return X_features, y_targets

# NEW: Enhanced Simulate_MultiTarget function with risk model integration
def Simulate_MultiTarget_Enhanced(target_etfs, feature_etfs, 
                                window_size=400, window_type='expanding',
                                model='ridge', model_params=None,
                                position_sizer='equal_weight',
                                start_date='2020-01-01', end_date=None,
                                riskmodels_api_key=None,
                                enable_risk_adjustment=True,
                                **kwargs):
    """
    Enhanced multi-target simulation with riskmodels.net integration
    
    This enhanced version incorporates risk models to improve
    portfolio construction, risk management, and performance attribution.
    
    NEW Parameters:
    --------------
    riskmodels_api_key : str, optional
        API key for riskmodels.net integration
    enable_risk_adjustment : bool, default True
        Whether to apply risk model adjustments to portfolio weights
    
    Educational Benefits:
    --------------------
    - Learn how asset managers integrate risk models
    - Understand factor-based portfolio optimization
    - Practice professional-quality risk management techniques
    - Master multi-dimensional financial data analysis
    
    Returns:
    --------
    Enhanced simulation results with risk factor attribution
    """
    
    logger.info("Starting enhanced multi-target simulation with risk model integration")
    
    # Load market data
    all_etfs = target_etfs + feature_etfs
    X_features, y_targets = load_and_prepare_multi_target_data(
        all_etfs, target_etfs, start_date, end_date
    )
    
    # Set up model
    if model_params is None:
        model_params = {}
    
    if model == 'ridge':
        fit_obj = MultiOutputRegressor(Ridge(**model_params))
    elif model == 'rf':
        fit_obj = MultiOutputRegressor(RandomForestRegressor(**model_params))
    else:
        fit_obj = MultiOutputRegressor(LinearRegression(**model_params))
    
    # Set up position sizing
    if position_sizer == 'equal_weight':
        position_func = L_func_multi_target_equal_weight_long_only  # Default to long-only
        position_params = []
    elif position_sizer == 'confidence_weighted':
        position_func = L_func_multi_target_confidence_weighted
        position_params = [2.0]  # max leverage
    elif position_sizer == 'long_short':
        position_func = L_func_multi_target_long_short
        position_params = [1.0]  # target leverage
    else:
        position_func = L_func_multi_target_equal_weight_long_only  # Default to long-only
        position_params = []
    
    # Load risk model data if API key provided
    risk_data = None
    if riskmodels_api_key and enable_risk_adjustment:
        logger.info("Loading riskmodels.net data for enhanced analysis")
        risk_data = load_riskmodels_data(
            target_etfs, 
            riskmodels_api_key,
            start_date=start_date,
            end_date=end_date
        )
        
        if risk_data is not None:
            logger.info("Risk model data loaded successfully")
        else:
            logger.warning("Risk model data not available, proceeding without risk adjustment")
    
    # Run standard multi-target simulation
    logger.info("Running core multi-target simulation...")
    
    regout = Simulate_MultiTarget(
        X_features, y_targets,
        train_frequency='monthly',
        window_size=window_size,
        window_type=window_type,
        fit_obj=fit_obj,
        position_func=position_func,
        tag=f"enhanced_{position_sizer}",
        position_params=position_params
    )
    
    # Prepare enhanced results dictionary
    enhanced_results = {
        'predictions': regout[[col for col in regout.columns if col.endswith('_pred')]],
        'returns': regout[[col for col in regout.columns if col.endswith('_actual')]],
        'portfolio_returns': regout['portfolio_return']
    }
    
    # Apply risk model enhancements if available
    if risk_data is not None and enable_risk_adjustment:
        logger.info("Applying risk model enhancements...")
        
        # Prepare predictions DataFrame for risk adjustment
        pred_cols = [col for col in regout.columns if col.endswith('_pred')]
        predictions_df = regout[pred_cols].copy()
        predictions_df.columns = [col.replace('_pred', '') for col in predictions_df.columns]
        
        # Enhanced portfolio optimization
        optimized_weights = risk_adjusted_portfolio_optimization(
            predictions_df,
            risk_data
        )
        
        enhanced_results['risk_adjusted_weights'] = optimized_weights
        enhanced_results['risk_data'] = risk_data
        
        # Calculate risk-adjusted portfolio returns
        actual_cols = [col for col in regout.columns if col.endswith('_actual')]
        actuals_df = regout[actual_cols].copy()
        actuals_df.columns = [col.replace('_actual', '') for col in actuals_df.columns]
        
        # Align weights and returns
        common_dates = optimized_weights.index.intersection(actuals_df.index)
        if len(common_dates) > 1:
            aligned_weights = optimized_weights.loc[common_dates]
            aligned_returns = actuals_df.loc[common_dates]
            
            # Calculate risk-adjusted returns (use previous day's weights)
            risk_adj_returns = (aligned_weights.shift(1) * aligned_returns).sum(axis=1).dropna()
            enhanced_results['risk_adjusted_returns'] = risk_adj_returns
        
        logger.info("Risk model enhancements applied successfully")
    
    return enhanced_results

def main():
    """
    Enhanced main demonstration with comprehensive results display
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime
    
    print("Multi-Target Quantitative Trading Simulation Framework")
    print("Enhanced with riskmodels.net Integration")
    print("=" * 60)
    
    # Configuration
    TARGET_ETFS = ['SPY', 'QQQ', 'IWM']
    FEATURE_ETFS = ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU']
    
    # NEW: riskmodels.net configuration
    RISKMODELS_API_KEY = os.getenv('RISKMODELS_API_KEY', 'demo_key_for_education')
    ENABLE_RISK_MODELS = False
    
    print(f"Target ETFs: {TARGET_ETFS}")
    print(f"Feature ETFs: {len(FEATURE_ETFS)} sector ETFs")
    print(f"Risk Models: {'Enabled' if ENABLE_RISK_MODELS else 'Disabled'}")
    
    if ENABLE_RISK_MODELS:
        print("\nRunning enhanced simulation")
        print("-" * 40)
        
        try:
            # Run enhanced simulation
            enhanced_results = Simulate_MultiTarget_Enhanced(
                TARGET_ETFS,
                FEATURE_ETFS,
                window_size=252,  # 1 year of training data
                riskmodels_api_key=RISKMODELS_API_KEY,
                enable_risk_adjustment=True
            )
            
            print(f"\nSimulation completed successfully.")
            print(f"=" * 50)
            
            # Extract results for analysis
            portfolio_returns = enhanced_results['portfolio_returns']
            risk_adjusted_returns = enhanced_results.get('risk_adjusted_returns', portfolio_returns)
            
            # Calculate performance metrics
            def calculate_metrics(returns):
                annual_return = returns.mean() * 252
                annual_vol = returns.std() * np.sqrt(252)
                sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                cumulative = (1 + returns).cumprod()
                max_dd = ((cumulative / cumulative.expanding().max()) - 1).min()
                return {
                    'Annual Return': f"{annual_return:.2%}",
                    'Annual Volatility': f"{annual_vol:.2%}",
                    'Sharpe Ratio': f"{sharpe:.3f}",
                    'Max Drawdown': f"{max_dd:.2%}",
                    'Total Return': f"{(cumulative.iloc[-1] - 1):.2%}",
                    'Trading Days': len(returns)
                }
            
            # Display performance table
            standard_metrics = calculate_metrics(portfolio_returns)
            risk_adj_metrics = calculate_metrics(risk_adjusted_returns)
            
            print(f"\nPerformance summary")
            print(f"{'Metric':<20} {'Standard':<15} {'Risk-Adjusted':<15}")
            print("-" * 50)
            
            for metric in standard_metrics.keys():
                print(f"{metric:<20} {standard_metrics[metric]:<15} {risk_adj_metrics[metric]:<15}")
            
            # Create visualizations
            print(f"\nGenerating visualizations...")
            
            # Set up the plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Multi-Target Portfolio Simulation Results', fontsize=16, fontweight='bold')
            
            # Align data for visualization (use common dates)
            common_dates = portfolio_returns.index.intersection(risk_adjusted_returns.index)
            if len(common_dates) > 0:
                aligned_standard = portfolio_returns.loc[common_dates]
                aligned_risk_adj = risk_adjusted_returns.loc[common_dates]
            else:
                # Fallback: use standard returns only
                aligned_standard = portfolio_returns
                aligned_risk_adj = portfolio_returns.iloc[:0]  # Empty series
            
            # 1. Cumulative Returns
            dates = aligned_standard.index
            cum_standard = (1 + aligned_standard).cumprod()
            
            axes[0,0].plot(dates, cum_standard, label='Standard Portfolio', alpha=0.8, linewidth=2)
            
            if len(aligned_risk_adj) > 0:
                cum_risk_adj = (1 + aligned_risk_adj).cumprod()
                axes[0,0].plot(dates, cum_risk_adj, label='Risk-Adjusted Portfolio', alpha=0.8, linewidth=2)
            axes[0,0].set_title('Cumulative Returns')
            axes[0,0].set_ylabel('Cumulative Return')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # 2. Rolling Sharpe Ratio (60-day)
            rolling_sharpe_std = aligned_standard.rolling(30).mean() / aligned_standard.rolling(30).std() * np.sqrt(252)
            
            axes[0,1].plot(dates, rolling_sharpe_std, label='Standard', alpha=0.8)
            
            if len(aligned_risk_adj) > 0:
                rolling_sharpe_adj = aligned_risk_adj.rolling(30).mean() / aligned_risk_adj.rolling(30).std() * np.sqrt(252)
                axes[0,1].plot(dates, rolling_sharpe_adj, label='Risk-Adjusted', alpha=0.8)
            axes[0,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
            axes[0,1].set_title('Rolling 30-Day Sharpe Ratio')
            axes[0,1].set_ylabel('Sharpe Ratio')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # 3. Drawdown Analysis
            running_max_std = cum_standard.expanding().max()
            drawdown_std = (cum_standard - running_max_std) / running_max_std
            
            axes[1,0].fill_between(dates, drawdown_std, 0, alpha=0.3, color='red', label='Standard')
            
            if len(aligned_risk_adj) > 0:
                running_max_adj = cum_risk_adj.expanding().max()
                drawdown_adj = (cum_risk_adj - running_max_adj) / running_max_adj
                axes[1,0].fill_between(dates, drawdown_adj, 0, alpha=0.3, color='blue', label='Risk-Adjusted')
            axes[1,0].set_title('Portfolio Drawdowns')
            axes[1,0].set_ylabel('Drawdown %')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
            
            # 4. Portfolio Weights (if available)
            if 'risk_adjusted_weights' in enhanced_results:
                weights = enhanced_results['risk_adjusted_weights']
                for i, etf in enumerate(TARGET_ETFS):
                    if etf in weights.columns:
                        axes[1,1].plot(dates[:len(weights)], weights[etf], 
                                     label=etf, alpha=0.8, linewidth=2)
                
                axes[1,1].set_title('Risk-Adjusted Portfolio Weights')
                axes[1,1].set_ylabel('Weight')
                axes[1,1].legend()
                axes[1,1].grid(True, alpha=0.3)
            else:
                axes[1,1].text(0.5, 0.5, 'Portfolio weights\nnot available', 
                             ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Portfolio Weights')
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"reports/multi_target_simulation_results_{timestamp}.png"
            os.makedirs('reports', exist_ok=True)
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            
            print(f"Results visualization saved: {plot_filename}")
            
            # Display the plot
            plt.show()
            
            # Risk Analysis (if available)
            if 'risk_data' in enhanced_results:
                risk_data = enhanced_results['risk_data']
                print(f"\nRisk model analysis")
                print(f"   Factors analyzed: {list(risk_data.data_vars)}")
                print(f"   Assets covered: {list(risk_data.ticker.values)}")
                print(f"   📅 Risk data period: {len(risk_data.time)} days")
                
                # Show average factor exposures
                print(f"\n   Average Factor Exposures:")
                for etf in TARGET_ETFS:
                    if etf in risk_data.ticker.values:
                        print(f"   {etf}:")
                        etf_data = risk_data.sel(ticker=etf)
                        for factor in ['Market', 'Size', 'Value', 'Momentum']:
                            if factor in risk_data.data_vars:
                                avg_exposure = float(etf_data[factor].mean())
                                print(f"     {factor:10s}: {avg_exposure:+.3f}")
            
            # Summary and next steps
            print(f"\nSimulation complete.")
            print(f"Results saved to: {os.path.abspath(plot_filename)}")
            print(f"Performance Summary:")
            print(f"   • Standard Portfolio Sharpe: {standard_metrics['Sharpe Ratio']}")
            print(f"   • Risk-Adjusted Sharpe: {risk_adj_metrics['Sharpe Ratio']}")
            print(f"   • Total Trading Days: {standard_metrics['Trading Days']}")
            
        except Exception as e:
            print(f"Simulation failed: {e}")
            logger.error(f"Main simulation error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nNext steps:")
    print(f"   Explore Tutorial 1: Single-target analysis (notebooks/01_single_target_tutorial.ipynb)")
    print(f"   Try Tutorial 2: Multi-target strategies (notebooks/02_multi_target_tutorial.ipynb)")
    print(f"   Advanced Tutorial 4: riskmodels.net integration (notebooks/04_riskmodels_integration.ipynb)")
    print(f"   Get API access: https://riskmodels.net")
    print(f"   View results: {os.path.abspath('reports/')}")
    print(f"   Check logs: {os.path.abspath('logs/')}")

if __name__ == "__main__":
    main()