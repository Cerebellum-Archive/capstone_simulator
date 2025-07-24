"""
Enhanced Quantitative Trading Utilities with xarray Integration

This module provides essential utilities for quantitative trading simulation and analysis,
with native xarray support for multi-dimensional financial data handling. Designed for
educational use in financial engineering programs.

Key Features:
- Time-series data preprocessing and feature engineering
- Statistical analysis tools for feature importance and correlation
- xarray-based result storage and visualization for standardized reporting
- sklearn-compatible transformers for ML pipeline integration
- Educational docstrings explaining financial concepts

Blue Water Macro Corp. © 2025
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score as r_squared
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS
from typing import Dict, List, Optional, Union, Tuple


def simplify_teos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timezone-aware datetime index to timezone-naive normalized dates.
    
    This function is essential for financial data processing as it ensures consistent
    date handling across different data sources (e.g., yfinance, Bloomberg).
    
    Args:
        df: DataFrame with timezone-aware datetime index
        
    Returns:
        DataFrame with normalized timezone-naive datetime index
        
    Educational Note:
        Timezone handling is crucial in global markets. This function standardizes
        all timestamps to prevent alignment issues in multi-source data analysis.
    """
    df.index = df.index.tz_localize(None).normalize()
    return df


def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns from price data.
    
    Log returns are preferred in quantitative finance because:
    1. They're time-additive: log(P_t/P_0) = sum(log(P_i/P_{i-1}))
    2. They're approximately normally distributed for small changes
    3. They handle compounding effects naturally
    
    Args:
        df: DataFrame of price data
        
    Returns:
        DataFrame of log returns (first row will be NaN)
        
    Formula:
        log_return_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
    """
    log_prices = np.log(df)
    return log_prices - log_prices.shift(1)


def p_by_slice(X: pd.DataFrame, y: pd.Series, t_list: List, t_list_labels: List[str]) -> pd.DataFrame:
    """
    Analyze feature significance across different time periods.
    
    This function is crucial for understanding feature stability over time,
    which is essential for robust trading strategy development.
    
    Args:
        X: Feature matrix
        y: Target variable
        t_list: List of time slice indices
        t_list_labels: Labels for each time slice
        
    Returns:
        DataFrame with Pearson correlation coefficients for each feature across time slices
        
    Educational Note:
        Feature stability analysis helps identify regime changes and ensures
        your model's predictive power is consistent across different market conditions.
    """
    feat_stats = pd.DataFrame(index=X.columns)

    for n, idx in enumerate(t_list):
        X_fit = X.loc[idx, :].dropna()
        y_fit = y.reindex(X_fit.index)
        feat_stats.loc[:, t_list_labels[n]] = r_regression(X_fit, y_fit, center=True)

    print(f'Analysis period: {X_fit.index.min()} to {X_fit.index.max()}')
    return feat_stats


def p_by_year(X: pd.DataFrame, y: pd.Series, sort_by: str = 'p_value') -> pd.DataFrame:
    """
    Annual feature analysis for regime detection and stability assessment.
    
    This function helps identify:
    1. Features that work consistently across years
    2. Regime changes in feature importance
    3. Overfitting to specific time periods
    
    Args:
        X: Feature matrix with datetime index
        y: Target variable
        sort_by: Column to sort results by
        
    Returns:
        DataFrame with annual Pearson correlations for each feature
        
    Educational Application:
        Use this to identify which sector ETFs consistently predict SPY returns
        and detect structural breaks in market relationships.
    """
    feat_stats = pd.DataFrame(index=X.columns)

    for year in X.index.year.unique():
        X_fit = X.loc[str(year), :].dropna()
        y_fit = y.reindex(X_fit.index)
        feat_stats.loc[:, str(year)] = r_regression(X_fit, y_fit, center=True)

    print(f'Analysis period: {X_fit.index.min()} to {X_fit.index.max()}')
    return feat_stats


def feature_profiles(X: pd.DataFrame, y: pd.Series, sort_by: str = 'pearson', 
                    t_slice: Optional[slice] = None) -> pd.DataFrame:
    """
    Comprehensive feature analysis including multiple statistical tests.
    
    Provides a complete statistical profile of each feature's relationship
    with the target variable, essential for feature selection in quantitative strategies.
    
    Args:
        X: Feature matrix
        y: Target variable
        sort_by: Column to sort results by ('pearson', 'p_value', 'mutual_info', etc.)
        t_slice: Time slice for analysis (default: entire period)
        
    Returns:
        DataFrame with comprehensive feature statistics
        
    Statistics Included:
        - Pearson correlation: Linear relationship strength
        - P-values: Statistical significance
        - Mutual information: Non-linear relationship detection
        - T-statistics: Hypothesis testing
    """
    if not t_slice:
        t_slice = slice(X.index.min(), X.index.max())
        print(f'Analyzing period: {t_slice}')

    X_fit = X.loc[t_slice, :].dropna()
    y_fit = y.reindex(X_fit.index)
    
    # Calculate comprehensive statistics
    pear_test = r_regression(X_fit, y_fit, center=True)
    abs_pear_test = np.abs(pear_test)
    f_test, p_value = f_regression(X_fit, y_fit, center=False)
    t_test = np.sqrt(f_test)
    mi = mutual_info_regression(X_fit, y_fit)
    nobs = len(X_fit)
    
    feat_stats = pd.DataFrame({
        'nobs': nobs,
        'mutual_info': mi,
        'p_value': p_value,
        't_test': t_test,
        'pearson': pear_test,
        'abs_pearson': abs_pear_test
    }, index=X.columns)
    
    print(f'Feature analysis from {X_fit.index.min()} to {X_fit.index.max()}')
    return feat_stats.sort_values(by=[sort_by], ascending=False)


def generate_train_predict_calender(df: pd.DataFrame, window_type: str, 
                                  window_size: int) -> List[List]:
    """
    Generate training and prediction date ranges for walk-forward analysis.
    
    This is the heart of proper backtesting methodology, ensuring no look-ahead bias
    by using only historical data for each prediction.
    
    Args:
        df: DataFrame with datetime index
        window_type: 'fixed' (rolling window) or 'expanding' (growing window)
        window_size: Number of periods for training window
        
    Returns:
        List of [train_start, train_end, prediction_date] triplets
        
    Educational Note:
        Walk-forward analysis is crucial for realistic performance estimation.
        'expanding' windows show how strategies perform with more historical data,
        while 'fixed' windows test adaptation to recent market conditions.
    """
    date_ranges = []
    index = df.index
    num_days = len(index)

    if window_type == 'fixed':
        # Rolling window: Use only recent window_size periods for training
        for i in range(window_size, num_days):
            train_start_date = index[i - window_size]
            train_end_date = index[i - 1]
            prediction_date = index[i]
            date_ranges.append([train_start_date, train_end_date, prediction_date])

    elif window_type == 'expanding':
        # Expanding window: Use all available historical data
        for i in range(window_size, num_days):
            train_start_date = index[0]
            train_end_date = index[i - 1]
            prediction_date = index[i]
            date_ranges.append([train_start_date, train_end_date, prediction_date])

    return date_ranges


def graph_df(df: pd.DataFrame, w: int = 10, h: int = 15) -> None:
    """
    Plot multiple time series in a DataFrame with proper formatting.
    
    Args:
        df: DataFrame with time series data
        w: Figure width
        h: Figure height
    """
    fig, axes = plt.subplots(nrows=len(df.columns), ncols=1, figsize=(w, h))
    
    if len(df.columns) == 1:
        axes = [axes]  # Ensure axes is always iterable
    
    for i, col in enumerate(df.columns):
        axes[i].plot(df[col])
        axes[i].set_title(col)
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


class StatsModelsWrapper_with_OLS(BaseEstimator, RegressorMixin):
    """
    Sklearn-compatible wrapper for statsmodels OLS regression.
    
    This wrapper allows using statsmodels' rich statistical output
    within sklearn pipelines, providing detailed regression diagnostics
    essential for quantitative finance applications.
    
    Educational Value:
        Statsmodels provides R-squared, p-values, confidence intervals,
        and other statistics crucial for understanding model performance
        beyond simple prediction accuracy.
    """
    
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit OLS model with statsmodels for detailed statistics."""
        X_with_const = add_constant(X, has_constant='add')
        self.model_ = OLS(y, X_with_const)
        self.results_ = self.model_.fit()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using fitted model."""
        X_pred_constant = add_constant(X, has_constant='add')
        return self.results_.predict(X_pred_constant)

    def summary(self, title: Optional[str] = None):
        """Return detailed regression summary."""
        if title is not None:
            return self.results_.summary(title=title)
        else:
            return self.results_.summary(title="OLS Estimation Summary")


class EWMTransformer(BaseEstimator, TransformerMixin):
    """
    Exponentially Weighted Moving Average transformer for feature smoothing.
    
    EWM is widely used in quantitative finance for:
    1. Noise reduction in price/return data
    2. Trend following indicators
    3. Volatility estimation
    4. Feature engineering
    
    The exponential weighting gives more importance to recent observations,
    making it responsive to recent changes while maintaining historical context.
    
    Formula:
        EWM_t = α * X_t + (1-α) * EWM_{t-1}
        where α = 1 - exp(-ln(2) / halflife)
    """
    
    def __init__(self, halflife: float = 3):
        """
        Initialize EWM transformer.
        
        Args:
            halflife: Period for exponential decay (lower = more responsive to recent data)
        """
        self.halflife = halflife

    def fit(self, X, y=None):
        """Fit transformer (stateless for EWM)."""
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Apply exponential weighted moving average transformation.
        
        Args:
            X: Input features
            
        Returns:
            Smoothed features as DataFrame
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_transformed = X.ewm(halflife=self.halflife).mean()
        return X_transformed

    def set_output(self, *, transform="default"):
        """Set output format (sklearn 1.0+ compatibility)."""
        return self


# =============================================================================
# XARRAY INTEGRATION FOR STANDARDIZED FINANCIAL DATA HANDLING
# =============================================================================

def create_results_xarray(results_dict: Dict[str, Union[pd.Series, np.ndarray]], 
                         time_index: Optional[pd.DatetimeIndex] = None,
                         strategy_names: Optional[List[str]] = None,
                         asset_names: Optional[List[str]] = None) -> xr.Dataset:
    """
    Create standardized xarray Dataset for multi-dimensional financial results.
    
    xarray is ideal for financial data because it:
    1. Handles multi-dimensional data (time × assets × strategies)
    2. Provides built-in plotting and aggregation methods
    3. Ensures proper alignment of different data series
    4. Supports metadata and documentation
    
    Args:
        results_dict: Dictionary of result series/arrays
        time_index: DatetimeIndex for time dimension
        strategy_names: Names of strategies
        asset_names: Names of assets
        
    Returns:
        xarray Dataset with standardized dimensions and coordinates
        
    Example:
        >>> results = create_results_xarray({
        ...     'returns': portfolio_returns,
        ...     'positions': positions,
        ...     'predictions': predictions
        ... })
        >>> results.returns.plot()  # Built-in plotting
        >>> results.sel(strategy='long_short').mean('time')  # Easy aggregation
    """
    if time_index is None:
        # Create default time index
        first_series = next(iter(results_dict.values()))
        if isinstance(first_series, pd.Series):
            time_index = first_series.index
        else:
            time_index = pd.date_range(start='2020-01-01', periods=len(first_series))
    
    # Convert all results to consistent format
    data_vars = {}
    for key, values in results_dict.items():
        if isinstance(values, pd.Series):
            data_vars[key] = (['time'], values.values)
        elif isinstance(values, np.ndarray):
            if values.ndim == 1:
                data_vars[key] = (['time'], values)
            elif values.ndim == 2:
                data_vars[key] = (['time', 'asset'], values)
            else:
                data_vars[key] = (['time'], values.flatten())
        else:
            data_vars[key] = (['time'], np.array(values))
    
    # Create coordinates
    coords = {'time': time_index}
    if strategy_names:
        coords['strategy'] = strategy_names
    if asset_names:
        coords['asset'] = asset_names
    
    # Create Dataset with metadata
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            'title': 'Quantitative Trading Simulation Results',
            'description': 'Multi-strategy backtesting results with standardized metrics',
            'created_by': 'Blue Water Macro Quantitative Trading Framework',
            'creation_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    )
    
    return ds


def plot_xarray_results(ds: xr.Dataset, variables: Optional[List[str]] = None,
                       plot_type: str = 'line', **kwargs) -> None:
    """
    Generate publication-quality plots from xarray results.
    
    Args:
        ds: xarray Dataset with results
        variables: Variables to plot (default: all)
        plot_type: Type of plot ('line', 'heatmap', 'bar')
        **kwargs: Additional plotting arguments
        
    Educational Note:
        xarray's built-in plotting leverages matplotlib but handles
        multi-dimensional indexing automatically, making it ideal
        for financial time series visualization.
    """
    if variables is None:
        variables = list(ds.data_vars.keys())
    
    n_vars = len(variables)
    fig, axes = plt.subplots(nrows=n_vars, ncols=1, figsize=(12, 4*n_vars))
    
    if n_vars == 1:
        axes = [axes]
    
    for i, var in enumerate(variables):
        if plot_type == 'line':
            ds[var].plot(ax=axes[i], **kwargs)
        elif plot_type == 'heatmap':
            ds[var].plot.imshow(ax=axes[i], **kwargs)
        
        axes[i].set_title(f'{var.replace("_", " ").title()}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def calculate_performance_metrics(returns: Union[pd.Series, xr.DataArray]) -> Dict[str, float]:
    """
    Calculate standard quantitative finance performance metrics.
    
    Args:
        returns: Time series of strategy returns
        
    Returns:
        Dictionary of performance metrics
        
    Metrics Included:
        - Annualized Return: Geometric mean return scaled to annual
        - Volatility: Annualized standard deviation
        - Sharpe Ratio: Risk-adjusted return measure
        - Maximum Drawdown: Worst peak-to-trough decline
        - Calmar Ratio: Annual return / Max Drawdown
    """
    if isinstance(returns, xr.DataArray):
        returns = returns.to_pandas()
    
    # Remove NaN values
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {}
    
    # Calculate metrics
    annual_return = (1 + returns.mean()) ** 252 - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # Calculate maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'Annual Return': annual_return,
        'Volatility': volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown,
        'Calmar Ratio': calmar_ratio,
        'Total Return': (1 + returns).prod() - 1,
        'Win Rate': (returns > 0).mean(),
        'Observations': len(returns)
    }


def create_correlation_matrix(ds: xr.Dataset, variables: List[str]) -> pd.DataFrame:
    """
    Create correlation matrix for multiple strategies/assets.
    
    Args:
        ds: xarray Dataset with results
        variables: Variables to include in correlation analysis
        
    Returns:
        Correlation matrix as DataFrame
    """
    data = {}
    for var in variables:
        if var in ds.data_vars:
            data[var] = ds[var].values.flatten()
    
    df = pd.DataFrame(data)
    return df.corr()


def export_results_to_csv(ds: xr.Dataset, filepath: str, include_metadata: bool = True) -> None:
    """
    Export xarray results to CSV with optional metadata.
    
    Args:
        ds: xarray Dataset
        filepath: Output file path
        include_metadata: Whether to include dataset metadata
    """
    # Convert to DataFrame for CSV export
    df = ds.to_dataframe()
    
    if include_metadata:
        # Add metadata as comments
        with open(filepath, 'w') as f:
            f.write(f"# Quantitative Trading Simulation Results\n")
            f.write(f"# Generated: {pd.Timestamp.now()}\n")
            for key, value in ds.attrs.items():
                f.write(f"# {key}: {value}\n")
            f.write("\n")
        
        # Append data
        df.to_csv(filepath, mode='a')
    else:
        df.to_csv(filepath)
    
    print(f"Results exported to {filepath}")


# =============================================================================
# EDUCATIONAL HELPER FUNCTIONS
# =============================================================================

def explain_log_returns() -> None:
    """
    Educational function explaining why we use log returns in quantitative finance.
    """
    explanation = """
    WHY LOG RETURNS IN QUANTITATIVE FINANCE?
    
    1. TIME ADDITIVITY:
       log(P_T/P_0) = log(P_T/P_{T-1}) + log(P_{T-1}/P_{T-2}) + ... + log(P_1/P_0)
       This means log returns over multiple periods simply add up!
    
    2. APPROXIMATE NORMALITY:
       For small price changes, log returns are approximately normally distributed,
       which is crucial for statistical modeling and risk management.
    
    3. COMPOUNDING:
       Log returns naturally handle the compounding effect of reinvested gains.
    
    4. MATHEMATICAL CONVENIENCE:
       Many statistical models and optimization problems are easier with log returns.
    
    WHEN TO BE CAREFUL:
       - Log returns can be misleading for very large price movements
       - Some practitioners prefer simple returns for certain applications
       - Always check the assumptions of your specific use case
    """
    print(explanation)


def explain_walk_forward_analysis() -> None:
    """
    Educational function explaining walk-forward analysis methodology.
    """
    explanation = """
    WALK-FORWARD ANALYSIS: THE GOLD STANDARD OF BACKTESTING
    
    The Problem with Traditional Backtests:
    - Using future information to make past decisions (look-ahead bias)
    - Overfitting to the entire historical dataset
    - Unrealistic performance estimates
    
    Walk-Forward Solution:
    1. Use only historical data available at each point in time
    2. Retrain models periodically with new data
    3. Make predictions one step ahead
    4. Aggregate results for realistic performance estimates
    
    Two Common Approaches:
    
    EXPANDING WINDOW:
    - Training set grows over time
    - Uses all available historical data
    - Better for long-term stable relationships
    
    ROLLING (FIXED) WINDOW:
    - Training set size stays constant
    - Only uses recent historical data
    - Better for adapting to changing market conditions
    
    Implementation in This Framework:
    - generate_train_predict_calender() creates proper date ranges
    - Simulation engine respects temporal ordering
    - Results reflect realistic trading performance
    """
    print(explanation)


# Global registry for educational functions
EDUCATIONAL_FUNCTIONS = {
    'log_returns': explain_log_returns,
    'walk_forward': explain_walk_forward_analysis
}

def get_educational_help(topic: str) -> None:
    """
    Get educational explanations for quantitative finance concepts.
    
    Available topics: log_returns, walk_forward
    """
    if topic in EDUCATIONAL_FUNCTIONS:
        EDUCATIONAL_FUNCTIONS[topic]()
    else:
        print(f"Available educational topics: {list(EDUCATIONAL_FUNCTIONS.keys())}")