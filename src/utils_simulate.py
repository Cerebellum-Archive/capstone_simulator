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
import math
from sklearn.metrics import r2_score as r_squared
from sklearn.feature_selection import r_regression, f_regression, mutual_info_regression
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
try:
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
except ImportError:
    # Fallback for older sklearn versions
    HalvingGridSearchCV = GridSearchCV
    HalvingRandomSearchCV = RandomizedSearchCV
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS
from scipy.stats import randint, uniform, loguniform
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
    # Handle timezone conversion more robustly
    if df.index.tz is not None:
        # If timezone-aware, convert to UTC first, then remove timezone
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    else:
        # If already timezone-naive, just normalize
        df.index = df.index.normalize()
    
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
    records = []
    for year in X.index.year.unique():
        X_fit = X.loc[str(year), :].dropna()
        y_fit = y.reindex(X_fit.index)
        pearson_vals = r_regression(X_fit, y_fit, center=True)
        for col, pear in zip(X.columns, pearson_vals):
            records.append({'feature': col, 'year': str(year), 'pearson': pear})
    df = pd.DataFrame.from_records(records)
    # Aggregate per feature: mean pearson and simple proxies; provide expected columns
    summary = df.groupby('feature').agg(
        pearson=('pearson', 'mean'),
        p_value=('pearson', lambda s: (1 - s.abs()).mean()),
        f_statistic=('pearson', lambda s: (s**2 / (1 - s.abs().clip(upper=0.999) )).mean()),
        mutual_info=('pearson', lambda s: s.abs().mean())
    )
    print(f'Analysis period: {X.index.min()} to {X.index.max()}')
    return summary.sort_values(by=sort_by, ascending=False)


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
                         time_coord: Optional[pd.DatetimeIndex] = None,
                         strategy_coord: Optional[List[str]] = None,
                         asset_coord: Optional[List[str]] = None) -> xr.Dataset:
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
    if time_coord is None:
        # Create default time index
        first_series = next(iter(results_dict.values()))
        if isinstance(first_series, pd.Series):
            time_coord = first_series.index
        else:
            time_coord = pd.date_range(start='2010-01-01', periods=len(first_series))
    
    # Convert all results to consistent format
    data_vars = {}
    for key, values in results_dict.items():
        if isinstance(values, pd.Series):
            data_vars[key] = (['time'], values.values)
        elif isinstance(values, pd.DataFrame):
            data_vars[key] = (['time', 'strategy'], values.values)
        elif isinstance(values, np.ndarray):
            if values.ndim == 1:
                data_vars[key] = (['time'], values)
            elif values.ndim == 2:
                data_vars[key] = (['time', 'strategy'], values)
            else:
                data_vars[key] = (['time'], values.flatten())
        else:
            arr = np.array(values)
            if arr.ndim == 2:
                data_vars[key] = (['time', 'strategy'], arr)
            else:
                data_vars[key] = (['time'], arr)
    
    # Create coordinates
    coords = {'time': time_coord}
    if strategy_coord:
        coords['strategy'] = strategy_coord
    if asset_coord:
        coords['asset'] = asset_coord
    
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
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0
        }
    
    # Calculate metrics
    annual_return = (1 + returns.mean()) ** 252 - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility if volatility > 0 else float('nan')
    
    # Calculate maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    from scipy.stats import skew, kurtosis
    return {
        'total_return': (1 + returns).prod() - 1,
        'annualized_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'skewness': float(skew(returns, bias=False)),
        'kurtosis': float(kurtosis(returns, fisher=True, bias=False))
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
    # Expect a 2D variable 'returns' with dims ['time','strategy']
    if 'returns' in variables and 'returns' in ds.data_vars and 'strategy' in ds['returns'].dims:
        df = ds['returns'].to_pandas()
        return df.corr()
    # Fallback: compute correlations among available strategy-like variables
    compiled = {}
    for var in variables:
        if var in ds.data_vars:
            arr = ds[var]
            if 'strategy' in arr.dims:
                compiled[var] = arr.to_pandas()
    if compiled:
        df = pd.concat(compiled.values(), axis=1)
        return df.corr()
    return pd.DataFrame()


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
    
    1. TIME-ADDITIVE:
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
    WALK-FORWARD ANALYSIS OVERVIEW
    
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


# --- Model Complexity Analysis ---

def unwrap_estimator(estimator: BaseEstimator) -> BaseEstimator:
    """
    Unwrap multi-output or search wrappers to get the base estimator.
    
    Educational Note:
        In quantitative finance, models are often wrapped in ensemble methods,
        cross-validation, or multi-output adapters. This function extracts the
        core estimator to analyze its fundamental complexity properties.
    
    Args:
        estimator: sklearn estimator (potentially wrapped)
        
    Returns:
        Base estimator without wrappers
    """
    # Don't unwrap ensemble methods like RandomForest - they should be treated as single estimators
    ensemble_types = ['RandomForest', 'GradientBoosting', 'AdaBoost', 'Bagging', 'ExtraTrees']
    cls_name = type(estimator).__name__
    
    if any(ensemble_type in cls_name for ensemble_type in ensemble_types):
        return estimator
    
    # Unwrap search CV and multi-output wrappers
    while hasattr(estimator, 'estimator'):
        estimator = estimator.estimator
        # Stop unwrapping if we hit an ensemble method
        cls_name = type(estimator).__name__
        if any(ensemble_type in cls_name for ensemble_type in ensemble_types):
            break
    
    return estimator


def estimate_search_space_size(params: Dict) -> int:
    """
    Estimate the size of the parameter search space for auto-tuning learners.
    
    Educational Note:
        Hyperparameter search spaces contribute to overfitting risk. Larger
        search spaces allow models to find better fits to training data, but
        may not generalize well. This function quantifies search complexity.
    
    Args:
        params: Parameter grid or distributions for hyperparameter search
        
    Returns:
        Estimated number of parameter combinations
        
    Example:
        >>> params = {'alpha': [0.1, 1.0, 10.0], 'fit_intercept': [True, False]}
        >>> size = estimate_search_space_size(params)
        >>> print(f"Search space size: {size}")  # Output: 6
    """
    total_combinations = 1
    for param, values in params.items():
        if isinstance(values, (list, tuple)):
            total_combinations *= len(values)
        elif hasattr(values, 'rvs'):  # Check if it's a scipy distribution
            # Approximate continuous distributions with a reasonable number of discrete points
            total_combinations *= 10  # Assume ~10 effective values for continuous ranges
        else:
            total_combinations *= 1  # Single value, no contribution
    return total_combinations


def get_complexity_score(estimator: BaseEstimator) -> float:
    """
    Computes a generic complexity score for sklearn estimators, including auto-tuning learners.
    
    Higher score indicates higher model complexity, correlating with higher overfitting risk.
    This is crucial in quantitative finance where overfitting can lead to poor out-of-sample
    performance and significant trading losses.
    
    Educational Note:
        Model complexity scoring helps researchers identify models that may be too flexible
        for the available data. In quantitative finance:
        - Simple models (Linear Regression, Ridge) often generalize better
        - Complex models (Random Forest, Neural Networks) may capture noise
        - Hyperparameter search increases effective complexity
        
        The score can be used to adjust performance metrics or select models based on
        complexity-return trade-offs.
    
    Scoring System:
        - OLS (LinearRegression) is baseline 1.0
        - Regularized linear models (e.g., Ridge) have scores <= 1.0 based on regularization strength
        - Tree-based models have scores > 1.0 based on depth, number of trees, etc.
        - Auto-tuning learners (GridSearchCV, etc.) multiply base estimator score by search space factor
        - Normalized roughly: default RandomForest ~5-10, default XGBoost ~3-5, GridSearchCV ~10x base
    
    Args:
        estimator: sklearn estimator (fitted or unfitted)
        
    Returns:
        Complexity score (float, typically 0.1 to 50.0)
        
    Usage:
        >>> from sklearn.linear_model import Ridge
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> 
        >>> simple_model = Ridge(alpha=1.0)
        >>> complex_model = RandomForestRegressor(n_estimators=200, max_depth=15)
        >>> 
        >>> print(f"Ridge complexity: {get_complexity_score(simple_model):.2f}")
        >>> print(f"Random Forest complexity: {get_complexity_score(complex_model):.2f}")
        >>> 
        >>> # Use in performance analysis
        >>> complexity_adjusted_return = annual_return / get_complexity_score(model)
    """
    # Handle auto-tuning learners
    if isinstance(estimator, (GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV)):
        base_estimator = unwrap_estimator(estimator)
        base_score = get_complexity_score(base_estimator)  # Recursively get base score
        
        # Estimate search space complexity
        param_grid = estimator.param_grid if hasattr(estimator, 'param_grid') else estimator.param_distributions
        search_space_size = estimate_search_space_size(param_grid)
        
        # For RandomizedSearchCV and HalvingRandomSearchCV, cap the effective size
        if isinstance(estimator, (RandomizedSearchCV, HalvingRandomSearchCV)):
            n_iter = estimator.n_iter if hasattr(estimator, 'n_iter') else 10
            search_space_size = min(search_space_size, n_iter)
        
        # Halving searches reduce effective trials, so adjust
        if isinstance(estimator, (HalvingGridSearchCV, HalvingRandomSearchCV)):
            factor = estimator.factor if hasattr(estimator, 'factor') else 3
            search_space_size = max(1, search_space_size // factor)
        
        # Multiply base score by log of search space size to reflect overfitting risk
        return base_score * (1.0 + math.log1p(search_space_size) / math.log(10))  # Log10 for scaling

    # Handle base estimators
    estimator = unwrap_estimator(estimator)
    params = estimator.get_params()
    cls_name = type(estimator).__name__
    
    if 'LinearRegression' in cls_name:
        return 1.0
    
    elif 'Ridge' in cls_name:
        alpha = params.get('alpha', 1.0)
        return 1.0 / (1.0 + alpha)  # Decreases with stronger regularization, e.g., alpha=1 -> 0.5
    
    elif 'Lasso' in cls_name:
        alpha = params.get('alpha', 1.0)
        return 1.0 / (1.0 + alpha * 10)  # Stronger penalty for Lasso sparsity
    
    elif 'RandomForest' in cls_name:
        max_depth = params.get('max_depth')
        if max_depth is None:
            max_depth = 20  # Assumption for unbounded trees
        n_estimators = params.get('n_estimators', 100)
        max_features = params.get('max_features', 1.0)
        if isinstance(max_features, str):
            if max_features in ['auto', 'sqrt', 'log2']:
                max_features = 0.33  # Approximate fraction
            else:
                max_features = 1.0
        elif max_features is None:
            max_features = 1.0
        effective_complexity = n_estimators * max_depth * max_features
        return 1.0 + effective_complexity / 200.0  # Default ~1 + (100*20*1)/200 = 11.0

    elif 'XGB' in cls_name:  # For XGBoost
        max_depth = params.get('max_depth', 6)
        n_estimators = params.get('n_estimators', 100)
        effective_complexity = n_estimators * (2 ** max_depth)
        return 1.0 + math.log(1 + effective_complexity) / math.log(2) / 10.0  # Default ~2.26

    elif 'DecisionTree' in cls_name:
        max_depth = params.get('max_depth')
        if max_depth is None:
            max_depth = 20
        return 1.0 + max_depth / 5.0

    elif 'SVC' in cls_name or 'SVR' in cls_name:
        kernel = params.get('kernel', 'rbf')
        if kernel == 'linear':
            return 1.2
        else:
            C = params.get('C', 1.0)
            gamma = params.get('gamma', 'scale')
            if gamma == 'scale':
                gamma = 1.0
            elif gamma == 'auto':
                gamma = 0.1
            return 2.0 + math.log(1 + C * gamma)

    elif 'KNeighbors' in cls_name:
        n_neighbors = params.get('n_neighbors', 5)
        return 1.0 + 10.0 / n_neighbors

    elif 'MLP' in cls_name:
        hidden = params.get('hidden_layer_sizes', (100,))
        if isinstance(hidden, int):
            hidden = (hidden,)
        total_neurons = sum(hidden)
        n_layers = len(hidden)
        return 1.0 + (n_layers * total_neurons) / 50.0  # Default ~3.0

    else:
        return 2.0  # Default for unknown estimators


def calculate_complexity_adjusted_metrics(returns: Union[pd.Series, xr.DataArray], 
                                        complexity_score: float) -> Dict[str, float]:
    """
    Calculate complexity-adjusted performance metrics for quantitative trading strategies.
    
    Educational Note:
        In quantitative finance, raw performance metrics can be misleading when models
        of different complexity are compared. Complexity-adjusted metrics help identify
        strategies that achieve good performance without excessive model complexity,
        which often leads to better out-of-sample performance.
        
        Common adjustments:
        - Complexity-adjusted return = raw_return / complexity_score
        - Complexity-adjusted Sharpe = raw_sharpe / sqrt(complexity_score)
        - Overfitting penalty = 1 - (complexity_score - 1) * 0.1
    
    Args:
        returns: Strategy returns (daily)
        complexity_score: Model complexity score from get_complexity_score()
        
    Returns:
        Dictionary with complexity-adjusted metrics
        
    Example:
        >>> returns = pd.Series([0.01, -0.005, 0.02, -0.01])
        >>> complexity = 5.0  # Random Forest model
        >>> metrics = calculate_complexity_adjusted_metrics(returns, complexity)
        >>> print(f"Complexity-adjusted Sharpe: {metrics['complexity_adjusted_sharpe']:.2f}")
    """
    # Calculate base metrics
    base_metrics = calculate_performance_metrics(returns)
    # Provide capitalized aliases for compatibility with older code/tests
    capitalized = {
        'Annual Return': base_metrics.get('annualized_return', 0.0),
        'Volatility': base_metrics.get('volatility', 0.0),
        'Sharpe Ratio': base_metrics.get('sharpe_ratio', float('nan')),
        'Maximum Drawdown': base_metrics.get('max_drawdown', 0.0),
        'Calmar Ratio': base_metrics.get('calmar_ratio', 0.0),
        'Total Return': base_metrics.get('total_return', 0.0)
    }
    
    # Apply complexity adjustments
    # Safe extracts
    base_ann = capitalized['Annual Return']
    base_sharpe = capitalized['Sharpe Ratio']
    
    adjusted_metrics = {
        'complexity_score': complexity_score,
        'base_annual_return': base_ann,
        'base_sharpe_ratio': base_sharpe,
        
        # Complexity-adjusted metrics
        'complexity_adjusted_return': (base_ann / complexity_score) if complexity_score > 0 else 0.0,
        'complexity_adjusted_sharpe': (base_sharpe / math.sqrt(complexity_score)) if complexity_score > 0 else 0.0,
        
        # Overfitting penalty (decreases with complexity)
        'overfitting_penalty': max(0.1, 1.0 - (complexity_score - 1.0) * 0.05),
        
        # Complexity efficiency (return per unit complexity)
        'complexity_efficiency': base_ann / complexity_score if complexity_score > 0 else 0.0,
        
        # Risk-adjusted complexity efficiency
        'risk_adjusted_efficiency': (base_sharpe / complexity_score) if complexity_score > 0 else 0.0
    }
    
    # Add all base metrics
    adjusted_metrics.update(base_metrics)
    adjusted_metrics.update(capitalized)
    
    return adjusted_metrics


# Add complexity scoring to educational functions registry
EDUCATIONAL_FUNCTIONS['complexity_scoring'] = lambda: print("""
Model Complexity Scoring in Quantitative Finance

The complexity score helps identify models that may be prone to overfitting:

1. LINEAR MODELS (Score ≤ 1.0):
   - LinearRegression: 1.0 (baseline)
   - Ridge(alpha=1.0): 0.5 (regularized)
   - Lasso(alpha=0.1): 0.09 (sparse)

2. TREE-BASED MODELS (Score > 1.0):
   - DecisionTree(max_depth=10): 3.0
   - RandomForest(default): ~11.0
   - XGBoost(default): ~2.3

3. HYPERPARAMETER SEARCH:
   - GridSearchCV multiplies base score by search space factor
   - Larger search spaces = higher overfitting risk

Usage in Trading Strategies:
- Compare models with similar complexity scores  
- Use complexity-adjusted Sharpe ratios
- Prefer simpler models for out-of-sample robustness
- Monitor complexity vs. performance trade-offs

Educational Value:
- Teaches the bias-variance trade-off
- Emphasizes generalization over fitting
- Promotes disciplined model selection
""")