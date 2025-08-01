# -*- coding: utf-8 -*-
"""
Multi-Target Simulation Utilities

This module contains utility functions for the multi-target quantitative trading
simulation framework. Functions are organized by category for easy maintenance
and testing.

Categories:
- Logging and Configuration
- Data Caching and Storage (yfinance, zarr)
- Simulation Metadata and Hashing
- Risk Model Integration (riskmodels.net)
- Performance Metrics and Benchmarking
- Visualization and Reporting
- Position Sizing Functions
"""

import os
import warnings
import sys
from datetime import datetime, timedelta
import time
import pytz
import numpy as np
import pandas as pd
import xarray as xr
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle
import hashlib
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass
import logging
import requests
from urllib.parse import urljoin
import json

# Import zarr with error handling
try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    print("⚠️ zarr not available - install with: pip install zarr>=2.12.0")

# Scikit-learn imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as rmse, mean_absolute_error as mae, r2_score
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Constants
TRADING_DAYS_PER_YEAR = 252
QUARTERLY_WINDOW_DAYS = 63
MONTHLY_RETRAIN_DAYS = 21
WEEKLY_RETRAIN_DAYS = 5
DAILY_RETRAIN_DAYS = 1
BASE_LEVERAGE_DEFAULT = 1.0
MAX_LEVERAGE_DEFAULT = 2.0
LONG_SHORT_LEVERAGE_DEFAULT = 1.0

# --- Logging and Configuration ---

def setup_logging():
    """Set up logging to both console and file with rotation."""
    import logging.handlers
    from pathlib import Path
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation (10MB max, keep 5 files)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "multi_target_simulator.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Separate error log
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
    print(f"Logging configured - files will be written to: {log_dir.absolute()}")
    return root_logger

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

def format_benchmark_name(benchmark_col: str) -> str:
    """Format benchmark column name for display with abbreviations."""
    name = benchmark_col.replace('benchmark_', '').replace('_', ' ').title()
    # Apply abbreviations
    if name == 'Equal Weight Targets':
        return 'EQ Weight'
    return name

# --- Data Caching and Storage ---

def save_yfinance_data_to_zarr(data, tickers, start_date, end_date):
    """Save yfinance data to zarr format with metadata."""
    if not ZARR_AVAILABLE:
        logger.warning("zarr not available, skipping cache save")
        return None
    
    cache_dir = "cache/yfinance_data"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create filename with tickers and date range
    ticker_str = "_".join(sorted(tickers))
    filename = f"yf_{ticker_str}_{start_date}_{end_date}.zarr"
    filepath = os.path.join(cache_dir, filename)
    
    try:
        # Flatten column names for zarr compatibility
        if isinstance(data.columns, pd.MultiIndex):
            # Convert MultiIndex columns to string format: "TICKER_COLUMN"
            data_flat = data.copy()
            data_flat.columns = [f"{col[0]}_{col[1]}" for col in data.columns]
        else:
            data_flat = data
        
        # Convert to xarray for zarr storage
        data_xr = data_flat.to_xarray()
        
        # Add metadata including original column structure
        data_xr.attrs.update({
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'created': datetime.now().isoformat(),
            'source': 'yfinance',
            'has_multiindex': isinstance(data.columns, pd.MultiIndex),
            'original_columns': data.columns.tolist() if hasattr(data.columns, 'tolist') else list(data.columns)
        })
        
        # Save to zarr
        data_xr.to_zarr(filepath, mode='w')
        
        logger.info(f"Cached yfinance data to {filepath}")
        return filepath
        
    except Exception as e:
        logger.warning(f"Failed to cache yfinance data: {e}")
        return None

def load_yfinance_data_from_zarr(tickers, start_date, end_date, max_age_hours=24):
    """Load yfinance data from zarr cache if available and fresh."""
    if not ZARR_AVAILABLE:
        return None
    
    cache_dir = "cache/yfinance_data"
    if not os.path.exists(cache_dir):
        return None
    
    # Create expected filename
    ticker_str = "_".join(sorted(tickers))
    filename = f"yf_{ticker_str}_{start_date}_{end_date}.zarr"
    filepath = os.path.join(cache_dir, filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        # Check file age
        file_age_hours = (time.time() - os.path.getctime(filepath)) / 3600
        if file_age_hours > max_age_hours:
            logger.info(f"Cache file {filename} is {file_age_hours:.1f} hours old, refreshing")
            return None
        
        # Load from zarr
        data_xr = xr.open_zarr(filepath)
        
        # Convert back to pandas DataFrame
        data_df = data_xr.to_dataframe()
        
        # Restore MultiIndex columns if they were flattened
        if data_xr.attrs.get('has_multiindex', False):
            # Parse flattened column names back to MultiIndex
            new_columns = []
            for col in data_df.columns:
                if '_' in col:
                    ticker, metric = col.split('_', 1)
                    new_columns.append((ticker, metric))
                else:
                    new_columns.append(col)
            
            if new_columns:
                data_df.columns = pd.MultiIndex.from_tuples(new_columns)
        
        logger.info(f"Loaded cached yfinance data from {filepath}")
        return data_df
        
    except Exception as e:
        logger.warning(f"Failed to load cached data: {e}")
        return None

def download_etf_data_with_cache(tickers, start_date='2010-01-01', end_date=None, max_age_hours=24):
    """
    Download ETF data with intelligent caching to avoid API limits.
    
    This function first checks for cached data and only downloads fresh data
    if the cache is stale or missing. This dramatically improves development
    workflow by eliminating repeated API calls.
    
    Parameters:
    -----------
    tickers : list
        List of ETF symbols to download
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format (defaults to today)
    max_age_hours : int
        Maximum age of cached data in hours before refresh
        
    Returns:
    --------
    pd.DataFrame
        Multi-level DataFrame with ETF price data
        
    Benefits:
    ---------
    - Eliminates "ETF not found" warnings from repeated API calls
    - Speeds up development and testing cycles
    - Provides offline capability once data is cached
    - Reduces yfinance API usage and potential rate limiting
    """
    
    logger = logging.getLogger(__name__)
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Requesting ETF data: {len(tickers)} tickers, {start_date} to {end_date}")
    
    # Try to load from cache first
    cached_data = load_yfinance_data_from_zarr(tickers, start_date, end_date, max_age_hours)
    
    if cached_data is not None:
        logger.info("✅ Using cached data - no API call needed")
        return cached_data
    
    # Download fresh data
    logger.info("📡 Downloading fresh data from yfinance...")
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', progress=False)
        
        if data.empty:
            logger.error("No data returned from yfinance")
            return None
        
        # Handle single ticker case (yfinance returns different structure)
        if len(tickers) == 1:
            data.columns = pd.MultiIndex.from_product([tickers, data.columns])
        
        logger.info(f"✅ Downloaded {data.shape[0]} days of data for {len(tickers)} tickers")
        
        # Cache the data
        save_yfinance_data_to_zarr(data, tickers, start_date, end_date)
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to download ETF data: {e}")
        return None

def clean_yfinance_cache(max_age_days=7):
    """Clean old cached yfinance files to save disk space."""
    cache_dir = "cache/yfinance_data"
    if not os.path.exists(cache_dir):
        return
    
    cleaned_count = 0
    total_size_freed = 0
    
    cutoff_time = time.time() - (max_age_days * 24 * 3600)
    
    for filename in os.listdir(cache_dir):
        if filename.endswith('.zarr'):
            filepath = os.path.join(cache_dir, filename)
            if os.path.getctime(filepath) < cutoff_time:
                try:
                    # Get size before deletion
                    import shutil
                    if os.path.isdir(filepath):
                        size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                 for dirpath, dirnames, filenames in os.walk(filepath)
                                 for filename in filenames)
                        shutil.rmtree(filepath)
                    else:
                        size = os.path.getsize(filepath)
                        os.remove(filepath)
                    
                    total_size_freed += size
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to clean cache file {filename}: {e}")
    
    if cleaned_count > 0:
        size_mb = total_size_freed / (1024 * 1024)
        logger.info(f"Cleaned {cleaned_count} cache files, freed {size_mb:.1f} MB")
    else:
        logger.info("No cache files to clean")

def list_yfinance_cache():
    """List all cached yfinance datasets with metadata."""
    cache_dir = "cache/yfinance_data"
    if not os.path.exists(cache_dir):
        print("No yfinance cache directory found")
        return
    
    cache_files = []
    total_size = 0
    
    for filename in os.listdir(cache_dir):
        if filename.endswith('.zarr'):
            filepath = os.path.join(cache_dir, filename)
            
            try:
                # Parse filename to extract info
                parts = filename.replace('.zarr', '').split('_')
                if len(parts) >= 4:
                    # Extract tickers (between 'yf' and date parts)
                    start_idx = 1
                    end_idx = len(parts) - 2
                    tickers = parts[start_idx:end_idx]
                    start_date = parts[-2]
                    end_date = parts[-1]
                else:
                    tickers = ['unknown']
                    start_date = 'unknown'
                    end_date = 'unknown'
                
                # Get file info
                created_time = datetime.fromtimestamp(os.path.getctime(filepath))
                age_hours = (time.time() - os.path.getctime(filepath)) / 3600
                
                # Calculate size
                if os.path.isdir(filepath):
                    import shutil
                    size = sum(os.path.getsize(os.path.join(dirpath, filename))
                             for dirpath, dirnames, filenames in os.walk(filepath)
                             for filename in filenames)
                else:
                    size = os.path.getsize(filepath)
                
                total_size += size
                
                cache_files.append({
                    'filename': filename,
                    'tickers': tickers,
                    'start_date': start_date,
                    'end_date': end_date,
                    'created': created_time,
                    'age_hours': age_hours,
                    'size_mb': size / (1024 * 1024)
                })
                
            except Exception as e:
                logger.warning(f"Failed to read cache file {filename}: {e}")
    
    if not cache_files:
        print("No cached yfinance datasets found")
        return
    
    # Display summary
    print(f"\n📊 YFINANCE CACHE SUMMARY")
    print(f"{'Tickers':<20} {'Date Range':<25} {'Age':<10} {'Size':<8}")
    print("-" * 70)
    
    for file_info in sorted(cache_files, key=lambda x: x['age_hours']):
        tickers_str = ",".join(file_info['tickers'][:3])  # Show first 3 tickers
        if len(file_info['tickers']) > 3:
            tickers_str += f"+{len(file_info['tickers'])-3}"
        
        date_range = f"{file_info['start_date']} to {file_info['end_date']}"
        age_str = f"{file_info['age_hours']:.1f}h"
        size_str = f"{file_info['size_mb']:.1f}MB"
        
        print(f"{tickers_str:<20} {date_range:<25} {age_str:<10} {size_str:<8}")
    
    print(f"\nTotal: {len(cache_files)} files, {total_size/(1024*1024):.1f} MB")

# --- Simulation Metadata and Hashing ---

def generate_simulation_metadata(X, y_multi, window_size, window_type, pipe_steps, param_grid, 
                               tag, position_func, position_params, train_frequency,
                               etf_symbols=None, target_etfs=None, start_date=None, 
                               random_seed=None, feature_engineering_steps=None):
    """
    Generate comprehensive metadata for full simulation reconstruction.
    
    This enhanced version stores all information needed to perfectly reproduce
    a simulation, enabling full reproducibility and audit trails.
    """
    
    # Core simulation parameters
    metadata = {
        'tag': tag,
        'data_shape': X.shape,
        'target_shape': y_multi.shape,
        'window_size': window_size,
        'window_type': window_type,
        'train_frequency': train_frequency,
        'data_start_date': str(X.index.min()),
        'data_end_date': str(X.index.max()),
        'target_columns': list(y_multi.columns),
        'feature_columns': list(X.columns),
        
        # Model configuration
        'pipeline_string': str(pipe_steps),
        'pipe_steps': [step[0] for step in pipe_steps.steps] if hasattr(pipe_steps, 'steps') else str(pipe_steps),
        'param_grid': param_grid,
        
        # Position sizing
        'position_func_name': getattr(position_func, '__name__', str(position_func)),
        'position_params': position_params,
        
        # Data fingerprinting
        'data_fingerprint': hashlib.md5(pd.util.hash_pandas_object(X, index=True).values).hexdigest()[:16],
        'target_fingerprint': hashlib.md5(pd.util.hash_pandas_object(y_multi, index=True).values).hexdigest()[:16],
        
        # Simulation context
        'born_on_date': datetime.now().isoformat(),
        'framework_version': '2.1.0',  # Update this with actual version
        'python_version': sys.version,
        'random_seed': random_seed,
        
        # ETF information
        'etf_symbols': etf_symbols,
        'target_etfs': target_etfs,
        'n_features': len(X.columns),
        'n_targets': len(y_multi.columns),
        'n_observations': len(X),
        
        # Optional extensions
        'start_date': start_date,
        'feature_engineering_steps': feature_engineering_steps,
        
        # Hash for quick identification
        'metadata_hash': None  # Will be filled after metadata is complete
    }
    
    # Generate metadata hash
    metadata_str = json.dumps(metadata, sort_keys=True, default=str)
    metadata['metadata_hash'] = hashlib.md5(metadata_str.encode()).hexdigest()[:8]
    
    return metadata

def generate_simulation_hash(X, y_multi, window_size, window_type, pipe_steps, param_grid, tag, 
                           position_func, position_params, train_frequency):
    """Generate a unique hash for simulation configuration to enable caching and reproducibility."""
    
    # Create a string representation of all simulation parameters
    hash_components = [
        str(X.shape),
        str(y_multi.shape),
        str(window_size),
        str(window_type),
        str(pipe_steps),
        str(param_grid),
        str(tag),
        getattr(position_func, '__name__', str(position_func)),
        str(position_params),
        str(train_frequency),
        # Add data fingerprint for uniqueness
        hashlib.md5(pd.util.hash_pandas_object(X, index=True).values).hexdigest()[:8],
        hashlib.md5(pd.util.hash_pandas_object(y_multi, index=True).values).hexdigest()[:8]
    ]
    
    hash_string = '|'.join(hash_components)
    simulation_hash = hashlib.md5(hash_string.encode()).hexdigest()[:8]
    
    return simulation_hash

def save_simulation_results(regout_df, simulation_hash, tag, metadata=None):
    """
    Save simulation results with complete metadata for reproducibility.
    
    Enhanced version that stores both results and complete simulation metadata
    in zarr format for efficient access and perfect reproducibility.
    """
    
    if not ZARR_AVAILABLE:
        logger.warning("zarr not available, saving to pickle instead")
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        filename = f"simulation_{simulation_hash}_{tag}.pkl"
        filepath = os.path.join(cache_dir, filename)
        
        save_data = {
            'regout_df': regout_df,
            'metadata': metadata,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved simulation results (pickle) to {filepath}")
        return filepath
    
    # Use zarr for efficient storage
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    filename = f"simulation_{simulation_hash}_{tag}.zarr"
    filepath = os.path.join(cache_dir, filename)
    
    try:
        # Convert DataFrame to xarray for zarr storage
        regout_xr = regout_df.to_xarray()
        
        # Add metadata as attributes
        if metadata:
            regout_xr.attrs.update(metadata)
        
        # Add save timestamp
        regout_xr.attrs['saved_at'] = datetime.now().isoformat()
        regout_xr.attrs['simulation_hash'] = simulation_hash
        regout_xr.attrs['tag'] = tag
        
        # Special handling for born_on_date - store as coordinate for fast access
        if metadata and 'born_on_date' in metadata:
            regout_xr.coords['born_on_date'] = metadata['born_on_date']
        
        # Save to zarr
        regout_xr.to_zarr(filepath, mode='w')
        
        logger.info(f"Saved simulation results (zarr) to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Failed to save simulation results: {e}")
        # Fallback to pickle
        filename = f"simulation_{simulation_hash}_{tag}.pkl"
        filepath = os.path.join(cache_dir, filename)
        
        save_data = {
            'regout_df': regout_df,
            'metadata': metadata,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.warning(f"Saved simulation results (pickle fallback) to {filepath}")
        return filepath

def load_simulation_results(simulation_hash, tag):
    """
    Load simulation results and metadata from cache.
    
    Enhanced version that supports both zarr and pickle formats for maximum
    flexibility and backward compatibility.
    """
    
    cache_dir = "cache"
    
    # Try zarr first (preferred format)
    zarr_filename = f"simulation_{simulation_hash}_{tag}.zarr"
    zarr_filepath = os.path.join(cache_dir, zarr_filename)
    
    if os.path.exists(zarr_filepath) and ZARR_AVAILABLE:
        try:
            # Load from zarr
            regout_xr = xr.open_zarr(zarr_filepath)
            regout_df = regout_xr.to_dataframe()
            
            # Extract metadata from attributes
            metadata = dict(regout_xr.attrs)
            
            logger.info(f"Loaded simulation results (zarr) from {zarr_filepath}")
            return regout_df, metadata
            
        except Exception as e:
            logger.warning(f"Failed to load zarr format: {e}")
    
    # Try pickle format (fallback)
    pkl_filename = f"simulation_{simulation_hash}_{tag}.pkl"
    pkl_filepath = os.path.join(cache_dir, pkl_filename)
    
    if os.path.exists(pkl_filepath):
        try:
            with open(pkl_filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            regout_df = save_data['regout_df']
            metadata = save_data.get('metadata', {})
            
            logger.info(f"Loaded simulation results (pickle) from {pkl_filepath}")
            return regout_df, metadata
            
        except Exception as e:
            logger.error(f"Failed to load pickle format: {e}")
    
    logger.warning(f"No cached results found for hash {simulation_hash} with tag {tag}")
    return None, None

def get_born_on_date_from_zarr(simulation_hash, tag):
    """
    Fast extraction of born_on_date from zarr file without loading full dataset.
    
    This function leverages xarray's coordinate system to quickly access
    the born_on_date without loading the entire simulation results.
    """
    
    if not ZARR_AVAILABLE:
        return None
    
    cache_dir = "cache"
    filename = f"simulation_{simulation_hash}_{tag}.zarr"
    filepath = os.path.join(cache_dir, filename)
    
    if not os.path.exists(filepath):
        return None
    
    try:
        # Open zarr file and check for born_on_date coordinate
        regout_xr = xr.open_zarr(filepath)
        
        if 'born_on_date' in regout_xr.coords:
            return str(regout_xr.coords['born_on_date'].values)
        elif 'born_on_date' in regout_xr.attrs:
            return regout_xr.attrs['born_on_date']
        else:
            return None
            
    except Exception as e:
        logger.warning(f"Failed to extract born_on_date: {e}")
        return None

def reconstruct_pipeline_from_metadata(simulation_hash, tag):
    """
    Reconstruct complete pipeline configuration from cached metadata.
    
    This function enables perfect reproduction of any historical simulation
    by reconstructing all parameters from stored metadata.
    """
    
    regout_df, metadata = load_simulation_results(simulation_hash, tag)
    
    if metadata is None:
        logger.error(f"No metadata found for simulation {simulation_hash}_{tag}")
        return None
    
    # Reconstruct pipeline configuration
    pipeline_config = {
        'simulation_hash': simulation_hash,
        'tag': tag,
        'window_size': metadata.get('window_size'),
        'window_type': metadata.get('window_type'),
        'train_frequency': metadata.get('train_frequency'),
        'pipe_steps': metadata.get('pipe_steps'),
        'param_grid': metadata.get('param_grid'),
        'position_func_name': metadata.get('position_func_name'),
        'position_params': metadata.get('position_params'),
        'target_columns': metadata.get('target_columns'),
        'feature_columns': metadata.get('feature_columns'),
        'born_on_date': metadata.get('born_on_date'),
        'data_fingerprint': metadata.get('data_fingerprint'),
        'target_fingerprint': metadata.get('target_fingerprint'),
        'etf_symbols': metadata.get('etf_symbols'),
        'target_etfs': metadata.get('target_etfs'),
        'framework_version': metadata.get('framework_version'),
        'random_seed': metadata.get('random_seed')
    }
    
    logger.info(f"Reconstructed pipeline config for simulation {simulation_hash}")
    return pipeline_config

# --- Risk Model Integration (riskmodels.net) ---

def load_riskmodels_data(tickers: List[str], api_key: str, 
                        start_date: str = None, end_date: str = None,
                        forecast_horizon: str = 'daily',
                        cache_dir: str = 'cache/riskmodels') -> Optional[xr.Dataset]:
    """
    Load institutional-grade risk model data from riskmodels.net
    
    This function fetches factor attribution data, variance decomposition, and
    beta coefficients from riskmodels.net's proprietary risk models. The data
    includes exposures to market, size, value, momentum, quality, and low-volatility
    factors, enabling sophisticated risk analysis and portfolio optimization.
    
    Parameters:
    -----------
    tickers : List[str]
        List of equity symbols to fetch risk data for (e.g., ['SPY', 'QQQ', 'IWM'])
    api_key : str
        Your riskmodels.net API key (register at riskmodels.net)
    start_date : str, optional
        Start date for historical data (YYYY-MM-DD format)
    end_date : str, optional
        End date for historical data (YYYY-MM-DD format)
    forecast_horizon : str, default 'daily'
        Risk model forecast horizon ('daily', 'weekly', 'monthly')
    cache_dir : str, default 'cache/riskmodels'
        Directory for caching risk model data using zarr format
        
    Returns:
    --------
    xarray.Dataset or None
        Multi-dimensional dataset with risk factors, betas, and residual risks
        Dimensions: ['ticker', 'time', 'factor']
        Variables: factor betas, residual volatility, total risk
        
    Educational Benefits:
    --------------------
    - Learn institutional-grade risk model implementation
    - Understand factor attribution and variance decomposition
    - Practice professional-quality portfolio risk management
    - Master multi-dimensional financial data analysis with xarray
    
    Example Usage:
    --------------
    >>> risk_data = load_riskmodels_data(['SPY', 'QQQ'], 'your_api_key')
    >>> print(f"Available factors: {list(risk_data.data_vars)}")
    >>> spy_market_beta = risk_data['Market'].sel(ticker='SPY')
    """
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading riskmodels.net data for {len(tickers)} tickers")
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key for this request
    cache_key = hashlib.md5(
        f"{'-'.join(sorted(tickers))}_{start_date}_{end_date}_{forecast_horizon}".encode()
    ).hexdigest()[:8]
    
    cache_path = os.path.join(cache_dir, f"risk_data_{cache_key}.zarr")
    
    # Check cache first
    if os.path.exists(cache_path) and ZARR_AVAILABLE:
        try:
            logger.info(f"Loading cached risk data from {cache_path}")
            return xr.open_zarr(cache_path)
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
    
    # Fetch from API
    try:
        logger.info("Fetching fresh data from riskmodels.net API")
        
        # API configuration
        base_url = "https://api.riskmodels.net/v1/"
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # PLACEHOLDER: This demonstrates the API structure
        # In production, this would make actual HTTP requests
        
        # For educational purposes, generate realistic mock data
        logger.info("Generating educational mock data (replace with actual API calls)")
        
        # Generate date range
        if start_date and end_date:
            dates = pd.date_range(start_date, end_date, freq='D')
        else:
            dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        
        # Define risk factors (aligned with industry standards)
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'LowVol', 'Residual']
        
        # Generate realistic factor data for each ticker
        np.random.seed(42)  # Reproducible for educational consistency
        
        risk_dataset = xr.Dataset()
        
        for factor in factors:
            if factor == 'Residual':
                # Residual volatility (always positive)
                factor_data = np.random.gamma(2, 0.05, (len(tickers), len(dates)))
            else:
                # Factor betas (can be positive or negative)
                if factor == 'Market':
                    # Market betas typically close to 1.0 for broad market ETFs
                    means = [0.95 if 'SPY' in ticker else 1.1 for ticker in tickers]
                    factor_data = np.array([np.random.normal(mean, 0.15, len(dates)) for mean in means])
                elif factor == 'Size':
                    # Size factor: negative for large-cap ETFs
                    means = [-0.3 if 'SPY' in ticker else 0.2 for ticker in tickers]
                    factor_data = np.array([np.random.normal(mean, 0.2, len(dates)) for mean in means])
                else:
                    # Other factors: centered around zero
                    factor_data = np.random.normal(0, 0.25, (len(tickers), len(dates)))
            
            risk_dataset[factor] = xr.DataArray(
                factor_data,
                coords={'ticker': tickers, 'time': dates},
                dims=['ticker', 'time'],
                attrs={
                    'description': f'{factor} factor exposure',
                    'source': 'riskmodels.net (educational mock)',
                    'forecast_horizon': forecast_horizon
                }
            )
        
        # Add metadata
        risk_dataset.attrs.update({
            'created': datetime.now().isoformat(),
            'source': 'riskmodels.net API (educational demonstration)',
            'tickers': tickers,
            'factors': factors,
            'forecast_horizon': forecast_horizon,
            'api_version': 'v1',
            'cache_key': cache_key
        })
        
        # Cache the results
        if ZARR_AVAILABLE:
            try:
                risk_dataset.to_zarr(cache_path, mode='w')
                logger.info(f"Cached risk data to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
        
        logger.info(f"Successfully loaded risk data for {len(tickers)} tickers")
        return risk_dataset
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        logger.info("💡 Check your API key and network connection")
        return None
    except Exception as e:
        logger.error(f"Error loading risk model data: {e}")
        return None

def risk_adjusted_portfolio_optimization(predictions: pd.DataFrame, 
                                       risk_data: xr.Dataset,
                                       risk_budget: float = 0.15,
                                       max_factor_exposure: float = 0.5) -> pd.DataFrame:
    """
    Optimize portfolio weights using risk model constraints
    
    This function demonstrates institutional-grade portfolio optimization using
    factor-based risk models. Instead of simple prediction-based weights, it
    incorporates risk factor exposures to create more robust, diversified portfolios.
    
    Parameters:
    -----------
    predictions : pd.DataFrame
        Expected returns for each asset (index: dates, columns: tickers)
    risk_data : xarray.Dataset
        Risk factor data from riskmodels.net
    risk_budget : float, default 0.15
        Target portfolio risk level (annualized volatility)
    max_factor_exposure : float, default 0.5
        Maximum absolute exposure to any single factor
        
    Returns:
    --------
    pd.DataFrame
        Optimized portfolio weights (index: dates, columns: tickers)
        
    Educational Value:
    -----------------
    - Learn industry-standard portfolio optimization techniques
    - Understand factor-based risk budgeting
    - Practice constraint-based optimization
    - Master integration of external risk models with portfolio construction
    
    Risk Model Benefits:
    -------------------
    - Reduces unintended factor concentrations
    - Improves diversification beyond simple correlation
    - Enables risk budgeting across systematic factors
    - Provides transparency into portfolio risk sources
    """
    
    logger = logging.getLogger(__name__)
    logger.info("Applying risk model constraints to portfolio optimization")
    
    if risk_data is None:
        logger.warning("No risk data available, using simple prediction-based weights")
        # Fallback to prediction-based weights
        return predictions.div(predictions.abs().sum(axis=1), axis=0).fillna(0)
    
    try:
        # Align dates between predictions and risk data
        common_dates = pd.Index(predictions.index).intersection(
            pd.Index(risk_data.time.values)
        )
        
        if len(common_dates) == 0:
            logger.warning("No overlapping dates, using simple weights")
            return predictions.div(predictions.abs().sum(axis=1), axis=0).fillna(0)
        
        # Focus on common period
        pred_aligned = predictions.loc[common_dates]
        risk_aligned = risk_data.sel(time=common_dates)
        
        # Get common tickers
        common_tickers = list(set(predictions.columns) & set(risk_data.ticker.values))
        if not common_tickers:
            logger.warning("No common tickers, using simple weights")
            return predictions.div(predictions.abs().sum(axis=1), axis=0).fillna(0)
        
        pred_common = pred_aligned[common_tickers]
        risk_common = risk_aligned.sel(ticker=common_tickers)
        
        optimized_weights = []
        
        for date in pred_common.index:
            try:
                # Get predictions and risk factors for this date
                date_predictions = pred_common.loc[date]
                date_risk = risk_common.sel(time=date, method='nearest')
                
                # Simple risk-adjusted optimization
                # In production, this would use sophisticated optimization (cvxpy, etc.)
                
                # Step 1: Start with long-only prediction-based weights
                # Apply same long-only logic as standard portfolio
                positive_predictions = date_predictions[date_predictions > 0]
                if len(positive_predictions) > 0:
                    # Use only positive predictions
                    raw_weights = pd.Series(0.0, index=date_predictions.index)
                    raw_weights[positive_predictions.index] = 1.0 / len(positive_predictions)
                else:
                    # If all negative, use equal weights (long-only)
                    raw_weights = pd.Series(1.0 / len(date_predictions), index=date_predictions.index)
                
                # Step 2: Apply risk adjustments
                market_betas = date_risk['Market'].values
                residual_vols = date_risk['Residual'].values
                
                # Calculate total risk for each asset
                total_risks = np.abs(market_betas) + residual_vols
                
                # Risk budgeting: inverse risk weighting
                risk_adjustments = 1.0 / (1.0 + total_risks)
                risk_adjustments = risk_adjustments / risk_adjustments.sum()
                
                # Step 3: Combine prediction signal with risk adjustment (both long-only)
                combined_weights = 0.7 * raw_weights + 0.3 * pd.Series(risk_adjustments, index=raw_weights.index)
                
                # Step 4: Apply long-only constraints
                # Ensure all weights are non-negative (long-only)
                combined_weights = combined_weights.clip(0.0, 0.4)
                
                # Ensure weights sum to 1.0 for long-only portfolio
                weight_sum = combined_weights.sum()
                if weight_sum > 0:
                    combined_weights = combined_weights / weight_sum
                else:
                    # Fallback to equal weights if something went wrong
                    combined_weights = pd.Series(1.0 / len(date_predictions), index=date_predictions.index)
                
                optimized_weights.append(combined_weights)
                
            except Exception as e:
                logger.warning(f"Optimization failed for {date}: {e}")
                # Fallback to equal weights
                n_assets = len(common_tickers)
                fallback_weights = pd.Series(
                    [1.0/n_assets] * n_assets, 
                    index=common_tickers
                )
                optimized_weights.append(fallback_weights)
        
        # Convert to DataFrame
        weights_df = pd.DataFrame(optimized_weights, index=pred_common.index)
        
        # Add missing tickers with zero weights
        for ticker in predictions.columns:
            if ticker not in weights_df.columns:
                weights_df[ticker] = 0.0
        
        # Reorder columns to match original predictions
        weights_df = weights_df[predictions.columns].fillna(0)
        
        logger.info(f"Portfolio optimization completed for {len(weights_df)} dates")
        logger.info(f"Average weight concentration: {weights_df.abs().max(axis=1).mean():.3f}")
        
        return weights_df
        
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        # Return simple prediction-based weights as fallback
        return predictions.div(predictions.abs().sum(axis=1), axis=0).fillna(0)

# --- Position Sizing Functions ---

def L_func_multi_target_equal_weight_long_only(predictions_df, params=[]):
    """
    Long-only equal weight position sizing for multi-target strategies.
    
    Assigns equal capital allocation across all assets. If all predictions are negative,
    invests in the assets with the least negative (most positive) predictions to avoid
    staying in cash too often and missing market returns.
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with predictions for each target asset (columns = assets)
    params : list
        Additional parameters (not used in equal weight)
        
    Returns:
    --------
    pd.DataFrame
        Position weights for each asset (long-only, sums to ~1.0)
        
    Educational Benefits:
    --------------------
    - Simple, interpretable long-only position sizing
    - Good baseline for performance comparison
    - Reduces concentration risk
    - More conservative than long-short strategies
    - Avoids staying in cash too often (participates in market uptrends)
    - Suitable for beginner educational use
    """
    
    positions = predictions_df.copy()
    
    # For each date, determine position allocation
    for idx in positions.index:
        row = positions.loc[idx]
        positive_count = (row > 0).sum()
        
        if positive_count > 0:
            # If we have positive predictions, use only those
            positions.loc[idx] = 0
            positions.loc[idx, row > 0] = 1.0 / positive_count
        else:
            # If all predictions are negative, invest equally in all assets
            # This prevents staying in cash and missing market returns
            positions.loc[idx] = 1.0 / len(row)
    
    return positions.fillna(0)

def L_func_multi_target_equal_weight(predictions_df, params=[]):
    """
    Equal weight position sizing for multi-target strategies (long-short).
    
    Assigns equal capital allocation across all assets with positive predictions
    and equal short allocation across assets with negative predictions.
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with predictions for each target asset (columns = assets)
    params : list
        Additional parameters (not used in equal weight)
        
    Returns:
    --------
    pd.DataFrame
        Position weights for each asset
        
    Educational Benefits:
    --------------------
    - Simple, interpretable position sizing
    - Good baseline for performance comparison
    - Reduces concentration risk
    - Allows both long and short positions
    """
    
    # Simple equal weighting based on prediction direction
    positions = predictions_df.copy()
    
    # Convert predictions to position directions (+1, 0, -1)
    positions = np.sign(positions)
    
    # Normalize to equal weights within long and short sides
    for idx in positions.index:
        row = positions.loc[idx]
        long_count = (row > 0).sum()
        short_count = (row < 0).sum()
        
        if long_count > 0:
            positions.loc[idx, row > 0] = 1.0 / long_count
        if short_count > 0:
            positions.loc[idx, row < 0] = -1.0 / short_count
    
    return positions.fillna(0)

def L_func_multi_target_confidence_weighted(predictions_df, params=[]):
    """
    Confidence-weighted position sizing for multi-target strategies.
    
    Allocates capital proportional to prediction magnitude, with stronger
    predictions receiving larger position sizes.
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with predictions for each target asset
    params : list
        [max_leverage] - maximum total leverage (default: 2.0)
        
    Returns:
    --------
    pd.DataFrame
        Position weights proportional to prediction confidence
        
    Educational Benefits:
    --------------------
    - Incorporates prediction strength into position sizing
    - Dynamic allocation based on model confidence
    - More sophisticated than equal weighting
    """
    
    max_leverage = params[0] if params else 2.0
    
    positions = predictions_df.copy()
    
    # Scale positions by prediction magnitude
    for idx in positions.index:
        row = positions.loc[idx]
        
        # Calculate total absolute prediction strength
        total_strength = row.abs().sum()
        
        if total_strength > 0:
            # Scale positions to use full leverage budget
            scaled_positions = (row / total_strength) * max_leverage
            positions.loc[idx] = scaled_positions
        else:
            # No predictions - zero positions
            positions.loc[idx] = 0
    
    return positions.fillna(0)

def L_func_multi_target_long_short(predictions_df, params=[]):
    """
    Long-short position sizing for multi-target strategies.
    
    Creates market-neutral portfolios by maintaining equal dollar amounts
    in long and short positions, regardless of prediction magnitudes.
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with predictions for each target asset
    params : list
        [target_leverage] - target leverage for each side (default: 1.0)
        
    Returns:
    --------
    pd.DataFrame
        Market-neutral position weights
        
    Educational Benefits:
    --------------------
    - Market-neutral strategy construction
    - Risk management through balanced exposure
    - Advanced portfolio construction technique
    """
    
    target_leverage = params[0] if params else 1.0
    
    positions = predictions_df.copy()
    
    for idx in positions.index:
        row = positions.loc[idx]
        
        # Separate long and short predictions
        long_preds = row[row > 0]
        short_preds = row[row < 0]
        
        # Calculate position sizes
        new_positions = pd.Series(0.0, index=row.index)
        
        if len(long_preds) > 0:
            # Long side: distribute target_leverage across positive predictions
            long_weights = long_preds / long_preds.sum() * target_leverage
            new_positions[long_preds.index] = long_weights
        
        if len(short_preds) > 0:
            # Short side: distribute -target_leverage across negative predictions
            short_weights = short_preds / short_preds.abs().sum() * (-target_leverage)
            new_positions[short_preds.index] = short_weights
        
        positions.loc[idx] = new_positions
    
    return positions.fillna(0)

# --- Performance Metrics ---

def calculate_performance_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for a return series.
    
    Returns standard risk-adjusted performance metrics used in
    institutional portfolio management and academic research.
    """
    
    if len(returns) == 0 or returns.isna().all():
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0
        }
    
    # Basic return metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + returns).prod() ** (TRADING_DAYS_PER_YEAR / len(returns)) - 1
    volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    # Risk-adjusted metrics
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Calmar ratio (return/max_drawdown)
    calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown < 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio
    }

def calculate_information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculate information ratio (excess return / tracking error).
    
    The information ratio measures risk-adjusted excess return relative
    to a benchmark, widely used in institutional portfolio management.
    """
    
    if len(strategy_returns) != len(benchmark_returns):
        return 0.0
    
    excess_returns = strategy_returns - benchmark_returns
    
    if excess_returns.std() == 0:
        return 0.0
    
    information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return information_ratio

# Initialize logger
logger = setup_logging()