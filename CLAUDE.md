# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **educational quantitative trading simulation framework** designed for financial engineering students. The refactored codebase emphasizes the **full research cycle** (data inputs, feature preprocessing, model exploration, simulation, and reporting) with native **xarray integration** for standardized multi-dimensional data handling.

## Refactored Repository Structure

```
capstone_simulator/
├── src/                    # Core production modules
│   ├── utils_simulate.py   # Enhanced utilities with xarray transformers
│   ├── single_target_simulator.py  # Educational single-asset (SPY) prediction
│   └── multi_target_simulator.py   # Advanced multi-asset portfolio strategies
├── notebooks/              # Interactive educational tutorials
│   ├── 01_single_target_tutorial.ipynb    # Basics: Walk-forward simulation
│   ├── 02_multi_target_tutorial.ipynb     # Advanced: Portfolio construction
│   └── 03_full_research_cycle.ipynb       # End-to-end project demo
├── docs/                   # Documentation and guides
│   ├── README.md           # Student-focused educational guide
│   └── ERM3_2e_Data_Dictionary.md         # Blue Water Macro data schema
├── data/                   # Sample datasets (user-provided, git-ignored)
├── reports/                # Auto-generated outputs (git-ignored)
├── requirements.txt        # Updated dependencies (includes jupyter, xarray)
└── .gitignore             # Standard project ignores
```

## Development Commands

**Educational Workflow:**
```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# Start educational tutorials
jupyter notebook notebooks/01_single_target_tutorial.ipynb

# Run standalone simulations
cd src/
python3 single_target_simulator.py
python3 multi_target_simulator.py
```

**Key Changes from Original:**
- **Modular src/** structure instead of root-level scripts
- **Educational notebooks/** with step-by-step tutorials and exercises
- **Enhanced utils_simulate.py** with xarray functions and educational docstrings
- **Import path changes:** Use `from .utils_simulate import ...` within src/ modules

## Key Architecture Patterns

**Enhanced xarray Integration:**
- `create_results_xarray()` - Standardized multi-dimensional result storage
- `plot_xarray_results()` - Native visualization with proper indexing
- `calculate_performance_metrics()` - Comprehensive risk-adjusted metrics
- Multi-strategy comparison with time/asset/strategy dimensions

**Educational Framework:**
- **Tutorial Notebooks:** Progressive learning from single-target to multi-asset portfolios
- **Student Exercises:** Hands-on coding challenges with solutions
- **Educational Functions:** `get_educational_help()` for concept explanations
- **Blue Water Macro Branding:** Professional context and career development

**Multi-Target Simulation Engine:**
- Uses `sklearn.multioutput.MultiOutputRegressor` for simultaneous ETF prediction
- Three position sizing strategies: EqualWeight, ConfidenceWeighted, LongShort
- Portfolio-level performance analysis with proper return aggregation
- Enterprise-grade caching system for iterative development

**Walk-Forward Backtesting:**
- `generate_train_predict_calender()` ensures no look-ahead bias
- Rolling/expanding window training with configurable parameters
- Realistic transaction cost modeling for production readiness

**Feature Engineering Pipeline:**
- `EWMTransformer` for exponentially weighted moving averages with educational explanations
- `sklearn.Pipeline` integration for complex preprocessing workflows
- Feature importance analysis with `p_by_year()` for stability assessment

## Educational Data Flow (Full Research Cycle)

1. **Inputs**: `yfinance` downloads ETF price data with proper timezone handling
2. **Preprocessing**: `log_returns()` + `EWMTransformer` via sklearn pipelines with educational explanations
3. **Exploration**: `p_by_year()`, `feature_profiles()` for predictive power analysis across time periods
4. **Simulation**: Walk-forward backtesting with `generate_train_predict_calender()` to prevent look-ahead bias
5. **Reporting**: `create_results_xarray()` for standardized multi-dimensional storage and native plotting

## Student-Focused Configuration

**Tutorial Progression:**
```python
# Tutorial 1: Single-target SPY prediction
TARGET_ETF = 'SPY'
FEATURE_ETFS = ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU']

# Tutorial 2: Multi-target portfolio strategies  
TARGET_ETFS = ['SPY', 'QQQ', 'IWM']
position_strategies = ['equal_weight', 'confidence_weighted', 'long_short']
```

**xarray Results Structure:**
```python
# Multi-dimensional dataset with proper coordinates
results_xr = xr.Dataset({
    'portfolio_returns': (['time', 'strategy'], returns_array),
    'individual_returns': (['time', 'asset'], asset_returns),
    'predictions': (['time', 'asset'], predictions_array)
}, coords={'time': dates, 'strategy': strategy_names, 'asset': target_etfs})
```

## Educational Features

**Interactive Learning:**
- **Jupyter Notebooks**: Step-by-step tutorials with explanations and exercises
- **Educational Helper Functions**: `explain_log_returns()`, `explain_walk_forward_analysis()`
- **Student Exercises**: Hands-on coding challenges to implement advanced techniques
- **Blue Water Macro Context**: Professional applications and career development guidance

**Performance Analysis Tools:**
- `calculate_performance_metrics()` - Comprehensive risk-adjusted metrics (Sharpe, Calmar, drawdown)
- `create_correlation_matrix()` - Multi-strategy correlation analysis
- `export_results_to_csv()` - Production-ready result export with metadata

## Production Considerations

**Transaction Cost Modeling:**
- `apply_transaction_costs()` function for realistic performance estimates
- Position change tracking for turnover analysis
- Slippage and market impact considerations

**Enterprise Integration:**
- **netCDF Export**: `results.to_netcdf()` for standardized data exchange
- **Metadata Tracking**: Complete audit trail in xarray attributes
- **Scalability**: Modular design supports additional assets and strategies

## Important Educational Notes

- **Emphasis on Understanding**: Every function includes educational docstrings explaining financial concepts
- **Real-World Applications**: Examples demonstrate institutional-quality practices
- **Career Development**: Framework positions students for quant finance roles
- **Blue Water Macro Branding**: Professional context with enterprise-grade methodologies
- **Progressive Complexity**: From basic single-target to advanced multi-asset portfolio construction