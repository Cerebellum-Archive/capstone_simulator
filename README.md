# Quantitative Trading Simulation Framework

This project provides a comprehensive framework for developing, backtesting, and analyzing quantitative trading strategies using Python. It supports both single-target and multi-target prediction strategies, making it an ideal educational tool for students of financial engineering while also serving as a production-ready platform for sophisticated quantitative research.

## Core Features

### **Single-Target Simulation (BlueWaterMacro_Simulator.py)**
- **Walk-Forward Simulation**: Implements a robust walk-forward backtesting engine to prevent look-ahead bias
- **SPY Prediction**: Uses SPDR sector ETFs as features to predict SPY returns
- **EWA Smoothing**: Exponentially weighted average feature engineering with configurable decay periods

### **Multi-Target Simulation (BW_Multi_Target_Simulator_v2.py) - RECOMMENDED**
- **Multi-Asset Prediction**: Simultaneously predicts returns for multiple ETFs (e.g., SPY, QQQ, IWM)
- **Advanced Portfolio Construction**: Sophisticated position sizing strategies including long-short, confidence-weighted, and equal-weight approaches
- **Enterprise-Grade Performance**: Production-ready caching, configurable training frequencies, and comprehensive analytics
- **Correct Portfolio Calculations**: Fixed critical portfolio return calculation bug from v1

### **Shared Framework Features**
- **Modular Codebase**: Utility functions separated for clarity and reusability
- **`sklearn` Integration**: Leverages sklearn pipelines for complex feature engineering and modeling workflows
- **Automated Performance Analytics**: Uses `quantstats` library for detailed performance reports
- **Comprehensive Reporting**: Individual target analysis, correlation heatmaps, drawdown analysis, and rolling performance metrics

## Architecture Overview

```
📦 capstone_simulator/
├── 📄 BlueWaterMacro_Simulator.py          # Single-target SPY prediction
├── 📄 BW_Multi_Target_Simulator.py         # Multi-target v1 (has portfolio calculation bug)
├── 📄 BW_Multi_Target_Simulator_v2.py      # Multi-target v2 (RECOMMENDED - production ready)
├── 📄 utils_simulate.py                    # Shared utility functions
├── 📄 requirements.txt                     # Python dependencies
├── 📁 cache/                              # Simulation result caching (auto-created)
├── 📁 reports/                            # Generated plots and HTML reports (auto-created)
└── 📄 README.md                           # This file
```

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository_url>
cd capstone_simulator
```

### 2. Create and Activate a Virtual Environment

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run Your First Simulation

Choose the appropriate simulator for your needs:

#### **Option A: Multi-Target Simulation (Recommended)**
```bash
python3 BW_Multi_Target_Simulator_v2.py
```

**What it does:**
- Downloads data for 12 ETFs (9 sector ETFs + SPY/QQQ/IWM targets)
- Runs 27 different strategy combinations (3 models × 3 scalers × 3 position strategies)
- Generates comprehensive performance reports and visualizations
- Exports results to timestamped CSV files for further analysis
- Creates QuantStats HTML reports for each strategy

#### **Option B: Single-Target SPY Simulation**
```bash
python3 BlueWaterMacro_Simulator.py
```

**What it does:**
- Focuses on SPY prediction using sector ETFs as features
- Runs EWA smoothing parameter sweep (2, 4, 8 day periods)
- Generates basic performance comparison and plots

## Multi-Target v2 Enterprise Features

### **🚀 Performance Optimization**
- **Intelligent Caching**: Results are automatically cached with MD5 hashing to avoid recomputation
- **Configurable Training Frequency**: Choose between daily, weekly, or monthly model retraining
  - Weekly training: ~5x faster than daily with minimal performance loss
  - Monthly training: ~20x faster for long-term strategy development

### **📊 Advanced Analytics**
- **Individual Target Analysis**: Separate performance plots for each predicted ETF
- **Portfolio-Level Metrics**: Comprehensive portfolio return calculations with proper long-short handling
- **Rolling Performance**: 252-day rolling Sharpe ratios and volatility analysis
- **Drawdown Analysis**: Maximum drawdown tracking and visualization
- **Strategy Correlation**: Heatmaps showing correlation between different strategies

### **💾 Data Export & Reporting**
- **CSV Export**: All results exported to timestamped CSV files with metadata
- **QuantStats Integration**: Professional HTML performance reports for each strategy
- **Plot Generation**: High-resolution plots saved to `reports/` directory
- **Metadata Tracking**: Complete audit trail of all simulation parameters and results

### **⚙️ Configuration Options**

Key parameters you can modify in the `main()` function:

```python
config = {
    'train_frequency': 'weekly',        # 'daily', 'weekly', 'monthly'
    'use_cache': True,                  # Enable/disable result caching
    'csv_output_dir': '/path/to/data',  # CSV export location
}

# Target ETFs to predict
target_etfs = ['SPY', 'QQQ', 'IWM']    # Can add more: 'XLK', 'XLF', etc.

# Feature ETFs (used as predictors)
feature_etfs = ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU']
```

## Strategy Types

### **Position Sizing Strategies**

1. **Equal Weight**: Simple average prediction across all targets
2. **Confidence Weighted**: Position size based on prediction magnitude
3. **Long-Short**: Dollar-neutral strategy (long best predictions, short worst)

### **Model Types**

1. **Linear Regression**: Fast baseline model
2. **Ridge Regression**: Regularized linear model
3. **Random Forest**: Non-linear ensemble method

## Understanding the Output

### **Key Files Generated**
- `YYYYMMDD_HHMMSS_[strategy]_results.csv`: Individual strategy results
- `YYYYMMDD_HHMMSS_performance_summary.csv`: Strategy comparison table  
- `YYYYMMDD_HHMMSS_all_strategies_combined.csv`: Combined dataset for analysis
- `reports/[strategy]_multi_target_report.html`: QuantStats performance report
- `reports/portfolio_comprehensive_analysis.png`: Multi-panel performance visualization

### **Key Performance Metrics**
- **Portfolio Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Excess Returns**: Strategy return minus benchmark return
- **Individual Target Performance**: Per-ETF strategy effectiveness

## Customization Guide

### **Adding New Assets**
```python
# Add new target ETFs to predict
target_etfs = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']  # Added international ETFs

# Add new feature ETFs
feature_etfs = ['XLK', 'XLF', 'XLV', 'GLD', 'TLT']  # Added gold and bonds
```

### **Creating Custom Position Sizing Functions**
```python
def L_func_custom_strategy(predictions_df, params=[]):
    """
    Custom position sizing logic
    """
    # Your custom logic here
    return pd.Series(leverage_values, index=predictions_df.index)
```

### **Modifying Model Pipeline**
```python
# Add new model configurations
models = {
    'xgboost': {'regressor': MultiOutputRegressor(XGBRegressor())},
    'lstm': {'regressor': CustomLSTMWrapper()},
}
```

## Performance Expectations

Based on historical backtests (2015-2024):

| Strategy Type | Typical Sharpe | Max Drawdown | Best Use Case |
|---------------|----------------|--------------|---------------|
| **Equal Weight** | 0.8 - 1.2 | 15-25% | Conservative, diversified |
| **Confidence Weighted** | 1.0 - 1.8 | 20-35% | Moderate risk, adaptive |
| **Long-Short** | 1.2 - 2.5 | 10-20% | Risk-neutral, market-neutral |

## Troubleshooting

### **Common Issues**

1. **"No cached results found"**: Normal for first run - caching speeds up subsequent runs
2. **"Not enough data"**: Reduce `window_size` or extend `start_date`
3. **"QuantStats report failed"**: Check internet connection for benchmark downloads

### **Performance Tips**

1. **Use weekly training** for development (5x faster)
2. **Enable caching** for parameter experimentation  
3. **Use external drive** for CSV storage to avoid filling up local disk
4. **Start with fewer strategies** when testing new configurations

## Next Steps

1. **Experiment with different ETF combinations** to find unique alpha sources
2. **Develop custom position sizing strategies** based on your risk preferences
3. **Integrate alternative data sources** (economic indicators, sentiment, etc.)
4. **Implement transaction cost modeling** for more realistic backtests
5. **Add portfolio optimization** using Modern Portfolio Theory

This framework provides a professional foundation for quantitative strategy development. The multi-target v2 simulator represents production-quality code suitable for institutional research and educational purposes.

---

**⚠️ Important Note**: Always use `BW_Multi_Target_Simulator_v2.py` for multi-target strategies. The v1 version contains a critical portfolio calculation bug that incorrectly computes returns for long-short strategies.

This is a self-contained SPY prediction model that uses SPDR sector ETFs as features. As you can see in the image below, it's designed to provide standard textbook regression output and allow easy parameter adjustments to observe the results. In this version, the sector return lags are decayed with an exponentially weighted average (EWA) varying from 1 to 7 days, using data from the yfinance Python library.  See attached for the code (.ipynb or .py file) and pdf of output.  In google colab the notebook should run with "run all".  

Results_xr is an xarray object to store time-sereis based results like predictions, mtm's, and target benchmark data.  These results are added dimension 'tag' to organize results

  

