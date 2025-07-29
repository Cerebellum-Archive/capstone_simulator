# 2023 Update to Capstone Project - Quantitative Trading Simulation Framework

<div align="center">

![Blue Water Macro Logo](docs/transparent-logo.png)

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange) ![License](https://img.shields.io/badge/License-BWMEL_Educational-green) ![xarray](https://img.shields.io/badge/xarray-Multi--Dimensional-red) ![sklearn](https://img.shields.io/badge/sklearn-ML--Pipeline-yellowgreen)

**Keywords:** `quantitative trading` · `backtesting` · `Python` · `ETFs` · `financial engineering` · `machine learning` · `portfolio optimization` · `xarray` · `educational framework`

</div>

> **Updated quantitative trading simulation framework for backtesting strategies using Python, ideal for financial engineering education and research.**

A hands-on educational platform for **financial engineering students** to develop, backtest, and analyze trading strategies. Built with Python, xarray, and scikit-learn, this framework guides you through the **full quantitative research cycle**: data loading, feature engineering, model exploration, simulation, and reporting.

**Developed by Conrad Gann for Blue Water Macro Corp. © 2025**

## 📋 Table of Contents
- [Why This Framework?](#why-this-framework)
- [Learning Objectives](#learning-objectives)
- [Quick Start](#quick-start)
- [Quick Wins - Copy & Paste Examples](#quick-wins---copy--paste-examples)
- [Repository Structure](#repository-structure)
- [Framework Components](#framework-components)
- [Example Workflow](#example-workflow-full-research-cycle)
- [Student Exercises & Capstone Ideas](#student-exercises--capstone-ideas)
- [Performance Expectations](#performance-expectations)
- [Educational Resources](#educational-resources)
- [Getting Help](#getting-help)

## Why This Framework?

- **Educational Focus**: Step-by-step tutorials teach core concepts like time-series cross-validation, multi-dimensional data handling with xarray, and risk-adjusted performance metrics
- **Full Research Cycle**: From raw data inputs to publication-quality reports—learn how quants at hedge funds structure their workflow
- **xarray for Finance**: Native use of xarray for standardized, multi-dimensional reporting (e.g., results across time/assets/strategies)—a skill increasingly valued in quant roles
- **Real-World Strategies**: Simulate single-asset (e.g., SPY) and multi-asset (e.g., SPY/QQQ/IWM) predictions with position sizing, leverage, and portfolio optimization
- **Capstone-Ready**: Ideal for financial engineering projects—includes exercises, extensions, and resources from QuantNet/CFA
- **Enterprise-Grade Analytics**: Leveraging Blue Water Macro's institutional insights and data science expertise

### Learning Objectives

By completing this framework, students will:

1. **Master time-series simulation** to avoid common pitfalls like look-ahead bias
2. **Use xarray for efficient data organization** and visualization in finance
3. **Build ML pipelines** for feature preprocessing and learner exploration
4. **Generate professional graphs and reports** for strategy evaluation
5. **Understand portfolio construction** and risk management principles
6. **Apply quantitative methods** used in institutional settings

## Quick Start

### Option 1: Install as Python Package (Recommended)
```bash
# Install directly from GitHub
pip install git+https://github.com/Cerebellum-Archive/capstone_simulator.git

# Or install in development mode after cloning
git clone https://github.com/Cerebellum-Archive/capstone_simulator.git
cd capstone_simulator
pip install -e .

# Launch tutorials
jupyter notebook notebooks/01_single_target_tutorial.ipynb
```

### Option 2: Traditional Setup
```bash
# 1. Clone the repository
git clone https://github.com/Cerebellum-Archive/capstone_simulator.git
cd capstone_simulator

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter and start learning
jupyter notebook notebooks/01_single_target_tutorial.ipynb

# Optional: Generate offline PDF tutorial
python scripts/generate_pdf_simple.py
```

### Using the Package in Your Code
```python
# After installation, import anywhere in your Python environment
import capstone_simulator as cs

# Access key functions directly
from capstone_simulator import (
    log_returns, 
    EWMTransformer, 
    create_results_xarray,
    calculate_performance_metrics
)

# Run simulations programmatically
data = cs.utils_simulate.download_etf_data(['SPY', 'QQQ'])
results = cs.single_target_simulator.run_simulation(data)
```

## 🚀 Quick Wins - Copy & Paste Examples

### Add a New ETF to Your Strategy
```python
# In notebooks/01_single_target_tutorial.ipynb or src/single_target_simulator.py
FEATURE_ETFS = ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU', 'VTI']  # Added VTI
```

### Switch to Different Target Assets
```python
# For single-target simulation
TARGET_ETF = 'QQQ'  # Instead of 'SPY'

# For multi-target portfolio
TARGET_ETFS = ['SPY', 'QQQ', 'IWM', 'EFA']  # Added international exposure
```

### Quick Model Comparison
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

# Compare multiple models in one run
models = {
    'ridge': Ridge(alpha=1.0),
    'rf': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    results[name] = simulate_strategy(features, targets, model=model)
```

### Generate Performance Report in 3 Lines
```python
from src.utils_simulate import create_results_xarray, calculate_performance_metrics

results_xr = create_results_xarray(your_results)
metrics = calculate_performance_metrics(results_xr)
print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
```

## Repository Structure

```
quant_trading_simulator/
├── src/                    # Core production code (modular for extensions)
│   ├── utils_simulate.py   # Utilities (expanded with xarray transformers)
│   ├── single_target_simulator.py  # Basic SPY prediction simulator
│   └── multi_target_simulator.py   # Advanced multi-asset simulator
├── notebooks/              # Interactive educational tutorials
│   ├── 01_single_target_tutorial.ipynb    # Basics: Single-asset simulation
│   ├── 02_multi_target_tutorial.ipynb     # Advanced: Multi-asset strategies
│   └── 03_full_research_cycle.ipynb       # End-to-end project demo
├── docs/                   # Documentation and guides
│   ├── ERM3_2e_Data_Dictionary.md         # Blue Water Macro data schema
│   └── README.md                          # This guide
├── data/                   # Sample datasets (user-provided)
├── reports/                # Auto-generated outputs (plots, CSVs)
├── requirements.txt        # Python dependencies
└── .gitignore             # Standard project ignores
```

## Framework Components

### Core Modules (`src/`)
- **`utils_simulate.py`**: Essential utilities for data preparation, returns calculation, and xarray operations
- **`single_target_simulator.py`**: Educational single-asset (SPY) prediction framework
- **`multi_target_simulator.py`**: Production-ready multi-asset prediction and portfolio construction

### Educational Materials (`notebooks/`)
- **Tutorial 1**: Single-target simulation walkthrough with sector ETFs
- **Tutorial 2**: Multi-asset portfolio strategies with advanced position sizing
- **Tutorial 3**: Complete research cycle demonstration with exercises

### Documentation (`docs/`)
- **📖 Complete PDF Tutorial**: Professional offline guide (`Blue_Water_Macro_Tutorial.pdf`)
- **🌐 Interactive HTML Tutorial**: Browser-based tutorial with print-to-PDF capability
- **Data Dictionary**: Blue Water Macro's ERM3 model documentation
- **Implementation Guides**: Best practices and advanced techniques

## Example Workflow (Full Research Cycle)

```python
# 1. INPUTS: Load ETF data via yfinance
import yfinance as yf
data = yf.download(['SPY', 'XLK', 'XLF'], start='2020-01-01')

# 2. PREPROCESSING: Feature engineering with xarray
from src.utils_simulate import EWMTransformer, create_results_xarray
transformer = EWMTransformer(halflife=5)
features_smooth = transformer.fit_transform(features)

# 3. EXPLORATION: Analyze feature importance
from src.utils_simulate import p_by_year
feat_analysis = p_by_year(X_features, y_target)

# 4. SIMULATION: Walk-forward backtesting
results = simulate_strategy(features_smooth, targets, model='ridge')

# 5. REPORTING: Generate publication-quality outputs
results_xr = create_results_xarray(results)
results_xr.plot.line(x='time', col='strategy', col_wrap=2)
```

See `notebooks/03_full_research_cycle.ipynb` for a complete implementation.

## 📖 Offline PDF Tutorial

For comprehensive offline study, we provide multiple options:

### Option 1: Direct PDF Generation (Recommended)
```bash
# Generate professional PDF directly
python scripts/generate_pdf_simple.py
```

### Option 2: HTML + Browser Print
```bash
# Generate interactive HTML tutorial
python scripts/generate_html_tutorial.py

# Then use your browser's "Print to PDF" function (Ctrl+P or Cmd+P)
```

**What's included:**
- ✅ **Complete Framework Guide**: All concepts from basic to advanced
- ✅ **Professional Styling**: Blue Water Macro corporate design
- ✅ **Code Examples**: Syntax-highlighted Python with copy buttons  
- ✅ **Career Guidance**: Pathways to quantitative finance roles
- ✅ **Mathematical Foundations**: Formulas and derivations
- ✅ **Offline Access**: No internet required once generated

**Perfect for:**
- 📚 **Student Study Materials**: Comprehensive reference guide
- 💼 **Career Portfolios**: Demonstrate quantitative expertise
- 🎓 **Academic Courses**: Ready-to-use educational content
- 📋 **Quick Reference**: Formulas and code templates

## Student Exercises & Capstone Ideas

### Beginner Exercises
1. **Volatility Targeting**: Implement position sizing based on realized volatility
2. **Feature Engineering**: Add momentum indicators and technical analysis features
3. **Model Comparison**: Compare Ridge, Random Forest, and Linear models using xarray coordinates

### Advanced Capstone Projects
1. **Regime Detection**: Implement bull/bear market switching using hidden Markov models
2. **Transaction Costs**: Add realistic trading costs and slippage to backtests
3. **Risk Management**: Implement VaR-based position limits and drawdown controls
4. **Alternative Data**: Integrate sentiment analysis or economic indicators
5. **Portfolio Optimization**: Apply Modern Portfolio Theory for asset allocation

## 🔬 Benefits of Enhanced Reproducibility

The expanded metadata hashing in the Capstone Simulator Framework ensures full reproducibility by storing all key parameters, such as data sources, training configurations, model pipelines, and simulation details. This structured approach, combined with the framework's comprehensive caching and metadata tracking, delivers significant benefits for quantitative trading research, particularly in addressing overfitting risks and fostering reliable alpha discovery.

### Key Benefits

#### **Mitigating Overfitting through a Reference Base**
By storing complete experiment configurations (e.g., ETF symbols, training windows, hyperparameters), the framework creates a robust reference base of all simulations. This allows researchers to systematically compare strategies across runs, identifying patterns that may indicate overfitting (e.g., strategies that perform well only on specific parameter sets). A comprehensive record of experiments helps distinguish true alpha—signals with consistent predictive power—from spurious results driven by data-specific noise or over-optimized parameters.

#### **Enhanced Research Rigor**
Full reproducibility ensures every simulation can be recreated exactly, enabling researchers to validate results and build confidence in findings. The metadata (e.g., `random_seed=42`, `window_size=400`) acts as a blueprint, allowing users to revisit past experiments and assess their stability across different time periods or market conditions, reducing the risk of overfitting to a single dataset or period.

#### **Audit Trail for Transparency**
The detailed metadata structure supports regulatory compliance and academic scrutiny by providing a complete audit trail. This transparency is critical in financial research, where documenting the exact data (e.g., `start_date`, `feature_columns`) and model configurations (e.g., `pipe_steps`, `param_grid`) helps verify that performance metrics reflect genuine predictive power rather than overfitting artifacts.

#### **Facilitating Collaboration and Knowledge Sharing**
Researchers and students can share exact simulation setups via metadata, fostering collaboration without ambiguity. This shared reference base allows teams to replicate experiments, refine strategies, and collectively evaluate whether discovered signals are robust or likely overfitted to historical data, accelerating the identification of true alpha.

#### **Version Control and Backward Compatibility**
Tracking `framework_version` and `python_version` ensures experiments remain reproducible even as the codebase evolves. This preserves the integrity of historical results, allowing users to compare new strategies against a baseline of past runs to assess whether improvements are genuine or artifacts of overfitting to new data or configurations.

#### **Educational Value for Understanding Alpha Discovery**
For students, the metadata-driven approach highlights the importance of reproducibility in financial modeling. By cataloging all experiments, it teaches the discipline of systematic testing, helping learners recognize overfitting pitfalls (e.g., cherry-picking parameters) and focus on strategies with repeatable performance, which are more likely to yield true alpha in live trading.

### Linking to Alpha Discovery

The reference base created by storing all experiments is a powerful tool for evaluating the likelihood of discovering true alpha. By maintaining a comprehensive record of simulations—including data fingerprints, model parameters, and performance metrics—researchers can analyze the consistency of strategies across diverse conditions. Strategies that perform well across varied training windows, asset sets, or market regimes are less likely to be overfitted and more likely to capture genuine market inefficiencies. This systematic approach enables users to prioritize robust signals, filter out noise-driven results, and build confidence in strategies that hold up under scrutiny, ultimately increasing the probability of uncovering true alpha.

## Performance Expectations

Based on historical backtests using Blue Water Macro's methodology (2015-2024):

| Strategy Type | Typical Sharpe | Max Drawdown | Best Use Case |
|---------------|----------------|--------------|---------------|
| **Equal Weight** | 0.8 - 1.2 | 15-25% | Conservative, diversified |
| **Confidence Weighted** | 1.0 - 1.8 | 20-35% | Moderate risk, adaptive |
| **Long-Short** | 1.2 - 2.5 | 10-20% | Market-neutral strategies |

## Educational Resources

### Academic References
- **QuantNet Forums**: [quantnet.com](https://quantnet.com) - Premier quantitative finance community
- **CFA Institute**: [cfainstitute.org](https://cfainstitute.org) - Professional certification and readings
- **xarray Documentation**: [xarray.pydata.org](https://xarray.pydata.org/en/stable/examples/finance.html) - Financial data handling
- **Risk Models**: [riskmodels.net](https://riskmodels.net) - Institutional beta factors, grade gross and residual returns

### Professional Development
- **Blue Water Macro**: Enterprise-grade data and analytics for institutional research
- **Industry Applications**: Portfolio management, risk analytics, algorithmic trading
- **Career Paths**: Quantitative researcher, portfolio manager, risk analyst

## 🧪 Running Tests

The framework includes comprehensive unit tests to ensure reliability and correctness of quantitative trading simulations. Tests cover core utilities, simulation engines, caching, and educational functions.

### Quick Test Commands

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=src --cov-report=html

# Run specific test modules
pytest tests/test_utils_simulate.py
pytest tests/test_single_target_simulator.py
pytest tests/test_multi_target_simulator.py

# Run only fast tests (exclude slow integration tests)
pytest -m "not slow"

# Run tests with verbose output
pytest -v

# Run tests in parallel (if you have pytest-xdist installed)
pytest -n auto
```

### Test Categories

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test complete workflows and component interactions  
- **Performance Tests**: Verify reasonable execution times for large datasets
- **Error Handling Tests**: Ensure graceful handling of edge cases and invalid inputs

### Educational Value of Testing

The test suite serves as:
- **Living Documentation**: Examples of how to use each function correctly
- **Quality Assurance**: Confidence that simulations produce reliable results
- **Regression Prevention**: Ensures changes don't break existing functionality
- **Learning Tool**: Students can study tests to understand expected behaviors

## 🔧 Troubleshooting

### Common Installation Issues
```bash
# If you get permission errors on macOS/Linux
pip install --user -r requirements.txt

# If Jupyter won't start
pip install --upgrade jupyter notebook

# If yfinance fails to download data
pip install --upgrade yfinance requests
```

### Runtime Issues
- **"Module not found" errors**: Ensure you're in the correct directory and virtual environment is activated
- **Data download failures**: Check your internet connection and try running the download cell again
- **Memory errors with large datasets**: Reduce the date range or number of ETFs in your configuration
- **Plot not displaying**: Run `%matplotlib inline` in your Jupyter notebook

### Performance Tips
- **Faster backtests**: Reduce `TRAINING_WINDOW_SIZE` or increase `REBALANCE_FREQ` in configuration
- **Better predictions**: Experiment with different `EWM_HALFLIFE` values (try 3, 5, 10, 20 days)
- **Cleaner results**: Use `results_xr.sel(strategy='equal_weight')` to focus on one strategy

## Getting Help

### Technical Support
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides in `docs/` folder
- **Code Comments**: Detailed inline explanations throughout codebase

### Academic Guidance
- **Office Hours**: Contact [Conrad.Gann@BlueWaterMacro.com] for capstone project guidance
- **Collaboration**: Connect with other students on QuantNet forums
- **Industry Mentorship**: Blue Water Macro internship and career opportunities

## Disclaimer

This educational framework is provided for learning purposes. Past performance does not guarantee future results. Always conduct thorough due diligence before making investment decisions. Blue Water Macro Corp. provides no warranty regarding the accuracy or completeness of this software.

## License

This project is licensed under the **Blue Water Macro Educational License (BWMEL)** - see the [LICENSE](LICENSE) file for complete terms.

### 📚 **Educational Use**
- ✅ **Free for students and educators** at accredited institutions
- ✅ **Research and academic publication** permitted with attribution
- ✅ **Capstone projects and dissertations** encouraged
- ✅ **Modifications for educational purposes** allowed

### 🏢 **Commercial Use**
- 📧 **Commercial licensing available** - contact licensing@bluewatermacro.com
- 🤝 **Enterprise partnerships welcome** for institutional adoption
- 💼 **Career opportunities** for exceptional student contributors

### ⚖️ **Key Requirements**
- **Attribution**: Must credit "Blue Water Macro Quantitative Trading Framework"
- **Educational Focus**: Commercial use requires separate license
- **Share-Alike**: Educational improvements should benefit the community
- **Financial Disclaimer**: For educational purposes only - not investment advice

---

**Build your quantitative finance career with enterprise-grade tools and institutional insights.**

*Leveraging Blue Water Macro's ERM3 model for comprehensive market analysis and strategy development.*