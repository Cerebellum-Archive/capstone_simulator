# Quantitative Trading Simulation Framework

This project provides a comprehensive framework for developing, backtesting, and analyzing quantitative trading strategies using Python. It is designed to be an educational tool for students of financial engineering, demonstrating best practices in time series analysis, machine learning pipeline construction, and performance evaluation.

## Core Features

- **Walk-Forward Simulation**: Implements a robust walk-forward backtesting engine to prevent look-ahead bias.
- **Modular Codebase**: Utility functions are separated from the main simulation logic for clarity and reusability.
- **`sklearn` Integration**: Leverages `sklearn` pipelines for building and evaluating complex feature engineering and modeling workflows.
- **Automated Performance Analytics**: Uses the `quantstats` library to generate detailed performance reports, including metrics like Sharpe Ratio, Sortino Ratio, max drawdown, and more.
- **Configurability**: Easily configure simulation parameters, such as tickers, date ranges, and model hyperparameters, from a central `config` dictionary.

## Getting Started

Follow these instructions to set up your local environment and run a simulation.

### 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

You will know the environment is active when you see `(venv)` at the beginning of your command prompt.

### 3. Install Required Libraries

Install all the necessary libraries using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

This command will install all the correct versions of the packages used in this project.

### 4. Run the Simulation

With the environment set up and dependencies installed, you can now run the simulation script:

```bash
python3 BlueWaterMacro_Simulator.py
```

The script will:
1.  Download the necessary ETF price data from Yahoo Finance.
2.  Run the walk-forward simulation for each parameter sweep defined in the `main()` function.
3.  Print detailed performance reports from `quantstats` for each strategy.
4.  Print a summary table comparing the key performance metrics of all strategies.
5.  Display a plot of the cumulative returns for each strategy.

## How to Customize a Simulation

To run your own experiments, you can modify the `config` dictionary and parameter sweep setup within the `main()` function in `BlueWaterMacro_Simulator.py`.

- **Change the Assets**: Modify the `target_etf` and `feature_etfs` in the `config` dictionary.
- **Adjust the Model**: Change the `pipe_steps` to use different `sklearn`-compatible models or feature transformers.
- **Tune Hyperparameters**: Modify the `n_ewa_lags_list` or other parameter lists to test different model configurations.

This framework provides a solid foundation for you to build upon and explore the fascinating world of quantitative finance. Good luck!

This is a self-contained SPY prediction model that uses SPDR sector ETFs as features. As you can see in the image below, it's designed to provide standard textbook regression output and allow easy parameter adjustments to observe the results. In this version, the sector return lags are decayed with an exponentially weighted average (EWA) varying from 1 to 7 days, using data from the yfinance Python library.  See attached for the code (.ipynb or .py file) and pdf of output.  In google colab the notebook should run with "run all".  

Results_xr is an xarray object to store time-sereis based results like predictions, mtm's, and target benchmark data.  These results are added dimension 'tag' to organize results

  

