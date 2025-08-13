### A Practical Way To Backtest Time‑Series Models: Rolling Simulations, Not Just .fit()/.predict()

Most sklearn tutorials stop at a single .fit() on train and .predict() on test. That’s fine for static prediction tasks, but it’s not enough for quantitative trading and other time‑series applications where the data and the model both evolve over time.

To evaluate a trading idea, you need to re‑train your model on information available up to each point in time and predict the next step repeatedly. That’s a rolling, walk‑forward simulation. This article introduces a simple notebook workflow that turns any sklearn‑style model into a time‑aligned backtest with minimal friction.

### What’s missing in .fit()/.predict() for quant

- Time moves. You shouldn’t train once and reuse the model forever. Regimes change.
- Features and targets are time‑indexed. You must enforce “past-only” training windows to avoid look‑ahead.
- Trading returns depend on a position sizing rule applied to predictions, not just the predictions themselves.

A productive backtesting workflow needs to:
- Generate a sequence of training ranges (expanding or rolling windows),
- Refit the model at those points,
- Predict one step ahead each time,
- Convert predictions to positions,
- Compute portfolio returns and metrics.

### The minimal simulation loop

The Capstone notebook provides a small set of utilities that keep the familiar sklearn ergonomics but add the missing walk‑forward structure.

- generate_train_predict_calender: build time‑aligned training windows
- Simulate: orchestrate fit → predict → position sizing → performance, on each step
- Position sizers: functions or classes to translate predictions into weights or leverage
- Performance metrics: Sharpe, drawdown, annualized return, and more

Here’s the essence of the loop you want (simplified):

```python
from single_target_simulator import Simulate, load_and_prepare_data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 1) Load features X and target y (e.g., SPY next-day return)
X, y, _ = load_and_prepare_data(
    etf_list=['SPY', 'XLK', 'XLF', 'XLV'],  # include target + features
    target_etf='SPY',
    start_date='2015-01-01'
)

# 2) Small pipeline using sklearn
pipe_steps = [
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
]

# 3) Walk-forward simulation
result_df, _ = Simulate(
    X=X,
    y=y,
    window_size=200,             # expanding training window
    window_type='expanding',
    pipe_steps=pipe_steps,
    param_grid={},               # no search for this example
    tag='linreg_expanding'
)
```

Under the hood, Simulate repeatedly:
- Fits `Pipeline(steps=pipe_steps)` on data up to date t,
- Predicts on date t+1,
- Applies a position rule (e.g., go long when predicted return > 0, short otherwise),
- Accumulates portfolio returns.

### Why this matters

- You evaluate how models adapt to new data, not just a single snapshot.
- You avoid look‑ahead bias by construction.
- You can compare models and parameters on equal footing: same training frequency, window size, and sizing rules.
- You get portfolio‑level performance, not just prediction error.

### Extend any sklearn model quickly

Any estimator that implements .fit() and .predict() can be dropped into the loop. Examples that work out of the box:
- LinearRegression and Ridge
- ElasticNetCV (for automatic alpha/l1 search)
- Tree‑based models and ensembles
- XGBoost via a thin wrapper

A minimal XGBoost baseline (linear booster, similar spirit to OLS):

```python
from utils_simulate import XGBoostRegressorWrapper
from sklearn.preprocessing import StandardScaler

pipe_steps = [
    ('scaler', StandardScaler()),
    ('model', XGBoostRegressorWrapper(booster='gblinear', n_estimators=1, learning_rate=1.0))
]
result_df, _ = Simulate(
    X=X, y=y, window_size=200, window_type='expanding',
    pipe_steps=pipe_steps, param_grid={}, tag='xgb_linear_like'
)
```

From there, you can gradually increase complexity (depth, trees) and watch how performance scales.

### Position sizing is first‑class

Trading strategies need a mapping from predictions to positions. That’s decoupled here:
- Binary sizer: long if predicted return > 0, short otherwise
- Quartile/Proportional sizers: weight by confidence or thresholds
- You can plug in your own function or class

Because sizing is independent, you can test the same predictions under different portfolio constructions.

### Metrics that matter

Besides error metrics, you’ll want returns‑driven KPIs:
- Annualized return and volatility
- Sharpe ratio
- Max drawdown
- Information ratio against optional benchmarks

Those are computed on the walk‑forward portfolio returns, not on a static test split. This makes model comparisons meaningful for trading applications.

### A quick notebook workflow

- Load data (features + target)
- Pick a model and minimal pipeline
- Choose expanding vs rolling window and size
- Run Simulate and inspect:
  - `result_df` with predictions, positions, and returns
  - performance summary/plots

If you prefer a lightweight UI, there’s also a Streamlit app that lets you select:
- ETFs and start date,
- Window type/size,
- Model (LinearRegression, Ridge, ElasticNetCV, or XGBoost variants),
- Then runs the same walk‑forward engine, returning metrics and charts.

Run it with:
```bash
streamlit run src/app_streamlit.py
```

### Keep parameter sweeps organized with xarray

When you iterate over model settings (e.g., Ridge alphas or XGBoost depth/trees), you’ll generate parallel time series of strategy returns. Converting these into an xarray Dataset (dimensions: time × strategy) keeps results labeled and consistent. You can then:
- Slice/aggregate by strategy, compute Sharpe/vol, and plot cumulative returns across the whole sweep
- Build a strategy correlation matrix from the same Dataset
- Store additional variables (predictions, weights) as separate data variables for richer analysis

This reduces ad‑hoc DataFrame juggling and makes comparisons reproducible. One helper, `create_results_xarray`, turns your sweep outputs into a standard structure you can plot, compare, and export.

Minimal pattern:
```python
# Assume `results` is dict[strategy_name] -> pd.Series of returns
from utils_simulate import create_results_xarray
import numpy as np
import pandas as pd

all_dates = sorted(set().union(*[s.index for s in results.values()]))
strategies = list(results.keys())
mat = np.array([results[name].reindex(all_dates).fillna(0.0).values for name in strategies]).T

ds = create_results_xarray(
    results_dict={'returns': mat},
    time_coord=pd.DatetimeIndex(all_dates),
    strategy_coord=strategies
)
# Now ds['returns'] has dims ['time','strategy']
```

### Where to go from here

- Swap in any sklearn‑compatible estimator to compare models fairly in a rolling setting.
- Add your own position sizer to express product constraints or risk preferences.
- Sweep hyperparameters and visualize performance vs model complexity (e.g., ElasticNetCV search space, XGBoost depth/trees), stored neatly in xarray.
- Integrate benchmarks and information ratios if you track against a specific index.

If you already know sklearn, the only new concept is replacing the one‑time .fit()/.predict() with a small loop that respects time. Once you do that, you’re testing strategies, not just models.

