## Parameter Sweeps and Organizing Results with xarray

This framework makes it easy to run parameter sweeps for time‑series simulations and to keep results organized using xarray. Below is a compact workflow you can adapt to any sklearn‑compatible model.

### Why sweeps in time‑series?

Static train/test splits don’t reflect how models behave over time. In quant backtesting, you want to measure performance as the model is re‑trained and re‑evaluated across many dates. Sweeps let you compare models or hyperparameters on equal footing (same windowing, same position sizing) and xarray helps keep the outputs tidy and comparable.

### Minimal sweep example (Ridge alpha)

```python
from single_target_simulator import load_and_prepare_data, Simulate
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Load features and target
X, y, _ = load_and_prepare_data(
    etf_list=['SPY','XLK','XLF','XLV'],
    target_etf='SPY',
    start_date='2015-01-01'
)

alphas = [0.1, 1.0, 10.0]
results = {}

for a in alphas:
    pipe = [('scaler', StandardScaler()), ('model', Ridge(alpha=a))]
    regout, _ = Simulate(
        X=X, y=y, window_size=200, window_type='expanding',
        pipe_steps=pipe, param_grid={}, tag=f"ridge_a{a}"
    )
    # Ensure we have performance returns
    if 'perf_ret' not in regout.columns:
        # For single-target flow, add simple binary sizing if needed
        from single_target_simulator import BinaryPositionSizer
        common = regout.index.intersection(y.index)
        regout = regout.loc[common].copy()
        regout['actual'] = y.loc[common]
        sizer = BinaryPositionSizer(-1.0, 1.0)
        regout['leverage'] = sizer.calculate_position(regout['prediction'])
        regout['perf_ret'] = regout['leverage'] * regout['actual']

    results[f"ridge_alpha_{a}"] = regout['perf_ret'].dropna()
```

### Turn the sweep into a structured dataset with xarray

The helper `create_results_xarray` builds a standard, labeled, multi‑dimensional dataset so you can slice, aggregate, and plot consistently across strategies.

```python
from utils_simulate import create_results_xarray

# Align time across series and build a matrix of returns
all_dates = sorted(set().union(*[s.index for s in results.values()]))
returns_mat = []
strategies = list(results.keys())
for name in strategies:
    s = results[name].reindex(all_dates).fillna(0.0)
    returns_mat.append(s.values)

import numpy as np
returns_mat = np.array(returns_mat).T  # shape: (time, strategy)

ds = create_results_xarray(
    results_dict={'returns': returns_mat},
    time_coord=pd.DatetimeIndex(all_dates),
    strategy_coord=strategies
)

print(ds)
```

This produces an xarray `Dataset` with labeled dimensions `time` and `strategy` and a variable `returns`. You can add more variables (e.g., predictions, weights) by extending `results_dict` with conforming arrays/frames.

### Analyze and visualize

```python
# Basic summary by strategy
mean_by_strategy = ds['returns'].mean('time').to_pandas()
vol_by_strategy = ds['returns'].std('time').to_pandas() * (252 ** 0.5)
sharpe_by_strategy = (ds['returns'].mean('time') / ds['returns'].std('time')).to_pandas() * (252 ** 0.5)

print('Annualized Volatility:\n', vol_by_strategy)
print('Sharpe (approx):\n', sharpe_by_strategy)

# Plot cumulative return per strategy
cum = (1.0 + ds['returns'].to_pandas()).cumprod()
ax = cum.plot(title='Cumulative Return by Strategy', figsize=(10, 4))
ax.set_ylabel('Cumulative Return')
```

### Correlations across strategies

```python
from utils_simulate import create_correlation_matrix
corr = create_correlation_matrix(ds, ['returns'])
print(corr)
```

### Notes

- You can include GridSearchCV or ElasticNetCV directly inside the pipeline. In time‑series, prefer explicit rolling/expanding schemes for honest evaluation.
- For multi‑target simulations, organize predictions/returns similarly, then store them with `create_results_xarray` using additional coordinates (e.g., `asset`).
- Once your results live in xarray, comparing strategies, plotting, and exporting become straightforward and reproducible.


