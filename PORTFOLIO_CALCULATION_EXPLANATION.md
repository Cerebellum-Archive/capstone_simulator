# Multi-Target Portfolio Return Calculation: The Correct Approach

## 🎯 **Understanding Multi-Target Portfolio Construction**

This document explains the methodology for calculating portfolio returns in multi-target trading strategies, particularly focusing on long-short strategies that simultaneously predict and trade multiple assets.

---

## 📊 **The Multi-Target Challenge**

### **What is Multi-Target Trading?**

Multi-target strategies simultaneously:
- **Predict returns** for multiple assets (e.g., SPY, QQQ, IWM)
- **Construct portfolios** across all predicted assets
- **Generate alpha** from relative performance differences

**Example Strategy:**
- Predict SPY: +2% return
- Predict QQQ: -1% return  
- Predict IWM: +1% return
- **Portfolio Action**: Long SPY, Short QQQ/IWM (dollar-neutral)

---

## 🧮 **Portfolio Return Calculation**

### **The Fundamental Formula:**

For any multi-target portfolio, the return calculation is:

```python
portfolio_return = Σ(weight_i × return_i) for all assets i
```

### **Why This Formula Works:**

1. **Individual Asset Tracking**: Each asset's contribution is weighted by its position size
2. **Proper Long-Short Accounting**: Long positions gain when asset rises, short positions gain when asset falls
3. **Dollar-Neutral Strategy**: Can generate returns even with zero net exposure

---

## 📈 **Step-by-Step Example: Long-Short Strategy**

### **Scenario: January 1, 2024**

**Input Data:**
- Predictions: SPY=+2%, QQQ=-1%, IWM=+1%
- Actual Returns: SPY=+1.5%, QQQ=-0.8%, IWM=+1.2%
- Strategy: Long-Short (Long SPY, Short QQQ/IWM)

**Step 1: Calculate Position Weights**
```
- Rank predictions: SPY (+2%, rank 1), IWM (+1%, rank 2), QQQ (-1%, rank 3)
- Long the top-ranked asset (SPY) with weight +1.0
- Short the bottom two assets (QQQ, IWM) with equal weights (-0.5 each) to ensure dollar neutrality
- Weights: SPY = +1.0, QQQ = -0.5, IWM = -0.5
- Sum: 1.0 + (-0.5) + (-0.5) = 0.0 (dollar-neutral: zero net exposure)
```

**Step 2: Calculate Individual Contributions**
```
SPY Contribution: +1.0 × +1.5% = +1.5%
QQQ Contribution: -0.5 × -0.8% = +0.4% (short gains when asset falls)
IWM Contribution: -0.5 × +1.2% = -0.6% (short loses when asset rises)
```

**Step 3: Sum for Portfolio Return**
```
Portfolio Return = +1.5% + 0.4% - 0.6% = +1.3%
```

**Result:** The strategy generated +1.3% alpha despite having zero net dollar exposure!

---

## 🎯 **Why This Matters: Real-World Examples**

### **Example 1: Market Crash Protection (2020 COVID)**

**Scenario:** March 2020 COVID crash
- **SPY**: -12% (large-cap stocks crash)
- **QQQ**: -15% (tech stocks crash harder)  
- **IWM**: -18% (small-cap stocks crash hardest)

**Long-Short Strategy Performance:**
- Long SPY: +1.0 × (-12%) = -12%
- Short QQQ: -0.5 × (-15%) = +7.5% (short gains when asset falls)
- Short IWM: -0.5 × (-18%) = +9% (short gains when asset falls)
- **Total Return: -12% + 7.5% + 9% = +4.5%** ✅

**Key Insight:** The strategy profited during a market crash by being short the worst-performing assets!

### **Example 2: Sector Rotation Alpha**

**Scenario:** Tech outperforms, small-caps underperform
- **SPY**: +3% (market average)
- **QQQ**: +5% (tech outperforms)  
- **IWM**: +1% (small-caps underperform)

**Long-Short Strategy Performance:**
- Long SPY: +1.0 × (+3%) = +3%
- Short QQQ: -0.5 × (+5%) = -2.5% (short loses when asset rises)
- Short IWM: -0.5 × (+1%) = -0.5% (short loses when asset rises)
- **Total Return: +3% - 2.5% - 0.5% = 0%** (neutral)

**Key Insight:** The strategy was neutral because it was short the outperforming asset (QQQ).

---

## 🔧 **Technical Implementation**

### **Multi-Target Portfolio Calculation:**

```python
def calculate_portfolio_return(predictions_row, actual_returns_row, position_func, position_params):
    """
    Calculate portfolio return using individual asset weights and returns.
    
    Args:
        predictions_row: Series with predictions for each target
        actual_returns_row: Series with actual returns for each target
        position_func: Position sizing function
        position_params: Parameters for position function
    
    Returns:
        float: Portfolio return for this date
    
    Raises:
        ValueError: If inputs are invalid or mismatched
    """
    if not isinstance(predictions_row, pd.Series) or not isinstance(actual_returns_row, pd.Series):
        raise ValueError("Inputs must be pandas Series")
    if len(predictions_row) != len(actual_returns_row):
        raise ValueError("Predictions and actual returns must have the same length")
    if predictions_row.isna().any() or actual_returns_row.isna().any():
        raise ValueError("Inputs cannot contain NaN values")
    
    # Calculate individual position weights
    weights = position_func(predictions_row, **position_params)
    
    # Calculate portfolio return as weighted sum of individual returns
    portfolio_return = np.sum(weights * actual_returns_row.values)
    
    return portfolio_return
```

### **Key Implementation Details:**

1. **Individual Weight Calculation**: Each asset gets a position weight based on predictions
2. **Weighted Return Sum**: Portfolio return = sum of (weight × return) for each asset
3. **Proper Short Handling**: Short positions have negative weights but can generate positive returns

### **Storing Results in xarray**

Use `xarray` to store multi-target results for analysis:

```python
import xarray as xr

# Example: Store portfolio returns and weights
portfolio_returns = pd.Series(portfolio_returns, index=dates)
weights = pd.DataFrame(weights_dict, index=dates, columns=['SPY', 'QQQ', 'IWM'])

ds = xr.Dataset(
    {
        'portfolio_returns': ('time', portfolio_returns),
        'weights': (['time', 'asset'], weights)
    },
    coords={'time': dates, 'asset': ['SPY', 'QQQ', 'IWM']}
)
ds.to_netcdf('multi_target_results.nc')

# Load and analyze results
ds = xr.open_dataset('multi_target_results.nc')
print(ds.portfolio_returns.mean())  # Average portfolio return
print(ds.weights.std(dim='time'))   # Weight volatility by asset
```

---

## 📊 **Portfolio Construction Strategies**

### **1. Equal Weight Strategy**
```python
# Simple average prediction across all targets
avg_prediction = predictions.mean()
if avg_prediction > 0:
    weights = np.full(n_assets, base_leverage / n_assets)  # Equal long
else:
    weights = np.full(n_assets, -base_leverage / n_assets)  # Equal short
```

### **2. Confidence Weighted Strategy**
```python
# Position size based on prediction magnitude
confidence = predictions.abs()
confidence_normalized = confidence / confidence.sum() if confidence.sum() > 0 else np.zeros_like(confidence)
weights = confidence_normalized * np.sign(predictions) * max_leverage / len(predictions)
```

### **3. Long-Short Strategy**
```python
def long_short_positions(predictions, base_leverage=1.0):
    """
    Long top third, short bottom third, dollar-neutral strategy.
    
    Args:
        predictions: Series of asset predictions
        base_leverage: Total leverage for long positions
    
    Returns:
        Series of position weights
    """
    n_assets = len(predictions)
    ranked = predictions.rank(ascending=False)
    long_mask = ranked <= n_assets / 3
    short_mask = ranked >= 2 * n_assets / 3
    
    n_long = long_mask.sum()
    n_short = short_mask.sum()
    
    if n_long == 0 or n_short == 0:
        return pd.Series(np.zeros(n_assets), index=predictions.index)
    
    long_weight = base_leverage / n_long
    short_weight = -(base_leverage * n_long) / (n_short * n_long)  # Ensure dollar neutrality
    
    weights = pd.Series(np.zeros(n_assets), index=predictions.index)
    weights[long_mask] = long_weight
    weights[short_mask] = short_weight
    
    return weights
```

---

## 🧮 **Mathematical Foundation**

### **Portfolio Return Formula:**

For a portfolio with `n` assets:

**R_portfolio = Σ(w_i × r_i) for i = 1 to n**

Where:
- `w_i` = weight of asset i  
- `r_i` = return of asset i
- `Σ` = summation over all assets from i=1 to i=n

### **Dollar-Neutral vs Market-Neutral:**

**Dollar-Neutral Constraint:**
```
Σ(w_i) = 0 (sum of weights equals zero)
```

**What Dollar-Neutral Means:**
- Total long exposure = Total short exposure in dollar terms
- Portfolio can generate returns even with zero net exposure
- Risk is primarily from relative performance, not market direction
- **Example**: Long $100K SPY, Short $50K QQQ, Short $50K IWM = $0 net exposure

**Dollar-Neutral vs Market-Neutral:**
- **Dollar-Neutral**: Zero net dollar exposure (longs = shorts)
- **Market-Neutral**: Zero beta exposure to market factors (may have non-zero dollar exposure)
- **Key Difference**: Dollar-neutral focuses on position sizing, market-neutral focuses on factor exposure

**Example Distinction:**
- **Dollar-Neutral Strategy**: Long $100K SPY, Short $100K QQQ (dollar-neutral but may have market beta)
- **Market-Neutral Strategy**: Long $100K SPY, Short $100K SPY futures (beta-neutral but may have dollar exposure)

---

## 📈 **Performance Metrics**

### **Key Metrics for Multi-Target Strategies:**

1. **Portfolio Sharpe Ratio**: Risk-adjusted returns
2. **Maximum Drawdown**: Worst peak-to-trough decline
3. **Information Ratio**: Excess return / tracking error
4. **Hit Rate**: Percentage of profitable trades
5. **Individual Asset Performance**: Per-asset strategy effectiveness

### **Example Performance Analysis:**

```
Strategy: Long-Short Multi-Target
Period: January 2020 - June 2025
Annual Return: 8.5%
Annual Volatility: 12.3%
Sharpe Ratio: 0.69
Maximum Drawdown: -15.2%
Hit Rate: 52.3%
```

---

## 🎯 **Best Practices**

### **1. Proper Position Sizing**
- Ensure dollar neutrality (zero net exposure) for long-short strategies
- Scale positions based on prediction confidence
- Consider transaction costs and liquidity

### **2. Risk Management**
- Monitor individual asset exposures
- Set maximum position limits
- Implement stop-loss mechanisms

### **3. Performance Attribution**
- Track contribution from each asset
- Analyze prediction accuracy by asset
- Monitor strategy performance across market regimes

### **4. Transaction Cost Modeling**
Transaction costs (e.g., commissions, bid-ask spreads) reduce net returns. For long-short strategies, model costs as a percentage of trade size:

```python
def apply_transaction_costs(weights, returns, cost_per_trade=0.001):
    position_changes = weights.diff().fillna(0)  # Change in positions
    costs = np.abs(position_changes) * cost_per_trade
    net_returns = returns - costs.sum(axis=1)
    return net_returns
```

- **Example**: A 0.1% cost per trade reduces gross returns by 0.2% per round-trip trade.
- **Recommendation**: Minimize turnover to reduce costs.

### **5. Leverage Constraints**
To manage risk, impose leverage limits:
- **Portfolio Leverage**: Sum of absolute weights (e.g., ≤ 2.0)
- **Asset Leverage**: Maximum weight per asset (e.g., ≤ 1.0)

```python
def apply_leverage_limits(weights, max_portfolio_leverage=2.0, max_asset_leverage=1.0):
    # Cap individual asset leverage
    weights = np.clip(weights, -max_asset_leverage, max_asset_leverage)
    
    # Scale portfolio leverage
    total_leverage = np.abs(weights).sum()
    if total_leverage > max_portfolio_leverage:
        weights = weights * (max_portfolio_leverage / total_leverage)
    return weights
```

---

## 🔬 **Testing Your Implementation**

### **Verification Steps:**

1. **Check Dollar Neutrality**: Sum of weights should be approximately zero (zero net exposure)
2. **Verify Short Positions**: Short positions should gain when asset falls
3. **Test Edge Cases**: Zero predictions, all positive/negative predictions
4. **Compare to Benchmarks**: Equal-weight, buy-and-hold strategies

### **Example Test Case:**
```python
import pandas as pd
import numpy as np

# Test data
predictions = pd.Series({'SPY': 0.02, 'QQQ': -0.01, 'IWM': 0.01})
actual_returns = pd.Series({'SPY': 0.015, 'QQQ': -0.008, 'IWM': 0.012})

# Calculate weights using long-short strategy
weights = long_short_positions(predictions, base_leverage=1.0)

# Expected weights (approximate due to ranking)
expected_weights = pd.Series({'SPY': 1.0, 'QQQ': -0.5, 'IWM': -0.5})

# Calculate portfolio return
portfolio_return = np.sum(weights * actual_returns)

# Expected return
expected_return = 1.0 * 0.015 + (-0.5) * (-0.008) + (-0.5) * 0.012
# = 0.015 + 0.004 - 0.006 = 0.013 (1.3%)

# Verify
np.testing.assert_almost_equal(portfolio_return, 0.013, decimal=4)
print("Test passed: Portfolio return = 1.3%")

### **Unit Tests**

Use `pytest` to verify the portfolio return calculation:

```python
import pytest
import pandas as pd
import numpy as np

def test_calculate_portfolio_return():
    predictions = pd.Series({'SPY': 0.02, 'QQQ': -0.01, 'IWM': 0.01})
    actual_returns = pd.Series({'SPY': 0.015, 'QQQ': -0.008, 'IWM': 0.012})
    
    def position_func(preds, **params):
        return pd.Series({'SPY': 1.0, 'QQQ': -0.5, 'IWM': -0.5})
    
    result = calculate_portfolio_return(predictions, actual_returns, position_func, {})
    expected = 0.013
    assert abs(result - expected) < 1e-6

def test_dollar_neutrality():
    predictions = pd.Series({'SPY': 0.02, 'QQQ': -0.01, 'IWM': 0.01})
    weights = long_short_positions(predictions, base_leverage=1.0)
    assert abs(weights.sum()) < 1e-6  # Should be approximately zero

def test_leverage_limits():
    weights = pd.Series({'SPY': 2.0, 'QQQ': -1.5, 'IWM': -0.5})
    limited_weights = apply_leverage_limits(weights, max_asset_leverage=1.0)
    assert (limited_weights.abs() <= 1.0).all()  # No asset > 1.0 leverage

# Run: pytest test_portfolio.py
```

---

## 📊 **Visualization**

Visualizing portfolio performance helps understand strategy behavior:

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Example: Plot cumulative returns
dates = pd.date_range('2024-01-01', periods=100)
returns = np.random.normal(0.0013, 0.01, 100).cumsum()  # Simulated returns
plt.plot(dates, returns)
plt.title('Cumulative Portfolio Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()

# Example: Plot weight distribution over time
weights_df = pd.DataFrame({
    'SPY': np.random.normal(0.5, 0.2, 100),
    'QQQ': np.random.normal(-0.3, 0.15, 100),
    'IWM': np.random.normal(-0.2, 0.15, 100)
}, index=dates)

plt.figure(figsize=(12, 6))
for asset in weights_df.columns:
    plt.plot(dates, weights_df[asset], label=asset, linewidth=2)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.title('Portfolio Weights Over Time')
plt.xlabel('Date')
plt.ylabel('Weight')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## ✅ **Summary**

The approach for multi-target portfolio return calculation is:

1. **Calculate individual position weights** based on predictions
2. **Multiply each weight by its corresponding return**
3. **Sum all weighted returns** to get portfolio return
4. **Ensure proper handling of short positions**

This methodology enables:
- ✅ Accurate performance measurement
- ✅ Proper risk-adjusted returns
- ✅ Meaningful strategy comparison
- ✅ Valid backtesting results

### **Assumptions and Extensions**
- **Daily Frequency**: Calculations assume daily returns. For other frequencies (e.g., weekly), adjust annualized metrics accordingly.
- **Dollar-Neutral**: The long-short strategy assumes zero net exposure. For non-neutral strategies, modify the weight sum constraint.
- **No Leverage Limits**: Examples assume unconstrained leverage. In practice, apply leverage caps (see Best Practices).

---

*This approach represents the industry standard for multi-target quantitative strategy development, ensuring that portfolio returns are calculated properly and can be used for real investment decisions.* 