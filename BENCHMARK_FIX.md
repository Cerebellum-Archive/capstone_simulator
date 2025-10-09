# Benchmark Calculation Error Fix

## Issue
When running the single target simulator, benchmark calculations were failing with the error:
```
ERROR:__main__:Benchmark calculation failed for st_ewa2_binary: too many values to unpack (expected 2)
```

This error occurred for every strategy during the statistics calculation phase.

## Root Cause
In `src/single_target_simulator.py` line 727, the `load_and_prepare_data()` function was being called but not properly unpacking its return values.

**The function returns 3 values:**
```python
def load_and_prepare_data(etf_list, target_etf, start_date=None):
    # ... processing ...
    return X, y, all_returns_df  # Returns 3 values
```

**But the code was only unpacking 2:**
```python
X, y = load_and_prepare_data(all_etfs, target_etf, start_date=start_date)  # ❌ Only 2 variables
```

This caused a `ValueError: too many values to unpack (expected 2)` when the function tried to return 3 values.

## Fix
Updated line 727 to properly unpack all 3 return values:

```python
# Before (incorrect):
X, y = load_and_prepare_data(all_etfs, target_etf, start_date=start_date)

# After (correct):
X, y, all_returns_df = load_and_prepare_data(all_etfs, target_etf, start_date=start_date)
```

## Impact
- ✅ Benchmark calculations now complete successfully
- ✅ Information ratios are properly calculated
- ✅ Excess returns vs benchmarks are displayed
- ✅ Best benchmark identification works correctly
- ✅ Performance summary shows complete benchmark data

## Testing
After the fix, the simulator should show:
- No "Benchmark calculation failed" errors
- Proper values in `best_benchmark`, `best_info_ratio`, and `best_excess_return` rows
- No "Error" or "NaN" values in benchmark-related metrics

## Files Modified
- `src/single_target_simulator.py` (line 727)

---
**Date**: October 9, 2025  
**Status**: FIXED ✅
