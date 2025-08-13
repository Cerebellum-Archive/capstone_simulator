# -*- coding: utf-8 -*-
"""
XGBoost Complexity Sweep in the Simulation Framework

- Wrap XGBoost for sklearn Pipeline via XGBoostRegressorWrapper
- Start with nearly-OLS baseline using 'gblinear' booster
- Gradually increase complexity (tree depth, n_estimators)
- Compute a simple complexity score from params
- Run single-target simulations and graph performance vs complexity
"""

import os
import sys
from pathlib import Path

# Ensure src is on path when executed directly
this_dir = Path(__file__).parent
sys.path.append(str(this_dir))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils_simulate import (
    XGBoostRegressorWrapper,
    calculate_performance_metrics
)
from single_target_simulator import (
    load_and_prepare_data,
    Simulate,
    BinaryPositionSizer
)


def xgb_complexity_score(params: dict) -> float:
    """A simple monotonic complexity score for XGBoost params.
    - For gblinear: base ~ 1.5 (slightly above OLS)
    - For gbtree/gbdart: grows with depth, estimators, and min_child_weight inverse.
    """
    booster = params.get('booster', 'gbtree')
    if booster == 'gblinear':
        return 1.5
    max_depth = int(params.get('max_depth', 3))
    n_estimators = int(params.get('n_estimators', 100))
    min_child_weight = float(params.get('min_child_weight', 1.0))
    # Simple heuristic: trees * depth scaled, penalize low min_child_weight
    return 1.0 + (max_depth * n_estimators) / 200.0 + (1.0 / max(1.0, min_child_weight)) * 0.2


def run_sweep():
    # Data setup
    etf_list = ['SPY', 'XLK', 'XLF', 'XLV']
    target_etf = 'SPY'
    start_date = '2015-01-01'

    X, y, _all_returns = load_and_prepare_data(etf_list, target_etf, start_date=start_date)

    # Define configurations from minimal to increasing complexity
    configs = [
        # Nearly OLS baseline using linear booster
        {
            'name': 'gblinear_minimal',
            'params': {
                'booster': 'gblinear',
                'n_estimators': 1,
                'learning_rate': 1.0,
                'reg_lambda': 0.0,
                'reg_alpha': 0.0,
            }
        },
        # Very shallow tree, single estimator
        {
            'name': 'gbtree_d1_t1',
            'params': {
                'booster': 'gbtree',
                'max_depth': 1,
                'n_estimators': 1,
                'learning_rate': 1.0,
                'min_child_weight': 1.0,
                'subsample': 1.0,
                'colsample_bytree': 1.0
            }
        },
        # Increase estimators
        {
            'name': 'gbtree_d1_t10',
            'params': {
                'booster': 'gbtree',
                'max_depth': 1,
                'n_estimators': 10,
                'learning_rate': 0.3,
                'min_child_weight': 1.0,
                'subsample': 1.0,
                'colsample_bytree': 1.0
            }
        },
        # Increase depth
        {
            'name': 'gbtree_d3_t50',
            'params': {
                'booster': 'gbtree',
                'max_depth': 3,
                'n_estimators': 50,
                'learning_rate': 0.2,
                'min_child_weight': 1.0,
                'subsample': 0.9,
                'colsample_bytree': 0.9
            }
        },
        # Higher depth and estimators
        {
            'name': 'gbtree_d5_t100',
            'params': {
                'booster': 'gbtree',
                'max_depth': 5,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'min_child_weight': 1.0,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        }
    ]

    results = []

    for cfg in configs:
        learner = XGBoostRegressorWrapper(**cfg['params'])
        pipe_steps = [
            ('scaler', StandardScaler()),
            ('model', learner)
        ]
        # Keep simulation reasonably small for speed
        result_df, _meta = Simulate(
            X=X.iloc[:750],
            y=y.iloc[:750],
            window_size=200,
            window_type='expanding',
            pipe_steps=pipe_steps,
            param_grid={},
            tag=f"xgb_{cfg['name']}"
        )
        if result_df.empty:
            continue
        # Compute performance on simulated portfolio using repo's position sizing
        common_index = result_df.index.intersection(y.index)
        regout = result_df.loc[common_index].copy()
        regout['actual'] = y.loc[common_index]
        # Use a simple binary sizer (consistent with single_target main flow)
        sizer = BinaryPositionSizer(-1.0, 1.0)
        regout['leverage'] = sizer.calculate_position(regout['prediction'])
        regout['perf_ret'] = regout['leverage'] * regout['actual']
        perf = regout['perf_ret'].dropna()

        metrics = calculate_performance_metrics(perf)
        comp = xgb_complexity_score(cfg['params'])
        results.append({
            'name': cfg['name'],
            'complexity': comp,
            'annualized_return': metrics.get('annualized_return', 0.0),
            'sharpe': metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': metrics.get('max_drawdown', 0.0)
        })
        print(f"{cfg['name']}: complexity={comp:.2f}, sharpe={metrics.get('sharpe_ratio', 0.0):.3f}")

    if not results:
        print("No results to plot.")
        return

    df = pd.DataFrame(results).sort_values('complexity')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(df['complexity'], df['annualized_return'], marker='o')
    axes[0].set_title('Annualized Return vs Complexity')
    axes[0].set_xlabel('Complexity')
    axes[0].set_ylabel('Annualized Return')

    axes[1].plot(df['complexity'], df['sharpe'], marker='o')
    axes[1].set_title('Sharpe vs Complexity')
    axes[1].set_xlabel('Complexity')
    axes[1].set_ylabel('Sharpe')

    axes[2].plot(df['complexity'], df['max_drawdown'], marker='o')
    axes[2].set_title('Max Drawdown vs Complexity')
    axes[2].set_xlabel('Complexity')
    axes[2].set_ylabel('Max Drawdown')

    plt.tight_layout()
    os.makedirs('reports', exist_ok=True)
    out_path = os.path.join('reports', 'xgboost_complexity_sweep.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot: {os.path.abspath(out_path)}")


if __name__ == '__main__':
    run_sweep()

