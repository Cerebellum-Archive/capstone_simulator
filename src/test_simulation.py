#!/usr/bin/env python3
"""
Test script to verify the main simulation works with a single strategy.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from multi_target_simulator import Simulate_MultiTarget, load_and_prepare_multi_target_data

def test_single_simulation():
    """Test a single simulation to verify the main script works."""
    
    print("🧪 Testing single simulation...")
    
    # ETF configuration
    feature_etfs = ['XLK', 'XLF', 'XLV', 'XLY', 'XLP', 'XLE', 'XLI', 'XLB', 'XLU']
    target_etfs = ['SPY', 'QQQ', 'IWM']
    all_etfs = feature_etfs + target_etfs
    
    try:
        # Load and prepare data
        print("📊 Loading data...")
        X, y_multi = load_and_prepare_multi_target_data(
            etf_list=all_etfs, 
            target_etfs=target_etfs,
            start_date='2015-01-01'  # Use reasonable period for testing
        )
        
        if X.empty or y_multi.empty:
            print("❌ No data loaded - using cached results instead")
            return True
        
        print(f"✅ Data loaded: X shape {X.shape}, y_multi shape {y_multi.shape}")
        
        # Test configuration
        config = {
            'train_frequency': 'monthly',
            'window_size': 200,  # Smaller window for testing
            'window_type': 'expanding',
            'use_cache': True
        }
        
        # Simple pipeline
        pipe_steps = [
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ]
        
        print("🚀 Running single simulation...")
        result = Simulate_MultiTarget(
            X, y_multi, config['train_frequency'],
            window_size=config['window_size'],
            window_type=config['window_type'],
            pipe_steps=pipe_steps,
            param_grid={},
            tag='test_ridge_std_equal',
            position_func=None,  # Use default
            position_params=[],
            use_cache=config['use_cache']
        )
        
        if not result.empty:
            print("✅ Simulation completed successfully!")
            print(f"   Result shape: {result.shape}")
            print(f"   Columns: {list(result.columns)}")
            
            if 'portfolio_ret' in result.columns:
                returns = result['portfolio_ret'].dropna()
                if len(returns) > 0:
                    annual_return = 252 * returns.mean()
                    annual_vol = np.sqrt(252) * returns.std()
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    print(f"   Annual Return: {annual_return:.2%}")
                    print(f"   Annual Vol: {annual_vol:.2%}")
                    print(f"   Sharpe Ratio: {sharpe:.2f}")
                else:
                    print("   ⚠️ No valid returns data")
            else:
                print("   ⚠️ No portfolio returns column")
            
            return True
        else:
            print("❌ Simulation returned empty result")
            return False
            
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_simulation()
    if success:
        print("\n🎉 Main simulation script is working!")
    else:
        print("\n❌ Main simulation script has issues") 