#!/usr/bin/env python3
"""
Test script to verify plotting functions work correctly.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add src to path
sys.path.append('src')

def create_test_data():
    """Create test data for plotting."""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create test returns
    returns = np.random.normal(0.001, 0.02, len(dates))
    benchmark_returns = np.random.normal(0.0008, 0.015, len(dates))
    
    # Create test DataFrame
    df = pd.DataFrame({
        'portfolio_ret': returns,
        'benchmark_spy_only': benchmark_returns,
        'benchmark_equal_weight_targets': benchmark_returns * 0.9,
        'benchmark_zero_return': np.zeros(len(dates))
    }, index=dates)
    
    return df

def test_plotting_functions():
    """Test that plotting functions work and save files correctly."""
    
    print("🧪 Testing plotting functions...")
    
    # Create test data
    test_df = create_test_data()
    
    # Create test config
    config = {
        'run_timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'colors': ['#1f77b4', '#ff7f0e', '#2ca02c']
    }
    
    # Test 1: Simple matplotlib save
    print("\n📊 Test 1: Simple matplotlib save")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(test_df.index, test_df['portfolio_ret'].cumsum(), label='Portfolio')
        ax.plot(test_df.index, test_df['benchmark_spy_only'].cumsum(), label='Benchmark')
        ax.set_title('Test Plot')
        ax.legend()
        
        # Save to reports directory
        current_dir = os.getcwd()
        reports_dir = os.path.join(current_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        test_filename = os.path.join(reports_dir, f'test_plot_{config["run_timestamp"]}.png')
        plt.savefig(test_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(test_filename):
            print(f"✅ Test plot saved successfully: {test_filename}")
        else:
            print(f"❌ Test plot was not created: {test_filename}")
            
    except Exception as e:
        print(f"❌ Error in test 1: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Import and test plotting_utils
    print("\n📊 Test 2: Import plotting_utils")
    try:
        from plotting_utils import create_simple_comparison_plot
        
        # Create test data list
        test_data_list = [test_df]
        test_tags = ['test_strategy']
        
        # Test the plotting function
        result = create_simple_comparison_plot(test_data_list, test_tags, config)
        print(f"✅ Plotting function completed: {result}")
        
    except Exception as e:
        print(f"❌ Error in test 2: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Check reports directory contents
    print("\n📁 Test 3: Check reports directory")
    try:
        reports_dir = os.path.join(os.getcwd(), 'reports')
        files = os.listdir(reports_dir)
        print(f"Reports directory: {reports_dir}")
        print(f"Files in reports directory: {files}")
        
        # Check for recent files
        recent_files = [f for f in files if config["run_timestamp"] in f]
        print(f"Recent files with timestamp {config['run_timestamp']}: {recent_files}")
        
    except Exception as e:
        print(f"❌ Error in test 3: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_plotting_functions() 