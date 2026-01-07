#!/usr/bin/env python3
"""
Script to view cached simulation results without retraining.
Enhanced to work with new benchmarking framework and configuration structure.
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from utils_simulate import calculate_performance_metrics
warnings.filterwarnings('ignore')

# Constants for annualization
TRADING_DAYS_PER_YEAR = 252

def load_cached_results():
    """Load and display cached simulation results with enhanced benchmarking support."""
    
    cache_dir = Path("cache")
    if not cache_dir.exists():
        print("❌ No cache directory found!")
        print("   Run a simulation first to generate cached results.")
        return
    
    # Find all cached files
    cache_files = list(cache_dir.glob("*.pkl"))
    if not cache_files:
        print("❌ No cached results found!")
        print("   Run a simulation with caching enabled to generate results.")
        return
    
    print(f"📁 Found {len(cache_files)} cached simulation files")
    print(f"📊 Analyzing cached results with enhanced benchmarking support...")
    
    # Load more results to get a better picture
    sample_files = cache_files[:20]  # Show first 20 results
    
    results_summary = []
    benchmark_summary = []
    
    for file_path in sample_files:
        try:
            with open(file_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            if isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                # Extract strategy name from filename
                strategy_name = file_path.stem.split('_', 2)[-1]  # Get part after second underscore
                
                # Determine return column (handle both single and multi-target)
                return_col = None
                if 'portfolio_ret' in cached_data.columns:
                    return_col = 'portfolio_ret'
                elif 'perf_ret' in cached_data.columns:
                    return_col = 'perf_ret'
                
                if return_col:
                    portfolio_returns = cached_data[return_col].dropna()
                    
                    if len(portfolio_returns) > 0:
                        # Core performance metrics using consistent utility
                        metrics = calculate_performance_metrics(portfolio_returns, is_log_returns=True)
                        annual_return = metrics['annualized_return']
                        annual_vol = metrics['volatility']
                        sharpe_ratio = metrics['sharpe_ratio']
                        max_drawdown = metrics['max_drawdown']
                        
                        # Enhanced metrics
                        win_rate = (portfolio_returns > 0).mean()
                        cumulative_return = metrics['total_return']
                        
                        strategy_result = {
                            'Strategy': strategy_name,
                            'Annual Return (%)': annual_return,
                            'Annual Vol (%)': annual_vol,
                            'Sharpe Ratio': sharpe_ratio,
                            'Max Drawdown (%)': max_drawdown,
                            'Win Rate (%)': win_rate,
                            'Cumulative Return (%)': cumulative_return,
                            'Data Points': len(portfolio_returns),
                            'Date Range': f"{portfolio_returns.index.min().date()} to {portfolio_returns.index.max().date()}"
                        }
                        
                        # Check for benchmark data
                        benchmark_cols = [col for col in cached_data.columns if col.startswith('benchmark_')]
                        if benchmark_cols:
                            strategy_result['Benchmarks Available'] = len(benchmark_cols)
                            strategy_result['Benchmark Types'] = ', '.join([col.replace('benchmark_', '') for col in benchmark_cols[:3]])
                            
                            # Calculate information ratio vs first available benchmark
                            first_benchmark = cached_data[benchmark_cols[0]].dropna()
                            if len(first_benchmark) > 0:
                                # Align returns
                                aligned_strategy = portfolio_returns.reindex(first_benchmark.index).dropna()
                                aligned_benchmark = first_benchmark.reindex(aligned_strategy.index).dropna()
                                
                                if len(aligned_strategy) > 0 and len(aligned_benchmark) > 0:
                                    excess_returns = aligned_strategy - aligned_benchmark
                                    excess_mean = excess_returns.mean() * TRADING_DAYS_PER_YEAR
                                    tracking_error = excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                                    info_ratio = excess_mean / tracking_error if tracking_error != 0 else 0
                                    
                                    strategy_result['Excess Return (%)'] = excess_mean
                                    strategy_result['Information Ratio'] = info_ratio
                                    strategy_result['Best Benchmark'] = benchmark_cols[0].replace('benchmark_', '').replace('_', ' ').title()
                        else:
                            strategy_result['Benchmarks Available'] = 0
                            strategy_result['Benchmark Types'] = 'None'
                        
                        results_summary.append(strategy_result)
                        
                        # Enhanced logging
                        bench_info = f", {strategy_result.get('Benchmarks Available', 0)} benchmarks" if benchmark_cols else ""
                        print(f"✅ {strategy_name}: {annual_return:.2%} return, {sharpe_ratio:.2f} Sharpe{bench_info}")
                    else:
                        print(f"⚠️  {strategy_name}: No valid returns data")
                else:
                    print(f"⚠️  {strategy_name}: No portfolio/performance returns column found")
                    print(f"     Available columns: {', '.join(cached_data.columns[:10])}")
            else:
                print(f"⚠️  {file_path.name}: Invalid cached data format")
                
        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {e}")
    
    # Display summary table
    if results_summary:
        print("\n" + "="*100)
        print("CACHED SIMULATION RESULTS SUMMARY")
        print("="*100)
        
        # Convert to DataFrame and sort by Sharpe ratio
        df = pd.DataFrame(results_summary)
        df_sorted = df.sort_values('Sharpe Ratio', ascending=False)
        
        # Format for display
        display_df = df_sorted.copy()
        display_df['Annual Return (%)'] = display_df['Annual Return (%)'].apply(lambda x: f"{x:.2%}")
        display_df['Annual Vol (%)'] = display_df['Annual Vol (%)'].apply(lambda x: f"{x:.2%}")
        display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        display_df['Max Drawdown (%)'] = display_df['Max Drawdown (%)'].apply(lambda x: f"{x:.2%}")
        display_df['Win Rate (%)'] = display_df['Win Rate (%)'].apply(lambda x: f"{x:.2%}")
        display_df['Cumulative Return (%)'] = display_df['Cumulative Return (%)'].apply(lambda x: f"{x:.2%}")
        
        # Format benchmark columns if present
        if 'Excess Return (%)' in display_df.columns:
            display_df['Excess Return (%)'] = display_df['Excess Return (%)'].apply(lambda x: f"{x:.2%}")
        if 'Information Ratio' in display_df.columns:
            display_df['Information Ratio'] = display_df['Information Ratio'].apply(lambda x: f"{x:.2f}")
        
        # Core metrics table
        core_columns = ['Strategy', 'Annual Return (%)', 'Annual Vol (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
        print(display_df[core_columns].to_string(index=False))
        
        # Enhanced metrics table (if benchmark data available)
        if 'Best Benchmark' in display_df.columns and display_df['Best Benchmark'].notna().any():
            print(f"\n📊 BENCHMARK ANALYSIS")
            print("=" * 60)
            benchmark_columns = ['Strategy', 'Best Benchmark', 'Excess Return (%)', 'Information Ratio', 'Win Rate (%)']
            available_benchmark_columns = [col for col in benchmark_columns if col in display_df.columns]
            if len(available_benchmark_columns) > 2:  # More than just Strategy and one other column
                print(display_df[available_benchmark_columns].to_string(index=False))
        
        print(f"\n📊 Loaded {len(results_summary)} successful simulations from cache")
        
        # Show top performers with benchmark information
        if len(results_summary) > 0:
            print(f"\n🏆 Top 3 by Sharpe Ratio:")
            for i, row in df_sorted.head(3).iterrows():
                benchmark_info = f", vs {row.get('Best Benchmark', 'N/A')}" if 'Best Benchmark' in row and pd.notna(row['Best Benchmark']) else ""
                print(f"   {i+1}. {row['Strategy']}: {row['Sharpe Ratio']:.2f} Sharpe, {row['Annual Return (%)']:.2%} return{benchmark_info}")
            
            print(f"\n📈 Top 3 by Annual Return:")
            df_return_sorted = df.sort_values('Annual Return (%)', ascending=False)
            for i, row in df_return_sorted.head(3).iterrows():
                excess_info = f", {row.get('Excess Return (%)', 0):.2%} excess" if 'Excess Return (%)' in row and pd.notna(row['Excess Return (%)']) else ""
                print(f"   {i+1}. {row['Strategy']}: {row['Annual Return (%)']:.2%} return, {row['Sharpe Ratio']:.2f} Sharpe{excess_info}")
            
            print(f"\n📉 Best Risk-Adjusted (Lowest Max Drawdown):")
            df_dd_sorted = df.sort_values('Max Drawdown (%)', ascending=True)
            for i, row in df_dd_sorted.head(3).iterrows():
                info_ratio_info = f", {row.get('Information Ratio', 0):.2f} Info Ratio" if 'Information Ratio' in row and pd.notna(row['Information Ratio']) else ""
                print(f"   {i+1}. {row['Strategy']}: {row['Max Drawdown (%)']:.2%} max DD, {row['Sharpe Ratio']:.2f} Sharpe{info_ratio_info}")
            
            # Additional benchmark-specific analysis if available
            if 'Information Ratio' in df.columns and df['Information Ratio'].notna().any():
                print(f"\n🎯 Top 3 by Information Ratio (vs Benchmarks):")
                df_info_sorted = df.sort_values('Information Ratio', ascending=False)
                for i, row in df_info_sorted.head(3).iterrows():
                    benchmark_name = row.get('Best Benchmark', 'N/A')
                    excess_return = row.get('Excess Return (%)', 0)
                    print(f"   {i+1}. {row['Strategy']}: {row['Information Ratio']:.2f} Info Ratio vs {benchmark_name}, {excess_return:.2%} excess return")
    else:
        print("❌ No valid results found in cache")

if __name__ == "__main__":
    load_cached_results() 