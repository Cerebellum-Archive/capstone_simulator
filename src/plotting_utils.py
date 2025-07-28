"""
Professional plotting utilities for financial strategy tear sheets.
Clean, modular approach to creating publication-quality visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple

# Constants
TRADING_DAYS_PER_YEAR = 252

def format_benchmark_name(benchmark_col: str) -> str:
    """Format benchmark column name for display with abbreviations."""
    name = benchmark_col.replace('benchmark_', '').replace('_', ' ').title()
    # Apply abbreviations
    if name == 'Equal Weight Targets':
        return 'EQ Weight'
    return name

@dataclass
class PlotConfig:
    """Configuration for plot styling and layout."""
    colors: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])
    figure_size: Tuple[int, int] = (11, 14)
    dpi: int = 300
    column_widths: List[float] = field(default_factory=lambda: [0.35, 0.15, 0.15, 0.15, 0.20])
    header_color: str = '#34495e'
    text_color: str = '#2c3e50'
    grid_color: str = '#95a5a6'

# Set professional plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5

def create_professional_tear_sheet(regout_list, sweep_tags, config):
    """
    Create a professional, publication-quality tear sheet.
    
    Args:
        regout_list: List of DataFrames with simulation results
        sweep_tags: List of strategy tags
        config: Configuration dictionary with run_timestamp
    
    Returns:
        str: Path to saved PDF file
    """
    # Prepare data
    cumulative_returns_data = []
    performance_data = []
    
    for regout_df, tag in zip(regout_list, sweep_tags):
        # Handle both single-target ('perf_ret') and multi-target ('portfolio_ret') column names
        returns_col = None
        if 'portfolio_ret' in regout_df.columns:
            returns_col = 'portfolio_ret'
        elif 'perf_ret' in regout_df.columns:
            returns_col = 'perf_ret'
        
        if returns_col:
            returns = regout_df[returns_col].dropna()
            if len(returns) > 0:
                cumulative_returns = returns.cumsum()
                cumulative_returns_data.append({
                    'strategy': tag,
                    'cumulative_returns': cumulative_returns,
                    'returns': returns
                })
                
                # Calculate performance metrics
                annual_return = TRADING_DAYS_PER_YEAR * returns.mean()
                annual_vol = np.sqrt(TRADING_DAYS_PER_YEAR) * returns.std()
                sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
                
                # Find best strategy-appropriate benchmark
                benchmark_cols = [col for col in regout_df.columns if col.startswith('benchmark_')]
                best_benchmark_name = "N/A"
                best_excess_return = 0
                best_info_ratio = 0
                
                if benchmark_cols:
                    # Determine strategy type and filter to appropriate benchmarks
                    strategy_type = 'equal_weight'  # default
                    if 'longshort' in tag.lower() or 'long_short' in tag.lower():
                        strategy_type = 'long_short'
                    elif 'confidenceweighted' in tag.lower() or 'confidence_weighted' in tag.lower() or 'confidence' in tag.lower():
                        strategy_type = 'confidence_weighted'
                    
                    # Filter benchmarks to show only the most relevant ones (exclude legacy 'benchmark_ret')
                    relevant_benchmarks = []
                    if strategy_type == 'long_short':
                        priority_benchmarks = ['zero_return', 'random_long_short', 'equal_weight_targets']
                    elif strategy_type == 'confidence_weighted':
                        priority_benchmarks = ['risk_parity', 'spy_only', 'equal_weight_targets']
                    else:  # equal_weight or single-target
                        priority_benchmarks = ['buy_and_hold', 'zero_return', 'spy_only', 'equal_weight_targets', 'vti_market', 'equal_weight_all']
                    
                    # Select benchmarks that exist in priority order
                    for priority in priority_benchmarks:
                        benchmark_col = f'benchmark_{priority}'
                        if benchmark_col in benchmark_cols:
                            relevant_benchmarks.append(benchmark_col)
                    
                    # If no priority benchmarks found, take available benchmarks (excluding legacy 'benchmark_ret')
                    if not relevant_benchmarks:
                        relevant_benchmarks = [col for col in benchmark_cols if col != 'benchmark_ret'][:1]
                    
                    # Find best benchmark among relevant ones
                    best_ir = -np.inf
                    for benchmark_col in relevant_benchmarks:
                        if benchmark_col in regout_df.columns:
                            benchmark_returns = regout_df[benchmark_col].dropna()
                            if len(benchmark_returns) > 0:
                                # Calculate information ratio
                                aligned_strategy, aligned_benchmark = returns.align(benchmark_returns, join='inner')
                                if len(aligned_strategy) > 0:
                                    excess_returns = aligned_strategy - aligned_benchmark
                                    excess_mean = excess_returns.mean() * TRADING_DAYS_PER_YEAR
                                    tracking_error = excess_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
                                    info_ratio = excess_mean / tracking_error if tracking_error != 0 else 0
                                    
                                    if info_ratio > best_ir:
                                        best_ir = info_ratio
                                        best_benchmark_name = benchmark_col.replace('benchmark_', '').replace('_', ' ').title()
                                        best_excess_return = excess_mean
                                        best_info_ratio = info_ratio
                
                performance_data.append({
                    'Strategy': tag.replace('mt_', '').replace('_', ' '),
                    'Annual Return (%)': f"{annual_return:.2%}",
                    'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                    'Max Drawdown (%)': f"{max_drawdown:.2%}",
                    'Best Bench': best_benchmark_name,
                    'Excess Return (%)': f"{best_excess_return:.2%}",
                    'Info Ratio': f"{best_info_ratio:.2f}"
                })
    
    if not cumulative_returns_data:
        print("No valid data for tear sheet")
        return None
    
    # Create figure with professional layout
    fig = plt.figure(figsize=(11, 14))
    
    # Define color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Create GridSpec for precise layout control with minimal spacing
    gs = GridSpec(4, 1, figure=fig, height_ratios=[0.4, 2.8, 0.3, 1.8], hspace=0.3)
    
    # 1. Header Section
    ax_header = fig.add_subplot(gs[0])
    ax_header.axis('off')
    
    # Add professional header with better spacing
    ax_header.text(0.5, 0.8, 'STRATEGY PERFORMANCE TEAR SHEET', 
                  fontsize=18, fontweight='bold', ha='center', va='center',
                  transform=ax_header.transAxes, color='#2c3e50')
    
    # Add subtitle with date and simulation identifier
    date_str = datetime.now().strftime('%B %d, %Y')
    ax_header.text(0.5, 0.4, f'Generated: {date_str}', 
                  fontsize=10, ha='center', va='center',
                  transform=ax_header.transAxes, color='#7f8c8d')
    
    # Add simulation identifier
    sim_id = config.get('run_timestamp', 'unknown')
    ax_header.text(0.5, 0.15, f'Simulation ID: {sim_id}', 
                  fontsize=9, ha='center', va='center',
                  transform=ax_header.transAxes, color='#95a5a6')
    
    # 2. Main Chart Section
    ax_chart = fig.add_subplot(gs[1])
    
    # Plot cumulative returns
    for i, data in enumerate(cumulative_returns_data):
        color = colors[i % len(colors)]
        ax_chart.plot(data['cumulative_returns'].index, data['cumulative_returns'].values,
                     linewidth=2.5, color=color, alpha=0.9, label=data['strategy'].replace('mt_', '').replace('_', ' '))
    
    # Professional chart styling - clean and minimal with minimal spacing
    ax_chart.set_title('Cumulative Strategy Returns', fontsize=14, fontweight='bold', pad=4, color='#2c3e50')
    ax_chart.set_ylabel('Cumulative Log-Return', fontsize=12, color='#2c3e50')
    ax_chart.tick_params(axis='both', which='major', labelsize=10, width=2, length=6)
    
    # Add year tick marks to x-axis with enhanced styling
    
    # Force exactly one tick per year and make them very prominent
    ax_chart.xaxis.set_major_locator(mdates.YearLocator(1))
    ax_chart.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Make tick marks very prominent and visible
    ax_chart.tick_params(axis='x', which='major', width=3, length=10, color='#2c3e50', labelsize=11)
    ax_chart.tick_params(axis='y', which='major', width=3, length=8, color='#2c3e50', labelsize=11)
    
    # Force tick mark visibility
    ax_chart.tick_params(axis='both', which='major', direction='out')
    
    # Remove box and grid, add clean x and y lines
    ax_chart.spines['top'].set_visible(False)
    ax_chart.spines['right'].set_visible(False)
    ax_chart.spines['left'].set_visible(True)
    ax_chart.spines['bottom'].set_visible(True)
    ax_chart.grid(False)
    
    # Style the visible spines to be thick and dark
    ax_chart.spines['left'].set_linewidth(3)
    ax_chart.spines['left'].set_color('#2c3e50')
    ax_chart.spines['bottom'].set_linewidth(3)
    ax_chart.spines['bottom'].set_color('#2c3e50')
    
    # Add clean reference lines - y=0 line thinner, x-axis line thick
    ax_chart.axhline(y=0, color='#2c3e50', linewidth=1.5, alpha=0.8, linestyle='-')
    ax_chart.axvline(x=ax_chart.get_xlim()[0], color='#2c3e50', linewidth=3, alpha=1.0, linestyle='-')
    
    # 3. Legend Section - positioned right under the plot with minimal spacing
    ax_legend = fig.add_subplot(gs[2])
    ax_legend.axis('off')
    
    # Create legend with clean styling (no background) positioned at top of legend section
    legend_elements = []
    for i, data in enumerate(cumulative_returns_data):
        color = colors[i % len(colors)]
        legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=3, 
                                        label=data['strategy'].replace('mt_', '').replace('_', ' ')))
    
    ax_legend.legend(handles=legend_elements, loc='upper center', ncol=3, 
                     fontsize=10, frameon=False, columnspacing=2.0)
    
    # 4. Performance Table Section
    ax_table = fig.add_subplot(gs[3])
    ax_table.axis('off')
    
    # Create performance table and sort by Excess Return
    df = pd.DataFrame(performance_data)
    
    # Convert percentage strings to numeric for sorting
    df['Excess_Return_Numeric'] = df['Excess Return (%)'].str.rstrip('%').astype(float)
    df_sorted = df.sort_values('Excess_Return_Numeric', ascending=False)
    df_sorted = df_sorted.drop('Excess_Return_Numeric', axis=1)  # Remove helper column
    
    # Add table title with better spacing
    ax_table.text(0.5, 0.98, 'PERFORMANCE SUMMARY', fontsize=14, fontweight='bold',
                 ha='center', va='top', transform=ax_table.transAxes, color='#2c3e50')
    
    # Create table with professional styling and proper column widths
    table = ax_table.table(cellText=df_sorted.values,
                          colLabels=df_sorted.columns,
                          cellLoc='center',
                          loc='center',
                          bbox=[0.05, 0.08, 0.9, 0.8])
    
    # PDF-safe column widths with extra padding to prevent overlap
    def calculate_pdf_safe_widths(df, headers):
        """Calculate column widths optimized for PDF rendering with padding."""
        col_widths = []
        
        # Calculate character width for each column with extra safety margin
        char_widths = []
        for i, col in enumerate(df.columns):
            if i < len(headers):
                header_len = len(headers[i].replace('\n', ' '))
                content_len = df[col].astype(str).str.len().max() if len(df) > 0 else 0
                max_len = max(header_len, content_len)
                # Add 20% padding for PDF safety
                padded_len = int(max_len * 1.2)
                char_widths.append(padded_len)
        
        total_chars = sum(char_widths)
        available_width = 0.85  # Reduced from 0.9 for more margin
        
        # Calculate proportional widths
        for char_width in char_widths:
            proportional_width = (char_width / total_chars) * available_width
            # Apply stricter constraints for PDF stability
            constrained_width = max(0.10, min(0.30, proportional_width))
            col_widths.append(constrained_width)
        
        # Ensure total doesn't exceed available space
        total_width = sum(col_widths)
        if total_width > available_width:
            col_widths = [w * (available_width / total_width) for w in col_widths]
        
        # Debug output
        print("PDF-safe column widths:")
        for i, (header, width) in enumerate(zip(headers, col_widths)):
            print(f"  Col {i}: {header.replace(chr(10), ' ')} -> {width:.3f}")
        print(f"  Total width: {sum(col_widths):.3f}")
            
        return col_widths
    
    # Calculate PDF-safe widths using header texts with line breaks
    header_texts_for_width = ['Strategy', 'Annual\nReturn (%)', 'Sharpe\nRatio', 'Max\nDrawdown (%)', 'Best\nBench', 'Excess\nReturn (%)', 'Info\nRatio']
    col_widths = calculate_pdf_safe_widths(df_sorted, header_texts_for_width)
    
    for i, width in enumerate(col_widths):
        if i < len(df_sorted.columns):  # Prevent index errors
            for j in range(len(df_sorted) + 1):  # +1 for header row
                table[(j, i)].set_width(width)
                # Set alignment: Strategy column left-aligned, others right-aligned
                if i == 0:  # Strategy column
                    table[(j, i)]._text.set_horizontalalignment('left')
                else:  # Data columns
                    table[(j, i)]._text.set_horizontalalignment('right')
    
    # Style the table with PDF-optimized settings
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # Slightly smaller font for better fit
    table.scale(1.0, 2.0)  # Reduced horizontal scaling to prevent overlap
    
    # Add explicit cell padding for PDF stability
    for key, cell in table.get_celld().items():
        cell.set_text_props(wrap=False)  # Prevent text wrapping
        cell.PAD = 0.02  # Add small padding
    
    # Add text wrapping for column headers (updated for 7 columns)
    header_texts = ['Strategy', 'Annual\nReturn (%)', 'Sharpe\nRatio', 'Max\nDrawdown (%)', 'Best\nBench', 'Excess\nReturn (%)', 'Info\nRatio']
    for i, header_text in enumerate(header_texts):
        if i < len(df_sorted.columns):  # Prevent index errors
            table[(0, i)]._text.set_text(header_text)
    
    # Color the header row and increase its height
    for i in range(len(df_sorted.columns)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.18)  # Increase header row height significantly
    
    # Color alternating rows
    for i in range(1, len(df_sorted) + 1):
        for j in range(len(df_sorted.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    # Save the tear sheet
    os.makedirs('reports', exist_ok=True)
    pdf_filename = f'reports/sim_tear_sheet_{config["run_timestamp"]}.pdf'
    png_filename = f'reports/sim_tear_sheet_{config["run_timestamp"]}.png'
    
    plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(png_filename, dpi=300, bbox_inches='tight', format='png')
    
    print(f"📊 Professional tear sheet saved:")
    print(f"   📄 PDF: {pdf_filename}")
    print(f"   🖼️  PNG: {png_filename}")
    
    plt.close()
    
    return pdf_filename

def plot_strategy_vs_benchmarks(regout_df: pd.DataFrame, strategy_name: str, config) -> str:
    """Plot strategy performance against strategy-appropriate benchmarks only."""
    benchmark_cols = [col for col in regout_df.columns if col.startswith('benchmark_')]
    
    if not benchmark_cols:
        print(f"No benchmarks found for {strategy_name}")
        return None
    
    # Determine strategy type and filter to appropriate benchmarks
    strategy_type = 'equal_weight'  # default
    if 'longshort' in strategy_name.lower() or 'long_short' in strategy_name.lower():
        strategy_type = 'long_short'
    elif 'confidenceweighted' in strategy_name.lower() or 'confidence_weighted' in strategy_name.lower() or 'confidence' in strategy_name.lower():
        strategy_type = 'confidence_weighted'
    
    # Filter benchmarks to show only the most relevant ones
    relevant_benchmarks = []
    if strategy_type == 'long_short':
        # For long-short: show zero return, random long-short, and equal weight for context
        priority_benchmarks = ['zero_return', 'random_long_short', 'equal_weight_targets']
    elif strategy_type == 'confidence_weighted':
        # For confidence weighted: show risk parity, spy, and equal weight (in that order)
        priority_benchmarks = ['risk_parity', 'spy_only', 'equal_weight_targets']
    else:  # equal_weight
        # For equal weight: show spy, equal weight, and vti
        priority_benchmarks = ['spy_only', 'equal_weight_targets', 'vti_market', 'equal_weight_all']
    
    # Select benchmarks that exist in priority order
    for priority in priority_benchmarks:
        benchmark_col = f'benchmark_{priority}'
        if benchmark_col in benchmark_cols:
            relevant_benchmarks.append(benchmark_col)
    
    # If no priority benchmarks found, take available benchmarks (excluding legacy 'benchmark_ret')
    if not relevant_benchmarks:
        relevant_benchmarks = [col for col in benchmark_cols if col != 'benchmark_ret'][:3]
    
    # Limit to max 3 benchmarks for readability
    relevant_benchmarks = relevant_benchmarks[:3]
    
    print(f"Strategy: {strategy_name}, Type: {strategy_type}")
    print(f"Available benchmarks: {[col.replace('benchmark_', '') for col in benchmark_cols]}")
    print(f"Selected relevant benchmarks: {[col.replace('benchmark_', '') for col in relevant_benchmarks]}")
    
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    plot_config = PlotConfig()
    
    # Cumulative returns comparison
    strategy_cumret = regout_df['portfolio_ret'].cumsum()
    ax1.plot(strategy_cumret.index, strategy_cumret.values, 
             linewidth=3, label=f'{strategy_name.replace("mt_", "").replace("_", " ")} Strategy', color='red')
    
    for i, benchmark_col in enumerate(relevant_benchmarks):
        if benchmark_col in regout_df.columns:
            benchmark_cumret = regout_df[benchmark_col].cumsum()
            benchmark_name = format_benchmark_name(benchmark_col)
            color = plot_config.colors[i % len(plot_config.colors)]
            ax1.plot(benchmark_cumret.index, benchmark_cumret.values,
                     linewidth=2, label=f'{benchmark_name} Benchmark', 
                     color=color, linestyle='--', alpha=0.7)
    
    ax1.set_title(f'{strategy_name.replace("mt_", "").replace("_", " ")} vs Relevant Benchmarks: Cumulative Returns', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Log-Return')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    
    # Rolling Sharpe ratio comparison (use same relevant benchmarks)
    window = TRADING_DAYS_PER_YEAR  # 1-year rolling
    strategy_ret = regout_df['portfolio_ret']
    strategy_rolling_sharpe = (strategy_ret.rolling(window).mean() / 
                              strategy_ret.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    ax2.plot(strategy_rolling_sharpe.index, strategy_rolling_sharpe.values,
             linewidth=3, label=f'{strategy_name.replace("mt_", "").replace("_", " ")} Strategy', color='red')
    
    for i, benchmark_col in enumerate(relevant_benchmarks):
        if benchmark_col in regout_df.columns:
            benchmark_ret = regout_df[benchmark_col]
            benchmark_rolling_sharpe = (benchmark_ret.rolling(window).mean() / 
                                       benchmark_ret.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            benchmark_name = format_benchmark_name(benchmark_col)
            color = plot_config.colors[i % len(plot_config.colors)]
            ax2.plot(benchmark_rolling_sharpe.index, benchmark_rolling_sharpe.values,
                     linewidth=2, label=f'{benchmark_name} Benchmark',
                     color=color, linestyle='--', alpha=0.7)
    
    ax2.set_title(f'{strategy_name.replace("mt_", "").replace("_", " ")} vs Relevant Benchmarks: Rolling Sharpe Ratio ({window}-day)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Rolling Sharpe Ratio')
    ax2.set_xlabel('Date')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('reports', exist_ok=True)
    filename = f'reports/{strategy_name}_vs_benchmarks_{config.get("run_timestamp", "unknown")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_benchmark_comparison_heatmap(regout_list, sweep_tags, config) -> str:
    """Create a heatmap comparing strategies vs their most relevant benchmarks."""
    # Collect benchmark performance data for relevant benchmarks only
    comparison_data = []
    
    for regout_df, tag in zip(regout_list, sweep_tags):
        benchmark_cols = [col for col in regout_df.columns if col.startswith('benchmark_')]
        strategy_returns = regout_df['portfolio_ret']
        
        # Determine strategy type and get relevant benchmarks
        strategy_type = 'equal_weight'  # default
        if 'longshort' in tag.lower() or 'long_short' in tag.lower():
            strategy_type = 'long_short'
        elif 'confidenceweighted' in tag.lower() or 'confidence_weighted' in tag.lower() or 'confidence' in tag.lower():
            strategy_type = 'confidence_weighted'
        
        # Select most relevant benchmarks for this strategy type (match what BenchmarkManager creates)
        if strategy_type == 'long_short':
            relevant_benchmark_names = ['zero_return', 'random_long_short', 'equal_weight_targets']
        elif strategy_type == 'confidence_weighted':
            relevant_benchmark_names = ['risk_parity', 'spy_only', 'equal_weight_targets']
        else:  # equal_weight
            relevant_benchmark_names = ['spy_only', 'equal_weight_targets', 'vti_market', 'equal_weight_all']
        
        for benchmark_name in relevant_benchmark_names:
            benchmark_col = f'benchmark_{benchmark_name}'
            if benchmark_col in benchmark_cols:
                benchmark_returns = regout_df[benchmark_col]
                display_name = format_benchmark_name(f'benchmark_{benchmark_name}')
                
                # Calculate information ratio
                from multi_target_simulator import calculate_information_ratio
                info_ratio = calculate_information_ratio(strategy_returns, benchmark_returns)
                
                comparison_data.append({
                    'Strategy': tag.replace('mt_', '').replace('_', ' '),
                    'Benchmark': display_name,
                    'Information_Ratio': info_ratio
                })
    
    if not comparison_data:
        return None
    
    # Create pivot table
    comparison_df = pd.DataFrame(comparison_data)
    pivot_table = comparison_df.pivot(index='Strategy', columns='Benchmark', values='Information_Ratio')
    
    # Create heatmap with appropriate size
    _, ax = plt.subplots(figsize=(10, 8))
    
    # Use diverging colormap centered at 0
    im = ax.imshow(pivot_table.values, cmap='RdYlBu_r', aspect='auto', vmin=-2, vmax=2)
    
    # Set ticks and labels
    ax.set_xticks(range(len(pivot_table.columns)))
    ax.set_yticks(range(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot_table.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Information Ratio', rotation=270, labelpad=20)
    
    # Add text annotations
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            value = pivot_table.iloc[i, j]
            if not np.isnan(value):
                text_color = 'white' if abs(value) > 1.0 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                        color=text_color, fontweight='bold')
    
    ax.set_title('Strategy vs Benchmark Information Ratios', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Benchmarks')
    ax.set_ylabel('Strategies')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('reports', exist_ok=True)
    filename = f'reports/benchmark_comparison_heatmap_{config.get("run_timestamp", "unknown")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_simple_comparison_plot(regout_list, sweep_tags, config):
    """
    Create a simple, clean comparison plot without complex layout.
    """
    # Prepare data
    cumulative_returns_data = []
    
    for regout_df, tag in zip(regout_list, sweep_tags):
        if 'portfolio_ret' in regout_df.columns:
            returns = regout_df['portfolio_ret'].dropna()
            if len(returns) > 0:
                cumulative_returns = returns.cumsum()
                cumulative_returns_data.append({
                    'strategy': tag,
                    'cumulative_returns': cumulative_returns
                })
    
    if not cumulative_returns_data:
        return None
    
    # Create simple figure
    _, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, data in enumerate(cumulative_returns_data):
        color = colors[i % len(colors)]
        ax.plot(data['cumulative_returns'].index, data['cumulative_returns'].values,
               linewidth=2, color=color, label=data['strategy'].replace('mt_', '').replace('_', ' '))
    
    ax.set_title('Strategy Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_ylabel('Cumulative Return')
    ax.set_xlabel('Date')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Save
    os.makedirs('reports', exist_ok=True)
    filename = f'reports/simple_comparison_{config["run_timestamp"]}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename 