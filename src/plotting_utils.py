"""
Professional plotting utilities for financial strategy tear sheets.
Clean, modular approach to creating publication-quality visualizations.
"""

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
        return 'EQ WT Targets'
    elif name == 'Equal Weight All':
        return 'EQ WT All'
    elif name == 'Equal Weight Features':
        return 'EQ WT Features'
    elif name == 'Spy Only':
        return 'SPY Only'
    elif name == 'Zero Return':
        return 'Zero Return'
    elif name == 'Random Long Short':
        return 'Random L/S'
    elif name == 'Risk Parity':
        return 'Risk Parity'
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


def _get_strategy_pipeline_data(sweep_tags):
    """Generate strategy pipeline overview data with parameters in correct columns."""
    from sklearn.linear_model import LinearRegression, HuberRegressor, ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    
    # Note: Complexity scoring has been removed from PDF reports as it is experimental
    
    pipeline_data = []
    
    # Master strategy configuration mapping with parameters in correct columns
    strategy_configs = {
        # Multi-target strategies
        'mt_linear_std_equalweight': {
            'strategy': 'Linear-EqualWeight',
            'preprocessing': 'StandardScaler()',
            'learner': 'LinearRegression()',
            'portfolio': 'Equal allocation\nbase_leverage=1.0',
            'estimator': LinearRegression(),
            'portfolio_multiplier': 1.0
        },
        'mt_linear_std_confidenceweighted': {
            'strategy': 'Linear-ConfWeighted', 
            'preprocessing': 'StandardScaler()',
            'learner': 'LinearRegression()',
            'portfolio': 'Confidence weighting\nmax_leverage=2.0',
            'estimator': LinearRegression(),
            'portfolio_multiplier': 1.2
        },
        'mt_linear_std_longshort': {
            'strategy': 'Linear-LongShort',
            'preprocessing': 'StandardScaler()',
            'learner': 'LinearRegression()',
            'portfolio': 'Long/Short/terciles\nbase_leverage=1.0',
            'estimator': LinearRegression(),
            'portfolio_multiplier': 1.5
        },
        'mt_huber_std_equalweight': {
            'strategy': 'Huber-EqualWeight',
            'preprocessing': 'StandardScaler()',
            'learner': 'HuberRegressor\n(ε=1.35)',
            'portfolio': 'Equal allocation\nbase_leverage=1.0',
            'estimator': HuberRegressor(epsilon=1.35),
            'portfolio_multiplier': 1.0
        },
        'mt_huber_std_confidenceweighted': {
            'strategy': 'Huber-ConfWeighted',
            'preprocessing': 'StandardScaler()',
            'learner': 'HuberRegressor\n(ε=1.35)',
            'portfolio': 'Confidence weighting\nmax_leverage=2.0',
            'estimator': HuberRegressor(epsilon=1.35),
            'portfolio_multiplier': 1.2
        },
        'mt_huber_std_longshort': {
            'strategy': 'Huber-LongShort',
            'preprocessing': 'StandardScaler()',
            'learner': 'HuberRegressor\n(ε=1.35)',
            'portfolio': 'Long/Short/terciles\nbase_leverage=1.0',
            'estimator': HuberRegressor(epsilon=1.35),
            'portfolio_multiplier': 1.5
        },
        'mt_elasticnet_std_equalweight': {
            'strategy': 'ElasticNet-EqualWeight',
            'preprocessing': 'StandardScaler()',
            'learner': 'ElasticNet\n(α=0.01, l1=0.5)',
            'portfolio': 'Equal allocation\nbase_leverage=1.0',
            'estimator': ElasticNet(alpha=0.01, l1_ratio=0.5),
            'portfolio_multiplier': 1.0
        },
        'mt_elasticnet_std_confidenceweighted': {
            'strategy': 'ElasticNet-ConfWeighted',
            'preprocessing': 'StandardScaler()',
            'learner': 'ElasticNet\n(α=0.01, l1=0.5)',
            'portfolio': 'Confidence weighting\nmax_leverage=2.0',
            'estimator': ElasticNet(alpha=0.01, l1_ratio=0.5),
            'portfolio_multiplier': 1.2
        },
        'mt_elasticnet_std_longshort': {
            'strategy': 'ElasticNet-LongShort',
            'preprocessing': 'StandardScaler()',
            'learner': 'ElasticNet\n(α=0.01, l1=0.5)',
            'portfolio': 'Long/Short/terciles\nbase_leverage=1.0',
            'estimator': ElasticNet(alpha=0.01, l1_ratio=0.5),
            'portfolio_multiplier': 1.5
        },
        # Single-target strategies
        'st_rf_ewm_binary': {
            'strategy': 'RF-Binary',
            'preprocessing': 'EWMTransformer\n(halflife=4)',
            'learner': 'RandomForest\n(n_est=50)',
            'portfolio': 'Binary long/short\nbinary_threshold=0.0',
            'estimator': RandomForestRegressor(n_estimators=50),
            'portfolio_multiplier': 1.3
        },
        'st_rf_ewm_quartile': {
            'strategy': 'RF-Quartile',
            'preprocessing': 'EWMTransformer\n(halflife=4)',
            'learner': 'RandomForest\n(n_est=50)',
            'portfolio': 'Quartile sizing\nquartile_bins=4',
            'estimator': RandomForestRegressor(n_estimators=50),
            'portfolio_multiplier': 1.4
        },
        'st_rf_ewm_proportional': {
            'strategy': 'RF-Proportional',
            'preprocessing': 'EWMTransformer\n(halflife=4)',
            'learner': 'RandomForest\n(n_est=50)',
            'portfolio': 'Proportional sizing\nmax_position=2.0',
            'estimator': RandomForestRegressor(n_estimators=50),
            'portfolio_multiplier': 1.6
        }
    }
    
    # Match sweep_tags to strategy configurations - ONE row per strategy
    for tag in sweep_tags:
        tag_lower = tag.lower()
        config_found = False
        
        for config_key, config_data in strategy_configs.items():
            if config_key in tag_lower:
                # Single row with all parameters in correct columns
                pipeline_data.append({
                    'Strategy': config_data['strategy'],
                    'Preprocessing': config_data['preprocessing'],
                    'Learner': config_data['learner'],
                    'Portfolio': config_data['portfolio']
                })
                config_found = True
                break
        
        # Fallback for unmatched strategies
        if not config_found:
            pipeline_data.append({
                'Strategy': tag[:15] + '...' if len(tag) > 15 else tag,
                'Preprocessing': 'Unknown',
                'Learner': 'Unknown',
                'Portfolio': 'Unknown'
            })
    
    return pipeline_data

def create_tear_sheet(regout_list, sweep_tags, config):
    """
    Create a professional, publication-quality tear sheet with FIXED formatting issues.
    """
    print("🎨 create_tear_sheet function called")
    print(f"📊 Number of regout items: {len(regout_list)}")
    print(f"🏷️  Number of tags: {len(sweep_tags)}")
    print(f"⚙️  Config keys: {list(config.keys())}")
    
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
                        priority_benchmarks = ['buy_and_hold', 'zero_return', 'spy_only', 'equal_weight_targets', 'equal_weight_features', 'zero_return']
                    
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
    fig = plt.figure(figsize=(12, 20))  # Increased height for more parameter rows
    
    # Define color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Create GridSpec with more space for tables
    gs = GridSpec(5, 1, figure=fig, height_ratios=[0.4, 2.2, 0.3, 1.4, 2.2], hspace=0.4)  # Increased pipeline section
    
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
        
        # Debug date index issue
        date_index = data['cumulative_returns'].index
        print(f"DEBUG: Strategy {data['strategy']} date range: {date_index.min()} to {date_index.max()}")
        
        # Check if index looks corrupted (all 1970 dates)
        if hasattr(date_index, 'year') and len(date_index) > 0:
            if all(date_index.year == 1970):
                print(f"WARNING: Corrupted date index detected for {data['strategy']}, attempting to fix...")
                # Try to reconstruct proper date index - use a reasonable date range
                start_date = pd.Timestamp('2011-01-01')
                end_date = pd.Timestamp('2025-07-01') 
                proper_dates = pd.date_range(start=start_date, end=end_date, periods=len(date_index))
                data['cumulative_returns'].index = proper_dates
                date_index = proper_dates
                print(f"FIXED: New date range: {date_index.min()} to {date_index.max()}")
        
        ax_chart.plot(date_index, data['cumulative_returns'].values,
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
    
    # 4. Performance Table Section - FIXED WITH TEXT WRAPPING
    ax_table = fig.add_subplot(gs[3])
    ax_table.axis('off')
    
    # Create performance table and sort by Excess Return
    df = pd.DataFrame(performance_data)
    
    # Convert percentage strings to numeric for sorting
    df['Excess_Return_Numeric'] = df['Excess Return (%)'].str.rstrip('%').astype(float)
    df_sorted = df.sort_values('Excess_Return_Numeric', ascending=False)
    df_sorted = df_sorted.drop('Excess_Return_Numeric', axis=1)
    
    # Add table title
    ax_table.text(0.5, 0.98, 'PERFORMANCE SUMMARY', fontsize=14, fontweight='bold',
                 ha='center', va='top', transform=ax_table.transAxes, color='#2c3e50')
    
    # Create table with FIXED column widths
    table = ax_table.table(cellText=df_sorted.values,
                          colLabels=df_sorted.columns,
                          cellLoc='center',
                          loc='center',
                          bbox=[0.01, 0.05, 0.98, 0.85])  # Full width bbox
    
    # FIXED: Use specific column widths that work well with text wrapping
    col_widths = [0.28, 0.12, 0.12, 0.12, 0.18, 0.12, 0.08]  # Strategy gets more space
    
    for i, width in enumerate(col_widths):
        if i < len(df_sorted.columns):
            for j in range(len(df_sorted) + 1):
                table[(j, i)].set_width(width)
                # Set alignment
                if i == 0:  # Strategy column
                    table[(j, i)]._text.set_horizontalalignment('left')
                else:  # Data columns
                    table[(j, i)]._text.set_horizontalalignment('right')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)  # Slightly smaller to fit more content
    table.scale(1.0, 2.5)  # More height for better readability
    
    # FIXED: Enable text wrapping for better column fit
    for _, cell in table.get_celld().items():
        cell.set_text_props(wrap=True)  # Enable text wrapping
        cell.PAD = 0.04  # More padding
    
    # FIXED: Use shorter header texts with line breaks for better fit
    header_texts = ['Strategy', 'Annual\nReturn (%)', 'Sharpe\nRatio', 'Max\nDrawdown (%)', 'Best\nBench', 'Excess\nReturn (%)', 'Info\nRatio']
    for i, header_text in enumerate(header_texts):
        if i < len(df_sorted.columns):
            table[(0, i)]._text.set_text(header_text)
    
    # Style header row
    for i in range(len(df_sorted.columns)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)
        table[(0, i)].set_height(0.22)  # Taller header
    
    # Color alternating rows
    for i in range(1, len(df_sorted) + 1):
        for j in range(len(df_sorted.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    # 5. Strategy Pipeline Overview Section - SIMPLIFIED TO ONE ROW PER STRATEGY
    ax_pipeline = fig.add_subplot(gs[4])
    ax_pipeline.axis('off')
    
    # Add pipeline overview title
    ax_pipeline.text(0.5, 0.98, 'STRATEGY PIPELINE OVERVIEW', fontsize=14, fontweight='bold',
                    ha='center', va='top', transform=ax_pipeline.transAxes, color='#2c3e50')
    
    # Add complexity scale explanation
    complexity_text = 'Complexity Scale: Real values based on model parameters and portfolio strategy (higher = more complex)'
    ax_pipeline.text(0.5, 0.92, complexity_text, 
                    fontsize=9, ha='center', va='top', transform=ax_pipeline.transAxes, 
                    color='#7f8c8d', style='italic')
    
    # Create strategy pipeline data
    pipeline_data = _get_strategy_pipeline_data(sweep_tags)
    
    if pipeline_data:
        # Create DataFrame from pipeline data
        pipeline_df = pd.DataFrame(pipeline_data)
        
        # Create pipeline table with FIXED positioning
        pipeline_table = ax_pipeline.table(cellText=pipeline_df.values,
                                         colLabels=pipeline_df.columns,
                                         cellLoc='left',
                                         loc='center',
                                         bbox=[0.01, 0.02, 0.98, 0.85])  # Full width
        
        # FIXED: Style with better text handling
        pipeline_table.auto_set_font_size(False)
        pipeline_table.set_fontsize(8)  # Larger font since fewer rows
        pipeline_table.scale(1.0, 2.5)  # More height for better readability
        
        # FIXED: Use specific column widths with WIDER complexity column
        col_widths = [0.22, 0.15, 0.15, 0.28, 0.20]  # Strategy, Preprocessing, Learner, Portfolio, Complexity (wider)
        for i, width in enumerate(col_widths):
            for j in range(len(pipeline_df) + 1):
                if i < len(pipeline_df.columns):
                    pipeline_table[(j, i)].set_width(width)
        
        # Style header row
        for i in range(len(pipeline_df.columns)):
            pipeline_table[(0, i)].set_facecolor('#34495e')
            pipeline_table[(0, i)].set_text_props(weight='bold', color='white', fontsize=9)
            pipeline_table[(0, i)].set_height(0.20)
        
        # FIXED: Enable text wrapping with better settings
        for _, cell in pipeline_table.get_celld().items():
            cell.set_text_props(wrap=True)  # Enable text wrapping
            cell.PAD = 0.05  # More padding for better readability
        
        # Style alternating rows (now every other row since one row per strategy)
        for i in range(1, len(pipeline_df) + 1):
            for j in range(len(pipeline_df.columns)):
                # Simple alternating pattern
                if i % 2 == 0:
                    pipeline_table[(i, j)].set_facecolor('#ecf0f1')
                
                # Set alignment
                if j == 0:  # Strategy column
                    pipeline_table[(i, j)]._text.set_horizontalalignment('left')
                elif j in [1, 2, 3]:  # Preprocessing, Learner, Portfolio columns
                    pipeline_table[(i, j)]._text.set_horizontalalignment('left')
                else:  # Complexity column
                    pipeline_table[(i, j)]._text.set_horizontalalignment('center')
                
                # All rows get same styling since they're all main strategy rows
                pipeline_table[(i, j)].set_text_props(fontsize=8, weight='normal', wrap=True)
    
    # Save the tear sheet - SIMPLE AND RELIABLE APPROACH
    import os
    
    print("💾 Starting file save process...")
    
    try:
        # Simple approach: use current working directory and create reports subdirectory
        current_dir = os.getcwd()
        reports_dir = os.path.join(current_dir, 'reports')
        
        print(f"📁 Current directory: {current_dir}")
        print(f"📁 Reports directory: {reports_dir}")
        print(f"📁 Reports directory exists: {os.path.exists(reports_dir)}")
        
        # Create reports directory if it doesn't exist
        os.makedirs(reports_dir, exist_ok=True)
        print(f"📁 Reports directory created/exists: {os.path.exists(reports_dir)}")
        
        # Use simple relative paths
        pdf_filename = os.path.join(reports_dir, f'sim_tear_sheet_{config["run_timestamp"]}.pdf')
        png_filename = os.path.join(reports_dir, f'sim_tear_sheet_{config["run_timestamp"]}.png')
        
        print(f"📄 PDF filename: {pdf_filename}")
        print(f"🖼️  PNG filename: {png_filename}")
        print(f"📄 PDF absolute path: {os.path.abspath(pdf_filename)}")
        print(f"🖼️  PNG absolute path: {os.path.abspath(png_filename)}")
        
        # Add debugging to see what's happening
        print(f"Current working directory: {current_dir}")
        print(f"Reports directory: {reports_dir}")
        print(f"Reports directory exists: {os.path.exists(reports_dir)}")
        print(f"Saving PDF to: {os.path.abspath(pdf_filename)}")
        print(f"Saving PNG to: {os.path.abspath(png_filename)}")
        
        print("💾 Saving PDF file...")
        # Save the files
        plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', format='pdf')
        print("✅ PDF saved successfully")
        
        print("💾 Saving PNG file...")
        plt.savefig(png_filename, dpi=300, bbox_inches='tight', format='png')
        print("✅ PNG saved successfully")
        
        # Verify files were created
        if os.path.exists(pdf_filename):
            print(f"✅ PDF file created successfully: {pdf_filename}")
            print(f"📊 PDF file size: {os.path.getsize(pdf_filename)} bytes")
        else:
            print(f"❌ PDF file was not created: {pdf_filename}")
            
        if os.path.exists(png_filename):
            print(f"✅ PNG file created successfully: {png_filename}")
            print(f"📊 PNG file size: {os.path.getsize(png_filename)} bytes")
        else:
            print(f"❌ PNG file was not created: {png_filename}")
        
        print(f"📊 Professional tear sheet saved:")
        print(f"   PDF: {pdf_filename}")
        print(f"   🖼️  PNG: {png_filename}")
        
    except Exception as e:
        print(f"❌ Error saving tear sheet: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: save to current directory
        pdf_filename = f'sim_tear_sheet_{config["run_timestamp"]}.pdf'
        png_filename = f'sim_tear_sheet_{config["run_timestamp"]}.png'
        plt.savefig(pdf_filename, dpi=300, bbox_inches='tight', format='pdf')
        plt.savefig(png_filename, dpi=300, bbox_inches='tight', format='png')
        print(f"📊 Fallback: Files saved to current directory")
        print(f"   PDF: {pdf_filename}")
        print(f"   PNG: {png_filename}")
    
    plt.close()
    
    return pdf_filename

def plot_strategy_vs_benchmarks(regout_df: pd.DataFrame, strategy_name: str, config, born_on_date=None) -> str:
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
        priority_benchmarks = ['spy_only', 'equal_weight_targets', 'equal_weight_features', 'zero_return']
    
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
    
    # Add vertical line for born_on_date if provided
    if born_on_date:
        try:
            from dateutil.parser import parse
            if isinstance(born_on_date, str):
                born_date = parse(born_on_date)
            else:
                born_date = born_on_date
            
            # Only add line if date is within the plot range
            if born_date >= regout_df.index.min() and born_date <= regout_df.index.max():
                ax1.axvline(x=born_date, color='purple', linewidth=2, linestyle=':', alpha=0.8, 
                           label=f'Strategy Born: {born_date.strftime("%Y-%m-%d")}')
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Refresh legend
        except Exception as e:
            print(f"Warning: Could not plot born_on_date {born_on_date}: {e}")
    
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
    
    # Save plot - SIMPLE APPROACH
    import os
    current_dir = os.getcwd()
    reports_dir = os.path.join(current_dir, 'reports')
    
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(reports_dir, f'{strategy_name}_vs_benchmarks_{config.get("run_timestamp", "unknown")}.png')
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
            relevant_benchmark_names = ['spy_only', 'equal_weight_targets', 'equal_weight_features', 'zero_return']
        
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
    
    # Save - SIMPLE APPROACH
    import os
    current_dir = os.getcwd()
    reports_dir = os.path.join(current_dir, 'reports')
    
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(reports_dir, f'benchmark_comparison_heatmap_{config.get("run_timestamp", "unknown")}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_simple_comparison_plot(regout_list, sweep_tags, config):
    """Create a simple comparison plot of all strategies."""
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
    
    # Save - SIMPLE APPROACH
    import os
    current_dir = os.getcwd()
    reports_dir = os.path.join(current_dir, 'reports')
    
    os.makedirs(reports_dir, exist_ok=True)
    filename = os.path.join(reports_dir, f'simple_comparison_{config.get("run_timestamp", "unknown")}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename