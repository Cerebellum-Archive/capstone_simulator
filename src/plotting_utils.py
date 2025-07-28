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
        if 'portfolio_ret' in regout_df.columns:
            returns = regout_df['portfolio_ret'].dropna()
            if len(returns) > 0:
                cumulative_returns = returns.cumsum()
                cumulative_returns_data.append({
                    'strategy': tag,
                    'cumulative_returns': cumulative_returns,
                    'returns': returns
                })
                
                # Calculate performance metrics
                annual_return = 252 * returns.mean()
                annual_vol = np.sqrt(252) * returns.std()
                sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
                max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
                
                performance_data.append({
                    'Strategy': tag.replace('mt_', '').replace('_', ' '),
                    'Annual Return (%)': f"{annual_return:.2%}",
                    'Annual Vol (%)': f"{annual_vol:.2%}",
                    'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                    'Max Drawdown (%)': f"{max_drawdown:.2%}"
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
    
    # Create performance table
    df = pd.DataFrame(performance_data)
    df_sorted = df.sort_values('Sharpe Ratio', 
                               key=lambda x: pd.to_numeric(x.str.rstrip('%'), errors='coerce'), 
                               ascending=False)
    
    # Add table title with better spacing
    ax_table.text(0.5, 0.98, 'PERFORMANCE SUMMARY', fontsize=14, fontweight='bold',
                 ha='center', va='top', transform=ax_table.transAxes, color='#2c3e50')
    
    # Create table with professional styling and proper column widths
    table = ax_table.table(cellText=df_sorted.values,
                          colLabels=df_sorted.columns,
                          cellLoc='center',
                          loc='center',
                          bbox=[0.05, 0.08, 0.9, 0.8])
    
    # Set custom column widths and alignment
    col_widths = [0.35, 0.15, 0.15, 0.15, 0.20]  # Strategy gets 35% width
    for i, width in enumerate(col_widths):
        for j in range(len(df_sorted) + 1):  # +1 for header row
            table[(j, i)].set_width(width)
            # Set alignment: Strategy column left-aligned, others right-aligned
            if i == 0:  # Strategy column
                table[(j, i)]._text.set_horizontalalignment('left')
            else:  # Data columns
                table[(j, i)]._text.set_horizontalalignment('right')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.2)  # Increased row height for better spacing
    
    # Add text wrapping for column headers
    header_texts = ['Strategy', 'Annual\nReturn (%)', 'Annual\nVol (%)', 'Sharpe\nRatio', 'Max\nDrawdown (%)']
    for i, header_text in enumerate(header_texts):
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