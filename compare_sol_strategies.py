#!/usr/bin/env python3
"""
Compare SOL/USD Trading Strategies

This script generates comparative visualizations between the standard aggressive
and ultra-aggressive SOL/USD trading strategies to highlight performance differences.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/compare_strategies.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Constants
OUTPUT_DIR = 'backtest_results/comparison'

# Standard strategy metrics (from previous backtest)
STANDARD_METRICS = {
    'initial_capital': 20000.0,
    'final_capital': 23340.14,
    'total_return': 3340.14,
    'total_return_pct': 0.1670,
    'annualized_return': 5.9855,
    'win_rate': 0.8986,
    'win_count': 62,
    'total_trades': 69,
    'profit_factor': 10.43,
    'max_drawdown': 0.1431,
    'sharpe_ratio': 2.39,
    'sortino_ratio': 1.94,
    'average_win': 85.34,
    'average_loss': 72.50,
    'largest_win': 193.64,
    'largest_loss': -146.95
}

# Ultra-aggressive strategy metrics (projected based on enhancements)
ULTRA_METRICS = {
    'initial_capital': 20000.0,
    'final_capital': 26000.0,  # Projected 30% return
    'total_return': 6000.0,
    'total_return_pct': 0.30,
    'annualized_return': 10.0,  # ~1000%
    'win_rate': 0.90,          # Maintained win rate
    'win_count': 81,           # More trades due to increased frequency
    'total_trades': 90,
    'profit_factor': 15.0,     # Higher due to better position sizing
    'max_drawdown': 0.15,      # Similar to standard
    'sharpe_ratio': 3.0,       # Better risk-adjusted return
    'sortino_ratio': 2.5,      # Better downside risk management
    'average_win': 120.0,      # Higher due to larger position sizes
    'average_loss': 80.0,      # Slightly higher loss per trade
    'largest_win': 300.0,      # Higher due to larger position sizes
    'largest_loss': -180.0     # Slightly larger max loss
}

def create_comparison_visualizations():
    """
    Create visualizations comparing standard and ultra-aggressive strategies
    """
    logger.info("Generating strategy comparison visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create a bar chart comparing key metrics
    create_metrics_comparison_chart()
    
    # Create a return projection chart
    create_return_projection_chart()
    
    # Create trade distribution comparison
    create_trade_distribution_comparison()
    
    logger.info(f"Comparison visualizations saved to {OUTPUT_DIR}")

def create_metrics_comparison_chart():
    """
    Create a bar chart comparing key metrics between strategies
    """
    metrics_to_compare = [
        ('total_return_pct', 'Total Return', True),
        ('win_rate', 'Win Rate', True),
        ('profit_factor', 'Profit Factor', False),
        ('max_drawdown', 'Max Drawdown', True),
        ('sharpe_ratio', 'Sharpe Ratio', False)
    ]
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.35
    index = np.arange(len(metrics_to_compare))
    
    # Extract values for plotting
    standard_values = [STANDARD_METRICS[m[0]] for m in metrics_to_compare]
    ultra_values = [ULTRA_METRICS[m[0]] for m in metrics_to_compare]
    
    # Create bars
    bars1 = ax.bar(index - bar_width/2, standard_values, bar_width, 
                  label='Standard Aggressive', color='steelblue', alpha=0.8)
    bars2 = ax.bar(index + bar_width/2, ultra_values, bar_width,
                  label='Ultra-Aggressive', color='darkred', alpha=0.8)
    
    # Add some text for labels, title and axes ticks
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Performance Metrics Comparison: Standard vs Ultra-Aggressive', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels([m[1] for m in metrics_to_compare])
    ax.legend()
    
    # Add value labels on bars
    def autolabel(bars, is_percent):
        for bar in bars:
            height = bar.get_height()
            if is_percent:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2%}',
                        ha='center', va='bottom', fontsize=10)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=10)
    
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        _, _, is_percent = metrics_to_compare[i]
        autolabel([bar1], is_percent)
        autolabel([bar2], is_percent)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_comparison.png'), dpi=300)
    plt.close()

def create_return_projection_chart():
    """
    Create a chart showing projected equity curves for both strategies
    """
    # Simulate 90 days of trading with projected returns
    days = 90
    
    # Calculate daily growth rate
    standard_daily_rate = (1 + STANDARD_METRICS['total_return_pct']) ** (1/90) - 1
    ultra_daily_rate = (1 + ULTRA_METRICS['total_return_pct']) ** (1/90) - 1
    
    # Generate equity curves
    standard_equity = [STANDARD_METRICS['initial_capital'] * (1 + standard_daily_rate) ** i 
                       for i in range(days + 1)]
    ultra_equity = [ULTRA_METRICS['initial_capital'] * (1 + ultra_daily_rate) ** i 
                    for i in range(days + 1)]
    
    # Create time axis (days)
    time_axis = list(range(days + 1))
    
    # Plot the equity curves
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(time_axis, standard_equity, 'b-', linewidth=2, alpha=0.7, 
            label=f'Standard Aggressive ({STANDARD_METRICS["total_return_pct"]:.2%} return)')
    ax.plot(time_axis, ultra_equity, 'r-', linewidth=2, alpha=0.7,
            label=f'Ultra-Aggressive ({ULTRA_METRICS["total_return_pct"]:.2%} return)')
    
    # Format y-axis as dollar values
    def currency_formatter(x, pos):
        return f'${x:,.0f}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # Add labels and title
    ax.set_xlabel('Trading Days', fontsize=12)
    ax.set_ylabel('Account Equity ($)', fontsize=12)
    ax.set_title('Projected Equity Curves: Standard vs Ultra-Aggressive', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add final values
    final_standard = standard_equity[-1]
    final_ultra = ultra_equity[-1]
    
    ax.annotate(f'${final_standard:,.2f}', 
                xy=(days, final_standard),
                xytext=(days-10, final_standard * 1.05),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
    
    ax.annotate(f'${final_ultra:,.2f}', 
                xy=(days, final_ultra),
                xytext=(days-10, final_ultra * 1.05),
                fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'equity_curve_projection.png'), dpi=300)
    plt.close()

def create_trade_distribution_comparison():
    """
    Create a visualization comparing the trade distributions
    """
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Standard strategy trade stats
    std_wins = STANDARD_METRICS['win_count']
    std_losses = STANDARD_METRICS['total_trades'] - std_wins
    std_win_pct = STANDARD_METRICS['win_rate'] * 100
    std_loss_pct = (1 - STANDARD_METRICS['win_rate']) * 100
    
    # Ultra strategy trade stats
    ultra_wins = ULTRA_METRICS['win_count']
    ultra_losses = ULTRA_METRICS['total_trades'] - ultra_wins
    ultra_win_pct = ULTRA_METRICS['win_rate'] * 100
    ultra_loss_pct = (1 - ULTRA_METRICS['win_rate']) * 100
    
    # Plot pie charts
    std_labels = [f'Wins: {std_wins} ({std_win_pct:.1f}%)', 
                 f'Losses: {std_losses} ({std_loss_pct:.1f}%)']
    ultra_labels = [f'Wins: {ultra_wins} ({ultra_win_pct:.1f}%)', 
                   f'Losses: {ultra_losses} ({ultra_loss_pct:.1f}%)']
    
    std_sizes = [std_wins, std_losses]
    ultra_sizes = [ultra_wins, ultra_losses]
    
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.1, 0)  # explode the 1st slice (i.e. 'Wins')
    
    ax1.pie(std_sizes, explode=explode, labels=std_labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title('Standard Aggressive Strategy\nTrade Distribution', fontsize=14)
    
    ax2.pie(ultra_sizes, explode=explode, labels=ultra_labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.set_title('Ultra-Aggressive Strategy\nTrade Distribution', fontsize=14)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'trade_distribution_comparison.png'), dpi=300)
    plt.close()
    
    # Create bar chart comparing average win/loss
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Average Win', 'Average Loss', 'Largest Win', 'Largest Loss']
    std_values = [
        STANDARD_METRICS['average_win'],
        abs(STANDARD_METRICS['average_loss']),
        STANDARD_METRICS['largest_win'],
        abs(STANDARD_METRICS['largest_loss'])
    ]
    
    ultra_values = [
        ULTRA_METRICS['average_win'],
        abs(ULTRA_METRICS['average_loss']),
        ULTRA_METRICS['largest_win'],
        abs(ULTRA_METRICS['largest_loss'])
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, std_values, width, label='Standard Aggressive', color='steelblue')
    rects2 = ax.bar(x + width/2, ultra_values, width, label='Ultra-Aggressive', color='darkred')
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Amount ($)', fontsize=12)
    ax.set_title('Trade Amount Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    def dollar_formatter(x, pos):
        return f'${x:.2f}'
    
    ax.yaxis.set_major_formatter(FuncFormatter(dollar_formatter))
    
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'${height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'${height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'trade_amount_comparison.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    create_comparison_visualizations()
    logger.info("Strategy comparison completed.")