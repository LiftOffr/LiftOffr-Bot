#!/usr/bin/env python3
"""
Generate Summary of Training Results

This script generates a summary of the training results for 9 cryptocurrency pairs,
showing the PnL, win rate, Sharpe ratio, and profit factor.

Usage:
    python training_summary.py
"""

import json

# Constants
PAIRS = [
    'BTC/USD', 'ETH/USD', 'SOL/USD', 'ADA/USD', 'DOT/USD',
    'LINK/USD', 'AVAX/USD', 'MATIC/USD', 'UNI/USD'
]

def load_metrics():
    """Load performance metrics from JSON file"""
    try:
        with open('training_results/performance_metrics.json', 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        print(f"Error loading metrics: {e}")
        return {}

def format_row(cols, col_widths):
    """Format a row with fixed column widths"""
    return "| " + " | ".join(str(col).ljust(width) for col, width in zip(cols, col_widths)) + " |"

def print_separator(col_widths):
    """Print a separator line"""
    parts = ["-" * (width + 2) for width in col_widths]
    return "+" + "+".join(parts) + "+"

def generate_summary(metrics):
    """Generate summary of training results"""
    if not metrics:
        print("No metrics available")
        return
    
    headers = ["Pair", "PnL", "Win Rate", "Sharpe Ratio", "Profit Factor", "Total Trades"]
    
    # Prepare data rows
    data = []
    for pair in PAIRS:
        if pair in metrics:
            pair_metrics = metrics[pair]
            data.append([
                pair,
                f"${pair_metrics['pnl']:.2f}",
                f"{pair_metrics['win_rate'] * 100:.1f}%",
                f"{pair_metrics['sharpe_ratio']:.2f}",
                f"{pair_metrics['profit_factor']:.2f}",
                f"{pair_metrics['total_trades']}"
            ])
    
    # Calculate average metrics
    avg_pnl = sum(metrics[pair]['pnl'] for pair in PAIRS if pair in metrics) / len(data)
    avg_win_rate = sum(metrics[pair]['win_rate'] for pair in PAIRS if pair in metrics) / len(data)
    avg_sharpe = sum(metrics[pair]['sharpe_ratio'] for pair in PAIRS if pair in metrics) / len(data)
    avg_profit_factor = sum(metrics[pair]['profit_factor'] for pair in PAIRS if pair in metrics) / len(data)
    avg_trades = sum(metrics[pair]['total_trades'] for pair in PAIRS if pair in metrics) / len(data)
    
    # Add average row
    data.append([
        "AVERAGE",
        f"${avg_pnl:.2f}",
        f"{avg_win_rate * 100:.1f}%",
        f"{avg_sharpe:.2f}",
        f"{avg_profit_factor:.2f}",
        f"{avg_trades:.0f}"
    ])
    
    # Calculate column widths
    col_widths = []
    for i in range(len(headers)):
        col_width = max(len(str(row[i])) for row in data)
        col_width = max(col_width, len(headers[i]))
        col_widths.append(col_width)
    
    # Print the table
    print("\n=== Training Results for 9 Cryptocurrency Pairs ===\n")
    print(print_separator(col_widths))
    print(format_row(headers, col_widths))
    print(print_separator(col_widths))
    
    for i, row in enumerate(data):
        print(format_row(row, col_widths))
        if i == len(data) - 2:  # Print a separator before the average row
            print(print_separator(col_widths))
    
    print(print_separator(col_widths))
    
    # Calculate distribution by win rate tier
    win_rates = [metrics[pair]['win_rate'] for pair in PAIRS if pair in metrics]
    tiers = {
        "90%+": sum(1 for wr in win_rates if wr >= 0.9),
        "80-89%": sum(1 for wr in win_rates if 0.8 <= wr < 0.9),
        "70-79%": sum(1 for wr in win_rates if 0.7 <= wr < 0.8),
        "<70%": sum(1 for wr in win_rates if wr < 0.7)
    }
    
    print("\nWin Rate Distribution:")
    for tier, count in tiers.items():
        print(f"  {tier}: {count} pairs")
    
    # Sort pairs by performance metrics
    print("\nRanking by Win Rate:")
    sorted_by_win_rate = sorted(
        [(pair, metrics[pair]['win_rate']) for pair in PAIRS if pair in metrics],
        key=lambda x: x[1],
        reverse=True
    )
    for i, (pair, win_rate) in enumerate(sorted_by_win_rate, 1):
        print(f"  {i}. {pair}: {win_rate * 100:.1f}%")
    
    print("\nRanking by Profit Factor:")
    sorted_by_pf = sorted(
        [(pair, metrics[pair]['profit_factor']) for pair in PAIRS if pair in metrics],
        key=lambda x: x[1],
        reverse=True
    )
    for i, (pair, pf) in enumerate(sorted_by_pf, 1):
        print(f"  {i}. {pair}: {pf:.2f}")

def main():
    """Main function"""
    metrics = load_metrics()
    generate_summary(metrics)

if __name__ == "__main__":
    main()