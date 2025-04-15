#!/usr/bin/env python3

"""
Display Pair Metrics

This script displays the current model accuracy and PnL metrics for each trading pair.
It extracts information from:
1. ML config for accuracy
2. Backtest results for historical PnL
3. Trading data for realized PnL

Usage:
    python display_pair_metrics.py [--pairs PAIRS]
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
BACKTEST_RESULTS_DIR = "backtest_results"
TRADING_DATA_DIR = "data"
PORTFOLIO_HISTORY_FILE = f"{TRADING_DATA_DIR}/sandbox_portfolio_history.json"
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Display Pair Metrics")
    parser.add_argument("--pairs", type=str, default=",".join(DEFAULT_PAIRS),
                        help="Comma-separated list of trading pairs")
    parser.add_argument("--full", action="store_true",
                        help="Display full metrics (including all available details)")
    return parser.parse_args()

def load_json_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default

def get_ml_metrics(pairs) -> Dict[str, Dict[str, Any]]:
    """Get ML metrics for each pair from the ML config"""
    metrics = {}
    
    # Load ML config
    ml_config = load_json_file(ML_CONFIG_FILE, {"pairs": {}})
    
    for pair in pairs:
        pair_config = ml_config.get("pairs", {}).get(pair, {})
        
        metrics[pair] = {
            "accuracy": pair_config.get("accuracy", 0.0),
            "backtest_return": pair_config.get("backtest_return", 0.0),
            "win_rate": pair_config.get("win_rate", 0.0),
            "sharpe_ratio": pair_config.get("sharpe_ratio", 0.0),
            "max_drawdown": pair_config.get("max_drawdown", 0.0),
            "confidence_threshold": pair_config.get("confidence_threshold", 0.65),
            "base_leverage": pair_config.get("base_leverage", 20.0),
            "max_leverage": pair_config.get("max_leverage", 125.0),
            "risk_percentage": pair_config.get("risk_percentage", 0.2)
        }
    
    return metrics

def get_backtest_metrics(pairs) -> Dict[str, Dict[str, Any]]:
    """Get backtest metrics for each pair"""
    metrics = {}
    
    for pair in pairs:
        pair_filename = pair.replace('/', '_')
        backtest_file = f"{BACKTEST_RESULTS_DIR}/{pair_filename}_backtest.json"
        backtest_results = load_json_file(backtest_file, {})
        
        metrics[pair] = {
            "total_return_pct": backtest_results.get("total_return_pct", 0.0),
            "max_drawdown": backtest_results.get("max_drawdown", 0.0),
            "sharpe_ratio": backtest_results.get("sharpe_ratio", 0.0),
            "win_rate": backtest_results.get("win_rate", 0.0),
            "total_trades": backtest_results.get("total_trades", 0),
            "winning_trades": backtest_results.get("winning_trades", 0)
        }
    
    return metrics

def get_trading_metrics(pairs) -> Dict[str, Dict[str, Any]]:
    """Get trading metrics for each pair from the portfolio history"""
    metrics = {}
    
    # Initialize metrics for each pair
    for pair in pairs:
        metrics[pair] = {
            "realized_pnl": 0.0,
            "realized_pnl_pct": 0.0,
            "trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0
        }
    
    # Load portfolio history
    portfolio_history = load_json_file(PORTFOLIO_HISTORY_FILE, [])
    
    # Extract completed trades from history
    completed_trades = []
    for entry in portfolio_history:
        trades = entry.get("completed_trades", [])
        for trade in trades:
            pair = trade.get("pair")
            if pair in pairs:
                completed_trades.append(trade)
    
    # Aggregate metrics by pair
    pair_trades = {pair: [] for pair in pairs}
    for trade in completed_trades:
        pair = trade.get("pair")
        if pair in pairs:
            pair_trades[pair].append(trade)
    
    # Calculate metrics for each pair
    for pair, trades in pair_trades.items():
        if not trades:
            continue
        
        total_pnl = sum(trade.get("pnl_amount", 0.0) for trade in trades)
        winning_trades = [t for t in trades if t.get("pnl_amount", 0.0) > 0]
        losing_trades = [t for t in trades if t.get("pnl_amount", 0.0) <= 0]
        
        # Calculate average profit and loss
        avg_profit = sum(t.get("pnl_amount", 0.0) for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t.get("pnl_amount", 0.0) for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
        
        # Calculate win rate
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
        
        # Calculate PnL percentage based on initial investment
        # Assuming a 20% risk per trade and an initial balance of $10,000
        initial_investment = 10000.0
        realized_pnl_pct = total_pnl / initial_investment
        
        metrics[pair] = {
            "realized_pnl": total_pnl,
            "realized_pnl_pct": realized_pnl_pct,
            "trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss
        }
    
    return metrics

def format_percentage(value, decimals=2):
    """Format a value as a percentage with +/- sign"""
    return f"{'+' if value > 0 else ''}{value * 100:.{decimals}f}%"

def format_dollar(value, decimals=2):
    """Format a value as a dollar amount with +/- sign"""
    return f"{'+' if value > 0 else ''}${value:.{decimals}f}"

def display_metrics(pairs, ml_metrics, backtest_metrics, trading_metrics, full=False):
    """Display metrics in a formatted table"""
    print("\n" + "=" * 100)
    print(f"{'TRADING PAIR':<12} | {'MODEL ACCURACY':<14} | {'BACKTEST PNL':<14} | {'REALIZED PNL':<14} | {'WIN RATE':<10} | {'SHARPE':<8}")
    print("-" * 100)
    
    for pair in pairs:
        ml = ml_metrics.get(pair, {})
        bt = backtest_metrics.get(pair, {})
        tr = trading_metrics.get(pair, {})
        
        accuracy = ml.get("accuracy", 0.0)
        backtest_pnl = bt.get("total_return_pct", 0.0)
        realized_pnl = tr.get("realized_pnl_pct", 0.0)
        win_rate = tr.get("win_rate", 0.0) or ml.get("win_rate", 0.0)
        sharpe = ml.get("sharpe_ratio", 0.0)
        
        print(f"{pair:<12} | {accuracy * 100:>12.2f}% | {format_percentage(backtest_pnl/100, 2):>12} | {format_percentage(realized_pnl, 2):>12} | {win_rate * 100:>8.2f}% | {sharpe:>6.2f}")
    
    print("=" * 100)
    
    # Display additional metrics if full mode is enabled
    if full:
        print("\nDetailed Metrics:")
        print("=" * 100)
        
        for pair in pairs:
            print(f"\n{pair} Detailed Metrics:")
            print("-" * 50)
            
            ml = ml_metrics.get(pair, {})
            bt = backtest_metrics.get(pair, {})
            tr = trading_metrics.get(pair, {})
            
            # ML Model Metrics
            print("ML Model Configuration:")
            print(f"  Accuracy:             {ml.get('accuracy', 0.0) * 100:.2f}%")
            print(f"  Confidence Threshold: {ml.get('confidence_threshold', 0.65):.2f}")
            print(f"  Base Leverage:        {ml.get('base_leverage', 20.0):.1f}x")
            print(f"  Max Leverage:         {ml.get('max_leverage', 125.0):.1f}x")
            print(f"  Risk Percentage:      {ml.get('risk_percentage', 0.2) * 100:.1f}%")
            
            # Backtest Metrics
            print("\nBacktest Results:")
            print(f"  Return:               {format_percentage(bt.get('total_return_pct', 0.0)/100)}")
            print(f"  Max Drawdown:         {format_percentage(bt.get('max_drawdown', 0.0))}")
            print(f"  Sharpe Ratio:         {bt.get('sharpe_ratio', 0.0):.2f}")
            print(f"  Win Rate:             {bt.get('win_rate', 0.0) * 100:.2f}%")
            print(f"  Total Trades:         {bt.get('total_trades', 0)}")
            print(f"  Winning Trades:       {bt.get('winning_trades', 0)}")
            
            # Trading Metrics
            print("\nActual Trading Performance:")
            print(f"  Realized PnL:         {format_dollar(tr.get('realized_pnl', 0.0))}")
            print(f"  Realized PnL %:       {format_percentage(tr.get('realized_pnl_pct', 0.0))}")
            print(f"  Trades:               {tr.get('trades', 0)}")
            print(f"  Winning Trades:       {tr.get('winning_trades', 0)}")
            print(f"  Losing Trades:        {tr.get('losing_trades', 0)}")
            print(f"  Win Rate:             {tr.get('win_rate', 0.0) * 100:.2f}%")
            print(f"  Avg Profit per Trade: {format_dollar(tr.get('avg_profit', 0.0))}")
            print(f"  Avg Loss per Trade:   {format_dollar(tr.get('avg_loss', 0.0))}")
        
        print("\n" + "=" * 100)

def main():
    """Main function"""
    args = parse_arguments()
    pairs = args.pairs.split(",")
    
    print(f"\nRetrieving metrics for {len(pairs)} pairs...\n")
    
    # Get metrics
    ml_metrics = get_ml_metrics(pairs)
    backtest_metrics = get_backtest_metrics(pairs)
    trading_metrics = get_trading_metrics(pairs)
    
    # Display metrics
    display_metrics(pairs, ml_metrics, backtest_metrics, trading_metrics, args.full)
    
    # Summary
    print("\nSummary:")
    avg_accuracy = sum(m.get("accuracy", 0.0) for m in ml_metrics.values()) / len(pairs) if pairs else 0.0
    avg_backtest_pnl = sum(m.get("total_return_pct", 0.0) for m in backtest_metrics.values()) / len(pairs) if pairs else 0.0
    total_realized_pnl = sum(m.get("realized_pnl", 0.0) for m in trading_metrics.values())
    
    print(f"Average Model Accuracy:      {avg_accuracy * 100:.2f}%")
    print(f"Average Backtest Return:     {avg_backtest_pnl:.2f}%")
    print(f"Total Realized PnL:          {format_dollar(total_realized_pnl)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())