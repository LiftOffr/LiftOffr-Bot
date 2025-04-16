#!/usr/bin/env python3
"""
Evaluate Improved Model Performance

This script analyzes the performance of the improved ML models by:
1. Evaluating model accuracy and trading metrics
2. Analyzing portfolio performance
3. Comparing performance before and after improvements
4. Displaying key metrics in a readable format

Usage:
    python evaluate_improved_performance.py [--pairs ALL|BTC/USD,ETH/USD,...]
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
DATA_DIR = "data"
SANDBOX_PORTFOLIO_PATH = f"{DATA_DIR}/sandbox_portfolio.json"
SANDBOX_POSITIONS_PATH = f"{DATA_DIR}/sandbox_positions.json"
SANDBOX_TRADES_PATH = f"{DATA_DIR}/sandbox_trades.json"
SANDBOX_PORTFOLIO_HISTORY_PATH = f"{DATA_DIR}/sandbox_portfolio_history.json"
RESULTS_DIR = "analysis_results"
ALL_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Create required directories
for directory in [CONFIG_DIR, DATA_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate improved model performance")
    parser.add_argument("--pairs", type=str, default="ALL",
                        help="Trading pairs to evaluate, comma-separated (default: ALL)")
    parser.add_argument("--save_plots", action="store_true", default=True,
                        help="Save plots to file (default: True)")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days to analyze (default: 7)")
    return parser.parse_args()

def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Load JSON data from file"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"File not found: {file_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def load_ml_config() -> Optional[Dict[str, Any]]:
    """Load ML configuration"""
    return load_json_file(ML_CONFIG_PATH)

def load_portfolio_data() -> Tuple[Optional[Dict], Optional[Dict], Optional[List]]:
    """Load portfolio, positions, and trades data"""
    portfolio = load_json_file(SANDBOX_PORTFOLIO_PATH)
    positions = load_json_file(SANDBOX_POSITIONS_PATH)
    trades = load_json_file(SANDBOX_TRADES_PATH)
    
    return portfolio, positions, trades

def load_portfolio_history() -> Optional[List[Dict]]:
    """Load portfolio history data"""
    return load_json_file(SANDBOX_PORTFOLIO_HISTORY_PATH)

def calculate_portfolio_metrics(portfolio: Dict, positions: Dict, trades: List) -> Dict[str, Any]:
    """Calculate portfolio performance metrics"""
    metrics = {}
    
    # Basic portfolio metrics
    metrics["total_balance"] = portfolio.get("total_balance", 0)
    metrics["available_balance"] = portfolio.get("available_balance", 0)
    metrics["unrealized_pnl"] = portfolio.get("unrealized_pnl", 0)
    metrics["realized_pnl"] = portfolio.get("realized_pnl", 0)
    metrics["portfolio_risk"] = portfolio.get("portfolio_risk", 0)
    
    # Count open positions
    metrics["open_positions"] = len(positions)
    
    # Calculate total trades
    metrics["total_trades"] = len(trades)
    
    # Calculate winning trades
    winning_trades = [t for t in trades if t.get("realized_profit", 0) > 0]
    metrics["winning_trades"] = len(winning_trades)
    
    # Calculate winning percentage
    metrics["win_rate"] = metrics["winning_trades"] / max(1, metrics["total_trades"])
    
    # Calculate average profit
    if metrics["total_trades"] > 0:
        profits = [t.get("realized_profit", 0) for t in trades]
        metrics["avg_profit"] = sum(profits) / len(profits)
        
        # Calculate profit factor
        winning_profits = sum(p for p in profits if p > 0)
        losing_profits = abs(sum(p for p in profits if p <= 0))
        metrics["profit_factor"] = winning_profits / max(0.01, losing_profits)
    else:
        metrics["avg_profit"] = 0
        metrics["profit_factor"] = 0
    
    # Calculate ROI
    starting_capital = 20000.0  # Default starting capital
    current_equity = metrics["total_balance"]
    metrics["roi"] = (current_equity - starting_capital) / starting_capital
    
    return metrics

def analyze_trades_by_pair(trades: List, pairs: List[str]) -> Dict[str, Dict]:
    """Analyze trades grouped by trading pair"""
    pair_analysis = {}
    
    # Initialize analysis for each pair
    for pair in pairs:
        pair_analysis[pair] = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_profit": 0,
            "avg_profit": 0,
            "max_profit": 0,
            "max_loss": 0
        }
    
    # Group trades by pair
    for trade in trades:
        pair = trade.get("pair", "Unknown")
        profit = trade.get("realized_profit", 0)
        
        # Skip if pair not in list
        if pair not in pair_analysis:
            continue
        
        # Update metrics
        pair_analysis[pair]["total_trades"] += 1
        
        if profit > 0:
            pair_analysis[pair]["winning_trades"] += 1
        else:
            pair_analysis[pair]["losing_trades"] += 1
        
        pair_analysis[pair]["total_profit"] += profit
        pair_analysis[pair]["max_profit"] = max(pair_analysis[pair]["max_profit"], profit)
        pair_analysis[pair]["max_loss"] = min(pair_analysis[pair]["max_loss"], profit)
    
    # Calculate derived metrics
    for pair, analysis in pair_analysis.items():
        if analysis["total_trades"] > 0:
            analysis["win_rate"] = analysis["winning_trades"] / analysis["total_trades"]
            analysis["avg_profit"] = analysis["total_profit"] / analysis["total_trades"]
        else:
            analysis["win_rate"] = 0
            analysis["avg_profit"] = 0
    
    return pair_analysis

def calculate_trading_performance(trades: List, days: int = 7) -> Dict[str, Any]:
    """Calculate trading performance metrics over the specified period"""
    performance = {}
    
    # Get current time
    now = datetime.now()
    start_time = now - timedelta(days=days)
    start_timestamp = start_time.timestamp()
    
    # Filter trades for the period
    period_trades = [t for t in trades if t.get("close_timestamp", 0) >= start_timestamp]
    
    # Calculate performance
    performance["period_days"] = days
    performance["period_trades"] = len(period_trades)
    
    if performance["period_trades"] > 0:
        # Calculate profits
        profits = [t.get("realized_profit", 0) for t in period_trades]
        performance["period_profit"] = sum(profits)
        
        # Calculate winning trades
        winning_trades = [t for t in period_trades if t.get("realized_profit", 0) > 0]
        performance["period_winning_trades"] = len(winning_trades)
        
        # Calculate win rate
        performance["period_win_rate"] = performance["period_winning_trades"] / performance["period_trades"]
        
        # Calculate average profit
        performance["period_avg_profit"] = performance["period_profit"] / performance["period_trades"]
        
        # Calculate max drawdown
        cumulative_profits = np.cumsum(profits)
        max_drawdown = 0
        peak = 0
        
        for profit in cumulative_profits:
            if profit > peak:
                peak = profit
            drawdown = peak - profit
            max_drawdown = max(max_drawdown, drawdown)
        
        performance["period_max_drawdown"] = max_drawdown
    else:
        performance["period_profit"] = 0
        performance["period_winning_trades"] = 0
        performance["period_win_rate"] = 0
        performance["period_avg_profit"] = 0
        performance["period_max_drawdown"] = 0
    
    return performance

def analyze_portfolio_history(history: List, days: int = 7) -> Dict[str, Any]:
    """Analyze portfolio history over time"""
    history_analysis = {}
    
    if not history:
        logger.warning("No portfolio history data available")
        return history_analysis
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    
    # Convert timestamp to datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    
    # Filter for specified period
    now = datetime.now()
    start_time = now - timedelta(days=days)
    df_period = df[df["datetime"] >= start_time]
    
    # Calculate metrics if data is available
    if not df_period.empty:
        # Calculate returns
        df_period["return"] = df_period["total_balance"].pct_change()
        
        # Calculate cumulative return
        start_balance = df_period["total_balance"].iloc[0]
        end_balance = df_period["total_balance"].iloc[-1]
        history_analysis["cumulative_return"] = (end_balance / start_balance) - 1
        
        # Calculate volatility
        history_analysis["volatility"] = df_period["return"].std() * np.sqrt(252)  # Annualized
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        avg_daily_return = df_period["return"].mean()
        daily_volatility = df_period["return"].std()
        if daily_volatility > 0:
            history_analysis["sharpe_ratio"] = (avg_daily_return / daily_volatility) * np.sqrt(252)
        else:
            history_analysis["sharpe_ratio"] = 0
        
        # Calculate max drawdown
        df_period["cumulative_return"] = (1 + df_period["return"]).cumprod() - 1
        df_period["cumulative_wealth"] = (1 + df_period["cumulative_return"]) * start_balance
        df_period["previous_peak"] = df_period["cumulative_wealth"].cummax()
        df_period["drawdown"] = (df_period["cumulative_wealth"] - df_period["previous_peak"]) / df_period["previous_peak"]
        history_analysis["max_drawdown"] = df_period["drawdown"].min()
        
        # Calculate Calmar ratio
        if abs(history_analysis["max_drawdown"]) > 0:
            history_analysis["calmar_ratio"] = history_analysis["cumulative_return"] / abs(history_analysis["max_drawdown"])
        else:
            history_analysis["calmar_ratio"] = 0
    else:
        logger.warning(f"No portfolio history data for the last {days} days")
        history_analysis["cumulative_return"] = 0
        history_analysis["volatility"] = 0
        history_analysis["sharpe_ratio"] = 0
        history_analysis["max_drawdown"] = 0
        history_analysis["calmar_ratio"] = 0
    
    return history_analysis, df_period if not df_period.empty else None

def plot_portfolio_history(df_period, save_path=None):
    """Plot portfolio history over time"""
    if df_period is None or df_period.empty:
        logger.warning("No portfolio history data to plot")
        return
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot total balance
    plt.subplot(2, 1, 1)
    plt.plot(df_period["datetime"], df_period["total_balance"], "b-", label="Total Balance")
    plt.title("Portfolio Balance Over Time")
    plt.ylabel("USD")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot drawdown
    plt.subplot(2, 1, 2)
    plt.plot(df_period["datetime"], df_period["drawdown"], "r-", label="Drawdown")
    plt.title("Portfolio Drawdown")
    plt.ylabel("Drawdown %")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Portfolio history plot saved to {save_path}")
    else:
        plt.show()

def plot_pair_performance(pair_analysis, save_path=None):
    """Plot performance metrics by trading pair"""
    if not pair_analysis:
        logger.warning("No pair analysis data to plot")
        return
    
    # Extract data
    pairs = list(pair_analysis.keys())
    win_rates = [analysis["win_rate"] * 100 for analysis in pair_analysis.values()]
    avg_profits = [analysis["avg_profit"] for analysis in pair_analysis.values()]
    total_profits = [analysis["total_profit"] for analysis in pair_analysis.values()]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot win rates
    plt.subplot(3, 1, 1)
    bars = plt.bar(pairs, win_rates)
    plt.title("Win Rate by Trading Pair")
    plt.ylabel("Win Rate (%)")
    plt.ylim(0, 100)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f"{height:.1f}%", ha="center", va="bottom")
    
    # Plot average profit
    plt.subplot(3, 1, 2)
    bars = plt.bar(pairs, avg_profits)
    plt.title("Average Profit by Trading Pair")
    plt.ylabel("USD")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f"${height:.2f}", ha="center", va="bottom")
    
    # Plot total profit
    plt.subplot(3, 1, 3)
    bars = plt.bar(pairs, total_profits)
    plt.title("Total Profit by Trading Pair")
    plt.ylabel("USD")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f"${height:.2f}", ha="center", va="bottom")
    
    plt.tight_layout()
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Pair performance plot saved to {save_path}")
    else:
        plt.show()

def print_performance_report(
    portfolio_metrics: Dict[str, Any],
    pair_analysis: Dict[str, Dict],
    trading_performance: Dict[str, Any],
    history_analysis: Dict[str, Any]
):
    """Print comprehensive performance report"""
    print("\n" + "=" * 80)
    print("IMPROVED MODEL PERFORMANCE REPORT")
    print("=" * 80)
    
    # Current Portfolio Status
    print("\nCURRENT PORTFOLIO STATUS:")
    print(f"Total Balance: ${portfolio_metrics['total_balance']:,.2f}")
    print(f"Available Balance: ${portfolio_metrics['available_balance']:,.2f}")
    print(f"Unrealized P&L: ${portfolio_metrics['unrealized_pnl']:,.2f}")
    print(f"Realized P&L: ${portfolio_metrics['realized_pnl']:,.2f}")
    print(f"Portfolio Risk: {portfolio_metrics['portfolio_risk']:.2%}")
    print(f"Open Positions: {portfolio_metrics['open_positions']}")
    
    # Overall Trading Performance
    print("\nOVERALL TRADING PERFORMANCE:")
    print(f"Total Trades: {portfolio_metrics['total_trades']}")
    print(f"Winning Trades: {portfolio_metrics['winning_trades']}")
    print(f"Win Rate: {portfolio_metrics['win_rate']:.2%}")
    print(f"Average Profit: ${portfolio_metrics['avg_profit']:,.2f}")
    print(f"Profit Factor: {portfolio_metrics['profit_factor']:.2f}")
    print(f"Return on Investment: {portfolio_metrics['roi']:.2%}")
    
    # Recent Trading Performance
    print(f"\nRECENT TRADING PERFORMANCE ({trading_performance['period_days']} DAYS):")
    print(f"Period Trades: {trading_performance['period_trades']}")
    print(f"Period Winning Trades: {trading_performance['period_winning_trades']}")
    print(f"Period Win Rate: {trading_performance['period_win_rate']:.2%}")
    print(f"Period Profit: ${trading_performance['period_profit']:,.2f}")
    print(f"Period Average Profit: ${trading_performance['period_avg_profit']:,.2f}")
    print(f"Period Max Drawdown: ${trading_performance['period_max_drawdown']:,.2f}")
    
    # Portfolio History Analysis
    print("\nPORTFOLIO HISTORY ANALYSIS:")
    print(f"Cumulative Return: {history_analysis.get('cumulative_return', 0):.2%}")
    print(f"Volatility (Annualized): {history_analysis.get('volatility', 0):.2%}")
    print(f"Sharpe Ratio: {history_analysis.get('sharpe_ratio', 0):.2f}")
    print(f"Maximum Drawdown: {history_analysis.get('max_drawdown', 0):.2%}")
    print(f"Calmar Ratio: {history_analysis.get('calmar_ratio', 0):.2f}")
    
    # Performance by Trading Pair
    print("\nPERFORMANCE BY TRADING PAIR:")
    for pair, analysis in pair_analysis.items():
        if analysis["total_trades"] > 0:
            print(f"\n{pair}:")
            print(f"  Total Trades: {analysis['total_trades']}")
            print(f"  Win Rate: {analysis['win_rate']:.2%}")
            print(f"  Total Profit: ${analysis['total_profit']:,.2f}")
            print(f"  Average Profit: ${analysis['avg_profit']:,.2f}")
            print(f"  Max Profit: ${analysis['max_profit']:,.2f}")
            print(f"  Max Loss: ${analysis['max_loss']:,.2f}")
    
    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Parse pairs
    pairs = ALL_PAIRS if args.pairs == "ALL" else args.pairs.split(",")
    
    print("\n" + "=" * 80)
    print("EVALUATING IMPROVED MODEL PERFORMANCE")
    print("=" * 80)
    print(f"Pairs: {', '.join(pairs)}")
    print(f"Analysis Period: {args.days} days")
    print(f"Save Plots: {args.save_plots}")
    print("=" * 80 + "\n")
    
    # Load ML configuration
    ml_config = load_ml_config()
    if not ml_config:
        print("ML configuration not found. Continuing with limited analysis.")
    
    # Load portfolio data
    portfolio, positions, trades = load_portfolio_data()
    if not portfolio or not positions or not trades:
        print("Portfolio data not found. Cannot proceed with analysis.")
        return False
    
    # Load portfolio history
    portfolio_history = load_portfolio_history()
    if not portfolio_history:
        print("Portfolio history not found. Continuing with limited analysis.")
    
    # Calculate portfolio metrics
    portfolio_metrics = calculate_portfolio_metrics(portfolio, positions, trades)
    
    # Analyze trades by pair
    pair_analysis = analyze_trades_by_pair(trades, pairs)
    
    # Calculate trading performance
    trading_performance = calculate_trading_performance(trades, args.days)
    
    # Analyze portfolio history
    if portfolio_history:
        history_analysis, df_period = analyze_portfolio_history(portfolio_history, args.days)
        
        # Plot portfolio history
        if args.save_plots and df_period is not None:
            history_plot_path = f"{RESULTS_DIR}/portfolio_history.png"
            plot_portfolio_history(df_period, history_plot_path)
    else:
        history_analysis = {}
        df_period = None
    
    # Plot pair performance
    if args.save_plots and pair_analysis:
        pair_plot_path = f"{RESULTS_DIR}/pair_performance.png"
        plot_pair_performance(pair_analysis, pair_plot_path)
    
    # Print performance report
    print_performance_report(
        portfolio_metrics,
        pair_analysis,
        trading_performance,
        history_analysis
    )
    
    # Save report to file
    report_path = f"{RESULTS_DIR}/performance_report.txt"
    try:
        # Redirect stdout to file
        original_stdout = sys.stdout
        with open(report_path, 'w') as f:
            sys.stdout = f
            print_performance_report(
                portfolio_metrics,
                pair_analysis,
                trading_performance,
                history_analysis
            )
        # Restore stdout
        sys.stdout = original_stdout
        print(f"\nPerformance report saved to {report_path}")
    except Exception as e:
        print(f"Error saving performance report: {e}")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error evaluating performance: {e}")
        import traceback
        traceback.print_exc()