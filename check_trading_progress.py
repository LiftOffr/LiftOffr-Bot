#!/usr/bin/env python3
"""
Check Trading Progress and Portfolio Status

This script provides a comprehensive overview of the current trading status:
- Portfolio value and performance
- Open positions across all pairs
- Completed trades and their outcomes
- Trading metrics and statistics

Use this to monitor the performance of the trading bot.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("trading_progress")

# Constants
SANDBOX_PORTFOLIO_PATH = "data/sandbox_portfolio.json"
SANDBOX_POSITIONS_PATH = "data/sandbox_positions.json"
SANDBOX_TRADES_PATH = "data/sandbox_trades.json"
SANDBOX_PORTFOLIO_HISTORY_PATH = "data/sandbox_portfolio_history.json"
STRATEGY_PERFORMANCE_PATH = "config/strategy_performance.json"


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Check trading progress and portfolio status.')
    parser.add_argument('--full', action='store_true', help='Show full details')
    parser.add_argument('--update-dashboard', action='store_true', help='Update dashboard before displaying')
    parser.add_argument('--pair', type=str, help='Filter by trading pair')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    return parser.parse_args()


def update_dashboard() -> bool:
    """
    Update the dashboard with latest data.
    
    Returns:
        success: Whether the update succeeded
    """
    # Check which script to use
    if os.path.exists("update_dashboard.py"):
        cmd = [sys.executable, "update_dashboard.py"]
    else:
        logger.warning("Dashboard update script not found")
        return False
    
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info("Dashboard updated successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to update dashboard: {e}")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def load_portfolio() -> Optional[Dict]:
    """
    Load portfolio data.
    
    Returns:
        portfolio: Portfolio data or None if not found
    """
    try:
        if os.path.exists(SANDBOX_PORTFOLIO_PATH):
            with open(SANDBOX_PORTFOLIO_PATH, 'r') as f:
                portfolio = json.load(f)
            return portfolio
        else:
            logger.warning(f"Portfolio file not found: {SANDBOX_PORTFOLIO_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading portfolio: {e}")
        return None


def load_positions() -> Optional[Dict]:
    """
    Load open positions data.
    
    Returns:
        positions: Open positions data or None if not found
    """
    try:
        if os.path.exists(SANDBOX_POSITIONS_PATH):
            with open(SANDBOX_POSITIONS_PATH, 'r') as f:
                positions = json.load(f)
            return positions
        else:
            logger.warning(f"Positions file not found: {SANDBOX_POSITIONS_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading positions: {e}")
        return None


def load_trades() -> Optional[List]:
    """
    Load completed trades data.
    
    Returns:
        trades: Completed trades data or None if not found
    """
    try:
        if os.path.exists(SANDBOX_TRADES_PATH):
            with open(SANDBOX_TRADES_PATH, 'r') as f:
                trades = json.load(f)
            return trades
        else:
            logger.warning(f"Trades file not found: {SANDBOX_TRADES_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading trades: {e}")
        return None


def load_portfolio_history() -> Optional[Dict]:
    """
    Load portfolio history data.
    
    Returns:
        history: Portfolio history data or None if not found
    """
    try:
        if os.path.exists(SANDBOX_PORTFOLIO_HISTORY_PATH):
            with open(SANDBOX_PORTFOLIO_HISTORY_PATH, 'r') as f:
                history = json.load(f)
            return history
        else:
            logger.warning(f"Portfolio history file not found: {SANDBOX_PORTFOLIO_HISTORY_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading portfolio history: {e}")
        return None


def load_strategy_performance() -> Optional[Dict]:
    """
    Load strategy performance data.
    
    Returns:
        performance: Strategy performance data or None if not found
    """
    try:
        if os.path.exists(STRATEGY_PERFORMANCE_PATH):
            with open(STRATEGY_PERFORMANCE_PATH, 'r') as f:
                performance = json.load(f)
            return performance
        else:
            logger.warning(f"Strategy performance file not found: {STRATEGY_PERFORMANCE_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading strategy performance: {e}")
        return None


def calculate_portfolio_metrics(portfolio: Dict, history: Optional[Dict], positions: Dict) -> Dict:
    """
    Calculate portfolio metrics.
    
    Args:
        portfolio: Portfolio data
        history: Portfolio history data
        positions: Open positions data
        
    Returns:
        metrics: Portfolio metrics
    """
    metrics = {}
    
    # Current portfolio value
    metrics["current_value"] = portfolio.get("total", 0)
    
    # Initial portfolio value
    metrics["initial_value"] = portfolio.get("initial_capital", 20000)
    
    # Overall profit/loss
    metrics["absolute_pnl"] = metrics["current_value"] - metrics["initial_value"]
    metrics["percentage_pnl"] = (metrics["absolute_pnl"] / metrics["initial_value"]) * 100
    
    # Calculate unrealized P&L from open positions
    unrealized_pnl = 0
    for pair, position in positions.items():
        if position.get("active", False):
            unrealized_pnl += position.get("unrealized_pnl", 0)
    
    metrics["unrealized_pnl"] = unrealized_pnl
    metrics["unrealized_pnl_percentage"] = (unrealized_pnl / metrics["current_value"]) * 100 if metrics["current_value"] > 0 else 0
    
    # Calculate daily performance
    if history:
        if len(history.get("dates", [])) >= 2:
            dates = history.get("dates", [])
            values = history.get("values", [])
            
            if len(dates) > 0 and len(values) > 0:
                latest_value = values[-1]
                previous_value = values[-2] if len(values) > 1 else metrics["initial_value"]
                
                daily_change = latest_value - previous_value
                daily_change_percentage = (daily_change / previous_value) * 100 if previous_value > 0 else 0
                
                metrics["daily_change"] = daily_change
                metrics["daily_change_percentage"] = daily_change_percentage
    
    return metrics


def calculate_trade_metrics(trades: List) -> Dict:
    """
    Calculate trade metrics.
    
    Args:
        trades: List of completed trades
        
    Returns:
        metrics: Trade metrics
    """
    metrics = {}
    
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "average_profit": 0,
            "average_loss": 0,
            "profit_factor": 0,
            "largest_profit": 0,
            "largest_loss": 0,
            "average_trade": 0
        }
    
    # Total trades
    metrics["total_trades"] = len(trades)
    
    # Winning and losing trades
    winning_trades = [t for t in trades if t.get("realized_pnl", 0) > 0]
    losing_trades = [t for t in trades if t.get("realized_pnl", 0) <= 0]
    
    metrics["winning_trades"] = len(winning_trades)
    metrics["losing_trades"] = len(losing_trades)
    
    # Win rate
    metrics["win_rate"] = (metrics["winning_trades"] / metrics["total_trades"]) * 100 if metrics["total_trades"] > 0 else 0
    
    # Average profit and loss
    metrics["average_profit"] = sum(t.get("realized_pnl", 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
    metrics["average_loss"] = sum(t.get("realized_pnl", 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
    
    # Profit factor
    total_profit = sum(t.get("realized_pnl", 0) for t in winning_trades)
    total_loss = abs(sum(t.get("realized_pnl", 0) for t in losing_trades))
    metrics["profit_factor"] = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Largest profit and loss
    metrics["largest_profit"] = max([t.get("realized_pnl", 0) for t in trades], default=0)
    metrics["largest_loss"] = min([t.get("realized_pnl", 0) for t in trades], default=0)
    
    # Average trade
    metrics["average_trade"] = sum(t.get("realized_pnl", 0) for t in trades) / len(trades)
    
    return metrics


def print_json_output(data: Dict):
    """
    Print data in JSON format.
    
    Args:
        data: Data to print
    """
    print(json.dumps(data, indent=2))


def print_portfolio_summary(portfolio: Dict, metrics: Dict):
    """
    Print portfolio summary.
    
    Args:
        portfolio: Portfolio data
        metrics: Portfolio metrics
    """
    print("\n" + "=" * 60)
    print(" PORTFOLIO SUMMARY")
    print("=" * 60)
    
    print(f"Current Value: ${metrics['current_value']:,.2f}")
    print(f"Initial Value: ${metrics['initial_value']:,.2f}")
    print(f"Overall P&L: ${metrics['absolute_pnl']:,.2f} ({metrics['percentage_pnl']:,.2f}%)")
    
    if "daily_change" in metrics:
        print(f"Daily Change: ${metrics['daily_change']:,.2f} ({metrics['daily_change_percentage']:,.2f}%)")
    
    print(f"Unrealized P&L: ${metrics['unrealized_pnl']:,.2f} ({metrics['unrealized_pnl_percentage']:,.2f}%)")
    print(f"Available Cash: ${portfolio.get('cash', 0):,.2f}")
    
    print("-" * 60)


def print_positions_summary(positions: Dict, args):
    """
    Print open positions summary.
    
    Args:
        positions: Open positions data
        args: Command line arguments
    """
    print("\n" + "=" * 60)
    print(" OPEN POSITIONS")
    print("=" * 60)
    
    active_positions = [(pair, pos) for pair, pos in positions.items() 
                      if pos.get("active", False) and 
                      (args.pair is None or args.pair == pair)]
    
    if not active_positions:
        print("No open positions")
        return
    
    for pair, position in active_positions:
        entry_price = position.get("entry_price", 0)
        current_price = position.get("current_price", 0)
        direction = position.get("direction", "")
        size = position.get("size", 0)
        leverage = position.get("leverage", 1)
        entry_time = position.get("entry_time", "")
        unrealized_pnl = position.get("unrealized_pnl", 0)
        pnl_percentage = position.get("pnl_percentage", 0)
        strategy = position.get("strategy", "unknown")
        
        print(f"Pair: {pair}")
        print(f"Direction: {direction.upper()}")
        print(f"Size: {size:.4f}")
        print(f"Leverage: {leverage:.1f}x")
        print(f"Entry Price: ${entry_price:.2f}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Entry Time: {entry_time}")
        print(f"Strategy: {strategy}")
        print(f"Unrealized P&L: ${unrealized_pnl:.2f} ({pnl_percentage:.2f}%)")
        print("-" * 60)


def print_recent_trades(trades: List, args):
    """
    Print recent trades.
    
    Args:
        trades: List of completed trades
        args: Command line arguments
    """
    print("\n" + "=" * 60)
    print(" RECENT TRADES")
    print("=" * 60)
    
    if not trades:
        print("No trades completed yet")
        return
    
    # Filter trades by pair if specified
    if args.pair:
        trades = [t for t in trades if t.get("pair", "") == args.pair]
    
    # Sort trades by exit time (most recent first)
    trades = sorted(trades, key=lambda t: t.get("exit_time", ""), reverse=True)
    
    # Limit to most recent 10 trades unless full details requested
    if not args.full:
        trades = trades[:10]
    
    if not trades:
        print(f"No trades found for {args.pair}")
        return
    
    for i, trade in enumerate(trades):
        pair = trade.get("pair", "")
        direction = trade.get("direction", "")
        entry_price = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        realized_pnl = trade.get("realized_pnl", 0)
        pnl_percentage = trade.get("pnl_percentage", 0)
        entry_time = trade.get("entry_time", "")
        exit_time = trade.get("exit_time", "")
        strategy = trade.get("strategy", "unknown")
        
        print(f"Trade #{i+1}")
        print(f"Pair: {pair}")
        print(f"Direction: {direction.upper()}")
        print(f"Entry Price: ${entry_price:.2f}")
        print(f"Exit Price: ${exit_price:.2f}")
        print(f"P&L: ${realized_pnl:.2f} ({pnl_percentage:.2f}%)")
        print(f"Entry Time: {entry_time}")
        print(f"Exit Time: {exit_time}")
        print(f"Strategy: {strategy}")
        print("-" * 60)


def print_trading_metrics(metrics: Dict):
    """
    Print trading metrics.
    
    Args:
        metrics: Trading metrics
    """
    print("\n" + "=" * 60)
    print(" TRADING METRICS")
    print("=" * 60)
    
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Average Profit: ${metrics['average_profit']:.2f}")
    print(f"Average Loss: ${metrics['average_loss']:.2f}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Largest Profit: ${metrics['largest_profit']:.2f}")
    print(f"Largest Loss: ${metrics['largest_loss']:.2f}")
    print(f"Average Trade: ${metrics['average_trade']:.2f}")
    print("-" * 60)


def print_strategy_performance(performance: Optional[Dict]):
    """
    Print strategy performance.
    
    Args:
        performance: Strategy performance data
    """
    print("\n" + "=" * 60)
    print(" STRATEGY PERFORMANCE")
    print("=" * 60)
    
    if not performance:
        print("No strategy performance data available")
        return
    
    # Print by category
    for category, strategies in performance.items():
        print(f"Category: {category}")
        print("-" * 40)
        
        for strategy, metrics in strategies.items():
            print(f"Strategy: {strategy}")
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.2f}%")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"Average Trade: ${metrics.get('average_trade', 0):.2f}")
            print("-" * 30)
        
        print()


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Update dashboard if requested
    if args.update_dashboard:
        update_dashboard()
    
    # Load data
    portfolio = load_portfolio()
    positions = load_positions()
    trades = load_trades()
    history = load_portfolio_history()
    performance = load_strategy_performance()
    
    if not portfolio or not positions:
        print("No trading data found")
        return 1
    
    # Calculate metrics
    portfolio_metrics = calculate_portfolio_metrics(portfolio, history, positions)
    trade_metrics = calculate_trade_metrics(trades or [])
    
    # Prepare data for JSON output
    if args.json:
        data = {
            "portfolio": portfolio,
            "positions": positions,
            "trades": trades,
            "history": history,
            "performance": performance,
            "portfolio_metrics": portfolio_metrics,
            "trade_metrics": trade_metrics,
            "timestamp": datetime.now().isoformat()
        }
        print_json_output(data)
        return 0
    
    # Print summaries
    print_portfolio_summary(portfolio, portfolio_metrics)
    print_positions_summary(positions, args)
    print_recent_trades(trades or [], args)
    print_trading_metrics(trade_metrics)
    print_strategy_performance(performance)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())