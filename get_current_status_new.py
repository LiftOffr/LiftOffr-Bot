#!/usr/bin/env python3
"""
Get Current Status New

This script provides detailed information about the current trading status
and portfolio performance, including USD return and percentage for each coin.

Usage:
    python get_current_status_new.py [--pair PAIR]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TRADES_FILE = "trades.csv"
DEFAULT_PORTFOLIO_FILE = "portfolio.json"
DEFAULT_INITIAL_CAPITAL = 20000.0  # USD

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Get Current Trading Status")
    parser.add_argument("--pair", help="Trading pair to check (e.g., SOL/USD)")
    parser.add_argument("--trades-file", default=DEFAULT_TRADES_FILE, help="Path to trades CSV file")
    parser.add_argument("--portfolio-file", default=DEFAULT_PORTFOLIO_FILE, help="Path to portfolio JSON file")
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL, help="Initial capital in USD")
    parser.add_argument("--output-json", action="store_true", help="Output results as JSON")
    return parser.parse_args()

def load_trades(trades_file: str) -> pd.DataFrame:
    """
    Load trades from CSV file
    
    Args:
        trades_file: Path to trades CSV file
        
    Returns:
        DataFrame with trades
    """
    try:
        if os.path.exists(trades_file):
            df = pd.read_csv(trades_file)
            # Convert timestamp to datetime if present
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        else:
            logger.warning(f"Trades file not found: {trades_file}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading trades: {e}")
        return pd.DataFrame()

def load_portfolio(portfolio_file: str) -> Dict[str, Any]:
    """
    Load portfolio from JSON file
    
    Args:
        portfolio_file: Path to portfolio JSON file
        
    Returns:
        Dictionary with portfolio information
    """
    try:
        if os.path.exists(portfolio_file):
            with open(portfolio_file, "r") as f:
                portfolio = json.load(f)
            return portfolio
        else:
            logger.warning(f"Portfolio file not found: {portfolio_file}")
            return {}
    except Exception as e:
        logger.error(f"Error loading portfolio: {e}")
        return {}

def get_current_market_prices() -> Dict[str, float]:
    """
    Get current market prices for all pairs
    
    Returns:
        Dictionary with pair -> price mapping
    """
    try:
        # Check if price cache file exists and is recent (within the last minute)
        cache_file = "market_prices_cache.json"
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
            
            cache_time = datetime.fromisoformat(cache_data.get("timestamp", "2000-01-01T00:00:00"))
            now = datetime.now()
            
            # If cache is recent, use it
            if (now - cache_time).total_seconds() < 60:
                logger.debug("Using cached market prices")
                return cache_data.get("prices", {})
        
        # Get market prices from API or other source
        import subprocess
        pairs = ["SOL/USD", "BTC/USD", "ETH/USD", "DOT/USD", "LINK/USD"]
        prices = {}
        
        for pair in pairs:
            try:
                # Use the display_status.py script to get current price
                result = subprocess.run(
                    f"python display_status.py --pair {pair} --quiet",
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Parse the output to get the price
                for line in result.stdout.splitlines():
                    if "Current price:" in line:
                        price_str = line.split("Current price:")[1].strip().replace("$", "")
                        prices[pair] = float(price_str)
                        break
            except Exception as e:
                logger.warning(f"Error getting price for {pair}: {e}")
        
        # Save to cache
        with open(cache_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "prices": prices
            }, f)
        
        return prices
    except Exception as e:
        logger.error(f"Error getting market prices: {e}")
        return {}

def calculate_pnl_for_pair(
    trades: pd.DataFrame,
    pair: str,
    current_price: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate P&L for a specific trading pair
    
    Args:
        trades: DataFrame with trades
        pair: Trading pair
        current_price: Current price of the pair (if None, will be fetched)
        
    Returns:
        Dictionary with P&L information
    """
    if trades.empty:
        return {
            "pair": pair,
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "win_rate": 0.0,
            "pnl_usd": 0.0,
            "pnl_pct": 0.0,
            "initial_value": 0.0,
            "current_value": 0.0,
            "open_positions": 0
        }
    
    # Filter trades for this pair (using the appropriate column name)
    if "trading_pair" in trades.columns:
        pair_trades = trades[trades["trading_pair"] == pair].copy()
    elif "pair" in trades.columns:
        pair_trades = trades[trades["pair"] == pair].copy()
    else:
        logger.error(f"Could not find trading pair column in trades file for {pair}")
        return {
            "pair": pair,
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "win_rate": 0.0,
            "pnl_usd": 0.0,
            "pnl_pct": 0.0,
            "initial_value": 0.0,
            "current_value": 0.0,
            "open_positions": 0
        }
    
    if pair_trades.empty:
        return {
            "pair": pair,
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "win_rate": 0.0,
            "pnl_usd": 0.0,
            "pnl_pct": 0.0,
            "initial_value": 0.0,
            "current_value": 0.0,
            "open_positions": 0
        }
    
    # Get current price if not provided
    if current_price is None:
        prices = get_current_market_prices()
        current_price = prices.get(pair, 0.0)
    
    # Check if status column exists
    if "status" in pair_trades.columns:
        # Calculate P&L for closed trades
        closed_trades = pair_trades[pair_trades["status"] == "CLOSED"]
        open_trades = pair_trades[pair_trades["status"] == "OPEN"]
    else:
        # Handle historical trades that might not have status column
        # Assume all trades with PnL are closed, others are open
        closed_trades = pair_trades[pair_trades["pnl"].notna() & (pair_trades["pnl"] != "")]
        open_trades = pair_trades[pair_trades["pnl"].isna() | (pair_trades["pnl"] == "")]
    
    total_trades = len(closed_trades)
    
    # Check if pnl column contains valid numeric values
    if "pnl" in closed_trades.columns:
        # Convert pnl to numeric, errors='coerce' will convert non-numeric values to NaN
        closed_trades["pnl_numeric"] = pd.to_numeric(closed_trades["pnl"], errors='coerce')
        win_trades = len(closed_trades[closed_trades["pnl_numeric"] > 0])
        loss_trades = len(closed_trades[closed_trades["pnl_numeric"] < 0])
        closed_pnl_usd = closed_trades["pnl_numeric"].sum()
    else:
        win_trades = 0
        loss_trades = 0
        closed_pnl_usd = 0.0
    
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    open_pnl_usd = 0.0
    
    # Set default values
    initial_investment = 0.0
    current_value = closed_pnl_usd
    
    # Initial investment calculation
    if "usd_amount" in pair_trades.columns:
        initial_investment = pair_trades["usd_amount"].sum()
    elif "value" in pair_trades.columns:
        # Extract numeric values from the 'value' column
        pair_trades["value_numeric"] = pd.to_numeric(pair_trades["value"], errors='coerce')
        initial_investment = pair_trades["value_numeric"].sum()
    elif "price" in pair_trades.columns and "amount" in pair_trades.columns:
        # Calculate value from price and amount
        pair_trades["value_calc"] = pair_trades["price"] * pair_trades["amount"]
        initial_investment = pair_trades["value_calc"].sum()
    
    # Process open trades if any
    for _, trade in open_trades.iterrows():
        # Set default values
        entry_price = 0.0
        quantity = 0.0
        direction = "BUY"  # Default to long
        
        # Extract entry price
        if "entry_price" in trade:
            entry_price = trade["entry_price"]
        elif "price" in trade:
            entry_price = trade["price"]
        
        # Extract quantity
        if "quantity" in trade:
            quantity = trade["quantity"]
        elif "amount" in trade:
            quantity = trade["amount"]
        
        # Extract direction
        if "direction" in trade:
            direction = trade["direction"]
        elif "side" in trade:
            direction = trade["side"]
        
        if direction == "BUY":
            # Long position
            position_value = quantity * current_price
            entry_value = quantity * entry_price
            trade_pnl = position_value - entry_value
        else:
            # Short position
            position_value = quantity * entry_price
            current_liability = quantity * current_price
            trade_pnl = position_value - current_liability
        
        open_pnl_usd += trade_pnl
        current_value += position_value
    
    total_pnl_usd = closed_pnl_usd + open_pnl_usd
    
    # Calculate percentage return
    pnl_pct = (total_pnl_usd / initial_investment) * 100 if initial_investment > 0 else 0.0
    
    return {
        "pair": pair,
        "total_trades": total_trades,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "win_rate": win_rate,
        "closed_pnl_usd": closed_pnl_usd,
        "open_pnl_usd": open_pnl_usd,
        "pnl_usd": total_pnl_usd,
        "pnl_pct": pnl_pct,
        "initial_value": initial_investment,
        "current_value": current_value,
        "open_positions": len(open_trades)
    }

def calculate_portfolio_metrics(
    trades: pd.DataFrame,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
) -> Dict[str, Any]:
    """
    Calculate overall portfolio metrics
    
    Args:
        trades: DataFrame with trades
        initial_capital: Initial capital in USD
        
    Returns:
        Dictionary with portfolio metrics
    """
    if trades.empty:
        return {
            "initial_capital": initial_capital,
            "current_value": initial_capital,
            "total_pnl_usd": 0.0,
            "total_pnl_pct": 0.0,
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "win_rate": 0.0,
            "open_positions": 0,
            "trading_pairs": []
        }
    
    # Get current market prices
    prices = get_current_market_prices()
    
    # Get unique trading pairs (column might be named 'pair' instead of 'trading_pair')
    if "trading_pair" in trades.columns:
        trading_pairs = trades["trading_pair"].unique().tolist()
    elif "pair" in trades.columns:
        trading_pairs = trades["pair"].unique().tolist()
    else:
        logger.error("Could not find trading pair column in trades file")
        return {
            "initial_capital": initial_capital,
            "current_value": initial_capital,
            "total_pnl_usd": 0.0,
            "total_pnl_pct": 0.0,
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "win_rate": 0.0,
            "open_positions": 0,
            "trading_pairs": []
        }
    
    # Calculate metrics for each pair
    pair_metrics = {}
    total_pnl_usd = 0.0
    total_trades = 0
    win_trades = 0
    loss_trades = 0
    open_positions = 0
    
    for pair in trading_pairs:
        current_price = prices.get(pair, 0.0)
        pair_result = calculate_pnl_for_pair(trades, pair, current_price)
        
        pair_metrics[pair] = pair_result
        total_pnl_usd += pair_result["pnl_usd"]
        total_trades += pair_result["total_trades"]
        win_trades += pair_result["win_trades"]
        loss_trades += pair_result["loss_trades"]
        open_positions += pair_result["open_positions"]
    
    # Calculate overall win rate
    win_rate = win_trades / total_trades if total_trades > 0 else 0.0
    
    # Calculate overall P&L percentage
    total_pnl_pct = (total_pnl_usd / initial_capital) * 100 if initial_capital > 0 else 0.0
    
    # Calculate current portfolio value
    current_value = initial_capital + total_pnl_usd
    
    return {
        "initial_capital": initial_capital,
        "current_value": current_value,
        "total_pnl_usd": total_pnl_usd,
        "total_pnl_pct": total_pnl_pct,
        "total_trades": total_trades,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "win_rate": win_rate,
        "open_positions": open_positions,
        "trading_pairs": trading_pairs,
        "pair_metrics": pair_metrics
    }

def display_results(metrics: Dict[str, Any], pair: Optional[str] = None, as_json: bool = False):
    """
    Display results
    
    Args:
        metrics: Portfolio metrics
        pair: Trading pair to display (if None, display all)
        as_json: Whether to output as JSON
    """
    if as_json:
        if pair:
            # Display metrics for specific pair
            pair_metrics = metrics["pair_metrics"].get(pair, {})
            print(json.dumps(pair_metrics, indent=2))
            
            # Also save to file
            with open(f"status_{pair.replace('/', '')}.json", "w") as f:
                json.dump(pair_metrics, f, indent=2)
        else:
            # Display all metrics
            print(json.dumps(metrics, indent=2))
        return
    
    # Display as formatted text
    if pair:
        # Display metrics for specific pair
        pair_metrics = metrics["pair_metrics"].get(pair, {})
        if not pair_metrics:
            print(f"No data available for {pair}")
            return
        
        print(f"\n===== {pair} Trading Performance =====")
        print(f"Total Trades: {pair_metrics['total_trades']}")
        print(f"Win Rate: {pair_metrics['win_rate']*100:.2f}%")
        print(f"P&L: ${pair_metrics['pnl_usd']:.2f} ({pair_metrics['pnl_pct']:.2f}%)")
        print(f"Initial Investment: ${pair_metrics['initial_value']:.2f}")
        print(f"Current Value: ${pair_metrics['current_value']:.2f}")
        print(f"Open Positions: {pair_metrics['open_positions']}")
        print("=====================================\n")
        
        # Save to file
        with open(f"status_{pair.replace('/', '')}.json", "w") as f:
            json.dump(pair_metrics, f, indent=2)
    else:
        # Display overall metrics
        print("\n===== Portfolio Performance =====")
        print(f"Initial Capital: ${metrics['initial_capital']:.2f}")
        print(f"Current Value: ${metrics['current_value']:.2f}")
        print(f"Total P&L: ${metrics['total_pnl_usd']:.2f} ({metrics['total_pnl_pct']:.2f}%)")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"Open Positions: {metrics['open_positions']}")
        print("==================================\n")
        
        # Display metrics for each pair
        print("===== Trading Pair Breakdown =====")
        
        for pair, pair_metrics in metrics["pair_metrics"].items():
            print(f"\n{pair}:")
            print(f"  Trades: {pair_metrics['total_trades']}")
            print(f"  Win Rate: {pair_metrics['win_rate']*100:.2f}%")
            print(f"  P&L: ${pair_metrics['pnl_usd']:.2f} ({pair_metrics['pnl_pct']:.2f}%)")
            
        print("\n==================================")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load trades
    trades = load_trades(args.trades_file)
    
    # Calculate portfolio metrics
    metrics = calculate_portfolio_metrics(trades, args.initial_capital)
    
    # Display results
    display_results(metrics, args.pair, args.output_json)

if __name__ == "__main__":
    main()