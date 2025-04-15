#!/usr/bin/env python3
"""
Check Sandbox Portfolio

This script checks the current status of the portfolio in sandbox mode.
It displays:
1. Current portfolio value
2. Open positions with unrealized P&L
3. Trade history with performance metrics
4. Risk metrics from risk management system
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_STARTING_CAPITAL = 20000.0  # $20,000
POSITION_DATA_FILE = "data/sandbox_positions.json"
TRADE_HISTORY_FILE = "data/sandbox_trades.json"
PORTFOLIO_HISTORY_FILE = "data/sandbox_portfolio_history.json"
RISK_METRICS_FILE = "data/risk_metrics.json"

# Load environment variables
load_dotenv()
KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.environ.get("KRAKEN_API_SECRET")

def get_current_prices(pairs: List[str]) -> Dict[str, float]:
    """
    Get current prices from Kraken API for multiple pairs.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        Dictionary mapping pairs to current prices
    """
    prices = {}
    
    try:
        # For sandbox mode, we'll check if real API access is available
        if KRAKEN_API_KEY and KRAKEN_API_SECRET:
            # Format pairs for Kraken API
            kraken_pairs = [pair.replace("/", "") for pair in pairs]
            pair_str = ",".join(kraken_pairs)
            
            # Make API request to Kraken
            url = f"https://api.kraken.com/0/public/Ticker?pair={pair_str}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                if "result" in data:
                    result = data["result"]
                    
                    # Extract prices from response
                    for kraken_pair, info in result.items():
                        # Convert back to original format
                        original_pair = None
                        for p in pairs:
                            if p.replace("/", "") in kraken_pair:
                                original_pair = p
                                break
                        
                        if original_pair:
                            # Use last trade price (c[0])
                            prices[original_pair] = float(info["c"][0])
                
                logger.info(f"Retrieved current prices from Kraken API: {prices}")
            else:
                logger.warning(f"Failed to get prices from Kraken API: {response.status_code}")
                # Fall back to local data
                prices = _get_prices_from_local_data(pairs)
        else:
            # No API keys, use local data
            logger.info("No API keys, using local data for prices")
            prices = _get_prices_from_local_data(pairs)
    except Exception as e:
        logger.error(f"Error getting prices: {e}")
        # Fall back to local data
        prices = _get_prices_from_local_data(pairs)
    
    return prices

def _get_prices_from_local_data(pairs: List[str]) -> Dict[str, float]:
    """Get prices from local data files"""
    prices = {}
    
    try:
        # Try to load from position data first
        if os.path.exists(POSITION_DATA_FILE):
            with open(POSITION_DATA_FILE, 'r') as f:
                position_data = json.load(f)
                
            for position in position_data:
                if position["pair"] in pairs and "last_price" in position:
                    prices[position["pair"]] = position["last_price"]
        
        # Fill in missing pairs with sample prices
        sample_prices = {
            "SOL/USD": 130.25,
            "BTC/USD": 60125.50,
            "ETH/USD": 3050.75,
            "ADA/USD": 0.45,
            "DOT/USD": 6.78,
            "LINK/USD": 15.25
        }
        
        for pair in pairs:
            if pair not in prices:
                prices[pair] = sample_prices.get(pair, 100.0)
    except Exception as e:
        logger.error(f"Error loading local price data: {e}")
        # Use sample prices as last resort
        prices = {
            "SOL/USD": 130.25,
            "BTC/USD": 60125.50,
            "ETH/USD": 3050.75,
            "ADA/USD": 0.45,
            "DOT/USD": 6.78,
            "LINK/USD": 15.25
        }
    
    return {p: prices[p] for p in pairs if p in prices}

def load_position_data() -> List[Dict[str, Any]]:
    """
    Load position data from file
    
    Returns:
        List of position dictionaries
    """
    if os.path.exists(POSITION_DATA_FILE):
        try:
            with open(POSITION_DATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading position data: {e}")
    
    return []

def load_trade_history() -> List[Dict[str, Any]]:
    """
    Load trade history from file
    
    Returns:
        List of trade dictionaries
    """
    if os.path.exists(TRADE_HISTORY_FILE):
        try:
            with open(TRADE_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
    
    return []

def load_portfolio_history() -> List[Dict[str, Any]]:
    """
    Load portfolio history from file
    
    Returns:
        List of portfolio history points
    """
    if os.path.exists(PORTFOLIO_HISTORY_FILE):
        try:
            with open(PORTFOLIO_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading portfolio history: {e}")
    
    return []

def load_risk_metrics() -> Dict[str, Any]:
    """
    Load risk metrics from file
    
    Returns:
        Dictionary of risk metrics
    """
    if os.path.exists(RISK_METRICS_FILE):
        try:
            with open(RISK_METRICS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading risk metrics: {e}")
    
    return {}

def format_currency(value: float) -> str:
    """Format a value as currency"""
    return f"${value:.2f}"

def format_percentage(value: float) -> str:
    """Format a value as percentage"""
    return f"{value:.2f}%"

def calculate_portfolio_value(positions: List[Dict[str, Any]], prices: Dict[str, float]) -> float:
    """
    Calculate current portfolio value based on positions and prices
    
    Args:
        positions: List of positions
        prices: Dictionary of current prices
        
    Returns:
        Total portfolio value
    """
    portfolio_value = DEFAULT_STARTING_CAPITAL
    
    for position in positions:
        pair = position["pair"]
        if pair in prices:
            current_price = prices[pair]
            entry_price = position["entry_price"]
            size = position["size"]
            leverage = position["leverage"]
            direction = position["direction"]
            
            # Calculate margin (may not exist in older positions)
            if "margin" in position:
                margin = position["margin"]
            else:
                # Calculate margin from position size and entry price
                notional_value = size * entry_price
                margin = notional_value / leverage
            
            # Calculate unrealized P&L
            if direction.lower() == "long":
                price_change_pct = (current_price / entry_price) - 1
            else:  # short
                price_change_pct = (entry_price / current_price) - 1
                
            pnl_pct = price_change_pct * leverage
            position_value = margin * (1 + pnl_pct)
            
            # Update portfolio value
            portfolio_value += position_value - margin
    
    return portfolio_value

def calculate_performance_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate performance metrics from trade history
    
    Args:
        trades: List of trades
        
    Returns:
        Dictionary of performance metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "net_pnl": 0.0,
            "average_profit": 0.0,
            "average_loss": 0.0,
            "largest_profit": 0.0,
            "largest_loss": 0.0
        }
    
    total_trades = len(trades)
    winning_trades = 0
    losing_trades = 0
    total_profit = 0.0
    total_loss = 0.0
    largest_profit = 0.0
    largest_loss = 0.0
    
    for trade in trades:
        pnl = trade.get("pnl_amount", 0)
        if pnl > 0:
            winning_trades += 1
            total_profit += pnl
            largest_profit = max(largest_profit, pnl)
        else:
            losing_trades += 1
            total_loss += abs(pnl)
            largest_loss = max(largest_loss, abs(pnl))
    
    # Calculate metrics
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    net_pnl = total_profit - total_loss
    average_profit = total_profit / winning_trades if winning_trades > 0 else 0
    average_loss = total_loss / losing_trades if losing_trades > 0 else 0
    
    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_profit": total_profit,
        "total_loss": total_loss,
        "net_pnl": net_pnl,
        "average_profit": average_profit,
        "average_loss": average_loss,
        "largest_profit": largest_profit,
        "largest_loss": largest_loss
    }

def check_portfolio():
    """Check and display portfolio information"""
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(POSITION_DATA_FILE), exist_ok=True)
    
    # Load position and trade data
    positions = load_position_data()
    trades = load_trade_history()
    portfolio_history = load_portfolio_history()
    risk_metrics = load_risk_metrics()
    
    # Get current prices for all pairs in positions
    pairs = list(set([p["pair"] for p in positions]))
    if not pairs:
        pairs = ["SOL/USD", "BTC/USD", "ETH/USD"]  # Default pairs to check
    
    prices = get_current_prices(pairs)
    
    # Calculate current portfolio value
    portfolio_value = calculate_portfolio_value(positions, prices)
    
    # Calculate performance metrics
    performance = calculate_performance_metrics(trades)
    
    # Display results
    print("\n" + "=" * 80)
    print(f"PORTFOLIO STATUS - SANDBOX MODE - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Portfolio summary
    print("\nPORTFOLIO SUMMARY")
    print("-" * 50)
    print(f"Starting Capital: {format_currency(DEFAULT_STARTING_CAPITAL)}")
    print(f"Current Portfolio Value: {format_currency(portfolio_value)}")
    print(f"Total Return: {format_percentage((portfolio_value / DEFAULT_STARTING_CAPITAL - 1) * 100)}")
    
    if portfolio_history:
        # Get peak portfolio value
        peak_value = max([p.get("portfolio_value", DEFAULT_STARTING_CAPITAL) for p in portfolio_history])
        current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
        print(f"Current Drawdown: {format_percentage(current_drawdown * 100)}")
    
    # Open positions
    print("\nOPEN POSITIONS")
    print("-" * 50)
    
    if positions:
        positions_table = []
        total_position_value = 0.0
        
        for position in positions:
            pair = position["pair"]
            if pair in prices:
                current_price = prices[pair]
                entry_price = position["entry_price"]
                size = position["size"]
                leverage = position["leverage"]
                direction = position["direction"]
                
                # Calculate margin (may not exist in older positions)
                if "margin" in position:
                    margin = position["margin"]
                else:
                    # Calculate margin from position size and entry price
                    notional_value = size * entry_price
                    margin = notional_value / leverage
                
                # Calculate unrealized P&L
                if direction.lower() == "long":
                    price_change_pct = (current_price / entry_price) - 1
                else:  # short
                    price_change_pct = (entry_price / current_price) - 1
                    
                pnl_pct = price_change_pct * leverage
                pnl_amount = margin * pnl_pct
                position_value = margin * (1 + pnl_pct)
                
                # Get stop loss if available
                stop_loss = position.get("stop_loss", "N/A")
                stop_loss_str = f"{stop_loss:.2f}" if isinstance(stop_loss, (int, float)) else "N/A"
                
                # Add to table
                positions_table.append({
                    "Pair": pair,
                    "Direction": direction.upper(),
                    "Entry Price": format_currency(entry_price),
                    "Current Price": format_currency(current_price),
                    "Size": f"{size:.6f}",
                    "Leverage": f"{leverage:.1f}x",
                    "Margin": format_currency(margin),
                    "Stop Loss": stop_loss_str,
                    "Unrealized P&L": format_currency(pnl_amount),
                    "P&L %": format_percentage(pnl_pct * 100)
                })
                
                total_position_value += position_value
        
        # Display positions table
        if positions_table:
            # Print as formatted table
            headers = ["Pair", "Direction", "Entry Price", "Current Price", "Size", 
                      "Leverage", "Margin", "Stop Loss", "Unrealized P&L", "P&L %"]
            
            # Calculate column widths
            col_widths = {}
            for header in headers:
                col_widths[header] = len(header)
                for pos in positions_table:
                    col_widths[header] = max(col_widths[header], len(str(pos[header])))
            
            # Print headers
            header_row = " | ".join(f"{h:{col_widths[h]}}" for h in headers)
            print(header_row)
            print("-" * len(header_row))
            
            # Print data rows
            for pos in positions_table:
                row = " | ".join(f"{str(pos[h]):{col_widths[h]}}" for h in headers)
                print(row)
            
            print("-" * len(header_row))
            print(f"Total Position Value: {format_currency(total_position_value)}")
        else:
            print("No open positions with valid price data")
    else:
        print("No open positions")
    
    # Performance metrics
    print("\nPERFORMANCE METRICS")
    print("-" * 50)
    print(f"Total Trades: {performance['total_trades']}")
    print(f"Winning Trades: {performance['winning_trades']}")
    print(f"Losing Trades: {performance['losing_trades']}")
    print(f"Win Rate: {format_percentage(performance['win_rate'] * 100)}")
    print(f"Profit Factor: {performance['profit_factor']:.2f}")
    print(f"Net P&L: {format_currency(performance['net_pnl'])}")
    
    if performance['winning_trades'] > 0:
        print(f"Average Profit: {format_currency(performance['average_profit'])}")
        print(f"Largest Profit: {format_currency(performance['largest_profit'])}")
    
    if performance['losing_trades'] > 0:
        print(f"Average Loss: {format_currency(performance['average_loss'])}")
        print(f"Largest Loss: {format_currency(performance['largest_loss'])}")
    
    # Risk metrics
    print("\nRISK MANAGEMENT METRICS")
    print("-" * 50)
    
    if risk_metrics:
        # Get key risk metrics
        current_risk = risk_metrics.get("current_risk_level", "Medium")
        volatility = risk_metrics.get("volatility", {})
        max_drawdown = risk_metrics.get("max_drawdown", 0.0)
        var = risk_metrics.get("value_at_risk", 0.0)
        kelly = risk_metrics.get("kelly_criterion", 0.0)
        optimal_leverage = risk_metrics.get("optimal_leverage", 20.0)
        
        print(f"Current Risk Level: {current_risk}")
        print(f"Historical Max Drawdown: {format_percentage(max_drawdown * 100)}")
        print(f"Daily Value at Risk (95%): {format_percentage(var * 100)}")
        print(f"Kelly Criterion Fraction: {kelly:.2f}")
        print(f"Optimal Leverage: {optimal_leverage:.1f}x")
        
        if volatility:
            vol_level = volatility.get("level", "Medium")
            vol_value = volatility.get("value", 0.0)
            print(f"Volatility Level: {vol_level} ({format_percentage(vol_value * 100)} daily)")
    else:
        print("No risk metrics available")
    
    # Recent trades
    print("\nRECENT TRADES")
    print("-" * 50)
    
    if trades:
        # Sort trades by exit time (most recent first)
        recent_trades = sorted(trades, key=lambda t: t.get("exit_time", ""), reverse=True)[:5]
        
        # Print as formatted table
        headers = ["Pair", "Direction", "Entry Price", "Exit Price", "Leverage", "P&L", "Exit Reason"]
        
        # Calculate column widths
        col_widths = {}
        for header in headers:
            col_widths[header] = len(header)
            for trade in recent_trades:
                value = ""
                if header == "Pair":
                    value = trade.get("pair", "")
                elif header == "Direction":
                    value = trade.get("direction", "").upper()
                elif header == "Entry Price":
                    value = format_currency(trade.get("entry_price", 0))
                elif header == "Exit Price":
                    value = format_currency(trade.get("exit_price", 0))
                elif header == "Leverage":
                    value = f"{trade.get('leverage', 0):.1f}x"
                elif header == "P&L":
                    pnl = trade.get("pnl_amount", 0)
                    value = format_currency(pnl)
                elif header == "Exit Reason":
                    value = trade.get("exit_reason", "")
                
                col_widths[header] = max(col_widths[header], len(str(value)))
        
        # Print headers
        header_row = " | ".join(f"{h:{col_widths[h]}}" for h in headers)
        print(header_row)
        print("-" * len(header_row))
        
        # Print data rows
        for trade in recent_trades:
            row_data = []
            
            # Pair
            row_data.append(f"{trade.get('pair', ''):{col_widths['Pair']}}")
            
            # Direction
            row_data.append(f"{trade.get('direction', '').upper():{col_widths['Direction']}}")
            
            # Entry Price
            entry_price_str = format_currency(trade.get("entry_price", 0))
            row_data.append(f"{entry_price_str:{col_widths['Entry Price']}}")
            
            # Exit Price
            exit_price_str = format_currency(trade.get("exit_price", 0))
            row_data.append(f"{exit_price_str:{col_widths['Exit Price']}}")
            
            # Leverage
            leverage_str = f"{trade.get('leverage', 0):.1f}x"
            row_data.append(f"{leverage_str:{col_widths['Leverage']}}")
            
            # P&L
            pnl = trade.get("pnl_amount", 0)
            pnl_str = format_currency(pnl)
            row_data.append(f"{pnl_str:{col_widths['P&L']}}")
            
            # Exit Reason
            exit_reason = trade.get("exit_reason", "")
            row_data.append(f"{exit_reason:{col_widths['Exit Reason']}}")
            
            print(" | ".join(row_data))
    else:
        print("No trade history available")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    check_portfolio()