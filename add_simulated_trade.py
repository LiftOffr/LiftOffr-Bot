#!/usr/bin/env python3

"""
Add Realistic Simulated Trade

This script adds a highly realistic simulated trade to the portfolio using:
1. Real-time market data from Kraken API (bid/ask prices, spreads, volume)
2. Realistic slippage modeling based on order book depth
3. Accurate fee calculations based on Kraken's fee structure
4. Historical volatility data for realistic price movements
5. Dynamic parameter adjustments based on market conditions

Usage:
    python add_simulated_trade.py [pair] [direction]
    
    pair: Optional trading pair (e.g., "SOL/USD")
    direction: Optional trade direction ("LONG" or "SHORT")
"""

import os
import sys
import json
import time
import logging
import random
import requests
import math
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"

# Trading constants
KRAKEN_TAKER_FEE = 0.0026  # 0.26% taker fee
KRAKEN_MAKER_FEE = 0.0016  # 0.16% maker fee
MAX_SLIPPAGE_PCT = 0.0050  # 0.5% max slippage
DEFAULT_STRATEGY_CATEGORIES = {
    "ARIMA": "those dudes",
    "Adaptive": "him all along",
    "TCN": "him all along",
    "LSTM": "those dudes",
    "Ensemble": "him all along",
    "Transformer": "those dudes"
}

# Environment variables
KRAKEN_API_KEY = os.environ.get("KRAKEN_API_KEY")
KRAKEN_API_SECRET = os.environ.get("KRAKEN_API_SECRET")

def load_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default

def save_file(filepath, data):
    """Save data to a JSON file"""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving {filepath}: {e}")
        return False

def get_market_data(pairs: List[str]) -> Dict[str, Dict]:
    """
    Get comprehensive market data from Kraken API including:
    - Bid/ask prices and spread
    - 24h volume
    - 24h price range
    - Order book depth
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        Dictionary mapping pairs to market data
    """
    market_data = {}
    
    try:
        # Format pairs for Kraken API
        kraken_pairs = [pair.replace("/", "") for pair in pairs]
        pair_str = ",".join(kraken_pairs)
        
        # Make API request to Kraken for ticker data
        url = f"https://api.kraken.com/0/public/Ticker?pair={pair_str}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            if "result" in data:
                result = data["result"]
                
                # Extract market data from response
                for kraken_pair, info in result.items():
                    # Convert back to original format
                    original_pair = None
                    for p in pairs:
                        if p.replace("/", "") in kraken_pair:
                            original_pair = p
                            break
                    
                    if original_pair:
                        # Extract data from ticker response
                        market_data[original_pair] = {
                            "ask_price": float(info["a"][0]),  # Ask price
                            "bid_price": float(info["b"][0]),  # Bid price
                            "last_price": float(info["c"][0]),  # Last trade price
                            "volume_24h": float(info["v"][1]),  # 24h volume
                            "vwap_24h": float(info["p"][1]),    # 24h volume weighted avg price
                            "trades_24h": int(info["t"][1]),    # 24h number of trades
                            "low_24h": float(info["l"][1]),     # 24h low
                            "high_24h": float(info["h"][1]),    # 24h high
                            "open_24h": float(info["o"]),       # 24h open price
                            "spread": float(info["a"][0]) - float(info["b"][0]),  # Current spread
                            "spread_pct": (float(info["a"][0]) - float(info["b"][0])) / float(info["b"][0]),  # Spread as percentage
                        }
                        
                        # Calculate volatility metrics
                        high = float(info["h"][1])
                        low = float(info["l"][1])
                        market_data[original_pair]["volatility_24h"] = (high - low) / low
                        
                # Get order book data for each pair
                for original_pair in market_data.keys():
                    kraken_pair = original_pair.replace("/", "")
                    order_book_url = f"https://api.kraken.com/0/public/Depth?pair={kraken_pair}&count=10"
                    order_book_response = requests.get(order_book_url)
                    
                    if order_book_response.status_code == 200:
                        order_book_data = order_book_response.json()
                        if "result" in order_book_data:
                            for k, v in order_book_data["result"].items():
                                # Calculate average sizes at different levels
                                asks = [[float(price), float(volume)] for price, volume, _ in v["asks"]]
                                bids = [[float(price), float(volume)] for price, volume, _ in v["bids"]]
                                
                                # Calculate metrics
                                market_data[original_pair]["order_book"] = {
                                    "asks": asks,
                                    "bids": bids,
                                    "asks_total_volume": sum(vol for _, vol in asks),
                                    "bids_total_volume": sum(vol for _, vol in bids),
                                    "ask_depth": sum(vol for _, vol in asks[:3]),  # Top 3 levels
                                    "bid_depth": sum(vol for _, vol in bids[:3])   # Top 3 levels
                                }
                
                logger.info(f"Retrieved comprehensive market data for {len(market_data)} pairs")
        else:
            logger.warning(f"Failed to get market data from Kraken API: {response.status_code}")
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
    
    return market_data

def calculate_historical_volatility(pair: str, lookback_days: int = 30) -> Optional[float]:
    """
    Calculate historical volatility for a pair using daily data.
    
    Args:
        pair: Trading pair
        lookback_days: Number of days to look back
        
    Returns:
        Historical volatility as standard deviation of daily returns
    """
    try:
        # Format pair for Kraken API
        kraken_pair = pair.replace("/", "")
        
        # Calculate timestamp for lookback period
        now = datetime.now()
        since = int((now - timedelta(days=lookback_days)).timestamp())
        
        # Make API request to Kraken for OHLC data
        url = f"https://api.kraken.com/0/public/OHLC?pair={kraken_pair}&interval=1440&since={since}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            if "result" in data:
                # Extract OHLC data - first key in result is the pair name
                pair_key = next(key for key in data["result"].keys() if key != "last")
                ohlc_data = data["result"][pair_key]
                
                # Calculate daily returns
                prices = [float(candle[4]) for candle in ohlc_data]  # Close price is at index 4
                if len(prices) < 2:
                    return None
                
                # Calculate log returns
                returns = [math.log(prices[i+1]) - math.log(prices[i]) for i in range(len(prices)-1)]
                
                # Calculate annualized volatility (standard deviation of returns * sqrt(365))
                mean = sum(returns) / len(returns)
                variance = sum((r - mean) ** 2 for r in returns) / len(returns)
                volatility = math.sqrt(variance) * math.sqrt(365)
                
                logger.info(f"Calculated historical volatility for {pair}: {volatility:.4f}")
                return volatility
        else:
            logger.warning(f"Failed to get OHLC data from Kraken API: {response.status_code}")
    except Exception as e:
        logger.error(f"Error calculating historical volatility: {e}")
    
    return None

def estimate_slippage(market_data: Dict, order_size: float, direction: str) -> float:
    """
    Estimate realistic slippage based on order book data.
    
    Args:
        market_data: Market data including order book
        order_size: Size of the order in USD
        direction: "LONG" or "SHORT"
        
    Returns:
        Estimated slippage as a percentage
    """
    # Default slippage if we can't calculate
    default_slippage = random.uniform(0.0005, 0.0025)  # 0.05% to 0.25%
    
    try:
        if "order_book" not in market_data:
            return default_slippage
        
        order_book = market_data["order_book"]
        if direction == "LONG":
            # For buys, we look at ask side
            asks = order_book["asks"]
            total_volume = order_book["asks_total_volume"]
        else:
            # For sells, we look at bid side
            asks = order_book["bids"]
            total_volume = order_book["bids_total_volume"]
        
        # If order size is larger than total volume, use maximum slippage
        if order_size > total_volume:
            return MAX_SLIPPAGE_PCT
        
        # Calculate what percentage of the order book would be consumed
        order_percentage = order_size / total_volume
        
        # Slippage increases exponentially with order size relative to book depth
        slippage = order_percentage * order_percentage * MAX_SLIPPAGE_PCT
        
        # Add a small random component
        slippage += random.uniform(0.0001, 0.0005)
        
        return min(slippage, MAX_SLIPPAGE_PCT)
    except Exception as e:
        logger.error(f"Error estimating slippage: {e}")
        return default_slippage

def calculate_fee(order_size: float, is_maker: bool = False) -> float:
    """
    Calculate trading fee based on Kraken's fee structure.
    
    Args:
        order_size: Size of the order in USD
        is_maker: Whether the order is a maker (limit) or taker (market)
        
    Returns:
        Fee amount in USD
    """
    fee_rate = KRAKEN_MAKER_FEE if is_maker else KRAKEN_TAKER_FEE
    return order_size * fee_rate

def get_price_with_slippage(base_price: float, slippage_pct: float, direction: str) -> float:
    """
    Apply slippage to the base price based on direction.
    
    Args:
        base_price: Base price
        slippage_pct: Slippage as a percentage
        direction: "LONG" or "SHORT"
        
    Returns:
        Price with slippage applied
    """
    if direction == "LONG":
        # For buys, price increases with slippage
        return base_price * (1 + slippage_pct)
    else:
        # For sells, price decreases with slippage
        return base_price * (1 - slippage_pct)

def simulate_price_movement(
    start_price: float,
    direction: str,
    volatility: float,
    timeframe_hours: float,
    win_probability: float,
    risk_reward_ratio: float = 1.5
) -> Tuple[float, bool, str]:
    """
    Simulate a realistic price movement based on volatility.
    
    Args:
        start_price: Starting price
        direction: "LONG" or "SHORT"
        volatility: Historical volatility
        timeframe_hours: How many hours the trade lasted
        win_probability: Probability of a winning trade
        risk_reward_ratio: Risk-reward ratio for target vs stop
        
    Returns:
        Tuple of (end_price, is_win, exit_reason)
    """
    # Determine if trade is a win based on win probability
    is_win = random.random() < win_probability
    
    # Calculate daily volatility
    daily_vol = volatility / math.sqrt(365)
    
    # Scale volatility to the timeframe (in days)
    timeframe_days = timeframe_hours / 24
    period_vol = daily_vol * math.sqrt(timeframe_days)
    
    # Calculate price targets based on volatility and risk-reward ratio
    if direction == "LONG":
        # For longs: stop below entry, target above entry
        stop_distance = period_vol * start_price
        target_distance = stop_distance * risk_reward_ratio
        
        stop_price = start_price - stop_distance
        target_price = start_price + target_distance
        
        if is_win:
            end_price = target_price
            exit_reason = "TP"
        else:
            end_price = stop_price
            exit_reason = "SL"
    else:
        # For shorts: stop above entry, target below entry
        stop_distance = period_vol * start_price
        target_distance = stop_distance * risk_reward_ratio
        
        stop_price = start_price + stop_distance
        target_price = start_price - target_distance
        
        if is_win:
            end_price = target_price
            exit_reason = "TP"
        else:
            end_price = stop_price
            exit_reason = "SL"
    
    # Add a small random component to make it look more natural
    random_factor = 1 + random.uniform(-0.0005, 0.0005)
    end_price *= random_factor
    
    return end_price, is_win, exit_reason

def determine_trade_direction(market_data: Dict, pair_config: Dict) -> str:
    """
    Determine optimal trade direction based on market data and strategy preferences.
    
    Args:
        market_data: Market data for the pair
        pair_config: Configuration for the pair
        
    Returns:
        "LONG" or "SHORT"
    """
    # Default bias from config or 50/50
    long_bias = pair_config.get("long_bias", 0.5)
    
    # Adjust based on market conditions
    if "ask_price" in market_data and "open_24h" in market_data:
        # If price is trending up, increase long bias
        daily_change = (market_data["ask_price"] / market_data["open_24h"]) - 1
        
        if daily_change > 0.02:  # >2% up
            long_bias += 0.2
        elif daily_change < -0.02:  # >2% down
            long_bias -= 0.2
            
    # Clamp to valid range
    long_bias = max(0.1, min(0.9, long_bias))
    
    # Determine direction
    return "LONG" if random.random() < long_bias else "SHORT"

def add_simulated_trade(pair=None, forced_direction=None):
    """
    Add a highly realistic simulated trade using real market data.
    
    Args:
        pair: Optional specific pair to trade
        forced_direction: Optional forced direction ("LONG" or "SHORT")
        
    Returns:
        True if successful, False otherwise
    """
    # Load portfolio, positions, and ML config
    portfolio = load_file(PORTFOLIO_FILE, {
        "balance": 20000.0, 
        "equity": 20000.0,
        "trades": [],
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "last_updated": datetime.now().isoformat()
    })
    
    positions = load_file(POSITIONS_FILE, [])
    
    ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
    
    # Get available pairs
    pairs = list(ml_config.get("pairs", {}).keys())
    if not pairs:
        pairs = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD", 
                "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]
    
    # If specific pair is provided, validate and use it
    if pair and pair in pairs:
        pairs = [pair]
    else:
        # Choose a random pair
        pair = random.choice(pairs)
        pairs = [pair]
    
    # Get detailed market data
    market_data = get_market_data(pairs)
    
    # If we can't get the market data, exit
    if not market_data or pair not in market_data:
        logger.error(f"Could not get market data for {pair}")
        return False
    
    # Get pair configuration
    pair_config = ml_config.get("pairs", {}).get(pair, {})
    
    # Determine trade parameters
    # Direction - use forced_direction if provided, otherwise determine based on market
    direction = forced_direction if forced_direction else determine_trade_direction(market_data[pair], pair_config)
    
    # Get confidence from ML model (simulated)
    confidence = random.uniform(0.70, 0.95)
    
    # Choose a strategy from available options
    strategies = ["ARIMA", "Adaptive", "TCN", "LSTM", "Ensemble", "Transformer"]
    strategy = random.choice(strategies)
    
    # Get strategy category
    strategy_category = DEFAULT_STRATEGY_CATEGORIES.get(strategy, "those dudes")
    
    # Get leverage from config, adjust by confidence
    base_leverage = pair_config.get("base_leverage", 38.0)
    max_leverage = pair_config.get("max_leverage", 100.0)
    
    leverage = base_leverage + (confidence * (max_leverage - base_leverage))
    leverage = min(max_leverage, max(base_leverage, leverage))
    
    # Calculate position size
    account_balance = portfolio.get("balance", 20000.0)
    risk_percentage = pair_config.get("risk_percentage", 0.20)  # Default 20% risk
    position_size = account_balance * risk_percentage * confidence
    
    # Apply position size limits
    max_position_size = account_balance * 0.5  # Max 50% of account per position
    position_size = min(position_size, max_position_size)
    
    # Calculate order value and estimate slippage
    slippage = estimate_slippage(market_data[pair], position_size, direction)
    
    # Get entry price with realistic spread and slippage
    if direction == "LONG":
        # For longs, use ask price + slippage
        base_price = market_data[pair]["ask_price"]
    else:
        # For shorts, use bid price - slippage
        base_price = market_data[pair]["bid_price"]
    
    entry_price = get_price_with_slippage(base_price, slippage, direction)
    
    # Calculate fees
    entry_fee = calculate_fee(position_size, is_maker=False)  # Assume market order
    
    # Get historical volatility or use a default
    volatility = calculate_historical_volatility(pair) or 0.85  # Default to 85% annual volatility
    
    # Get win rate from config or default to 75%
    win_rate = pair_config.get("win_rate", 0.75)
    
    # Adjust win rate based on confidence
    win_rate = win_rate * (0.8 + 0.4 * confidence)  # Scale between 80-120% of base win rate
    win_rate = min(max(win_rate, 0.5), 0.95)  # Constrain to reasonable range
    
    # Simulate trade duration (between 1 hour and 3 days)
    trade_duration_hours = random.uniform(1, 72)
    
    # Simulate price movement
    exit_price, is_win, exit_reason = simulate_price_movement(
        entry_price, 
        direction, 
        volatility, 
        trade_duration_hours,
        win_rate,
        risk_reward_ratio=1.5
    )
    
    # Calculate exit slippage
    exit_slippage = estimate_slippage(market_data[pair], position_size, "SHORT" if direction == "LONG" else "LONG")
    
    # Apply exit slippage to exit price
    exit_price = get_price_with_slippage(
        exit_price, 
        exit_slippage, 
        "SHORT" if direction == "LONG" else "LONG"
    )
    
    # Calculate exit fee
    exit_fee = calculate_fee(position_size, is_maker=False)  # Assume market order
    
    # Calculate total fees
    total_fees = entry_fee + exit_fee
    
    # Calculate PnL
    if direction == "LONG":
        price_change_pct = (exit_price / entry_price) - 1
    else:
        price_change_pct = (entry_price / exit_price) - 1
    
    # Apply leverage to percentage change
    leveraged_pnl_pct = price_change_pct * leverage
    
    # Calculate PnL amount, subtracting fees
    pnl_amount = (position_size * leveraged_pnl_pct) - total_fees
    
    # Create realistic timestamps
    now = datetime.now()
    entry_time = (now - timedelta(hours=trade_duration_hours)).isoformat()
    exit_time = now.isoformat()
    
    # Create a realistic trade
    trade = {
        "pair": pair,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "pnl_percentage": leveraged_pnl_pct,
        "pnl_amount": pnl_amount,
        "exit_reason": exit_reason,
        "position_size": position_size,
        "leverage": leverage,
        "confidence": confidence,
        "strategy": strategy,
        "strategy_category": strategy_category,
        "fees": total_fees,
        "slippage": (slippage + exit_slippage) * 100,  # Convert to percentage
        "trade_duration_hours": trade_duration_hours
    }
    
    # Add to portfolio
    trades = portfolio.get("trades", [])
    trades.append(trade)
    
    # Update portfolio statistics
    portfolio["total_trades"] = portfolio.get("total_trades", 0) + 1
    if is_win:
        portfolio["winning_trades"] = portfolio.get("winning_trades", 0) + 1
    else:
        portfolio["losing_trades"] = portfolio.get("losing_trades", 0) + 1
    
    # Update portfolio balance
    portfolio["balance"] = portfolio.get("balance", 20000.0) + pnl_amount
    portfolio["equity"] = portfolio.get("equity", 20000.0) + pnl_amount
    portfolio["last_updated"] = now.isoformat()
    
    # Save portfolio
    portfolio["trades"] = trades
    if not save_file(PORTFOLIO_FILE, portfolio):
        logger.error("Failed to save portfolio")
        return False
    
    # Update portfolio history
    try:
        history = load_file(HISTORY_FILE, {"timestamps": [], "values": []})
        
        # Make sure history has the correct structure
        if not isinstance(history, dict):
            history = {"timestamps": [], "values": []}
        if "timestamps" not in history:
            history["timestamps"] = []
        if "values" not in history:
            history["values"] = []
            
        history["timestamps"].append(now.isoformat())
        history["values"].append(portfolio["equity"])
        if not save_file(HISTORY_FILE, history):
            logger.warning("Failed to update portfolio history")
    except Exception as e:
        logger.warning(f"Error updating portfolio history: {e}")
    
    # Log the trade
    logger.info(
        f"Added trade for {pair} {direction}, "
        f"Entry: ${entry_price:.6f}, Exit: ${exit_price:.6f}, "
        f"PnL: ${pnl_amount:.2f} ({leveraged_pnl_pct*100:.2f}%), "
        f"Leverage: {leverage:.1f}x, Fees: ${total_fees:.2f}, "
        f"Duration: {trade_duration_hours:.1f}h"
    )
    
    print(f"Successfully added simulated trade for {pair}:")
    print(f"  Direction: {direction}")
    print(f"  Entry Price: ${entry_price:.6f}")
    print(f"  Exit Price: ${exit_price:.6f}")
    print(f"  PnL: ${pnl_amount:.2f} ({leveraged_pnl_pct*100:.2f}%)")
    print(f"  Position Size: ${position_size:.2f}")
    print(f"  Leverage: {leverage:.1f}x")
    print(f"  Strategy: {strategy} ({strategy_category})")
    print(f"  Fees: ${total_fees:.2f}")
    print(f"  Trade Duration: {trade_duration_hours:.1f} hours")
    
    return True

if __name__ == "__main__":
    # Get arguments
    args = sys.argv[1:]
    pair = args[0] if len(args) > 0 else None
    direction = args[1].upper() if len(args) > 1 and args[1].upper() in ["LONG", "SHORT"] else None
    
    start_time = time.time()
    if add_simulated_trade(pair, direction):
        elapsed = time.time() - start_time
        print(f"Trade simulation completed in {elapsed:.2f} seconds.")
    else:
        print("Failed to add simulated trade.")
        sys.exit(1)