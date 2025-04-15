#!/usr/bin/env python3
"""
Historical Data Trainer for Trade Optimizer

This script fetches and processes historical cryptocurrency data
to train the trade optimizer, generating optimal entry/exit parameters
for each trading pair.
"""
import os
import json
import logging
import argparse
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Import our modules
from trade_optimizer import TradeOptimizer
import kraken_api_client as kraken

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
HISTORICAL_DIR = f"{DATA_DIR}/historical_data"
PROCESSED_DIR = f"{DATA_DIR}/processed_data"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
OPTIMIZATION_FILE = f"{DATA_DIR}/trade_optimization.json"

# Default trading pairs
DEFAULT_PAIRS = [
    "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Historical Data Trainer for Trade Optimizer")
    
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        help="Trading pairs to train on"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days of historical data to use"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="OHLC candle interval in minutes"
    )
    
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save historical data to disk"
    )
    
    parser.add_argument(
        "--generate-trades",
        action="store_true",
        help="Generate simulated trades for training"
    )
    
    return parser.parse_args()

def create_directories():
    """Create necessary directories"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(HISTORICAL_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def fetch_historical_data(
    pair: str,
    days: int = 90,
    interval: int = 60,  # minutes
    save_data: bool = True
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLC data for a trading pair
    
    Args:
        pair: Trading pair
        days: Number of days of historical data
        interval: Candle interval in minutes
        save_data: Whether to save data to disk
        
    Returns:
        DataFrame with historical data or None on error
    """
    kraken_client = kraken.KrakenAPIClient()
    
    # Convert pair to Kraken format
    kraken_pair = pair.replace("/", "")
    
    # Calculate start time
    end_time = int(time.time())
    start_time = end_time - (days * 24 * 60 * 60)
    
    try:
        # Get OHLC data from Kraken API
        # Need to chunk requests to avoid API limits
        chunk_size = 720  # Approx 30 days of hourly data
        all_data = []
        
        current_start = start_time
        
        while current_start < end_time:
            try:
                logger.info(f"Fetching {pair} data from {datetime.fromtimestamp(current_start).strftime('%Y-%m-%d')}")
                
                # Get OHLC data
                ohlc_data = kraken_client._request(
                    kraken.PUBLIC_ENDPOINTS["ohlc"],
                    {
                        "pair": kraken_pair,
                        "interval": interval,
                        "since": current_start
                    }
                )
                
                # Find the key corresponding to the pair data
                pair_key = None
                for key in ohlc_data.keys():
                    if key != "last":
                        pair_key = key
                        break
                
                if not pair_key:
                    logger.warning(f"No data found for {pair}")
                    break
                
                # Extract OHLC data
                candles = ohlc_data[pair_key]
                if not candles:
                    logger.warning(f"No candles found for {pair}")
                    break
                
                all_data.extend(candles)
                
                # Update start time for next chunk
                last_time = int(candles[-1][0])
                if last_time <= current_start:
                    # No new data, break to avoid infinite loop
                    break
                
                current_start = last_time
                
                # Small delay to avoid API rate limiting
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error fetching {pair} data: {e}")
                break
        
        if not all_data:
            logger.warning(f"No historical data fetched for {pair}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_data,
            columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
            ]
        )
        
        # Convert types
        df['timestamp'] = df['timestamp'].astype(float)
        for col in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']:
            df[col] = df[col].astype(float)
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        # Sort index
        df.sort_index(inplace=True)
        
        # Save to disk if requested
        if save_data:
            # Create filename
            filename = f"{HISTORICAL_DIR}/{pair.replace('/', '_')}_{interval}m.csv"
            df.to_csv(filename)
            logger.info(f"Saved {pair} historical data to {filename}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching historical data for {pair}: {e}")
        return None

def fetch_all_historical_data(
    pairs: List[str],
    days: int = 90,
    interval: int = 60,
    save_data: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple pairs
    
    Args:
        pairs: List of trading pairs
        days: Number of days of historical data
        interval: Candle interval in minutes
        save_data: Whether to save data to disk
        
    Returns:
        Dictionary of DataFrame with historical data by pair
    """
    all_data = {}
    
    for pair in pairs:
        try:
            logger.info(f"Fetching historical data for {pair}")
            df = fetch_historical_data(pair, days, interval, save_data)
            
            if df is not None and not df.empty:
                all_data[pair] = df
            else:
                logger.warning(f"No valid data for {pair}")
                
                # Try to load from disk if save_data is True
                if save_data:
                    filename = f"{HISTORICAL_DIR}/{pair.replace('/', '_')}_{interval}m.csv"
                    if os.path.exists(filename):
                        try:
                            df = pd.read_csv(filename)
                            df['datetime'] = pd.to_datetime(df['datetime'])
                            df.set_index('datetime', inplace=True)
                            all_data[pair] = df
                            logger.info(f"Loaded {pair} historical data from disk")
                        except Exception as e:
                            logger.error(f"Error loading {pair} data from disk: {e}")
        
        except Exception as e:
            logger.error(f"Error processing {pair}: {e}")
    
    return all_data

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame
    
    Args:
        df: DataFrame with OHLC data
        
    Returns:
        DataFrame with technical indicators added
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate simple moving averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_100'] = df['close'].rolling(window=100).mean()
    
    # Calculate exponential moving averages
    df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
    
    # Calculate MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # Calculate Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()
    
    # Calculate price changes
    df['pct_change'] = df['close'].pct_change() * 100
    df['pct_change_1h'] = df['close'].pct_change(periods=1) * 100
    df['pct_change_4h'] = df['close'].pct_change(periods=4) * 100
    df['pct_change_24h'] = df['close'].pct_change(periods=24) * 100
    
    # Calculate volatility
    df['volatility_14'] = df['pct_change'].rolling(window=14).std()
    df['volatility_30'] = df['pct_change'].rolling(window=30).std()
    
    # Distance from moving averages (as percentage)
    df['dist_sma_50'] = ((df['close'] / df['sma_50']) - 1) * 100
    df['dist_sma_100'] = ((df['close'] / df['sma_100']) - 1) * 100
    df['dist_ema_50'] = ((df['close'] / df['ema_50']) - 1) * 100
    df['dist_ema_100'] = ((df['close'] / df['ema_100']) - 1) * 100
    
    # Fill NA values
    df.fillna(method='bfill', inplace=True)
    
    return df

def determine_market_state(df: pd.DataFrame) -> str:
    """
    Determine the current market state based on technical indicators
    
    Args:
        df: DataFrame with technical indicators
        
    Returns:
        Market state (trending_up, trending_down, ranging, volatile)
    """
    # Get latest row
    latest = df.iloc[-1]
    
    # Check for trending up
    if (latest['ema_20'] > latest['ema_50'] > latest['ema_100'] and
            latest['close'] > latest['ema_20'] and
            latest['macd'] > latest['macd_signal'] and
            latest['dist_ema_50'] > 2.0):
        return 'trending_up'
    
    # Check for trending down
    elif (latest['ema_20'] < latest['ema_50'] < latest['ema_100'] and
            latest['close'] < latest['ema_20'] and
            latest['macd'] < latest['macd_signal'] and
            latest['dist_ema_50'] < -2.0):
        return 'trending_down'
    
    # Check for volatility
    elif latest['volatility_14'] > 1.5 * df['volatility_14'].mean():
        return 'volatile'
    
    # Check for ranging
    elif (abs(latest['dist_ema_50']) < 1.0 and
            abs(latest['dist_ema_100']) < 1.0 and
            latest['volatility_14'] < 0.7 * df['volatility_14'].mean()):
        return 'ranging'
    
    # Default
    return 'normal'

def generate_simulated_trades(
    pairs: List[str],
    historical_data: Dict[str, pd.DataFrame],
    strategy: str = 'momentum'
) -> List[Dict[str, Any]]:
    """
    Generate simulated trades for training the optimizer
    
    Args:
        pairs: List of trading pairs
        historical_data: Dictionary of DataFrame with historical data by pair
        strategy: Trading strategy to use
        
    Returns:
        List of simulated trades
    """
    trades = []
    trade_id = 1
    
    for pair, df in historical_data.items():
        if df.empty:
            continue
        
        # Add technical indicators
        df_indicators = add_technical_indicators(df)
        
        # Generate signals based on strategy
        if strategy == 'momentum':
            # Simple momentum strategy
            df_indicators['long_signal'] = (
                (df_indicators['close'] > df_indicators['ema_20']) &
                (df_indicators['ema_20'] > df_indicators['ema_50']) &
                (df_indicators['macd'] > df_indicators['macd_signal']) &
                (df_indicators['rsi_14'] > 50)
            )
            
            df_indicators['short_signal'] = (
                (df_indicators['close'] < df_indicators['ema_20']) &
                (df_indicators['ema_20'] < df_indicators['ema_50']) &
                (df_indicators['macd'] < df_indicators['macd_signal']) &
                (df_indicators['rsi_14'] < 50)
            )
        
        elif strategy == 'mean_reversion':
            # Mean reversion strategy
            df_indicators['long_signal'] = (
                (df_indicators['close'] < df_indicators['bb_lower']) &
                (df_indicators['rsi_14'] < 30)
            )
            
            df_indicators['short_signal'] = (
                (df_indicators['close'] > df_indicators['bb_upper']) &
                (df_indicators['rsi_14'] > 70)
            )
        
        else:
            # Default strategy (moving average crossover)
            df_indicators['long_signal'] = (
                (df_indicators['ema_20'] > df_indicators['ema_50']) &
                (df_indicators['ema_20'].shift(1) <= df_indicators['ema_50'].shift(1))
            )
            
            df_indicators['short_signal'] = (
                (df_indicators['ema_20'] < df_indicators['ema_50']) &
                (df_indicators['ema_20'].shift(1) >= df_indicators['ema_50'].shift(1))
            )
        
        # Generate trades
        in_position = False
        entry_price = 0
        entry_time = None
        direction = None
        confidence = 0
        leverage = 0
        
        for i, row in df_indicators.iterrows():
            if not in_position:
                # Check for entry signals
                if row['long_signal']:
                    in_position = True
                    entry_price = row['close']
                    entry_time = i
                    direction = 'long'
                    
                    # Calculate confidence based on signal strength
                    dist_ema_50 = row['dist_ema_50']
                    rsi = row['rsi_14']
                    
                    # Stronger signal = higher confidence
                    if dist_ema_50 > 5.0 and rsi > 60:
                        confidence = 0.9
                        leverage = 100.0
                    elif dist_ema_50 > 3.0 and rsi > 55:
                        confidence = 0.8
                        leverage = 75.0
                    else:
                        confidence = 0.7
                        leverage = 50.0
                
                elif row['short_signal']:
                    in_position = True
                    entry_price = row['close']
                    entry_time = i
                    direction = 'short'
                    
                    # Calculate confidence based on signal strength
                    dist_ema_50 = row['dist_ema_50']
                    rsi = row['rsi_14']
                    
                    # Stronger signal = higher confidence
                    if dist_ema_50 < -5.0 and rsi < 40:
                        confidence = 0.9
                        leverage = 100.0
                    elif dist_ema_50 < -3.0 and rsi < 45:
                        confidence = 0.8
                        leverage = 75.0
                    else:
                        confidence = 0.7
                        leverage = 50.0
            
            else:
                # Check for exit signals
                exit_signal = False
                
                if direction == 'long':
                    # Exit long if short signal or take profit/stop loss
                    pnl_pct = ((row['close'] / entry_price) - 1) * 100 * leverage
                    
                    if (row['short_signal'] or 
                            pnl_pct >= 15.0 or  # Take profit
                            pnl_pct <= -4.0):   # Stop loss
                        exit_signal = True
                
                elif direction == 'short':
                    # Exit short if long signal or take profit/stop loss
                    pnl_pct = ((entry_price / row['close']) - 1) * 100 * leverage
                    
                    if (row['long_signal'] or 
                            pnl_pct >= 15.0 or  # Take profit
                            pnl_pct <= -4.0):   # Stop loss
                        exit_signal = True
                
                if exit_signal:
                    # Calculate PnL
                    if direction == 'long':
                        pnl_pct = ((row['close'] / entry_price) - 1) * 100 * leverage
                        pnl_amount = 1000 * ((row['close'] / entry_price) - 1) * leverage
                    else:
                        pnl_pct = ((entry_price / row['close']) - 1) * 100 * leverage
                        pnl_amount = 1000 * ((entry_price / row['close']) - 1) * leverage
                    
                    # Determine exit reason
                    if pnl_pct >= 15.0:
                        exit_reason = 'TAKE_PROFIT'
                    elif pnl_pct <= -4.0:
                        exit_reason = 'STOP_LOSS'
                    else:
                        exit_reason = f"SIGNAL_{direction.upper()}"
                    
                    # Create trade
                    trade = {
                        'id': f"trade_{trade_id}",
                        'pair': pair,
                        'direction': direction,
                        'entry_price': entry_price,
                        'position_size': 1000.0,  # $1000 per trade
                        'leverage': leverage,
                        'entry_time': entry_time.isoformat(),
                        'exit_time': i.isoformat(),
                        'exit_price': row['close'],
                        'pnl': pnl_amount,
                        'pnl_pct': pnl_pct,
                        'status': 'closed',
                        'exit_reason': exit_reason,
                        'confidence': confidence,
                        'model': 'Adaptive' if trade_id % 2 == 0 else 'ARIMA',
                        'category': 'those dudes' if trade_id % 3 == 0 else 'him all along',
                        'stop_loss_pct': 4.0,
                        'take_profit_pct': 15.0,
                        'market_state': determine_market_state(df_indicators.loc[:i])
                    }
                    
                    trades.append(trade)
                    trade_id += 1
                    
                    # Reset position state
                    in_position = False
                    entry_price = 0
                    entry_time = None
                    direction = None
    
    return trades

def create_optimizer_training_data(trades: List[Dict[str, Any]], pairs: List[str]):
    """
    Create training data for the trade optimizer
    
    Args:
        trades: List of trades
        pairs: List of trading pairs
    """
    # Create initial optimizer data structure
    optimizer_data = {
        'market_states': {},
        'optimal_hours': {},
        'volatility_data': {},
        'pair_specific_params': {},
        'last_updated': datetime.now().isoformat()
    }
    
    # Group trades by pair
    pair_trades = {}
    for pair in pairs:
        pair_trades[pair] = [t for t in trades if t.get('pair') == pair]
    
    # Process each pair
    for pair, ptrades in pair_trades.items():
        # Default market state
        optimizer_data['market_states'][pair] = 'normal'
        
        # Default volatility data
        optimizer_data['volatility_data'][pair] = {
            'current': 0.0,
            'historical': [],
            'regime': 'normal'
        }
        
        # Default pair-specific parameters
        optimizer_data['pair_specific_params'][pair] = {
            'min_confidence': 0.65,
            'ideal_entry_volatility': 'medium',
            'min_momentum_confirmation': 2,
            'optimal_take_profit': 15.0,
            'optimal_stop_loss': 4.0,
            'trailing_stop_activation': 5.0,
            'trailing_stop_distance': 2.5
        }
        
        # Process trades for this pair
        if ptrades:
            # Calculate optimal entry/exit hours
            entry_hours = []
            exit_hours = []
            profitable_entry_hours = []
            profitable_exit_hours = []
            
            for trade in ptrades:
                try:
                    entry_time = datetime.fromisoformat(trade.get('entry_time'))
                    exit_time = datetime.fromisoformat(trade.get('exit_time')) if trade.get('exit_time') else None
                    
                    if entry_time:
                        entry_hours.append(entry_time.hour)
                    
                    if exit_time:
                        exit_hours.append(exit_time.hour)
                    
                    # Track profitable trades separately
                    if trade.get('pnl', 0) > 0:
                        if entry_time:
                            profitable_entry_hours.append(entry_time.hour)
                        
                        if exit_time:
                            profitable_exit_hours.append(exit_time.hour)
                
                except (ValueError, TypeError):
                    continue
            
            # Find optimal hours (most frequent in profitable trades)
            if profitable_entry_hours:
                # Count occurrences of each hour
                hour_counts = {h: profitable_entry_hours.count(h) for h in range(24)}
                
                # Find hours with above-average frequency
                avg_count = sum(hour_counts.values()) / 24
                optimal_entry_hours = [
                    h for h, count in hour_counts.items()
                    if count > avg_count
                ]
                
                # Default to all hours if none found
                if not optimal_entry_hours:
                    optimal_entry_hours = list(range(24))
            else:
                optimal_entry_hours = list(range(24))
            
            if profitable_exit_hours:
                # Count occurrences of each hour
                hour_counts = {h: profitable_exit_hours.count(h) for h in range(24)}
                
                # Find hours with above-average frequency
                avg_count = sum(hour_counts.values()) / 24
                optimal_exit_hours = [
                    h for h, count in hour_counts.items()
                    if count > avg_count
                ]
                
                # Default to all hours if none found
                if not optimal_exit_hours:
                    optimal_exit_hours = list(range(24))
            else:
                optimal_exit_hours = list(range(24))
            
            # Set optimal hours
            optimizer_data['optimal_hours'][pair] = {
                'entry': optimal_entry_hours,
                'exit': optimal_exit_hours
            }
            
            # Calculate optimal risk parameters
            take_profits = [t.get('take_profit_pct', 15.0) for t in ptrades if t.get('pnl', 0) > 0]
            stop_losses = [t.get('stop_loss_pct', 4.0) for t in ptrades if t.get('pnl', 0) > 0]
            
            if take_profits:
                optimal_tp = sum(take_profits) / len(take_profits)
                optimizer_data['pair_specific_params'][pair]['optimal_take_profit'] = optimal_tp
            
            if stop_losses:
                optimal_sl = sum(stop_losses) / len(stop_losses)
                optimizer_data['pair_specific_params'][pair]['optimal_stop_loss'] = optimal_sl
        
        else:
            # No trades for this pair, use default values
            optimizer_data['optimal_hours'][pair] = {
                'entry': list(range(24)),
                'exit': list(range(24))
            }
    
    # Save optimizer data
    with open(OPTIMIZATION_FILE, 'w') as f:
        json.dump(optimizer_data, f, indent=2)
    
    logger.info(f"Created optimizer training data with {len(pairs)} pairs")

def initialize_optimizer_with_historical_data(
    pairs: List[str],
    days: int = 90,
    interval: int = 60,
    save_data: bool = True,
    generate_trades: bool = True
):
    """
    Initialize the trade optimizer with historical data
    
    Args:
        pairs: List of trading pairs
        days: Number of days of historical data
        interval: Candle interval in minutes
        save_data: Whether to save data to disk
        generate_trades: Whether to generate simulated trades
    """
    create_directories()
    
    # Fetch historical data
    logger.info(f"Fetching historical data for {len(pairs)} pairs")
    historical_data = fetch_all_historical_data(pairs, days, interval, save_data)
    
    # Check if we got data
    if not historical_data:
        logger.error("No historical data fetched")
        return
    
    logger.info(f"Fetched historical data for {len(historical_data)} pairs")
    
    # Load existing trades if available
    trades = []
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, 'r') as f:
                trades = json.load(f)
            logger.info(f"Loaded {len(trades)} existing trades")
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
    
    # Generate simulated trades if requested or no existing trades
    if generate_trades or not trades:
        logger.info("Generating simulated trades")
        
        # Generate trades using different strategies
        momentum_trades = generate_simulated_trades(pairs, historical_data, 'momentum')
        mean_rev_trades = generate_simulated_trades(pairs, historical_data, 'mean_reversion')
        
        # Combine trades
        simulated_trades = momentum_trades + mean_rev_trades
        
        # Save trades if requested
        if save_data:
            # Combine with existing trades
            combined_trades = trades + simulated_trades
            
            with open(TRADES_FILE, 'w') as f:
                json.dump(combined_trades, f, indent=2)
            
            logger.info(f"Saved {len(simulated_trades)} simulated trades")
        
        # Use simulated trades
        trades = simulated_trades
    
    # Create optimizer training data
    create_optimizer_training_data(trades, pairs)
    
    logger.info("Initialized trade optimizer with historical data")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Initialize optimizer with historical data
    initialize_optimizer_with_historical_data(
        pairs=args.pairs,
        days=args.days,
        interval=args.interval,
        save_data=args.save_data,
        generate_trades=args.generate_trades
    )
    
    return 0

if __name__ == "__main__":
    main()