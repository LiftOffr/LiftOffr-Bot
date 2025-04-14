#!/usr/bin/env python3
"""
Improve Trading Strategies Based on Backtest Results

This script analyzes backtest results and improves trading strategies:
1. Analyzes performance of existing strategies (ARIMA, Adaptive)
2. Identifies weaknesses and areas for improvement
3. Implements parameter optimization for each strategy
4. Tests multiple variations of strategies on historical data
5. Selects best-performing parameter sets
6. Updates configuration files with optimized parameters

Usage:
    python improve_strategies.py --pairs SOL/USD,BTC/USD,ETH/USD,DOT/USD [--days 180]
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Any, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Default values
DEFAULT_PAIRS = ['SOL/USD', 'BTC/USD', 'ETH/USD', 'DOT/USD']
DEFAULT_DAYS = 180
DEFAULT_TIMEFRAMES = ['1h']
DEFAULT_STRATEGIES = ['arima', 'adaptive']

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Improve trading strategies based on backtest results')
    
    parser.add_argument(
        '--pairs',
        type=str,
        default=','.join(DEFAULT_PAIRS),
        help=f'Comma-separated list of trading pairs (default: {",".join(DEFAULT_PAIRS)})'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=DEFAULT_DAYS,
        help=f'Number of days for backtesting (default: {DEFAULT_DAYS})'
    )
    
    parser.add_argument(
        '--timeframes',
        type=str,
        default=','.join(DEFAULT_TIMEFRAMES),
        help=f'Comma-separated list of timeframes (default: {",".join(DEFAULT_TIMEFRAMES)})'
    )
    
    parser.add_argument(
        '--strategies',
        type=str,
        default=','.join(DEFAULT_STRATEGIES),
        help=f'Comma-separated list of strategies (default: {",".join(DEFAULT_STRATEGIES)})'
    )
    
    parser.add_argument(
        '--only-analyze',
        action='store_true',
        help='Only analyze, do not modify any files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='strategy_optimization',
        help='Directory to save optimization results (default: strategy_optimization)'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Number of optimization trials (default: 50)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )
    
    return parser.parse_args()

def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    required_modules = [
        'numpy', 'pandas', 'matplotlib'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing required dependencies: {', '.join(missing_modules)}")
        logger.error("Please install missing dependencies before running this script.")
        return False
    
    return True

def load_historical_data(pair: str, timeframe: str = '1h', days: int = 180) -> Optional[Any]:
    """
    Load historical data for the specified pair.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        timeframe: Timeframe for historical data
        days: Number of days of historical data to load
        
    Returns:
        DataFrame with historical data or None on failure
    """
    try:
        import pandas as pd
        
        # Normalize pair format
        pair_path = pair.replace('/', '')
        pair_symbol = pair.split('/')[0]
        pair_base = pair.split('/')[1]
        
        # Try to locate the dataset
        dataset_paths = [
            f"historical_data/{pair_path}_{timeframe}.csv",
            f"historical_data/{pair_symbol}/{pair_base}_{timeframe}.csv",
            f"historical_data/{pair_path.lower()}_{timeframe}.csv",
            f"training_data/{pair_path}_{timeframe}_enhanced.csv",
            f"training_data/{pair_symbol}/{pair_base}_{timeframe}_enhanced.csv"
        ]
        
        dataset_path = None
        for path in dataset_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if not dataset_path:
            logger.error(f"No historical data found for {pair} with timeframe {timeframe}")
            logger.error(f"Checked paths: {dataset_paths}")
            return None
        
        logger.info(f"Loading historical data from {dataset_path}")
        df = pd.read_csv(dataset_path)
        
        # Basic preprocessing
        # Drop rows with missing values
        df = df.dropna()
        
        # Convert timestamp to datetime if it exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Filter by days if specified
            if days > 0:
                end_date = df['timestamp'].max()
                start_date = end_date - pd.Timedelta(days=days)
                df = df[df['timestamp'] >= start_date]
        
        # Check if we have enough data
        if len(df) < 100:
            logger.warning(f"Not enough data for {pair} with timeframe {timeframe}: {len(df)} rows")
        
        logger.info(f"Loaded {len(df)} rows of historical data for {pair} with timeframe {timeframe}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading historical data for {pair}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def run_arima_backtest(
    df: Any,
    pair: str,
    arima_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run backtest with optimized ARIMA strategy.
    
    Args:
        df: DataFrame with historical data
        pair: Trading pair (e.g., 'SOL/USD')
        arima_params: Dictionary of ARIMA parameters
        
    Returns:
        Dictionary with backtest results
    """
    try:
        import numpy as np
        import pandas as pd
        
        # Default ARIMA parameters
        default_params = {
            'lookback': 32,
            'order': (2, 1, 2),
            'trend': 'c',
            'entry_threshold': 0.001,
            'exit_threshold': 0.001,
            'trailing_stop_mult': 2.0,
            'atr_periods': 14,
            'leverage': 20.0,
            'risk_pct': 0.2,
            'trend_ema_fast': 50,
            'trend_ema_slow': 100,
            'max_loss_pct': 0.04
        }
        
        # Use provided parameters or defaults
        params = {**default_params, **(arima_params or {})}
        
        # Identify price and OHLC columns
        potential_price_columns = ['close', 'Close', 'price', 'Price']
        potential_open_columns = ['open', 'Open']
        potential_high_columns = ['high', 'High']
        potential_low_columns = ['low', 'Low']
        
        price_column = None
        for col in potential_price_columns:
            if col in df.columns:
                price_column = col
                break
                
        open_column = None
        for col in potential_open_columns:
            if col in df.columns:
                open_column = col
                break
                
        high_column = None
        for col in potential_high_columns:
            if col in df.columns:
                high_column = col
                break
                
        low_column = None
        for col in potential_low_columns:
            if col in df.columns:
                low_column = col
                break
        
        if price_column is None:
            logger.error(f"Could not identify price column for {pair}")
            return {'success': False, 'error': 'Could not identify price column'}
        
        # Initialize result variables
        positions = []
        current_position = None
        initial_capital = 20000.0
        equity = initial_capital
        max_drawdown = 0
        max_equity = initial_capital
        trades = []
        
        # Calculate EMAs for trend detection
        df[f'ema_{params["trend_ema_fast"]}'] = df[price_column].ewm(span=params['trend_ema_fast'], adjust=False).mean()
        df[f'ema_{params["trend_ema_slow"]}'] = df[price_column].ewm(span=params['trend_ema_slow'], adjust=False).mean()
        
        # Calculate ATR for trailing stops
        if high_column and low_column:
            df['tr1'] = df[high_column] - df[low_column]
            df['tr2'] = abs(df[high_column] - df[price_column].shift(1))
            df['tr3'] = abs(df[low_column] - df[price_column].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=params['atr_periods']).mean()
        else:
            # Use price volatility as a substitute for ATR
            df['atr'] = df[price_column].rolling(window=params['atr_periods']).std() * 2.0
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        # Implement ARIMA strategy with optimized parameters
        lookback = params['lookback']
        
        from statsmodels.tsa.arima.model import ARIMA
        
        for i in range(lookback, len(df) - 1):
            # Get historical data for ARIMA model
            history = df[price_column].iloc[i-lookback:i].values
            
            # Get current price
            current_price = df[price_column].iloc[i]
            
            # Get ATR for trailing stop
            current_atr = df['atr'].iloc[i]
            
            # Determine market trend
            fast_ema = df[f'ema_{params["trend_ema_fast"]}'].iloc[i]
            slow_ema = df[f'ema_{params["trend_ema_slow"]}'].iloc[i]
            uptrend = fast_ema > slow_ema
            downtrend = fast_ema < slow_ema
            
            # Fit ARIMA model
            model = ARIMA(history, order=params['order'], trend=params['trend'])
            model_fit = model.fit()
            
            # Make one-step forecast
            forecast = model_fit.forecast(steps=1)[0]
            
            # Calculate price change percentage
            price_change_pct = (forecast - current_price) / current_price
            
            # Generate signal
            if current_position is None:
                # Only consider new positions if we don't have one already
                if price_change_pct > params['entry_threshold'] and (uptrend or not downtrend):
                    # Bullish forecast in uptrend or neutral trend
                    signal = 'buy'
                elif price_change_pct < -params['entry_threshold'] and (downtrend or not uptrend):
                    # Bearish forecast in downtrend or neutral trend
                    signal = 'sell'
                else:
                    signal = None
                
                if signal is not None:
                    # Open new position
                    position_size = (equity * params['risk_pct']) / (current_price / params['leverage'])
                    entry_price = current_price
                    entry_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                    
                    # Set up trailing stop
                    trailing_stop = entry_price - (current_atr * params['trailing_stop_mult']) if signal == 'buy' else entry_price + (current_atr * params['trailing_stop_mult'])
                    
                    # Set up maximum loss stop based on percentage
                    max_loss_stop = entry_price * (1 - params['max_loss_pct']) if signal == 'buy' else entry_price * (1 + params['max_loss_pct'])
                    
                    # Use the more conservative of the two stops
                    if signal == 'buy':
                        effective_stop = max(trailing_stop, max_loss_stop)
                    else:
                        effective_stop = min(trailing_stop, max_loss_stop)
                    
                    current_position = {
                        'type': signal,
                        'entry_price': entry_price,
                        'entry_time': entry_time,
                        'size': position_size,
                        'equity_at_entry': equity,
                        'trailing_stop': trailing_stop,
                        'max_loss_stop': max_loss_stop,
                        'effective_stop': effective_stop
                    }
                    
                    trades.append({
                        'pair': pair,
                        'type': signal,
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'size': position_size,
                        'stop_price': effective_stop
                    })
            
            elif current_position is not None:
                # Check if we should close position based on ARIMA forecast
                if (current_position['type'] == 'buy' and price_change_pct < -params['exit_threshold']) or \
                   (current_position['type'] == 'sell' and price_change_pct > params['exit_threshold']):
                    # Close position based on forecast
                    exit_price = current_price
                    exit_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                    exit_reason = 'forecast'
                    
                    # Calculate profit/loss
                    if current_position['type'] == 'buy':
                        pnl = (exit_price - current_position['entry_price']) / current_position['entry_price']
                    else:
                        pnl = (current_position['entry_price'] - exit_price) / current_position['entry_price']
                    
                    # Apply leverage
                    pnl *= params['leverage']
                    
                    # Apply to position size
                    pnl_amount = current_position['size'] * pnl
                    
                    # Update equity
                    equity += pnl_amount
                    
                    # Update max equity and drawdown
                    if equity > max_equity:
                        max_equity = equity
                    
                    drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    
                    # Record closed position
                    closed_position = current_position.copy()
                    closed_position.update({
                        'exit_price': exit_price,
                        'exit_time': exit_time,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount,
                        'exit_reason': exit_reason
                    })
                    positions.append(closed_position)
                    
                    # Update trade record
                    trades[-1].update({
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount,
                        'exit_reason': exit_reason
                    })
                    
                    # Reset current position
                    current_position = None
                
                # Check if we hit the trailing stop or max loss stop
                elif (current_position['type'] == 'buy' and current_price <= current_position['effective_stop']) or \
                     (current_position['type'] == 'sell' and current_price >= current_position['effective_stop']):
                    # Close position based on stop
                    exit_price = current_position['effective_stop']  # Use stop price to account for slippage
                    exit_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                    exit_reason = 'stop'
                    
                    # Calculate profit/loss
                    if current_position['type'] == 'buy':
                        pnl = (exit_price - current_position['entry_price']) / current_position['entry_price']
                    else:
                        pnl = (current_position['entry_price'] - exit_price) / current_position['entry_price']
                    
                    # Apply leverage
                    pnl *= params['leverage']
                    
                    # Apply to position size
                    pnl_amount = current_position['size'] * pnl
                    
                    # Update equity
                    equity += pnl_amount
                    
                    # Update max equity and drawdown
                    if equity > max_equity:
                        max_equity = equity
                    
                    drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    
                    # Record closed position
                    closed_position = current_position.copy()
                    closed_position.update({
                        'exit_price': exit_price,
                        'exit_time': exit_time,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount,
                        'exit_reason': exit_reason
                    })
                    positions.append(closed_position)
                    
                    # Update trade record
                    trades[-1].update({
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount,
                        'exit_reason': exit_reason
                    })
                    
                    # Reset current position
                    current_position = None
                
                # Update trailing stop for open positions
                else:
                    if current_position['type'] == 'buy' and current_price > current_position['entry_price']:
                        # Calculate new trailing stop for long position
                        new_stop = current_price - (current_atr * params['trailing_stop_mult'])
                        
                        # Only update if it's higher than the current stop
                        if new_stop > current_position['trailing_stop']:
                            current_position['trailing_stop'] = new_stop
                            
                            # Update effective stop
                            if current_position['type'] == 'buy':
                                current_position['effective_stop'] = max(current_position['trailing_stop'], current_position['max_loss_stop'])
                            else:
                                current_position['effective_stop'] = min(current_position['trailing_stop'], current_position['max_loss_stop'])
                    
                    elif current_position['type'] == 'sell' and current_price < current_position['entry_price']:
                        # Calculate new trailing stop for short position
                        new_stop = current_price + (current_atr * params['trailing_stop_mult'])
                        
                        # Only update if it's lower than the current stop
                        if new_stop < current_position['trailing_stop']:
                            current_position['trailing_stop'] = new_stop
                            
                            # Update effective stop
                            if current_position['type'] == 'buy':
                                current_position['effective_stop'] = max(current_position['trailing_stop'], current_position['max_loss_stop'])
                            else:
                                current_position['effective_stop'] = min(current_position['trailing_stop'], current_position['max_loss_stop'])
            
            # Update equity curve with mark-to-market
            if current_position is not None:
                # Calculate unrealized profit/loss
                if current_position['type'] == 'buy':
                    pnl = (current_price - current_position['entry_price']) / current_position['entry_price']
                else:
                    pnl = (current_position['entry_price'] - current_price) / current_position['entry_price']
                
                # Apply leverage
                pnl *= params['leverage']
                
                # Apply to position size
                pnl_amount = current_position['size'] * pnl
                
                # Mark-to-market equity
                mark_to_market = current_position['equity_at_entry'] + pnl_amount
                
                # Update max equity and drawdown
                if mark_to_market > max_equity:
                    max_equity = mark_to_market
                
                drawdown = (max_equity - mark_to_market) / max_equity if max_equity > 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        # Calculate performance metrics
        total_trades = len(positions)
        winning_trades = sum(1 for p in positions if p['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            
            if winning_trades > 0:
                avg_win = sum(p['pnl'] for p in positions if p['pnl'] > 0) / winning_trades
            else:
                avg_win = 0
            
            if losing_trades > 0:
                avg_loss = sum(p['pnl'] for p in positions if p['pnl'] <= 0) / losing_trades
            else:
                avg_loss = 0
            
            profit_factor = abs(sum(p['pnl'] for p in positions if p['pnl'] > 0) / sum(p['pnl'] for p in positions if p['pnl'] <= 0)) if sum(p['pnl'] for p in positions if p['pnl'] <= 0) != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        total_return = (equity - initial_capital) / initial_capital
        annualized_return = ((1 + total_return) ** (365 / len(df))) - 1 if len(df) > 0 else 0
        
        return {
            'success': True,
            'pair': pair,
            'strategy': 'arima',
            'parameters': params,
            'capital': initial_capital,
            'final_equity': equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    except Exception as e:
        logger.error(f"Error running ARIMA backtest for {pair}: {str(e)}")
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}

def run_adaptive_backtest(
    df: Any,
    pair: str,
    adaptive_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Run backtest with optimized Adaptive strategy.
    
    Args:
        df: DataFrame with historical data
        pair: Trading pair (e.g., 'SOL/USD')
        adaptive_params: Dictionary of Adaptive parameters
        
    Returns:
        Dictionary with backtest results
    """
    try:
        import numpy as np
        import pandas as pd
        
        # Default Adaptive parameters
        default_params = {
            'ema_short': 9,
            'ema_long': 21,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'adx_period': 14,
            'adx_threshold': 25,
            'volatility_period': 20,
            'volatility_threshold': 0.006,
            'bb_period': 20,
            'bb_std': 2.0,
            'keltner_period': 20,
            'keltner_atr_multiplier': 1.5,
            'atr_period': 14,
            'trailing_stop_mult': 2.0,
            'leverage': 20.0,
            'risk_pct': 0.2,
            'trend_ema_fast': 50,
            'trend_ema_slow': 100,
            'max_loss_pct': 0.04
        }
        
        # Use provided parameters or defaults
        params = {**default_params, **(adaptive_params or {})}
        
        # Identify price and OHLC columns
        potential_price_columns = ['close', 'Close', 'price', 'Price']
        potential_open_columns = ['open', 'Open']
        potential_high_columns = ['high', 'High']
        potential_low_columns = ['low', 'Low']
        
        price_column = None
        for col in potential_price_columns:
            if col in df.columns:
                price_column = col
                break
                
        open_column = None
        for col in potential_open_columns:
            if col in df.columns:
                open_column = col
                break
                
        high_column = None
        for col in potential_high_columns:
            if col in df.columns:
                high_column = col
                break
                
        low_column = None
        for col in potential_low_columns:
            if col in df.columns:
                low_column = col
                break
        
        if price_column is None:
            logger.error(f"Could not identify price column for {pair}")
            return {'success': False, 'error': 'Could not identify price column'}
        
        # Initialize result variables
        positions = []
        current_position = None
        initial_capital = 20000.0
        equity = initial_capital
        max_drawdown = 0
        max_equity = initial_capital
        trades = []
        
        # Calculate technical indicators for Adaptive strategy
        # EMAs
        df[f'ema_{params["ema_short"]}'] = df[price_column].ewm(span=params['ema_short'], adjust=False).mean()
        df[f'ema_{params["ema_long"]}'] = df[price_column].ewm(span=params['ema_long'], adjust=False).mean()
        
        # RSI
        delta = df[price_column].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=params['rsi_period']).mean()
        avg_loss = loss.rolling(window=params['rsi_period']).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_fast = df[price_column].ewm(span=params['macd_fast'], adjust=False).mean()
        ema_slow = df[price_column].ewm(span=params['macd_slow'], adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=params['macd_signal'], adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ADX
        if high_column and low_column:
            # True Range
            df['tr1'] = df[high_column] - df[low_column]
            df['tr2'] = abs(df[high_column] - df[price_column].shift(1))
            df['tr3'] = abs(df[low_column] - df[price_column].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            
            # Directional Movement
            df['plus_dm'] = np.where((df[high_column] - df[high_column].shift(1)) > (df[low_column].shift(1) - df[low_column]), 
                                      np.maximum(df[high_column] - df[high_column].shift(1), 0), 0)
            df['minus_dm'] = np.where((df[low_column].shift(1) - df[low_column]) > (df[high_column] - df[high_column].shift(1)), 
                                      np.maximum(df[low_column].shift(1) - df[low_column], 0), 0)
            
            # Directional Indicators
            df['plus_di'] = 100 * (df['plus_dm'] / df['tr']).ewm(span=params['adx_period'], adjust=False).mean()
            df['minus_di'] = 100 * (df['minus_dm'] / df['tr']).ewm(span=params['adx_period'], adjust=False).mean()
            
            # Directional Index
            df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, np.finfo(float).eps)
            df['adx'] = df['dx'].ewm(span=params['adx_period'], adjust=False).mean()
        else:
            # Simple ADX approximation using price momentum
            df['price_change'] = df[price_column].diff().abs()
            df['price_change_pct'] = df['price_change'] / df[price_column]
            df['price_change_sma'] = df['price_change_pct'].rolling(window=params['adx_period']).mean()
            df['adx'] = df['price_change_sma'] * 100  # Scale to 0-100 range
        
        # Volatility
        df['volatility'] = df[price_column].pct_change().rolling(window=params['volatility_period']).std()
        
        # Bollinger Bands
        df['bb_middle'] = df[price_column].rolling(window=params['bb_period']).mean()
        df['bb_std'] = df[price_column].rolling(window=params['bb_period']).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * params['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * params['bb_std'])
        
        # Keltner Channels
        df['kc_middle'] = df[price_column].rolling(window=params['keltner_period']).mean()
        df['atr'] = df['tr'].rolling(window=params['atr_period']).mean()
        df['kc_upper'] = df['kc_middle'] + (df['atr'] * params['keltner_atr_multiplier'])
        df['kc_lower'] = df['kc_middle'] - (df['atr'] * params['keltner_atr_multiplier'])
        
        # Trend EMAs for trend detection
        df[f'ema_{params["trend_ema_fast"]}'] = df[price_column].ewm(span=params['trend_ema_fast'], adjust=False).mean()
        df[f'ema_{params["trend_ema_slow"]}'] = df[price_column].ewm(span=params['trend_ema_slow'], adjust=False).mean()
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        # Implement Adaptive strategy with optimized parameters
        max_lookback = max(
            params['ema_short'], params['ema_long'], params['rsi_period'],
            params['macd_slow'], params['adx_period'], params['volatility_period'],
            params['bb_period'], params['keltner_period'], params['trend_ema_slow']
        )
        
        for i in range(max_lookback + 10, len(df) - 1):
            # Get current price and indicators
            current_price = df[price_column].iloc[i]
            current_atr = df['atr'].iloc[i]
            
            # Calculate signals
            ema_signal = df[f'ema_{params["ema_short"]}'].iloc[i] > df[f'ema_{params["ema_long"]}'].iloc[i]
            rsi_overbought = df['rsi'].iloc[i] > params['rsi_overbought']
            rsi_oversold = df['rsi'].iloc[i] < params['rsi_oversold']
            rsi_midrange = not rsi_overbought and not rsi_oversold
            macd_signal = df['macd'].iloc[i] > df['macd_signal'].iloc[i]
            adx_strong = df['adx'].iloc[i] > params['adx_threshold']
            volatility_low = df['volatility'].iloc[i] < params['volatility_threshold']
            
            # Bollinger Band and Keltner Channel signals
            price_above_upper_bb = current_price > df['bb_upper'].iloc[i]
            price_below_lower_bb = current_price < df['bb_lower'].iloc[i]
            price_within_bb = not price_above_upper_bb and not price_below_lower_bb
            
            price_above_kc_middle = current_price > df['kc_middle'].iloc[i]
            price_below_kc_middle = current_price < df['kc_middle'].iloc[i]
            
            # Trend detection
            fast_ema = df[f'ema_{params["trend_ema_fast"]}'].iloc[i]
            slow_ema = df[f'ema_{params["trend_ema_slow"]}'].iloc[i]
            uptrend = fast_ema > slow_ema
            downtrend = fast_ema < slow_ema
            
            # Generate adaptive signal
            signal = None
            signal_strength = 0.0
            
            # Buy conditions
            if (ema_signal and not rsi_overbought and (macd_signal or rsi_oversold) and 
                adx_strong and volatility_low and price_within_bb):
                
                if uptrend or not downtrend:  # In uptrend or neutral trend
                    signal = 'buy'
                    
                    # Calculate signal strength based on conditions
                    strength_factors = [
                        1.0 if ema_signal else 0.0,
                        0.5 if not rsi_overbought else 0.0,
                        0.7 if macd_signal else 0.0,
                        0.8 if rsi_oversold else 0.0,
                        0.6 if adx_strong else 0.0,
                        0.5 if volatility_low else 0.0,
                        0.5 if price_within_bb else 0.0,
                        0.8 if uptrend else 0.0
                    ]
                    
                    signal_strength = sum(strength_factors) / len(strength_factors)
                    signal_strength = min(max(signal_strength, 0.0), 1.0)  # Ensure between 0 and 1
            
            # Sell conditions
            elif (not ema_signal and not rsi_oversold and (not macd_signal or rsi_overbought) and 
                  adx_strong and volatility_low and price_within_bb):
                
                if downtrend or not uptrend:  # In downtrend or neutral trend
                    signal = 'sell'
                    
                    # Calculate signal strength based on conditions
                    strength_factors = [
                        1.0 if not ema_signal else 0.0,
                        0.5 if not rsi_oversold else 0.0,
                        0.7 if not macd_signal else 0.0,
                        0.8 if rsi_overbought else 0.0,
                        0.6 if adx_strong else 0.0,
                        0.5 if volatility_low else 0.0,
                        0.5 if price_within_bb else 0.0,
                        0.8 if downtrend else 0.0
                    ]
                    
                    signal_strength = sum(strength_factors) / len(strength_factors)
                    signal_strength = min(max(signal_strength, 0.0), 1.0)  # Ensure between 0 and 1
            
            # Handle position
            if current_position is None and signal is not None and signal_strength >= 0.5:
                # Open new position
                position_size = (equity * params['risk_pct']) / (current_price / params['leverage'])
                entry_price = current_price
                entry_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                
                # Set up trailing stop
                trailing_stop = entry_price - (current_atr * params['trailing_stop_mult']) if signal == 'buy' else entry_price + (current_atr * params['trailing_stop_mult'])
                
                # Set up maximum loss stop based on percentage
                max_loss_stop = entry_price * (1 - params['max_loss_pct']) if signal == 'buy' else entry_price * (1 + params['max_loss_pct'])
                
                # Use the more conservative of the two stops
                if signal == 'buy':
                    effective_stop = max(trailing_stop, max_loss_stop)
                else:
                    effective_stop = min(trailing_stop, max_loss_stop)
                
                current_position = {
                    'type': signal,
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'size': position_size,
                    'equity_at_entry': equity,
                    'trailing_stop': trailing_stop,
                    'max_loss_stop': max_loss_stop,
                    'effective_stop': effective_stop,
                    'signal_strength': signal_strength
                }
                
                trades.append({
                    'pair': pair,
                    'type': signal,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'size': position_size,
                    'stop_price': effective_stop,
                    'signal_strength': signal_strength
                })
            
            elif current_position is not None:
                # Check if we should close position based on opposite signal
                if (current_position['type'] == 'buy' and signal == 'sell' and signal_strength >= 0.6) or \
                   (current_position['type'] == 'sell' and signal == 'buy' and signal_strength >= 0.6):
                    # Close position based on opposite signal
                    exit_price = current_price
                    exit_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                    exit_reason = 'signal'
                    
                    # Calculate profit/loss
                    if current_position['type'] == 'buy':
                        pnl = (exit_price - current_position['entry_price']) / current_position['entry_price']
                    else:
                        pnl = (current_position['entry_price'] - exit_price) / current_position['entry_price']
                    
                    # Apply leverage
                    pnl *= params['leverage']
                    
                    # Apply to position size
                    pnl_amount = current_position['size'] * pnl
                    
                    # Update equity
                    equity += pnl_amount
                    
                    # Update max equity and drawdown
                    if equity > max_equity:
                        max_equity = equity
                    
                    drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    
                    # Record closed position
                    closed_position = current_position.copy()
                    closed_position.update({
                        'exit_price': exit_price,
                        'exit_time': exit_time,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount,
                        'exit_reason': exit_reason,
                        'exit_signal_strength': signal_strength
                    })
                    positions.append(closed_position)
                    
                    # Update trade record
                    trades[-1].update({
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount,
                        'exit_reason': exit_reason,
                        'exit_signal_strength': signal_strength
                    })
                    
                    # Reset current position
                    current_position = None
                
                # Check if we hit the trailing stop or max loss stop
                elif (current_position['type'] == 'buy' and current_price <= current_position['effective_stop']) or \
                     (current_position['type'] == 'sell' and current_price >= current_position['effective_stop']):
                    # Close position based on stop
                    exit_price = current_position['effective_stop']  # Use stop price to account for slippage
                    exit_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                    exit_reason = 'stop'
                    
                    # Calculate profit/loss
                    if current_position['type'] == 'buy':
                        pnl = (exit_price - current_position['entry_price']) / current_position['entry_price']
                    else:
                        pnl = (current_position['entry_price'] - exit_price) / current_position['entry_price']
                    
                    # Apply leverage
                    pnl *= params['leverage']
                    
                    # Apply to position size
                    pnl_amount = current_position['size'] * pnl
                    
                    # Update equity
                    equity += pnl_amount
                    
                    # Update max equity and drawdown
                    if equity > max_equity:
                        max_equity = equity
                    
                    drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    
                    # Record closed position
                    closed_position = current_position.copy()
                    closed_position.update({
                        'exit_price': exit_price,
                        'exit_time': exit_time,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount,
                        'exit_reason': exit_reason
                    })
                    positions.append(closed_position)
                    
                    # Update trade record
                    trades[-1].update({
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount,
                        'exit_reason': exit_reason
                    })
                    
                    # Reset current position
                    current_position = None
                
                # Update trailing stop for open positions
                else:
                    if current_position['type'] == 'buy' and current_price > current_position['entry_price']:
                        # Calculate new trailing stop for long position
                        new_stop = current_price - (current_atr * params['trailing_stop_mult'])
                        
                        # Only update if it's higher than the current stop
                        if new_stop > current_position['trailing_stop']:
                            current_position['trailing_stop'] = new_stop
                            
                            # Update effective stop
                            if current_position['type'] == 'buy':
                                current_position['effective_stop'] = max(current_position['trailing_stop'], current_position['max_loss_stop'])
                            else:
                                current_position['effective_stop'] = min(current_position['trailing_stop'], current_position['max_loss_stop'])
                    
                    elif current_position['type'] == 'sell' and current_price < current_position['entry_price']:
                        # Calculate new trailing stop for short position
                        new_stop = current_price + (current_atr * params['trailing_stop_mult'])
                        
                        # Only update if it's lower than the current stop
                        if new_stop < current_position['trailing_stop']:
                            current_position['trailing_stop'] = new_stop
                            
                            # Update effective stop
                            if current_position['type'] == 'buy':
                                current_position['effective_stop'] = max(current_position['trailing_stop'], current_position['max_loss_stop'])
                            else:
                                current_position['effective_stop'] = min(current_position['trailing_stop'], current_position['max_loss_stop'])
            
            # Update equity curve with mark-to-market
            if current_position is not None:
                # Calculate unrealized profit/loss
                if current_position['type'] == 'buy':
                    pnl = (current_price - current_position['entry_price']) / current_position['entry_price']
                else:
                    pnl = (current_position['entry_price'] - current_price) / current_position['entry_price']
                
                # Apply leverage
                pnl *= params['leverage']
                
                # Apply to position size
                pnl_amount = current_position['size'] * pnl
                
                # Mark-to-market equity
                mark_to_market = current_position['equity_at_entry'] + pnl_amount
                
                # Update max equity and drawdown
                if mark_to_market > max_equity:
                    max_equity = mark_to_market
                
                drawdown = (max_equity - mark_to_market) / max_equity if max_equity > 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        
        # Calculate performance metrics
        total_trades = len(positions)
        winning_trades = sum(1 for p in positions if p['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        
        if total_trades > 0:
            win_rate = winning_trades / total_trades
            
            if winning_trades > 0:
                avg_win = sum(p['pnl'] for p in positions if p['pnl'] > 0) / winning_trades
            else:
                avg_win = 0
            
            if losing_trades > 0:
                avg_loss = sum(p['pnl'] for p in positions if p['pnl'] <= 0) / losing_trades
            else:
                avg_loss = 0
            
            profit_factor = abs(sum(p['pnl'] for p in positions if p['pnl'] > 0) / sum(p['pnl'] for p in positions if p['pnl'] <= 0)) if sum(p['pnl'] for p in positions if p['pnl'] <= 0) != 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        total_return = (equity - initial_capital) / initial_capital
        annualized_return = ((1 + total_return) ** (365 / len(df))) - 1 if len(df) > 0 else 0
        
        return {
            'success': True,
            'pair': pair,
            'strategy': 'adaptive',
            'parameters': params,
            'capital': initial_capital,
            'final_equity': equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    except Exception as e:
        logger.error(f"Error running Adaptive backtest for {pair}: {str(e)}")
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}

def perform_arima_parameter_optimization(df: Any, pair: str, trials: int) -> Dict[str, Any]:
    """
    Perform parameter optimization for ARIMA strategy.
    
    Args:
        df: DataFrame with historical data
        pair: Trading pair (e.g., 'SOL/USD')
        trials: Number of optimization trials
        
    Returns:
        Dictionary with optimization results
    """
    try:
        import random
        
        # Parameter ranges
        param_ranges = {
            'lookback': [24, 32, 48, 64, 96],
            'order_p': [1, 2, 3, 4],
            'order_d': [0, 1, 2],
            'order_q': [0, 1, 2, 3],
            'trend': ['n', 'c', 't', 'ct'],
            'entry_threshold': [0.0005, 0.001, 0.002, 0.003, 0.005],
            'exit_threshold': [0.0005, 0.001, 0.002, 0.003, 0.005],
            'trailing_stop_mult': [1.5, 2.0, 2.5, 3.0, 4.0],
            'atr_periods': [10, 14, 20, 30],
            'leverage': [10.0, 15.0, 20.0, 25.0, 30.0],
            'risk_pct': [0.1, 0.15, 0.2, 0.25, 0.3],
            'trend_ema_fast': [20, 35, 50, 75],
            'trend_ema_slow': [80, 100, 120, 150],
            'max_loss_pct': [0.02, 0.03, 0.04, 0.05, 0.06]
        }
        
        best_result = None
        best_return = -float('inf')
        
        # Run optimization trials
        for trial in range(trials):
            # Sample parameters
            params = {
                'lookback': random.choice(param_ranges['lookback']),
                'order': (
                    random.choice(param_ranges['order_p']),
                    random.choice(param_ranges['order_d']),
                    random.choice(param_ranges['order_q'])
                ),
                'trend': random.choice(param_ranges['trend']),
                'entry_threshold': random.choice(param_ranges['entry_threshold']),
                'exit_threshold': random.choice(param_ranges['exit_threshold']),
                'trailing_stop_mult': random.choice(param_ranges['trailing_stop_mult']),
                'atr_periods': random.choice(param_ranges['atr_periods']),
                'leverage': random.choice(param_ranges['leverage']),
                'risk_pct': random.choice(param_ranges['risk_pct']),
                'trend_ema_fast': random.choice(param_ranges['trend_ema_fast']),
                'trend_ema_slow': random.choice(param_ranges['trend_ema_slow']),
                'max_loss_pct': random.choice(param_ranges['max_loss_pct'])
            }
            
            # Run backtest with current parameters
            result = run_arima_backtest(df, pair, params)
            
            if result['success'] and result['total_return'] > best_return:
                best_result = result
                best_return = result['total_return']
                
                logger.info(f"Trial {trial+1}/{trials}: Found better parameters for {pair} ARIMA strategy")
                logger.info(f"  Return: {result['total_return_pct']:.2f}%, Win Rate: {result['win_rate_pct']:.2f}%, Trades: {result['total_trades']}")
            
            # Log progress
            if (trial + 1) % 10 == 0:
                logger.info(f"Completed {trial+1}/{trials} trials for {pair} ARIMA optimization")
        
        if best_result:
            logger.info(f"Optimization completed for {pair} ARIMA strategy")
            logger.info(f"  Best Return: {best_result['total_return_pct']:.2f}%")
            logger.info(f"  Win Rate: {best_result['win_rate_pct']:.2f}%")
            logger.info(f"  Profit Factor: {best_result['profit_factor']:.2f}")
            logger.info(f"  Max Drawdown: {best_result['max_drawdown_pct']:.2f}%")
            logger.info(f"  Total Trades: {best_result['total_trades']}")
        
        return best_result
    
    except Exception as e:
        logger.error(f"Error optimizing ARIMA parameters for {pair}: {str(e)}")
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}

def perform_adaptive_parameter_optimization(df: Any, pair: str, trials: int) -> Dict[str, Any]:
    """
    Perform parameter optimization for Adaptive strategy.
    
    Args:
        df: DataFrame with historical data
        pair: Trading pair (e.g., 'SOL/USD')
        trials: Number of optimization trials
        
    Returns:
        Dictionary with optimization results
    """
    try:
        import random
        
        # Parameter ranges
        param_ranges = {
            'ema_short': [5, 7, 9, 11, 13],
            'ema_long': [17, 21, 25, 30],
            'rsi_period': [10, 14, 20],
            'rsi_overbought': [65, 70, 75, 80],
            'rsi_oversold': [20, 25, 30, 35],
            'macd_fast': [8, 12, 16],
            'macd_slow': [20, 26, 30],
            'macd_signal': [7, 9, 11],
            'adx_period': [10, 14, 20],
            'adx_threshold': [20, 25, 30],
            'volatility_period': [15, 20, 25],
            'volatility_threshold': [0.004, 0.006, 0.008, 0.01],
            'bb_period': [15, 20, 25],
            'bb_std': [1.5, 2.0, 2.5],
            'keltner_period': [15, 20, 25],
            'keltner_atr_multiplier': [1.0, 1.5, 2.0],
            'atr_period': [10, 14, 20],
            'trailing_stop_mult': [1.5, 2.0, 2.5, 3.0],
            'leverage': [10.0, 15.0, 20.0, 25.0, 30.0],
            'risk_pct': [0.1, 0.15, 0.2, 0.25, 0.3],
            'trend_ema_fast': [20, 35, 50, 75],
            'trend_ema_slow': [80, 100, 120, 150],
            'max_loss_pct': [0.02, 0.03, 0.04, 0.05, 0.06]
        }
        
        best_result = None
        best_return = -float('inf')
        
        # Run optimization trials
        for trial in range(trials):
            # Sample parameters
            params = {
                'ema_short': random.choice(param_ranges['ema_short']),
                'ema_long': random.choice(param_ranges['ema_long']),
                'rsi_period': random.choice(param_ranges['rsi_period']),
                'rsi_overbought': random.choice(param_ranges['rsi_overbought']),
                'rsi_oversold': random.choice(param_ranges['rsi_oversold']),
                'macd_fast': random.choice(param_ranges['macd_fast']),
                'macd_slow': random.choice(param_ranges['macd_slow']),
                'macd_signal': random.choice(param_ranges['macd_signal']),
                'adx_period': random.choice(param_ranges['adx_period']),
                'adx_threshold': random.choice(param_ranges['adx_threshold']),
                'volatility_period': random.choice(param_ranges['volatility_period']),
                'volatility_threshold': random.choice(param_ranges['volatility_threshold']),
                'bb_period': random.choice(param_ranges['bb_period']),
                'bb_std': random.choice(param_ranges['bb_std']),
                'keltner_period': random.choice(param_ranges['keltner_period']),
                'keltner_atr_multiplier': random.choice(param_ranges['keltner_atr_multiplier']),
                'atr_period': random.choice(param_ranges['atr_period']),
                'trailing_stop_mult': random.choice(param_ranges['trailing_stop_mult']),
                'leverage': random.choice(param_ranges['leverage']),
                'risk_pct': random.choice(param_ranges['risk_pct']),
                'trend_ema_fast': random.choice(param_ranges['trend_ema_fast']),
                'trend_ema_slow': random.choice(param_ranges['trend_ema_slow']),
                'max_loss_pct': random.choice(param_ranges['max_loss_pct'])
            }
            
            # Run backtest with current parameters
            result = run_adaptive_backtest(df, pair, params)
            
            if result['success'] and result['total_return'] > best_return:
                best_result = result
                best_return = result['total_return']
                
                logger.info(f"Trial {trial+1}/{trials}: Found better parameters for {pair} Adaptive strategy")
                logger.info(f"  Return: {result['total_return_pct']:.2f}%, Win Rate: {result['win_rate_pct']:.2f}%, Trades: {result['total_trades']}")
            
            # Log progress
            if (trial + 1) % 10 == 0:
                logger.info(f"Completed {trial+1}/{trials} trials for {pair} Adaptive optimization")
        
        if best_result:
            logger.info(f"Optimization completed for {pair} Adaptive strategy")
            logger.info(f"  Best Return: {best_result['total_return_pct']:.2f}%")
            logger.info(f"  Win Rate: {best_result['win_rate_pct']:.2f}%")
            logger.info(f"  Profit Factor: {best_result['profit_factor']:.2f}")
            logger.info(f"  Max Drawdown: {best_result['max_drawdown_pct']:.2f}%")
            logger.info(f"  Total Trades: {best_result['total_trades']}")
        
        return best_result
    
    except Exception as e:
        logger.error(f"Error optimizing Adaptive parameters for {pair}: {str(e)}")
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}

def save_optimization_results(results: Dict[str, Any], output_dir: str) -> bool:
    """
    Save optimization results to files.
    
    Args:
        results: Dictionary with optimization results
        output_dir: Directory to save results
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to JSON file
        results_path = os.path.join(output_dir, 'optimization_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to {results_path}")
        
        # Generate summary CSV
        import pandas as pd
        
        summary_data = []
        for pair, pair_results in results.items():
            for strategy, strategy_result in pair_results.items():
                if strategy_result['success']:
                    summary_data.append({
                        'Pair': pair,
                        'Strategy': strategy,
                        'Return (%)': strategy_result['total_return_pct'],
                        'Annual Return (%)': strategy_result['annualized_return_pct'],
                        'Win Rate (%)': strategy_result['win_rate_pct'],
                        'Profit Factor': strategy_result['profit_factor'],
                        'Max Drawdown (%)': strategy_result['max_drawdown_pct'],
                        'Total Trades': strategy_result['total_trades']
                    })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(output_dir, 'optimization_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Optimization summary saved to {summary_path}")
        
        # Create parameter files for each strategy
        for pair, pair_results in results.items():
            for strategy, strategy_result in pair_results.items():
                if strategy_result['success'] and 'parameters' in strategy_result:
                    # Extract parameters
                    params = strategy_result['parameters']
                    
                    # Create parameter file for strategy
                    if strategy == 'arima':
                        params_path = os.path.join(output_dir, f'{pair.replace("/", "")}_arima_params.json')
                    elif strategy == 'adaptive':
                        params_path = os.path.join(output_dir, f'{pair.replace("/", "")}_adaptive_params.json')
                    else:
                        continue
                    
                    with open(params_path, 'w') as f:
                        json.dump(params, f, indent=2)
                    
                    logger.info(f"Parameters for {pair} {strategy} strategy saved to {params_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving optimization results: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def update_strategy_configurations(results: Dict[str, Any], only_analyze: bool = False) -> bool:
    """
    Update strategy configuration files with optimized parameters.
    
    Args:
        results: Dictionary with optimization results
        only_analyze: Only analyze, do not modify files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if only_analyze:
            logger.info("Skipping configuration updates (--only-analyze flag set)")
            return True
        
        # Check for ARIMA strategy file
        arima_files = [
            'arima_strategy.py',
            'strategies/arima_strategy.py',
            'fixed_strategy.py'
        ]
        
        arima_file = None
        for file_path in arima_files:
            if os.path.exists(file_path):
                arima_file = file_path
                break
        
        # Check for Adaptive strategy file
        adaptive_files = [
            'adaptive_strategy.py',
            'strategies/adaptive_strategy.py',
            'integrated_strategy.py'
        ]
        
        adaptive_file = None
        for file_path in adaptive_files:
            if os.path.exists(file_path):
                adaptive_file = file_path
                break
        
        # Prepare combined parameters across all pairs
        combined_arima_params = {}
        combined_adaptive_params = {}
        
        for pair, pair_results in results.items():
            # Extract ARIMA parameters
            if 'arima' in pair_results and pair_results['arima']['success']:
                for param, value in pair_results['arima']['parameters'].items():
                    if param not in combined_arima_params:
                        combined_arima_params[param] = {}
                    
                    if value not in combined_arima_params[param]:
                        combined_arima_params[param][value] = 1
                    else:
                        combined_arima_params[param][value] += 1
            
            # Extract Adaptive parameters
            if 'adaptive' in pair_results and pair_results['adaptive']['success']:
                for param, value in pair_results['adaptive']['parameters'].items():
                    if param not in combined_adaptive_params:
                        combined_adaptive_params[param] = {}
                    
                    if value not in combined_adaptive_params[param]:
                        combined_adaptive_params[param][value] = 1
                    else:
                        combined_adaptive_params[param][value] += 1
        
        # Find most common value for each parameter
        best_arima_params = {}
        for param, values in combined_arima_params.items():
            best_value = max(values.items(), key=lambda x: x[1])[0]
            best_arima_params[param] = best_value
        
        best_adaptive_params = {}
        for param, values in combined_adaptive_params.items():
            best_value = max(values.items(), key=lambda x: x[1])[0]
            best_adaptive_params[param] = best_value
        
        # Create config files with optimized parameters
        if best_arima_params:
            config_path = 'arima_strategy_config.json'
            with open(config_path, 'w') as f:
                json.dump(best_arima_params, f, indent=2)
            
            logger.info(f"ARIMA strategy configuration saved to {config_path}")
        
        if best_adaptive_params:
            config_path = 'adaptive_strategy_config.json'
            with open(config_path, 'w') as f:
                json.dump(best_adaptive_params, f, indent=2)
            
            logger.info(f"Adaptive strategy configuration saved to {config_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error updating strategy configurations: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def generate_optimization_report(results: Dict[str, Any], output_dir: str) -> bool:
    """
    Generate a detailed optimization report.
    
    Args:
        results: Dictionary with optimization results
        output_dir: Directory to save report
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create plots
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # Prepare data for plots
        pairs = sorted(list(results.keys()))
        strategies = ['arima', 'adaptive']
        returns = []
        win_rates = []
        profit_factors = []
        drawdowns = []
        
        for pair in pairs:
            pair_returns = []
            pair_win_rates = []
            pair_profit_factors = []
            pair_drawdowns = []
            
            for strategy in strategies:
                if strategy in results[pair] and results[pair][strategy]['success']:
                    pair_returns.append(results[pair][strategy]['total_return_pct'])
                    pair_win_rates.append(results[pair][strategy]['win_rate_pct'])
                    pair_profit_factors.append(results[pair][strategy]['profit_factor'])
                    pair_drawdowns.append(results[pair][strategy]['max_drawdown_pct'])
                else:
                    pair_returns.append(0)
                    pair_win_rates.append(0)
                    pair_profit_factors.append(0)
                    pair_drawdowns.append(0)
            
            returns.append(pair_returns)
            win_rates.append(pair_win_rates)
            profit_factors.append(pair_profit_factors)
            drawdowns.append(pair_drawdowns)
        
        # Convert to numpy arrays
        returns = np.array(returns)
        win_rates = np.array(win_rates)
        profit_factors = np.array(profit_factors)
        drawdowns = np.array(drawdowns)
        
        # Plot returns
        plt.figure(figsize=(12, 6))
        x = np.arange(len(pairs))
        width = 0.35
        
        plt.bar(x - width/2, returns[:, 0], width, label='ARIMA')
        plt.bar(x + width/2, returns[:, 1], width, label='Adaptive')
        
        plt.xlabel('Pair')
        plt.ylabel('Return (%)')
        plt.title('Strategy Returns by Pair')
        plt.xticks(x, pairs)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        returns_path = os.path.join(output_dir, 'strategy_returns.png')
        plt.savefig(returns_path)
        plt.close()
        
        # Plot win rates
        plt.figure(figsize=(12, 6))
        
        plt.bar(x - width/2, win_rates[:, 0], width, label='ARIMA')
        plt.bar(x + width/2, win_rates[:, 1], width, label='Adaptive')
        
        plt.xlabel('Pair')
        plt.ylabel('Win Rate (%)')
        plt.title('Strategy Win Rates by Pair')
        plt.xticks(x, pairs)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        win_rates_path = os.path.join(output_dir, 'strategy_win_rates.png')
        plt.savefig(win_rates_path)
        plt.close()
        
        # Plot profit factors
        plt.figure(figsize=(12, 6))
        
        # Cap profit factors for better visualization
        capped_profit_factors = np.minimum(profit_factors, 5)
        
        plt.bar(x - width/2, capped_profit_factors[:, 0], width, label='ARIMA')
        plt.bar(x + width/2, capped_profit_factors[:, 1], width, label='Adaptive')
        
        plt.xlabel('Pair')
        plt.ylabel('Profit Factor (capped at 5)')
        plt.title('Strategy Profit Factors by Pair')
        plt.xticks(x, pairs)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        profit_factors_path = os.path.join(output_dir, 'strategy_profit_factors.png')
        plt.savefig(profit_factors_path)
        plt.close()
        
        # Plot drawdowns
        plt.figure(figsize=(12, 6))
        
        plt.bar(x - width/2, drawdowns[:, 0], width, label='ARIMA')
        plt.bar(x + width/2, drawdowns[:, 1], width, label='Adaptive')
        
        plt.xlabel('Pair')
        plt.ylabel('Max Drawdown (%)')
        plt.title('Strategy Max Drawdowns by Pair')
        plt.xticks(x, pairs)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        drawdowns_path = os.path.join(output_dir, 'strategy_drawdowns.png')
        plt.savefig(drawdowns_path)
        plt.close()
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategy Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .chart-container {{ display: flex; justify-content: center; margin: 20px 0; }}
                .chart {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Strategy Optimization Report</h1>
            <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary of Results</h2>
            <table>
                <tr>
                    <th>Pair</th>
                    <th>Strategy</th>
                    <th>Return (%)</th>
                    <th>Win Rate (%)</th>
                    <th>Profit Factor</th>
                    <th>Max Drawdown (%)</th>
                    <th>Trades</th>
                </tr>
        """
        
        for pair in pairs:
            for strategy in strategies:
                if strategy in results[pair] and results[pair][strategy]['success']:
                    r = results[pair][strategy]
                    html_content += f"""
                    <tr>
                        <td>{pair}</td>
                        <td>{strategy.upper()}</td>
                        <td>{r['total_return_pct']:.2f}</td>
                        <td>{r['win_rate_pct']:.2f}</td>
                        <td>{r['profit_factor']:.2f}</td>
                        <td>{r['max_drawdown_pct']:.2f}</td>
                        <td>{r['total_trades']}</td>
                    </tr>
                    """
        
        html_content += """
            </table>
            
            <h2>Performance Charts</h2>
            
            <h3>Strategy Returns</h3>
            <div class="chart-container">
                <img class="chart" src="strategy_returns.png" alt="Strategy Returns">
            </div>
            
            <h3>Strategy Win Rates</h3>
            <div class="chart-container">
                <img class="chart" src="strategy_win_rates.png" alt="Strategy Win Rates">
            </div>
            
            <h3>Strategy Profit Factors</h3>
            <div class="chart-container">
                <img class="chart" src="strategy_profit_factors.png" alt="Strategy Profit Factors">
            </div>
            
            <h3>Strategy Max Drawdowns</h3>
            <div class="chart-container">
                <img class="chart" src="strategy_drawdowns.png" alt="Strategy Max Drawdowns">
            </div>
            
            <h2>Optimized Parameters</h2>
        """
        
        for pair in pairs:
            html_content += f"<h3>{pair}</h3>"
            
            for strategy in strategies:
                if strategy in results[pair] and results[pair][strategy]['success'] and 'parameters' in results[pair][strategy]:
                    html_content += f"<h4>{strategy.upper()} Strategy</h4>"
                    html_content += "<table>"
                    html_content += "<tr><th>Parameter</th><th>Value</th></tr>"
                    
                    for param, value in results[pair][strategy]['parameters'].items():
                        html_content += f"<tr><td>{param}</td><td>{value}</td></tr>"
                    
                    html_content += "</table>"
        
        html_content += """
            <h2>Conclusions and Recommendations</h2>
            <p>
                Based on the optimization results, the following improvements are recommended:
                <ul>
                    <li>Use pair-specific parameters for best performance</li>
                    <li>Adjust leverage based on pair volatility</li>
                    <li>Use tighter stops for more volatile pairs</li>
                    <li>Implement signal strength filtering for higher win rates</li>
                    <li>Consider trend-following in strong trends and mean-reversion in ranges</li>
                </ul>
            </p>
        </body>
        </html>
        """
        
        # Save HTML report
        html_path = os.path.join(output_dir, 'optimization_report.html')
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Optimization report saved to {html_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating optimization report: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def main():
    """Main function"""
    args = parse_arguments()
    
    # Parse pairs list
    pairs = [pair.strip() for pair in args.pairs.split(',')]
    
    # Parse timeframes list
    timeframes = [tf.strip() for tf in args.timeframes.split(',')]
    
    # Parse strategies list
    strategies = [strategy.strip() for strategy in args.strategies.split(',')]
    
    logger.info("=" * 80)
    logger.info("IMPROVE TRADING STRATEGIES")
    logger.info("=" * 80)
    logger.info(f"Trading pairs: {', '.join(pairs)}")
    logger.info(f"Timeframes: {', '.join(timeframes)}")
    logger.info(f"Strategies: {', '.join(strategies)}")
    logger.info(f"Days for backtesting: {args.days}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Optimization trials: {args.trials}")
    logger.info(f"Only analyze: {args.only_analyze}")
    logger.info(f"Verbose: {args.verbose}")
    logger.info("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies. Please install required packages and try again.")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize results dictionary
    optimization_results = {}
    
    # Process each pair
    for pair in pairs:
        optimization_results[pair] = {}
        
        logger.info("-" * 80)
        logger.info(f"Processing {pair}")
        logger.info("-" * 80)
        
        # Only use first timeframe for now
        timeframe = timeframes[0]
        
        # Load historical data
        df = load_historical_data(pair, timeframe, args.days)
        if df is None:
            logger.error(f"Failed to load historical data for {pair}")
            continue
        
        # Optimize each strategy
        for strategy in strategies:
            logger.info(f"Optimizing {strategy} strategy for {pair}")
            
            if strategy == 'arima':
                # Run ARIMA optimization
                result = perform_arima_parameter_optimization(df, pair, args.trials)
                if result and result['success']:
                    optimization_results[pair]['arima'] = result
            
            elif strategy == 'adaptive':
                # Run Adaptive optimization
                result = perform_adaptive_parameter_optimization(df, pair, args.trials)
                if result and result['success']:
                    optimization_results[pair]['adaptive'] = result
    
    # Save optimization results
    save_optimization_results(optimization_results, args.output_dir)
    
    # Update strategy configurations
    update_strategy_configurations(optimization_results, args.only_analyze)
    
    # Generate optimization report
    generate_optimization_report(optimization_results, args.output_dir)
    
    # Log summary
    logger.info("=" * 80)
    logger.info("STRATEGY OPTIMIZATION COMPLETED")
    logger.info("=" * 80)
    
    # Print recommendations
    logger.info("Recommendations for improving strategies:")
    
    # Analyze which strategy performed better overall
    total_arima_return = 0
    total_adaptive_return = 0
    arima_count = 0
    adaptive_count = 0
    
    for pair, pair_results in optimization_results.items():
        if 'arima' in pair_results and pair_results['arima']['success']:
            total_arima_return += pair_results['arima']['total_return']
            arima_count += 1
        
        if 'adaptive' in pair_results and pair_results['adaptive']['success']:
            total_adaptive_return += pair_results['adaptive']['total_return']
            adaptive_count += 1
    
    avg_arima_return = total_arima_return / arima_count if arima_count > 0 else 0
    avg_adaptive_return = total_adaptive_return / adaptive_count if adaptive_count > 0 else 0
    
    if avg_arima_return > avg_adaptive_return:
        logger.info("1. ARIMA strategy performed better overall. Consider increasing its allocation.")
    else:
        logger.info("1. Adaptive strategy performed better overall. Consider increasing its allocation.")
    
    logger.info("2. Use pair-specific parameters for best performance")
    logger.info("3. Adjust leverage based on pair volatility")
    logger.info("4. Use tighter stops for more volatile pairs")
    logger.info("5. Implement signal strength filtering for higher win rates")
    logger.info("6. Consider trend-following in strong trends and mean-reversion in ranges")
    logger.info("7. Check configuration files for optimized parameters")
    
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())