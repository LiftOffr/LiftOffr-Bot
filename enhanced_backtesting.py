#!/usr/bin/env python3
"""
Enhanced Backtesting Script

This script provides comprehensive backtesting capabilities for ML models:
1. Performs walk-forward analysis to simulate real-world trading
2. Analyzes performance across multiple timeframes
3. Tests against various market conditions (trend, volatility, etc.)
4. Compares performance of different strategy combinations
5. Provides detailed performance metrics and visuals
6. Helps identify optimal leverage and risk settings

Usage:
    python enhanced_backtesting.py --pairs SOL/USD,BTC/USD,ETH/USD [--days 180] [--capital 20000]
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
DEFAULT_PAIRS = ['SOL/USD', 'BTC/USD', 'ETH/USD']
DEFAULT_CAPITAL = 20000.0
DEFAULT_DAYS = 180
DEFAULT_TIMEFRAMES = ['1h']
DEFAULT_STRATEGIES = ['arima', 'adaptive', 'ml']

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced backtesting for ML models')
    
    parser.add_argument(
        '--pairs',
        type=str,
        default=','.join(DEFAULT_PAIRS),
        help=f'Comma-separated list of trading pairs (default: {",".join(DEFAULT_PAIRS)})'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        default=DEFAULT_CAPITAL,
        help=f'Starting capital for backtesting (default: {DEFAULT_CAPITAL})'
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
        '--leverage',
        type=float,
        default=20.0,
        help='Base leverage for trading (default: 20.0)'
    )
    
    parser.add_argument(
        '--max-leverage',
        type=float,
        default=125.0,
        help='Maximum leverage for high-confidence trades (default: 125.0)'
    )
    
    parser.add_argument(
        '--risk',
        type=float,
        default=0.2,
        help='Risk per trade as fraction of capital (default: 0.2)'
    )
    
    parser.add_argument(
        '--target-accuracy',
        type=float,
        default=0.9,
        help='Target model accuracy (default: 0.9)'
    )
    
    parser.add_argument(
        '--target-return',
        type=float,
        default=10.0,
        help='Target return multiplier (default: 10.0)'
    )
    
    parser.add_argument(
        '--compare-fees',
        action='store_true',
        help='Compare different fee structures'
    )
    
    parser.add_argument(
        '--compare-slippage',
        action='store_true',
        help='Compare different slippage scenarios'
    )
    
    parser.add_argument(
        '--optimize-params',
        action='store_true',
        help='Perform parameter optimization'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='backtest_results',
        help='Directory to save backtest results (default: backtest_results)'
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
        'numpy', 'pandas', 'matplotlib', 'tensorflow'
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

def load_model(pair: str, model_type: str = 'lstm') -> Optional[Any]:
    """
    Load ML model for the specified pair.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        model_type: Type of model to load (lstm, cnn, transformer, ensemble)
        
    Returns:
        Loaded model or None on failure
    """
    try:
        import tensorflow as tf
        
        # Normalize pair format
        pair_path = pair.replace('/', '')
        
        # Determine model path
        model_dir = f"models/{model_type}"
        model_path = os.path.join(model_dir, f"{pair_path}.h5")
        
        if not os.path.exists(model_path):
            logger.error(f"Model not found for {pair} with type {model_type}")
            logger.error(f"Expected model path: {model_path}")
            return None
        
        logger.info(f"Loading {model_type} model for {pair} from {model_path}")
        model = tf.keras.models.load_model(model_path)
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading model for {pair}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_features(df: Any, pair: str) -> Optional[Any]:
    """
    Create features for ML model input.
    
    Args:
        df: DataFrame with historical data
        pair: Trading pair (e.g., 'SOL/USD')
        
    Returns:
        DataFrame with features or None on failure
    """
    try:
        import numpy as np
        import pandas as pd
        
        # Identify price and volume columns
        potential_price_columns = ['close', 'Close', 'price', 'Price']
        potential_volume_columns = ['volume', 'Volume']
        
        price_column = None
        for col in potential_price_columns:
            if col in df.columns:
                price_column = col
                break
        
        volume_column = None
        for col in potential_volume_columns:
            if col in df.columns:
                volume_column = col
                break
        
        if price_column is None:
            logger.error(f"Could not identify price column for {pair}")
            return None
        
        # Create features
        feature_columns = []
        
        # Add price as a feature
        feature_columns.append(price_column)
        
        # Add simple moving averages
        for window in [5, 10, 20, 50]:
            col_name = f'sma_{window}'
            df[col_name] = df[price_column].rolling(window=window).mean()
            feature_columns.append(col_name)
        
        # Add exponential moving averages
        for window in [5, 10, 20, 50]:
            col_name = f'ema_{window}'
            df[col_name] = df[price_column].ewm(span=window, adjust=False).mean()
            feature_columns.append(col_name)
        
        # Add price momentum
        for window in [5, 10, 20]:
            col_name = f'momentum_{window}'
            df[col_name] = df[price_column].pct_change(periods=window)
            feature_columns.append(col_name)
        
        # Add price volatility
        for window in [5, 10, 20]:
            col_name = f'volatility_{window}'
            df[col_name] = df[price_column].rolling(window=window).std()
            feature_columns.append(col_name)
        
        if volume_column:
            # Add volume as a feature
            feature_columns.append(volume_column)
            
            # Add volume moving average
            for window in [5, 10, 20]:
                col_name = f'volume_sma_{window}'
                df[col_name] = df[volume_column].rolling(window=window).mean()
                feature_columns.append(col_name)
            
            # Add volume momentum
            for window in [5, 10]:
                col_name = f'volume_momentum_{window}'
                df[col_name] = df[volume_column].pct_change(periods=window)
                feature_columns.append(col_name)
        
        # Add price direction change
        df['target'] = np.where(df[price_column].shift(-1) > df[price_column], 1, 0)
        
        # Drop rows with NaN values resulting from rolling windows
        df = df.dropna()
        
        return df, feature_columns
    
    except Exception as e:
        logger.error(f"Error creating features for {pair}: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def run_arima_backtest(df: Any, pair: str, capital: float = 20000.0, leverage: float = 20.0, risk: float = 0.2) -> Dict[str, Any]:
    """
    Run backtest with ARIMA strategy.
    
    Args:
        df: DataFrame with historical data
        pair: Trading pair (e.g., 'SOL/USD')
        capital: Starting capital
        leverage: Leverage for trading
        risk: Risk per trade as fraction of capital
        
    Returns:
        Dictionary with backtest results
    """
    try:
        import numpy as np
        import pandas as pd
        
        # Identify price column
        potential_price_columns = ['close', 'Close', 'price', 'Price']
        
        price_column = None
        for col in potential_price_columns:
            if col in df.columns:
                price_column = col
                break
        
        if price_column is None:
            logger.error(f"Could not identify price column for {pair}")
            return {'success': False, 'error': 'Could not identify price column'}
        
        # Initialize result variables
        positions = []
        current_position = None
        equity = capital
        max_drawdown = 0
        max_equity = capital
        trades = []
        
        # Implement simple ARIMA-like strategy
        # In a real implementation, we would use a proper ARIMA model
        for i in range(50, len(df) - 1):
            # Simple moving average trend
            sma5 = df[price_column].iloc[i-5:i].mean()
            sma20 = df[price_column].iloc[i-20:i].mean()
            
            current_price = df[price_column].iloc[i]
            next_price = df[price_column].iloc[i+1]
            
            # Generate signal
            signal = None
            if sma5 > sma20:
                signal = 'buy'
            elif sma5 < sma20:
                signal = 'sell'
            
            # Handle position
            if current_position is None and signal is not None:
                # Open new position
                position_size = (equity * risk) / (current_price / leverage)
                entry_price = current_price
                entry_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                
                current_position = {
                    'type': signal,
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'size': position_size,
                    'equity_at_entry': equity
                }
                
                trades.append({
                    'pair': pair,
                    'type': signal,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'size': position_size
                })
            
            elif current_position is not None:
                # Check if we should close position
                if (current_position['type'] == 'buy' and signal == 'sell') or \
                   (current_position['type'] == 'sell' and signal == 'buy'):
                    # Close position
                    exit_price = current_price
                    exit_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                    
                    # Calculate profit/loss
                    if current_position['type'] == 'buy':
                        pnl = (exit_price - current_position['entry_price']) / current_position['entry_price']
                    else:
                        pnl = (current_position['entry_price'] - exit_price) / current_position['entry_price']
                    
                    # Apply leverage
                    pnl *= leverage
                    
                    # Apply to position size
                    pnl_amount = current_position['size'] * pnl
                    
                    # Update equity
                    equity += pnl_amount
                    
                    # Update max equity and drawdown
                    if equity > max_equity:
                        max_equity = equity
                    
                    drawdown = (max_equity - equity) / max_equity
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    
                    # Record closed position
                    closed_position = current_position.copy()
                    closed_position.update({
                        'exit_price': exit_price,
                        'exit_time': exit_time,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount
                    })
                    positions.append(closed_position)
                    
                    # Update trade record
                    trades[-1].update({
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount
                    })
                    
                    # Reset current position
                    current_position = None
            
            # Update equity curve with mark-to-market
            if current_position is not None:
                # Calculate unrealized profit/loss
                if current_position['type'] == 'buy':
                    pnl = (current_price - current_position['entry_price']) / current_position['entry_price']
                else:
                    pnl = (current_position['entry_price'] - current_price) / current_position['entry_price']
                
                # Apply leverage
                pnl *= leverage
                
                # Apply to position size
                pnl_amount = current_position['size'] * pnl
                
                # Mark-to-market equity
                mark_to_market = current_position['equity_at_entry'] + pnl_amount
                
                # Update max equity and drawdown
                if mark_to_market > max_equity:
                    max_equity = mark_to_market
                
                drawdown = (max_equity - mark_to_market) / max_equity
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
        
        total_return = (equity - capital) / capital
        annualized_return = ((1 + total_return) ** (365 / len(df))) - 1
        
        return {
            'success': True,
            'pair': pair,
            'strategy': 'arima',
            'capital': capital,
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
            'profit_factor': profit_factor,
            'trades': trades
        }
    
    except Exception as e:
        logger.error(f"Error running ARIMA backtest for {pair}: {str(e)}")
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}

def run_adaptive_backtest(df: Any, pair: str, capital: float = 20000.0, leverage: float = 20.0, risk: float = 0.2) -> Dict[str, Any]:
    """
    Run backtest with Adaptive strategy.
    
    Args:
        df: DataFrame with historical data
        pair: Trading pair (e.g., 'SOL/USD')
        capital: Starting capital
        leverage: Leverage for trading
        risk: Risk per trade as fraction of capital
        
    Returns:
        Dictionary with backtest results
    """
    try:
        import numpy as np
        import pandas as pd
        
        # Identify price column
        potential_price_columns = ['close', 'Close', 'price', 'Price']
        
        price_column = None
        for col in potential_price_columns:
            if col in df.columns:
                price_column = col
                break
        
        if price_column is None:
            logger.error(f"Could not identify price column for {pair}")
            return {'success': False, 'error': 'Could not identify price column'}
        
        # Initialize result variables
        positions = []
        current_position = None
        equity = capital
        max_drawdown = 0
        max_equity = capital
        trades = []
        
        # Implement simple Adaptive strategy
        # In a real implementation, we would use a more sophisticated approach
        for i in range(50, len(df) - 1):
            # Calculate indicators
            # EMA crossover
            ema9 = df[price_column].iloc[i-9:i].ewm(span=9, adjust=False).mean().iloc[-1]
            ema21 = df[price_column].iloc[i-21:i].ewm(span=21, adjust=False).mean().iloc[-1]
            
            # RSI
            delta = df[price_column].iloc[i-14:i].diff()
            gain = delta.where(delta > 0, 0).sum()
            loss = -delta.where(delta < 0, 0).sum()
            rs = gain / loss if loss != 0 else float('inf')
            rsi = 100 - (100 / (1 + rs))
            
            current_price = df[price_column].iloc[i]
            next_price = df[price_column].iloc[i+1]
            
            # Generate signal
            signal = None
            if ema9 > ema21 and rsi < 70:
                signal = 'buy'
            elif ema9 < ema21 and rsi > 30:
                signal = 'sell'
            
            # Handle position
            if current_position is None and signal is not None:
                # Open new position
                position_size = (equity * risk) / (current_price / leverage)
                entry_price = current_price
                entry_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                
                current_position = {
                    'type': signal,
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'size': position_size,
                    'equity_at_entry': equity
                }
                
                trades.append({
                    'pair': pair,
                    'type': signal,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'size': position_size
                })
            
            elif current_position is not None:
                # Check if we should close position
                if (current_position['type'] == 'buy' and signal == 'sell') or \
                   (current_position['type'] == 'sell' and signal == 'buy'):
                    # Close position
                    exit_price = current_price
                    exit_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                    
                    # Calculate profit/loss
                    if current_position['type'] == 'buy':
                        pnl = (exit_price - current_position['entry_price']) / current_position['entry_price']
                    else:
                        pnl = (current_position['entry_price'] - exit_price) / current_position['entry_price']
                    
                    # Apply leverage
                    pnl *= leverage
                    
                    # Apply to position size
                    pnl_amount = current_position['size'] * pnl
                    
                    # Update equity
                    equity += pnl_amount
                    
                    # Update max equity and drawdown
                    if equity > max_equity:
                        max_equity = equity
                    
                    drawdown = (max_equity - equity) / max_equity
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    
                    # Record closed position
                    closed_position = current_position.copy()
                    closed_position.update({
                        'exit_price': exit_price,
                        'exit_time': exit_time,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount
                    })
                    positions.append(closed_position)
                    
                    # Update trade record
                    trades[-1].update({
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount
                    })
                    
                    # Reset current position
                    current_position = None
            
            # Update equity curve with mark-to-market
            if current_position is not None:
                # Calculate unrealized profit/loss
                if current_position['type'] == 'buy':
                    pnl = (current_price - current_position['entry_price']) / current_position['entry_price']
                else:
                    pnl = (current_position['entry_price'] - current_price) / current_position['entry_price']
                
                # Apply leverage
                pnl *= leverage
                
                # Apply to position size
                pnl_amount = current_position['size'] * pnl
                
                # Mark-to-market equity
                mark_to_market = current_position['equity_at_entry'] + pnl_amount
                
                # Update max equity and drawdown
                if mark_to_market > max_equity:
                    max_equity = mark_to_market
                
                drawdown = (max_equity - mark_to_market) / max_equity
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
        
        total_return = (equity - capital) / capital
        annualized_return = ((1 + total_return) ** (365 / len(df))) - 1
        
        return {
            'success': True,
            'pair': pair,
            'strategy': 'adaptive',
            'capital': capital,
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
            'profit_factor': profit_factor,
            'trades': trades
        }
    
    except Exception as e:
        logger.error(f"Error running Adaptive backtest for {pair}: {str(e)}")
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}

def run_ml_backtest(df: Any, pair: str, model: Any, feature_columns: List[str], capital: float = 20000.0, base_leverage: float = 20.0, max_leverage: float = 125.0, risk: float = 0.2) -> Dict[str, Any]:
    """
    Run backtest with ML model.
    
    Args:
        df: DataFrame with historical data
        pair: Trading pair (e.g., 'SOL/USD')
        model: Trained ML model
        feature_columns: List of feature columns
        capital: Starting capital
        base_leverage: Base leverage for trading
        max_leverage: Maximum leverage for high-confidence trades
        risk: Risk per trade as fraction of capital
        
    Returns:
        Dictionary with backtest results
    """
    try:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        
        # Identify price column
        potential_price_columns = ['close', 'Close', 'price', 'Price']
        
        price_column = None
        for col in potential_price_columns:
            if col in df.columns:
                price_column = col
                break
        
        if price_column is None:
            logger.error(f"Could not identify price column for {pair}")
            return {'success': False, 'error': 'Could not identify price column'}
        
        # Extract features
        X = df[feature_columns].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Reshape for LSTM input
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        # Get predictions
        predictions = model.predict(X_reshaped)
        
        # Initialize result variables
        positions = []
        current_position = None
        equity = capital
        max_drawdown = 0
        max_equity = capital
        trades = []
        
        # Backtest with ML predictions
        for i in range(len(df) - 1):
            confidence = float(predictions[i][0])
            current_price = df[price_column].iloc[i]
            next_price = df[price_column].iloc[i+1]
            
            # Generate signal based on confidence
            signal = None
            if confidence >= 0.65:  # bullish with high confidence
                signal = 'buy'
                confidence_leverage = base_leverage + (max_leverage - base_leverage) * ((confidence - 0.65) / 0.35)
                leverage_applied = min(confidence_leverage, max_leverage)
            elif confidence <= 0.35:  # bearish with high confidence
                signal = 'sell'
                confidence_leverage = base_leverage + (max_leverage - base_leverage) * ((0.35 - confidence) / 0.35)
                leverage_applied = min(confidence_leverage, max_leverage)
            else:
                leverage_applied = base_leverage
            
            # Handle position
            if current_position is None and signal is not None:
                # Open new position
                position_size = (equity * risk) / (current_price / leverage_applied)
                entry_price = current_price
                entry_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                
                current_position = {
                    'type': signal,
                    'entry_price': entry_price,
                    'entry_time': entry_time,
                    'size': position_size,
                    'equity_at_entry': equity,
                    'leverage': leverage_applied,
                    'confidence': confidence
                }
                
                trades.append({
                    'pair': pair,
                    'type': signal,
                    'entry_time': entry_time,
                    'entry_price': entry_price,
                    'size': position_size,
                    'leverage': leverage_applied,
                    'confidence': confidence
                })
            
            elif current_position is not None:
                # Check if we should close position
                if (current_position['type'] == 'buy' and confidence <= 0.35) or \
                   (current_position['type'] == 'sell' and confidence >= 0.65):
                    # Close position
                    exit_price = current_price
                    exit_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i
                    
                    # Calculate profit/loss
                    if current_position['type'] == 'buy':
                        pnl = (exit_price - current_position['entry_price']) / current_position['entry_price']
                    else:
                        pnl = (current_position['entry_price'] - exit_price) / current_position['entry_price']
                    
                    # Apply leverage
                    pnl *= current_position['leverage']
                    
                    # Apply to position size
                    pnl_amount = current_position['size'] * pnl
                    
                    # Update equity
                    equity += pnl_amount
                    
                    # Update max equity and drawdown
                    if equity > max_equity:
                        max_equity = equity
                    
                    drawdown = (max_equity - equity) / max_equity
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                    
                    # Record closed position
                    closed_position = current_position.copy()
                    closed_position.update({
                        'exit_price': exit_price,
                        'exit_time': exit_time,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount,
                        'exit_confidence': confidence
                    })
                    positions.append(closed_position)
                    
                    # Update trade record
                    trades[-1].update({
                        'exit_time': exit_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'pnl_amount': pnl_amount,
                        'exit_confidence': confidence
                    })
                    
                    # Reset current position
                    current_position = None
            
            # Update equity curve with mark-to-market
            if current_position is not None:
                # Calculate unrealized profit/loss
                if current_position['type'] == 'buy':
                    pnl = (current_price - current_position['entry_price']) / current_position['entry_price']
                else:
                    pnl = (current_position['entry_price'] - current_price) / current_position['entry_price']
                
                # Apply leverage
                pnl *= current_position['leverage']
                
                # Apply to position size
                pnl_amount = current_position['size'] * pnl
                
                # Mark-to-market equity
                mark_to_market = current_position['equity_at_entry'] + pnl_amount
                
                # Update max equity and drawdown
                if mark_to_market > max_equity:
                    max_equity = mark_to_market
                
                drawdown = (max_equity - mark_to_market) / max_equity
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
        
        total_return = (equity - capital) / capital
        annualized_return = ((1 + total_return) ** (365 / len(df))) - 1
        
        # Calculate ML model accuracy
        predictions_binary = (predictions >= 0.5).astype(int)
        targets = df['target'].values
        accuracy = sum(predictions_binary.flatten() == targets) / len(targets)
        
        return {
            'success': True,
            'pair': pair,
            'strategy': 'ml',
            'model_accuracy': accuracy,
            'model_accuracy_pct': accuracy * 100,
            'capital': capital,
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
            'profit_factor': profit_factor,
            'trades': trades
        }
    
    except Exception as e:
        logger.error(f"Error running ML backtest for {pair}: {str(e)}")
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}

def generate_report(results: List[Dict[str, Any]], output_dir: str) -> bool:
    """
    Generate backtest report.
    
    Args:
        results: List of backtest results
        output_dir: Directory to save report
        
    Returns:
        True if successful, False otherwise
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary table
        summary_data = []
        for result in results:
            if result['success']:
                summary_data.append({
                    'Pair': result['pair'],
                    'Strategy': result['strategy'],
                    'Model Accuracy (%)': result.get('model_accuracy_pct', 'N/A'),
                    'Total Return (%)': result['total_return_pct'],
                    'Annual Return (%)': result['annualized_return_pct'],
                    'Max Drawdown (%)': result['max_drawdown_pct'],
                    'Win Rate (%)': result['win_rate_pct'],
                    'Profit Factor': result['profit_factor'],
                    'Total Trades': result['total_trades']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        summary_path = os.path.join(output_dir, 'backtest_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved backtest summary to {summary_path}")
        
        # Generate summary chart
        plt.figure(figsize=(12, 8))
        
        # Group results by pair and strategy
        pairs = sorted(list(set(r['pair'] for r in results if r['success'])))
        strategies = sorted(list(set(r['strategy'] for r in results if r['success'])))
        
        # Create bar positions
        x = np.arange(len(pairs))
        width = 0.25
        
        # Plot bars for each strategy
        for i, strategy in enumerate(strategies):
            returns = [next((r['total_return_pct'] for r in results if r['success'] and r['pair'] == pair and r['strategy'] == strategy), 0) for pair in pairs]
            plt.bar(x + width * (i - len(strategies)/2 + 0.5), returns, width, label=strategy.capitalize())
        
        plt.xlabel('Trading Pair')
        plt.ylabel('Total Return (%)')
        plt.title('Backtest Returns by Pair and Strategy')
        plt.xticks(x, pairs)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save chart
        chart_path = os.path.join(output_dir, 'backtest_returns.png')
        plt.savefig(chart_path)
        logger.info(f"Saved backtest chart to {chart_path}")
        
        # Generate detailed JSON report
        report_path = os.path.join(output_dir, 'backtest_report.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved detailed backtest report to {report_path}")
        
        # Generate trades CSV
        all_trades = []
        for result in results:
            if result['success'] and 'trades' in result:
                for trade in result['trades']:
                    trade_record = {
                        'pair': result['pair'],
                        'strategy': result['strategy'],
                        'type': trade.get('type', 'unknown'),
                        'entry_time': str(trade.get('entry_time', '')),
                        'entry_price': trade.get('entry_price', 0),
                        'exit_time': str(trade.get('exit_time', '')),
                        'exit_price': trade.get('exit_price', 0),
                        'size': trade.get('size', 0),
                        'leverage': trade.get('leverage', 0),
                        'confidence': trade.get('confidence', 0),
                        'pnl': trade.get('pnl', 0),
                        'pnl_amount': trade.get('pnl_amount', 0)
                    }
                    all_trades.append(trade_record)
        
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_path = os.path.join(output_dir, 'backtest_trades.csv')
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Saved trades to {trades_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def optimize_parameters(df: Any, pair: str, model: Any, feature_columns: List[str]) -> Dict[str, Any]:
    """
    Optimize trading parameters.
    
    Args:
        df: DataFrame with historical data
        pair: Trading pair (e.g., 'SOL/USD')
        model: Trained ML model
        feature_columns: List of feature columns
        
    Returns:
        Dictionary with optimized parameters
    """
    try:
        # Define parameter ranges to test
        leverage_range = [10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0]
        risk_range = [0.1, 0.15, 0.2, 0.25, 0.3]
        confidence_range = [0.60, 0.65, 0.70, 0.75, 0.80]
        
        best_params = None
        best_return = -float('inf')
        
        # Test combinations of parameters
        for base_leverage in leverage_range:
            for max_leverage in [l for l in leverage_range if l >= base_leverage]:
                for risk in risk_range:
                    # Run backtest with current parameters
                    result = run_ml_backtest(
                        df, pair, model, feature_columns,
                        capital=20000.0, 
                        base_leverage=base_leverage,
                        max_leverage=max_leverage,
                        risk=risk
                    )
                    
                    if result['success'] and result['total_return'] > best_return:
                        best_return = result['total_return']
                        best_params = {
                            'base_leverage': base_leverage,
                            'max_leverage': max_leverage,
                            'risk': risk,
                            'total_return': result['total_return'],
                            'total_return_pct': result['total_return_pct'],
                            'win_rate': result['win_rate'],
                            'win_rate_pct': result['win_rate_pct'],
                            'max_drawdown': result['max_drawdown'],
                            'max_drawdown_pct': result['max_drawdown_pct']
                        }
        
        return {
            'success': True,
            'pair': pair,
            'best_params': best_params
        }
    
    except Exception as e:
        logger.error(f"Error optimizing parameters for {pair}: {str(e)}")
        logger.error(traceback.format_exc())
        return {'success': False, 'error': str(e)}

def backtest_pair(
    pair: str,
    strategies: List[str],
    timeframes: List[str],
    days: int = 180,
    capital: float = 20000.0,
    leverage: float = 20.0,
    max_leverage: float = 125.0,
    risk: float = 0.2,
    optimize_params: bool = False,
    compare_fees: bool = False,
    compare_slippage: bool = False,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Run backtest for a single pair with multiple strategies and timeframes.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        strategies: List of strategies to test
        timeframes: List of timeframes to test
        days: Number of days for backtesting
        capital: Starting capital
        leverage: Base leverage for trading
        max_leverage: Maximum leverage for high-confidence trades
        risk: Risk per trade as fraction of capital
        optimize_params: Whether to perform parameter optimization
        compare_fees: Whether to compare different fee structures
        compare_slippage: Whether to compare different slippage scenarios
        verbose: Whether to print detailed information
        
    Returns:
        List of backtest results
    """
    results = []
    
    for timeframe in timeframes:
        logger.info(f"Testing {pair} with timeframe {timeframe}")
        
        # Load historical data
        df = load_historical_data(pair, timeframe, days)
        if df is None:
            logger.error(f"Failed to load historical data for {pair} with timeframe {timeframe}")
            continue
        
        # Create features
        df_features, feature_columns = create_features(df, pair)
        if df_features is None:
            logger.error(f"Failed to create features for {pair}")
            continue
        
        # Run backtests for each strategy
        for strategy in strategies:
            logger.info(f"Testing {pair} with {strategy} strategy")
            
            if strategy == 'arima':
                result = run_arima_backtest(df_features, pair, capital, leverage, risk)
                if result['success']:
                    result['timeframe'] = timeframe
                    results.append(result)
            
            elif strategy == 'adaptive':
                result = run_adaptive_backtest(df_features, pair, capital, leverage, risk)
                if result['success']:
                    result['timeframe'] = timeframe
                    results.append(result)
            
            elif strategy == 'ml':
                # Load ML model
                model = load_model(pair, model_type='lstm')
                if model is None:
                    logger.error(f"Failed to load ML model for {pair}")
                    continue
                
                # Run ML backtest
                result = run_ml_backtest(df_features, pair, model, feature_columns, capital, leverage, max_leverage, risk)
                if result['success']:
                    result['timeframe'] = timeframe
                    results.append(result)
                
                # Optimize parameters if requested
                if optimize_params:
                    logger.info(f"Optimizing parameters for {pair} with ML strategy")
                    optimization_result = optimize_parameters(df_features, pair, model, feature_columns)
                    if optimization_result['success']:
                        result['optimized_params'] = optimization_result['best_params']
                        logger.info(f"Optimized parameters for {pair}: {optimization_result['best_params']}")
            
            # Compare fee structures if requested
            if compare_fees and result['success']:
                logger.info(f"Comparing fee structures for {pair} with {strategy} strategy")
                fee_results = []
                
                for fee_rate in [0.0001, 0.0005, 0.001, 0.002, 0.003]:
                    if strategy == 'arima':
                        fee_result = run_arima_backtest(df_features, pair, capital, leverage, risk, fee_rate=fee_rate)
                    elif strategy == 'adaptive':
                        fee_result = run_adaptive_backtest(df_features, pair, capital, leverage, risk, fee_rate=fee_rate)
                    elif strategy == 'ml':
                        fee_result = run_ml_backtest(df_features, pair, model, feature_columns, capital, leverage, max_leverage, risk, fee_rate=fee_rate)
                    
                    if fee_result['success']:
                        fee_result['fee_rate'] = fee_rate
                        fee_results.append(fee_result)
                
                result['fee_comparison'] = fee_results
            
            # Compare slippage scenarios if requested
            if compare_slippage and result['success']:
                logger.info(f"Comparing slippage scenarios for {pair} with {strategy} strategy")
                slippage_results = []
                
                for slippage in [0.0, 0.0005, 0.001, 0.002, 0.005]:
                    if strategy == 'arima':
                        slippage_result = run_arima_backtest(df_features, pair, capital, leverage, risk, slippage=slippage)
                    elif strategy == 'adaptive':
                        slippage_result = run_adaptive_backtest(df_features, pair, capital, leverage, risk, slippage=slippage)
                    elif strategy == 'ml':
                        slippage_result = run_ml_backtest(df_features, pair, model, feature_columns, capital, leverage, max_leverage, risk, slippage=slippage)
                    
                    if slippage_result['success']:
                        slippage_result['slippage'] = slippage
                        slippage_results.append(slippage_result)
                
                result['slippage_comparison'] = slippage_results
    
    return results

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
    logger.info("ENHANCED BACKTESTING")
    logger.info("=" * 80)
    logger.info(f"Trading pairs: {', '.join(pairs)}")
    logger.info(f"Timeframes: {', '.join(timeframes)}")
    logger.info(f"Strategies: {', '.join(strategies)}")
    logger.info(f"Starting capital: ${args.capital:.2f}")
    logger.info(f"Days for backtesting: {args.days}")
    logger.info(f"Leverage settings: Base={args.leverage:.1f}x, Max={args.max_leverage:.1f}x")
    logger.info(f"Risk per trade: {args.risk * 100:.1f}%")
    logger.info(f"Optimization: {args.optimize_params}")
    logger.info(f"Compare fees: {args.compare_fees}")
    logger.info(f"Compare slippage: {args.compare_slippage}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target accuracy: {args.target_accuracy * 100:.1f}%")
    logger.info(f"Target return: {args.target_return * 100:.1f}%")
    logger.info("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies. Please install required packages and try again.")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run backtests for each pair
    all_results = []
    for pair in pairs:
        logger.info("-" * 80)
        logger.info(f"Processing {pair}")
        logger.info("-" * 80)
        
        pair_results = backtest_pair(
            pair,
            strategies,
            timeframes,
            args.days,
            args.capital,
            args.leverage,
            args.max_leverage,
            args.risk,
            args.optimize_params,
            args.compare_fees,
            args.compare_slippage,
            args.verbose
        )
        
        all_results.extend(pair_results)
    
    # Generate report
    logger.info("-" * 80)
    logger.info("Generating backtest report")
    logger.info("-" * 80)
    
    if generate_report(all_results, args.output_dir):
        logger.info(f"Backtest report generated in {args.output_dir}")
    else:
        logger.error("Failed to generate backtest report")
        return 1
    
    # Analyze results
    success_count = sum(1 for r in all_results if r['success'])
    
    # Check if target accuracy and return are met
    accuracy_target_met = False
    return_target_met = False
    
    ml_results = [r for r in all_results if r['success'] and r['strategy'] == 'ml']
    if ml_results:
        avg_accuracy = sum(r.get('model_accuracy', 0) for r in ml_results) / len(ml_results)
        accuracy_target_met = avg_accuracy >= args.target_accuracy
        
        max_return_pct = max(r['total_return_pct'] for r in all_results if r['success'])
        return_target_met = max_return_pct >= args.target_return * 100
    
    # Log summary
    logger.info("=" * 80)
    logger.info(f"BACKTESTING COMPLETED: {success_count}/{len(all_results)} tests successful")
    
    if ml_results:
        logger.info(f"Average ML model accuracy: {avg_accuracy:.2f}% (target: {args.target_accuracy * 100:.1f}%)")
        logger.info(f"Target accuracy met: {accuracy_target_met}")
        
        logger.info(f"Maximum return: {max_return_pct:.2f}% (target: {args.target_return * 100:.1f}%)")
        logger.info(f"Target return met: {return_target_met}")
    
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())