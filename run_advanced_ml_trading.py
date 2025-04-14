#!/usr/bin/env python3
"""
Run Advanced ML Trading Script

This script runs the entire advanced ML trading system:
1. Loads trained models (Temporal Fusion Transformer, etc.)
2. Processes market data with Multi-Asset Feature Fusion
3. Applies Adaptive Hyperparameter Tuning for optimization
4. Uses Explainable AI to provide insights
5. Incorporates Sentiment Analysis for improved predictions

It can be run in:
- Backtest mode: Testing models on historical data
- Live trading mode: Running with real-time market data
- Sandbox mode: Trading simulation with real-time data
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import advanced ML components
from advanced_ml_integration import (
    advanced_ml,
    ml_model_integrator,
    TARGET_ACCURACY,
    TARGET_RETURN
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("advanced_ml_trading.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TRADING_PAIRS = ["SOL/USD", "ETH/USD", "BTC/USD", "DOT/USD", "LINK/USD"]
HISTORICAL_DATA_DIR = "historical_data"
BACKTEST_RESULTS_DIR = "backtest_results"
os.makedirs(BACKTEST_RESULTS_DIR, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run advanced ML trading system")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["backtest", "live", "sandbox"], default="backtest",
                        help="Trading mode")
    
    # Asset selection
    parser.add_argument("--trading-pairs", nargs="+", default=DEFAULT_TRADING_PAIRS,
                        help="Trading pairs to use")
    
    # Backtest parameters
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date for backtest (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date for backtest (YYYY-MM-DD)")
    
    # ML parameters
    parser.add_argument("--use-sentiment", action="store_true", default=True,
                        help="Whether to use sentiment analysis")
    parser.add_argument("--explain-trades", action="store_true", default=True,
                        help="Whether to generate trade explanations")
    
    # Trading parameters
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Initial capital for trading")
    parser.add_argument("--max-leverage", type=float, default=10.0,
                        help="Maximum leverage to use")
    parser.add_argument("--risk-per-trade", type=float, default=0.02,
                        help="Percentage of capital to risk per trade")
    
    # Output parameters
    parser.add_argument("--results-path", type=str, default=None,
                        help="Path to save trading results")
    
    return parser.parse_args()

def load_market_data(args):
    """
    Load market data based on the selected mode
    
    Args:
        args (Namespace): Command line arguments
        
    Returns:
        dict: Dictionary of DataFrames with market data
    """
    if args.mode == "backtest":
        # Load historical data for backtest
        return load_backtest_data(args.trading_pairs, args.start_date, args.end_date)
    else:
        # Load real-time data for live/sandbox trading
        return load_realtime_data(args.trading_pairs)

def load_backtest_data(trading_pairs, start_date=None, end_date=None):
    """
    Load historical data for backtesting
    
    Args:
        trading_pairs (list): List of trading pairs to load data for
        start_date (str): Start date for backtest (YYYY-MM-DD)
        end_date (str): End date for backtest (YYYY-MM-DD)
        
    Returns:
        dict: Dictionary of DataFrames with historical data
    """
    data_dict = {}
    
    # Convert dates to datetime
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_dt = datetime.now() - timedelta(days=365)  # Default to 1 year
    
    if end_date:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end_dt = datetime.now()
    
    logger.info(f"Loading backtest data from {start_dt.date()} to {end_dt.date()}")
    
    for pair in trading_pairs:
        formatted_pair = pair.replace("/", "")
        
        # Construct file path for daily data
        file_path = os.path.join(HISTORICAL_DATA_DIR, formatted_pair, f"{formatted_pair}_1d.csv")
        
        if os.path.exists(file_path):
            try:
                # Load CSV data
                df = pd.read_csv(file_path)
                
                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Filter by date range
                if 'timestamp' in df.columns:
                    df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
                
                # Add to data dictionary if we have data
                if len(df) > 0:
                    data_dict[pair] = df
                    logger.info(f"Loaded {len(df)} rows of backtest data for {pair}")
                else:
                    logger.warning(f"No data in the selected date range for {pair}")
            except Exception as e:
                logger.error(f"Error loading backtest data for {pair}: {e}")
        else:
            logger.warning(f"Historical data file not found: {file_path}")
    
    # Add technical indicators
    if data_dict:
        try:
            import talib
            
            for pair, df in data_dict.items():
                if len(df) < 50:
                    logger.warning(f"Not enough data for {pair} to calculate indicators, skipping")
                    continue
                
                try:
                    # Make sure we have OHLCV data
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        logger.warning(f"Missing columns for {pair}: {missing_cols}")
                        continue
                    
                    # Extract columns with appropriate types
                    open_prices = df['open'].values.astype(float)
                    high_prices = df['high'].values.astype(float)
                    low_prices = df['low'].values.astype(float)
                    close_prices = df['close'].values.astype(float)
                    volumes = df['volume'].values.astype(float)
                    
                    # Add trend indicators
                    df['ema9'] = talib.EMA(close_prices, timeperiod=9)
                    df['ema21'] = talib.EMA(close_prices, timeperiod=21)
                    df['ema50'] = talib.EMA(close_prices, timeperiod=50)
                    df['ema100'] = talib.EMA(close_prices, timeperiod=100)
                    df['ema200'] = talib.EMA(close_prices, timeperiod=200)
                    df['sma20'] = talib.SMA(close_prices, timeperiod=20)
                    df['sma50'] = talib.SMA(close_prices, timeperiod=50)
                    
                    # Add momentum indicators
                    df['rsi'] = talib.RSI(close_prices, timeperiod=14)
                    macd, df['macd_signal'], df['macd_hist'] = talib.MACD(
                        close_prices, fastperiod=12, slowperiod=26, signalperiod=9
                    )
                    df['macd'] = macd
                    df['stoch_k'], df['stoch_d'] = talib.STOCH(
                        high_prices, low_prices, close_prices,
                        fastk_period=14, slowk_period=3, slowd_period=3
                    )
                    
                    # Add volatility indicators
                    df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
                    upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
                    df['bb_upper'] = upper
                    df['bb_middle'] = middle
                    df['bb_lower'] = lower
                    df['bb_width'] = (upper - lower) / middle
                    
                    # Handle NaN values
                    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
                    
                    # Update the data dictionary
                    data_dict[pair] = df
                    
                    logger.info(f"Added technical indicators for {pair}, now has {len(df.columns)} features")
                except Exception as e:
                    logger.error(f"Error adding technical indicators for {pair}: {e}")
        except ImportError:
            logger.warning("TA-Lib not available, skipping technical indicators")
    
    return data_dict

def load_realtime_data(trading_pairs):
    """
    Load real-time market data for live/sandbox trading
    
    Args:
        trading_pairs (list): List of trading pairs to load data for
        
    Returns:
        dict: Dictionary of DataFrames with real-time data
    """
    # Load the current trading bot functions for accessing market data
    try:
        from kraken_api import KrakenAPI
        from historical_data_fetcher import fetch_historical_data
        
        api = KrakenAPI()
        data_dict = {}
        
        for pair in trading_pairs:
            try:
                # Fetch recent OHLCV data
                ohlcv_data = fetch_historical_data(pair, interval=1440, count=200)  # Daily data
                
                if ohlcv_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time'])
                    
                    # Convert types
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = df[col].astype(float)
                    
                    # Add technical indicators
                    try:
                        import talib
                        
                        # Extract columns with appropriate types
                        open_prices = df['open'].values
                        high_prices = df['high'].values
                        low_prices = df['low'].values
                        close_prices = df['close'].values
                        volumes = df['volume'].values
                        
                        # Add trend indicators
                        df['ema9'] = talib.EMA(close_prices, timeperiod=9)
                        df['ema21'] = talib.EMA(close_prices, timeperiod=21)
                        df['ema50'] = talib.EMA(close_prices, timeperiod=50)
                        df['ema100'] = talib.EMA(close_prices, timeperiod=100)
                        df['ema200'] = talib.EMA(close_prices, timeperiod=200)
                        df['sma20'] = talib.SMA(close_prices, timeperiod=20)
                        df['sma50'] = talib.SMA(close_prices, timeperiod=50)
                        
                        # Add momentum indicators
                        df['rsi'] = talib.RSI(close_prices, timeperiod=14)
                        macd, df['macd_signal'], df['macd_hist'] = talib.MACD(
                            close_prices, fastperiod=12, slowperiod=26, signalperiod=9
                        )
                        df['macd'] = macd
                        df['stoch_k'], df['stoch_d'] = talib.STOCH(
                            high_prices, low_prices, close_prices,
                            fastk_period=14, slowk_period=3, slowd_period=3
                        )
                        
                        # Add volatility indicators
                        df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
                        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
                        df['bb_upper'] = upper
                        df['bb_middle'] = middle
                        df['bb_lower'] = lower
                        df['bb_width'] = (upper - lower) / middle
                    except ImportError:
                        logger.warning("TA-Lib not available, skipping technical indicators")
                    
                    # Handle NaN values
                    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
                    
                    # Add to data dictionary
                    data_dict[pair] = df
                    logger.info(f"Loaded {len(df)} rows of real-time data for {pair}")
                else:
                    logger.warning(f"No real-time data available for {pair}")
            except Exception as e:
                logger.error(f"Error loading real-time data for {pair}: {e}")
        
        return data_dict
    except ImportError:
        logger.error("Kraken API modules not found, cannot load real-time data")
        return {}

def run_backtest(data_dict, args):
    """
    Run backtest with the advanced ML models
    
    Args:
        data_dict (dict): Dictionary of DataFrames with market data
        args (Namespace): Command line arguments
        
    Returns:
        dict: Backtest results
    """
    # Initialize performance tracking
    results = {
        'trades': [],
        'daily_returns': [],
        'equity_curve': [],
        'drawdowns': [],
        'metrics': {}
    }
    
    # Track portfolio value
    portfolio = {
        'cash': args.capital,
        'positions': {},
        'equity_history': [],
        'trades_history': []
    }
    
    # Get all timestamps from the first asset (assuming all assets have the same timestamps)
    if not data_dict:
        logger.error("No data available for backtesting")
        return results
    
    first_asset = list(data_dict.keys())[0]
    timestamps = data_dict[first_asset]['timestamp'].values
    
    # Initialize models if not already done
    logger.info("Initializing advanced ML models")
    advanced_ml.initialize_models()
    
    # Loop through each timestamp (day)
    logger.info(f"Running backtest from {timestamps[0]} to {timestamps[-1]}")
    
    for i, current_time in enumerate(timestamps):
        if i < 60:  # Skip the first N days to have enough historical data
            continue
        
        # Create a slice of data up to the current timestamp
        current_data = {}
        for asset, df in data_dict.items():
            current_idx = df[df['timestamp'] <= current_time].index
            if len(current_idx) > 0:
                current_data[asset] = df.loc[current_idx]
        
        # Generate predictions
        predictions = advanced_ml.predict(current_data, include_explanations=args.explain_trades)
        
        # Convert predictions to trading signals
        signals = ml_model_integrator.get_trading_signals(current_data)
        
        # Execute trades based on signals
        for asset, signal in signals.items():
            # Get current price
            current_price = current_data[asset]['close'].iloc[-1]
            
            # Check if we have an open position for this asset
            position = portfolio['positions'].get(asset)
            
            if position:
                # Check if we should exit the position
                if (position['side'] == 'long' and signal['signal'] == 'SELL') or \
                   (position['side'] == 'short' and signal['signal'] == 'BUY'):
                    # Close the position
                    pnl = 0
                    if position['side'] == 'long':
                        pnl = (current_price - position['entry_price']) * position['size'] * position['leverage']
                    else:
                        pnl = (position['entry_price'] - current_price) * position['size'] * position['leverage']
                    
                    # Update portfolio
                    portfolio['cash'] += position['value'] + pnl
                    
                    # Record the trade
                    trade = {
                        'asset': asset,
                        'side': position['side'],
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'size': position['size'],
                        'leverage': position['leverage'],
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'pnl': pnl,
                        'pnl_percent': (pnl / position['value']) * 100,
                        'signal_confidence': signal['confidence']
                    }
                    
                    # Add explanation if available
                    if args.explain_trades and asset in predictions and 'explanation' in predictions[asset]:
                        trade['explanation'] = predictions[asset]['explanation'].get('explanation_text', '')
                    
                    results['trades'].append(trade)
                    portfolio['trades_history'].append(trade)
                    
                    # Remove the position
                    del portfolio['positions'][asset]
                    
                    logger.info(f"Closed {position['side']} position on {asset} at {current_price}: PnL {pnl:.2f} ({(pnl / position['value']) * 100:.2f}%)")
            
            else:
                # Check if we should open a new position
                if signal['signal'] in ['BUY', 'SELL']:
                    # Determine position size based on risk
                    risk_amount = portfolio['cash'] * args.risk_per_trade
                    
                    # Calculate leverage based on confidence (higher confidence = higher leverage)
                    leverage = min(args.max_leverage, 1 + (args.max_leverage - 1) * signal['confidence'])
                    
                    # Calculate position size
                    position_value = risk_amount * 10  # Risk 10% of position value
                    size = position_value / current_price
                    
                    # Check if we have enough cash
                    if position_value <= portfolio['cash']:
                        # Open the position
                        side = 'long' if signal['signal'] == 'BUY' else 'short'
                        
                        # Create position
                        position = {
                            'asset': asset,
                            'side': side,
                            'entry_price': current_price,
                            'size': size,
                            'value': position_value,
                            'leverage': leverage,
                            'entry_time': current_time
                        }
                        
                        # Update portfolio
                        portfolio['cash'] -= position_value
                        portfolio['positions'][asset] = position
                        
                        logger.info(f"Opened {side} position on {asset} at {current_price}: Size {size:.6f}, Leverage {leverage:.1f}x")
        
        # Calculate portfolio value
        portfolio_value = portfolio['cash']
        
        # Add value of open positions
        for asset, position in portfolio['positions'].items():
            current_price = current_data[asset]['close'].iloc[-1]
            
            position_value = position['value']
            if position['side'] == 'long':
                unrealized_pnl = (current_price - position['entry_price']) * position['size'] * position['leverage']
            else:
                unrealized_pnl = (position['entry_price'] - current_price) * position['size'] * position['leverage']
            
            portfolio_value += position_value + unrealized_pnl
        
        # Record portfolio value
        portfolio['equity_history'].append({
            'timestamp': current_time,
            'equity': portfolio_value
        })
        
        # Calculate daily return
        if len(portfolio['equity_history']) > 1:
            prev_equity = portfolio['equity_history'][-2]['equity']
            daily_return = (portfolio_value - prev_equity) / prev_equity
            results['daily_returns'].append(daily_return)
        
        # Calculate drawdown
        if len(portfolio['equity_history']) > 0:
            equity_series = [e['equity'] for e in portfolio['equity_history']]
            peak = max(equity_series)
            drawdown = (peak - portfolio_value) / peak if peak > 0 else 0
            results['drawdowns'].append(drawdown)
        
        # Update equity curve
        results['equity_curve'].append({
            'timestamp': current_time,
            'equity': portfolio_value
        })
    
    # Calculate performance metrics
    if results['daily_returns']:
        total_return = (portfolio_value / args.capital) - 1
        sharpe_ratio = np.mean(results['daily_returns']) / np.std(results['daily_returns']) * np.sqrt(252) if np.std(results['daily_returns']) > 0 else 0
        max_drawdown = max(results['drawdowns']) if results['drawdowns'] else 0
        win_rate = sum(1 for t in results['trades'] if t['pnl'] > 0) / max(1, len(results['trades']))
        
        results['metrics'] = {
            'total_return': total_return,
            'total_return_percent': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown * 100,
            'win_rate': win_rate,
            'win_rate_percent': win_rate * 100,
            'total_trades': len(results['trades']),
            'final_equity': portfolio_value
        }
        
        logger.info(f"Backtest completed: Return {total_return*100:.2f}%, Sharpe {sharpe_ratio:.2f}, Win Rate {win_rate*100:.2f}%")
    else:
        logger.warning("No trades executed during backtest")
    
    # Save results
    if args.results_path:
        results_path = args.results_path
    else:
        results_path = os.path.join(BACKTEST_RESULTS_DIR, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Convert numpy and datetime values to Python native types for JSON serialization
    def json_serialize(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, np.datetime64)):
            return str(obj)
        raise TypeError(f"Type {type(obj)} not serializable")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, default=json_serialize, indent=2)
    
    logger.info(f"Saved backtest results to {results_path}")
    
    return results

def run_live_trading(data_dict, args):
    """
    Run live trading with the advanced ML models
    
    Args:
        data_dict (dict): Dictionary of DataFrames with market data
        args (Namespace): Command line arguments
        
    Returns:
        bool: Success status
    """
    try:
        # Import trading modules
        from kraken_api import KrakenAPI
        import bot_manager
        
        # Initialize Kraken API
        api = KrakenAPI()
        
        # Initialize models if not already done
        logger.info("Initializing advanced ML models")
        advanced_ml.initialize_models()
        
        # Generate predictions
        predictions = advanced_ml.predict(data_dict, include_explanations=args.explain_trades)
        
        # Convert predictions to trading signals
        signals = ml_model_integrator.get_trading_signals(data_dict)
        
        # Process signals with bot manager
        for asset, signal in signals.items():
            logger.info(f"{asset} signal: {signal['signal']} (strength: {signal['strength']:.2f}, confidence: {signal['confidence']:.2f})")
            
            # Add explanation if available
            if args.explain_trades and asset in predictions and 'explanation' in predictions[asset]:
                explanation = predictions[asset]['explanation'].get('explanation_text', '')
                logger.info(f"Explanation: {explanation}")
            
            # Register signal with bot manager
            try:
                signal_type = bot_manager.SignalType.BUY if signal['signal'] == 'BUY' else \
                             bot_manager.SignalType.SELL if signal['signal'] == 'SELL' else \
                             bot_manager.SignalType.NEUTRAL
                
                bot_manager.register_signal(
                    asset, 
                    "MLModel", 
                    signal_type, 
                    signal['strength']
                )
                
                logger.info(f"Registered {signal['signal']} signal for {asset} with bot manager")
            except Exception as e:
                logger.error(f"Error registering signal with bot manager: {e}")
        
        return True
    
    except ImportError:
        logger.error("Trading modules not found, cannot run live trading")
        return False
    except Exception as e:
        logger.error(f"Error in live trading: {e}")
        return False

def main():
    """Main function"""
    args = parse_arguments()
    
    logger.info(f"Starting advanced ML trading in {args.mode} mode")
    logger.info(f"Trading pairs: {args.trading_pairs}")
    
    try:
        # Load market data
        data_dict = load_market_data(args)
        
        if not data_dict:
            logger.error("No market data available, aborting")
            return 1
        
        # Update sentiment data if enabled
        if args.use_sentiment:
            logger.info("Updating sentiment data")
            advanced_ml.update_sentiment_data()
        
        # Run in appropriate mode
        if args.mode == "backtest":
            results = run_backtest(data_dict, args)
            
            # Print summary
            if results and 'metrics' in results:
                print("\n" + "="*80)
                print("BACKTEST RESULTS")
                print("="*80)
                print(f"Total Return: {results['metrics']['total_return_percent']:.2f}%")
                print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
                print(f"Max Drawdown: {results['metrics']['max_drawdown_percent']:.2f}%")
                print(f"Win Rate: {results['metrics']['win_rate_percent']:.2f}%")
                print(f"Total Trades: {results['metrics']['total_trades']}")
                print(f"Final Equity: ${results['metrics']['final_equity']:.2f}")
                print("="*80)
                
                # Check if we're meeting performance targets
                accuracy_target_met = results['metrics']['win_rate'] >= TARGET_ACCURACY
                return_target_met = results['metrics']['total_return'] >= TARGET_RETURN
                
                if accuracy_target_met and return_target_met:
                    print("🎉 All performance targets met! Ready for live trading.")
                elif accuracy_target_met:
                    print("📈 Accuracy target met, but return target not yet reached.")
                elif return_target_met:
                    print("💰 Return target met, but accuracy target not yet reached.")
                else:
                    print("🔄 Continue training to reach performance targets.")
                
                print("="*80)
        else:
            # Live or sandbox trading
            success = run_live_trading(data_dict, args)
            
            if not success:
                logger.error(f"Failed to run {args.mode} trading")
                return 1
            
            logger.info(f"Successfully ran {args.mode} trading")
        
        return 0
    except Exception as e:
        logger.error(f"Error in main function: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())