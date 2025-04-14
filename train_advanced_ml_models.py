#!/usr/bin/env python3
"""
Train Advanced ML Models Script

This script trains all the advanced machine learning models:
1. Temporal Fusion Transformer
2. Multi-Asset Feature Fusion
3. Adaptive Hyperparameter Tuning
4. Explainable AI
5. Sentiment Analysis

It handles data preparation, model training, and performance evaluation
to achieve the target of 90% prediction accuracy and 1000%+ returns.
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
        logging.FileHandler("advanced_ml_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_TRADING_PAIRS = ["SOL/USD", "ETH/USD", "BTC/USD", "DOT/USD", "LINK/USD"]
HISTORICAL_DATA_DIR = "historical_data"
TRAINING_RESULTS_DIR = "training_results"
os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train advanced ML models for trading")
    
    # General parameters
    parser.add_argument("--trading-pairs", nargs="+", default=DEFAULT_TRADING_PAIRS,
                        help="Trading pairs to train models for")
    parser.add_argument("--force-retrain", action="store_true",
                        help="Force retraining of models even if they exist")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom configuration file")
    
    # Training parameters
    parser.add_argument("--max-lookback-days", type=int, default=365,
                        help="Maximum number of days of historical data to use")
    parser.add_argument("--timeframes", nargs="+", default=["1d", "4h", "1h"],
                        help="Timeframes to use for training")
    
    # Feature parameters
    parser.add_argument("--use-cross-asset", action="store_true", default=True,
                        help="Whether to use cross-asset features")
    parser.add_argument("--use-sentiment", action="store_true", default=True,
                        help="Whether to use sentiment analysis")
    
    # Output parameters
    parser.add_argument("--report-path", type=str, default=None,
                        help="Path to save the training report")
    
    return parser.parse_args()

def load_historical_data(trading_pairs, max_lookback_days=365, timeframes=None):
    """
    Load historical data for training
    
    Args:
        trading_pairs (list): List of trading pairs to load data for
        max_lookback_days (int): Maximum number of days of historical data to use
        timeframes (list): List of timeframes to load
        
    Returns:
        dict: Dictionary of DataFrames with historical data
    """
    if timeframes is None:
        timeframes = ["1d"]
    
    data_dict = {}
    cutoff_date = datetime.now() - timedelta(days=max_lookback_days)
    
    for pair in trading_pairs:
        pair_data = {}
        formatted_pair = pair.replace("/", "")
        
        for timeframe in timeframes:
            # Construct file path
            file_path = os.path.join(HISTORICAL_DATA_DIR, formatted_pair, f"{formatted_pair}_{timeframe}.csv")
            
            if os.path.exists(file_path):
                try:
                    # Load CSV data
                    df = pd.read_csv(file_path)
                    
                    # Convert timestamp to datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Filter by cutoff date
                    if 'timestamp' in df.columns:
                        df = df[df['timestamp'] >= cutoff_date]
                    
                    # Add to data dictionary
                    pair_data[timeframe] = df
                    logger.info(f"Loaded {len(df)} rows of {timeframe} data for {pair}")
                except Exception as e:
                    logger.error(f"Error loading {timeframe} data for {pair}: {e}")
            else:
                logger.warning(f"Historical data file not found: {file_path}")
        
        # Combine timeframes if we have multiple
        if len(pair_data) > 0:
            # Use the largest timeframe as base
            base_timeframe = timeframes[0]
            if base_timeframe in pair_data:
                combined_df = pair_data[base_timeframe].copy()
                
                # Add features from other timeframes
                for tf, df in pair_data.items():
                    if tf != base_timeframe:
                        for col in df.columns:
                            if col != 'timestamp' and col not in combined_df.columns:
                                # Prefix with timeframe
                                combined_df[f"{tf}_{col}"] = np.nan
                                
                                # Find matching timestamps and copy values
                                for i, row in combined_df.iterrows():
                                    if 'timestamp' in row and 'timestamp' in df.columns:
                                        # Find closest timestamp in other timeframe
                                        closest_idx = (df['timestamp'] - row['timestamp']).abs().idxmin()
                                        combined_df.loc[i, f"{tf}_{col}"] = df.loc[closest_idx, col]
                
                data_dict[pair] = combined_df
                logger.info(f"Combined data for {pair} with {len(combined_df.columns)} features")
            else:
                # Just use the first available timeframe
                first_tf = list(pair_data.keys())[0]
                data_dict[pair] = pair_data[first_tf]
                logger.info(f"Using {first_tf} data for {pair} with {len(pair_data[first_tf].columns)} features")
    
    return data_dict

def add_technical_indicators(data_dict):
    """
    Add technical indicators to historical data
    
    Args:
        data_dict (dict): Dictionary of DataFrames with historical data
        
    Returns:
        dict: Dictionary of DataFrames with added technical indicators
    """
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
            
            # Add volume indicators
            df['obv'] = talib.OBV(close_prices, volumes)
            df['volume_ma'] = talib.SMA(volumes, timeperiod=20)
            df['volume_ratio'] = volumes / df['volume_ma']
            
            # Add price action indicators
            for i in range(1, min(5, len(df))):
                df[f'return_{i}d'] = df['close'].pct_change(i)
            
            # Add custom indicators
            df['price_volatility'] = df['close'].pct_change().rolling(window=14).std()
            df['price_to_ema200'] = df['close'] / df['ema200']
            
            # Handle NaN values
            df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
            
            # Update the data dictionary
            data_dict[pair] = df
            
            logger.info(f"Added technical indicators for {pair}, now has {len(df.columns)} features")
        
        except Exception as e:
            logger.error(f"Error adding technical indicators for {pair}: {e}")
    
    return data_dict

def train_models(args):
    """
    Train all advanced ML models
    
    Args:
        args (Namespace): Command line arguments
        
    Returns:
        dict: Training results
    """
    # Load historical data
    logger.info(f"Loading historical data for {args.trading_pairs}")
    data_dict = load_historical_data(
        args.trading_pairs,
        max_lookback_days=args.max_lookback_days,
        timeframes=args.timeframes
    )
    
    if not data_dict:
        logger.error("No historical data available, aborting")
        return None
    
    # Add technical indicators
    try:
        data_dict = add_technical_indicators(data_dict)
    except ImportError:
        logger.warning("TA-Lib not available, skipping technical indicators")
    
    # Initialize models if needed
    logger.info("Initializing advanced ML models")
    advanced_ml.initialize_models()
    
    # Train models
    logger.info("Training models with historical data")
    training_results = advanced_ml.train_models(
        data_dict,
        force_retrain=args.force_retrain,
        save_models=True
    )
    
    # Update sentiment data if enabled
    if args.use_sentiment:
        logger.info("Updating sentiment data for all assets")
        advanced_ml.update_sentiment_data(force_refresh=True)
    
    # Generate performance report
    logger.info("Generating performance report")
    report = advanced_ml.generate_performance_report()
    
    # Save report
    if args.report_path:
        report_path = args.report_path
    else:
        report_path = os.path.join(TRAINING_RESULTS_DIR, f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved training report to {report_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Overall Accuracy: {report['overall']['accuracy']:.2%} (Target: {TARGET_ACCURACY:.2%})")
    print(f"Overall Return: {report['overall']['return']:.2f}% (Target: {TARGET_RETURN:.2f}%)")
    print(f"Sharpe Ratio: {report['overall']['sharpe_ratio']:.2f}")
    print("="*80)
    print("Asset Performance:")
    
    for asset, metrics in report['assets'].items():
        print(f"  {asset}:")
        print(f"    Accuracy: {metrics['accuracy']:.2%}")
        print(f"    Return: {metrics['return_percent']:.2f}%")
        print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    print("="*80)
    print("Next Steps:")
    print("1. Run backtesting to validate model performance")
    print("2. Deploy models for live trading")
    print("3. Continue training to improve accuracy and returns")
    print("="*80)
    
    return report

def main():
    """Main function"""
    args = parse_arguments()
    
    logger.info("Starting advanced ML model training")
    logger.info(f"Trading pairs: {args.trading_pairs}")
    logger.info(f"Force retrain: {args.force_retrain}")
    logger.info(f"Max lookback days: {args.max_lookback_days}")
    logger.info(f"Timeframes: {args.timeframes}")
    
    try:
        report = train_models(args)
        
        if report is None:
            logger.error("Training failed")
            return 1
        
        # Check if performance targets were met
        accuracy_target_met = report['overall']['accuracy'] >= TARGET_ACCURACY
        return_target_met = report['overall']['return'] >= TARGET_RETURN
        
        if accuracy_target_met and return_target_met:
            logger.info("🎉 All performance targets met! Ready for production.")
        elif accuracy_target_met:
            logger.info("📈 Accuracy target met, but return target not yet reached. Continue optimizing.")
        elif return_target_met:
            logger.info("💰 Return target met, but accuracy target not yet reached. Continue training.")
        else:
            logger.info("🔄 Continue training to reach performance targets.")
        
        return 0
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())