#!/usr/bin/env python3
"""
Enhanced Model Backtesting Script

This script provides comprehensive backtesting for ML models with the trading system.
It allows testing of model predictions and trading strategy performance on historical data
before deploying changes to the live trading system.

Features:
- Tests multiple model architectures and ensemble combinations
- Validates strategy integration with ML signals
- Generates detailed performance metrics and visualization
- Helps identify optimal configuration parameters

Usage:
    python enhanced_model_backtesting.py --pair SOL/USD --days 30 --models lstm,tcn,transformer
"""

import os
import sys
import json
import logging
import argparse
import importlib
from datetime import datetime, timedelta
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('model_backtesting.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
PANDAS_AVAILABLE = False
NUMPY_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not available, some functionality will be limited")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not available, some functionality will be limited")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available, visualization will be disabled")

# Constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_MODEL_TYPES = ['lstm', 'tcn', 'transformer', 'cnn', 'gru']
DEFAULT_BACKTEST_DAYS = 30
DEFAULT_INITIAL_CAPITAL = 20000
DEFAULT_RISK_PER_TRADE = 0.2  # 20% of capital per trade

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced ML model backtesting')
    parser.add_argument('--pair', type=str, default='SOL/USD',
                        help='Trading pair to backtest (e.g., SOL/USD)')
    parser.add_argument('--days', type=int, default=DEFAULT_BACKTEST_DAYS,
                        help=f'Number of days to backtest (default: {DEFAULT_BACKTEST_DAYS})')
    parser.add_argument('--models', type=str, default='lstm,tcn,transformer',
                        help='Comma-separated list of models to test')
    parser.add_argument('--capital', type=float, default=DEFAULT_INITIAL_CAPITAL,
                        help=f'Initial capital for backtesting (default: ${DEFAULT_INITIAL_CAPITAL})')
    parser.add_argument('--risk', type=float, default=DEFAULT_RISK_PER_TRADE,
                        help=f'Risk per trade as fraction of capital (default: {DEFAULT_RISK_PER_TRADE})')
    parser.add_argument('--confidence', type=float, default=DEFAULT_CONFIDENCE_THRESHOLD,
                        help=f'Confidence threshold for ML signals (default: {DEFAULT_CONFIDENCE_THRESHOLD})')
    parser.add_argument('--leverage', type=float, default=25.0,
                        help='Max leverage to use (default: 25.0)')
    parser.add_argument('--include-strategies', action='store_true',
                        help='Include traditional ARIMA and Adaptive strategies')
    parser.add_argument('--output', type=str, default='backtest_results',
                        help='Directory to save backtest results')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def load_historical_data(pair: str, days: int = DEFAULT_BACKTEST_DAYS):
    """
    Load historical data for backtesting.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        days: Number of days of historical data to load
        
    Returns:
        Dictionary with historical data
    """
    logger.info(f"Loading {days} days of historical data for {pair}")
    
    # Format pair for filenames
    formatted_pair = pair.replace('/', '')
    
    # Find the most suitable historical data file
    data_dir = 'historical_data'
    if not os.path.exists(data_dir):
        logger.error(f"Directory not found: {data_dir}")
        return None
    
    # List all historical data files for this pair
    pair_files = []
    for filename in os.listdir(data_dir):
        if formatted_pair in filename and filename.endswith('.csv'):
            pair_files.append(os.path.join(data_dir, filename))
    
    if not pair_files:
        logger.error(f"No historical data files found for {pair}")
        return None
    
    # Sort files by modification time (newest first)
    pair_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Load the newest file
    if not PANDAS_AVAILABLE:
        logger.error("Pandas is required to load historical data")
        return None
    
    try:
        data = pd.read_csv(pair_files[0])
        logger.info(f"Loaded {len(data)} records from {pair_files[0]}")
        
        # Filter to the requested number of days
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            cutoff_date = datetime.now() - timedelta(days=days)
            data = data[data['timestamp'] >= cutoff_date]
            logger.info(f"Filtered to {len(data)} records within last {days} days")
        
        return data
    except Exception as e:
        logger.error(f"Error loading historical data: {str(e)}")
        return None

def load_models(pair: str, model_types: List[str]):
    """
    Load ML models for backtesting.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        model_types: List of model types to load
        
    Returns:
        Dictionary of loaded models
    """
    logger.info(f"Loading models for {pair}: {', '.join(model_types)}")
    
    # Format pair for filenames
    formatted_pair = pair.replace('/', '')
    
    loaded_models = {}
    
    try:
        # Try to dynamically import TensorFlow
        tf_spec = importlib.util.find_spec('tensorflow')
        keras_spec = importlib.util.find_spec('tensorflow.keras')
        
        if tf_spec is None or keras_spec is None:
            logger.warning("TensorFlow/Keras not available, using simulated models")
            
            # Create simulated models for testing without TensorFlow
            for model_type in model_types:
                loaded_models[model_type] = {
                    'type': model_type,
                    'simulated': True,
                    'features': 20,
                    'lookback': 60
                }
            
            return loaded_models
        
        # If TensorFlow is available, try to load actual models
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        
        # Look for model files for each type
        for model_type in model_types:
            model_dir = f'models/{model_type}'
            if not os.path.exists(model_dir):
                logger.warning(f"Model directory not found: {model_dir}")
                continue
            
            # Check for model files with various naming patterns
            model_paths = [
                f'{model_dir}/{formatted_pair}.h5',
                f'{model_dir}/{formatted_pair}_{model_type}.h5'
            ]
            
            model_found = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        model = load_model(model_path)
                        loaded_models[model_type] = {
                            'model': model,
                            'path': model_path,
                            'type': model_type,
                            'input_shape': model.input_shape,
                            'simulated': False
                        }
                        logger.info(f"Loaded {model_type} model from {model_path}")
                        model_found = True
                        break
                    except Exception as e:
                        logger.error(f"Error loading model {model_path}: {str(e)}")
            
            if not model_found:
                logger.warning(f"No model file found for {model_type}, using simulated model")
                loaded_models[model_type] = {
                    'type': model_type,
                    'simulated': True,
                    'features': 20,
                    'lookback': 60
                }
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        
        # Fall back to simulated models
        for model_type in model_types:
            loaded_models[model_type] = {
                'type': model_type,
                'simulated': True,
                'features': 20,
                'lookback': 60
            }
    
    return loaded_models

def load_ensemble_config(pair: str):
    """
    Load ensemble configuration for the specified pair.
    
    Args:
        pair: Trading pair (e.g., 'SOL/USD')
        
    Returns:
        Dictionary with ensemble configuration
    """
    logger.info(f"Loading ensemble configuration for {pair}")
    
    # Format pair for filenames
    formatted_pair = pair.replace('/', '')
    
    # Check for ensemble configuration files
    ensemble_dir = 'models/ensemble'
    if not os.path.exists(ensemble_dir):
        logger.warning(f"Ensemble directory not found: {ensemble_dir}")
        return None
    
    ensemble_paths = [
        f'{ensemble_dir}/{formatted_pair}_ensemble.json',
        f'{ensemble_dir}/{formatted_pair}_weights.json'
    ]
    
    ensemble_config = {}
    
    for path in ensemble_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                
                if 'models' in config:
                    # This is the ensemble models configuration
                    ensemble_config['models'] = config['models']
                    ensemble_config['parameters'] = config.get('parameters', {})
                elif isinstance(config, dict):
                    # This is likely the weights configuration
                    ensemble_config['weights'] = config
                
                logger.info(f"Loaded ensemble configuration from {path}")
            except Exception as e:
                logger.error(f"Error loading ensemble configuration from {path}: {str(e)}")
    
    if not ensemble_config:
        logger.warning(f"No ensemble configuration found for {pair}")
        return None
    
    return ensemble_config

def load_ml_config():
    """
    Load the ML configuration file.
    
    Returns:
        Dictionary with ML configuration
    """
    logger.info("Loading ML configuration")
    
    config_path = 'ml_config.json'
    if not os.path.exists(config_path):
        logger.warning(f"ML configuration file not found: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded ML configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading ML configuration: {str(e)}")
        return None

def prepare_features(data, lookback=60, feature_cols=None):
    """
    Prepare features for model prediction.
    
    Args:
        data: DataFrame with historical data
        lookback: Number of lookback periods
        feature_cols: List of feature column names
        
    Returns:
        Dictionary with processed features
    """
    logger.info(f"Preparing features with lookback={lookback}")
    
    if not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
        logger.error("Pandas and NumPy are required for feature preparation")
        return None
    
    try:
        # If feature columns not specified, use all numeric columns
        if feature_cols is None:
            feature_cols = data.select_dtypes(include=np.number).columns.tolist()
            
            # Exclude common non-feature columns
            exclude_cols = ['timestamp', 'volume', 'target_direction', 'target_magnitude']
            for col in exclude_cols:
                if col in feature_cols:
                    feature_cols.remove(col)
        
        # Ensure we have enough lookback periods
        if len(data) < lookback:
            logger.error(f"Not enough data ({len(data)} rows) for lookback ({lookback})")
            return None
        
        # Create sequences
        X = []
        timestamps = []
        
        for i in range(len(data) - lookback):
            X.append(data[feature_cols].iloc[i:i+lookback].values)
            if 'timestamp' in data.columns:
                timestamps.append(data['timestamp'].iloc[i+lookback])
            else:
                timestamps.append(i+lookback)
        
        X = np.array(X)
        
        # Adjust feature dimensions if needed
        target_features = 20  # Target number of features
        
        if X.shape[2] != target_features:
            logger.info(f"Reshaping features from {X.shape[2]} to {target_features} features")
            
            if X.shape[2] > target_features:
                # If we have more features than needed, select the first N features
                X = X[:, :, :target_features]
            else:
                # If we have fewer features than needed, pad with zeros
                X_padded = np.zeros((X.shape[0], X.shape[1], target_features))
                X_padded[:, :, :X.shape[2]] = X
                X = X_padded
        
        logger.info(f"Prepared {X.shape[0]} sequences with shape {X.shape}")
        
        return {
            'X': X,
            'timestamps': timestamps,
            'feature_cols': feature_cols[:min(len(feature_cols), target_features)]
        }
    except Exception as e:
        logger.error(f"Error preparing features: {str(e)}")
        return None

def simulate_model_prediction(X, model_info, confidence_level=0.6):
    """
    Simulate model predictions when actual models aren't available.
    
    Args:
        X: Input features
        model_info: Model information
        confidence_level: Base confidence level
        
    Returns:
        Tuple of (predictions, confidence scores)
    """
    if not NUMPY_AVAILABLE:
        # Fall back to random predictions
        import random
        n_samples = len(X)
        predictions = [1 if random.random() > 0.5 else 0 for _ in range(n_samples)]
        confidence = [0.5 + random.random() * 0.3 for _ in range(n_samples)]
        return predictions, confidence
    
    # Use NumPy for more controlled simulation
    n_samples = len(X)
    
    # Generate prediction probabilities with a slight bias towards accurate predictions
    probabilities = np.random.normal(0.52, 0.15, n_samples)
    
    # Convert to binary predictions (1 for up, 0 for down)
    predictions = (probabilities > 0.5).astype(int)
    
    # Generate confidence scores
    base_confidence = confidence_level
    confidence = np.clip(np.abs(probabilities - 0.5) * 2 * base_confidence, 0.1, 0.95)
    
    return predictions, confidence

def generate_predictions(data, models, ensemble_config=None, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Generate predictions from models.
    
    Args:
        data: Historical data
        models: Dictionary of loaded models
        ensemble_config: Ensemble configuration
        confidence_threshold: Confidence threshold for signals
        
    Returns:
        Dictionary with model predictions
    """
    logger.info("Generating model predictions")
    
    if data is None or not models:
        logger.error("Cannot generate predictions without data and models")
        return None
    
    # Determine the lookback period to use
    lookback = 60  # Default lookback
    for model_info in models.values():
        if 'lookback' in model_info:
            lookback = model_info['lookback']
            break
    
    # Prepare features
    features = prepare_features(data, lookback=lookback)
    if not features:
        logger.error("Failed to prepare features")
        return None
    
    X = features['X']
    timestamps = features['timestamps']
    
    # Generate predictions for each model
    all_predictions = {}
    
    for model_type, model_info in models.items():
        logger.info(f"Generating predictions for {model_type} model")
        
        try:
            if model_info.get('simulated', True):
                # Simulate predictions for testing
                predictions, confidence = simulate_model_prediction(
                    X, model_info, confidence_level=confidence_threshold
                )
            else:
                # Use actual model for predictions
                if 'model' in model_info:
                    model = model_info['model']
                    
                    # Get predictions from the model
                    raw_predictions = model.predict(X)
                    
                    # Convert to binary predictions and confidence scores
                    if raw_predictions.ndim > 1 and raw_predictions.shape[1] > 1:
                        # Multi-class output (one-hot encoded)
                        predictions = np.argmax(raw_predictions, axis=1)
                        confidence = np.max(raw_predictions, axis=1)
                    else:
                        # Binary output
                        raw_predictions = raw_predictions.flatten()
                        predictions = (raw_predictions > 0.5).astype(int)
                        confidence = np.abs(raw_predictions - 0.5) * 2
                else:
                    # Fall back to simulated predictions
                    predictions, confidence = simulate_model_prediction(
                        X, model_info, confidence_level=confidence_threshold
                    )
            
            # Store predictions
            all_predictions[model_type] = {
                'predictions': predictions,
                'confidence': confidence,
                'timestamps': timestamps
            }
            
            logger.info(f"Generated {len(predictions)} predictions for {model_type} model")
        except Exception as e:
            logger.error(f"Error generating predictions for {model_type} model: {str(e)}")
    
    # Generate ensemble predictions if ensemble configuration is available
    if ensemble_config and 'weights' in ensemble_config:
        logger.info("Generating ensemble predictions")
        
        try:
            # Get ensemble weights
            weights = ensemble_config['weights']
            
            # Normalize weights to sum to 1.0
            weight_sum = sum(weights.values())
            normalized_weights = {k: v / weight_sum for k, v in weights.items()}
            
            # Calculate weighted predictions
            ensemble_predictions = np.zeros(len(timestamps))
            ensemble_confidence = np.zeros(len(timestamps))
            
            for model_type, weight in normalized_weights.items():
                if model_type in all_predictions:
                    model_predictions = all_predictions[model_type]
                    # Add weighted contribution to ensemble prediction
                    for i in range(len(timestamps)):
                        if model_predictions['predictions'][i] == 1:
                            ensemble_predictions[i] += weight
                        ensemble_confidence[i] += weight * model_predictions['confidence'][i]
            
            # Convert to binary predictions based on weighted majority
            binary_predictions = (ensemble_predictions > 0.5).astype(int)
            
            # Store ensemble predictions
            all_predictions['ensemble'] = {
                'predictions': binary_predictions,
                'confidence': ensemble_confidence,
                'timestamps': timestamps,
                'weights': normalized_weights
            }
            
            logger.info(f"Generated {len(binary_predictions)} ensemble predictions")
        except Exception as e:
            logger.error(f"Error generating ensemble predictions: {str(e)}")
    
    return all_predictions

def generate_trading_signals(predictions, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Generate trading signals from model predictions.
    
    Args:
        predictions: Dictionary with model predictions
        confidence_threshold: Confidence threshold for signals
        
    Returns:
        Dictionary with trading signals
    """
    logger.info(f"Generating trading signals with confidence threshold {confidence_threshold}")
    
    if not predictions:
        logger.error("Cannot generate signals without predictions")
        return None
    
    # Generate signals for each model
    all_signals = {}
    
    for model_type, model_predictions in predictions.items():
        logger.info(f"Generating signals for {model_type} model")
        
        try:
            # Extract predictions and confidence scores
            binary_predictions = model_predictions['predictions']
            confidence = model_predictions['confidence']
            timestamps = model_predictions['timestamps']
            
            # Generate signals based on predictions and confidence
            signals = []
            
            for i in range(len(binary_predictions)):
                # Determine signal direction
                if confidence[i] >= confidence_threshold:
                    if binary_predictions[i] == 1:
                        signal = 'BUY'
                    else:
                        signal = 'SELL'
                else:
                    signal = 'NEUTRAL'
                
                signals.append({
                    'timestamp': timestamps[i],
                    'signal': signal,
                    'confidence': confidence[i],
                    'prediction': binary_predictions[i]
                })
            
            # Store signals
            all_signals[model_type] = signals
            
            # Count signal types
            signal_counts = {
                'BUY': sum(1 for s in signals if s['signal'] == 'BUY'),
                'SELL': sum(1 for s in signals if s['signal'] == 'SELL'),
                'NEUTRAL': sum(1 for s in signals if s['signal'] == 'NEUTRAL')
            }
            
            logger.info(f"Generated {len(signals)} signals for {model_type} model: "
                       f"{signal_counts['BUY']} BUY, {signal_counts['SELL']} SELL, "
                       f"{signal_counts['NEUTRAL']} NEUTRAL")
        except Exception as e:
            logger.error(f"Error generating signals for {model_type} model: {str(e)}")
    
    return all_signals

def simulate_trading(data, signals, pair, initial_capital=DEFAULT_INITIAL_CAPITAL,
                    risk_per_trade=DEFAULT_RISK_PER_TRADE, max_leverage=25.0):
    """
    Simulate trading based on signals.
    
    Args:
        data: Historical data
        signals: Dictionary with trading signals
        pair: Trading pair
        initial_capital: Initial capital for trading
        risk_per_trade: Risk per trade as fraction of capital
        max_leverage: Maximum leverage to use
        
    Returns:
        Dictionary with trading results
    """
    logger.info(f"Simulating trading with initial capital ${initial_capital}")
    
    if data is None or not signals:
        logger.error("Cannot simulate trading without data and signals")
        return None
    
    # Prepare price data
    if not PANDAS_AVAILABLE:
        logger.error("Pandas is required for trading simulation")
        return None
    
    try:
        # Ensure we have price columns
        if 'close' not in data.columns:
            if 'price' in data.columns:
                data['close'] = data['price']
            else:
                logger.error("No price column found in data")
                return None
        
        # Create a mapping of timestamps to prices
        price_map = {}
        
        if 'timestamp' in data.columns:
            for _, row in data.iterrows():
                price_map[row['timestamp']] = row['close']
        else:
            # If no timestamp column, use index
            for i, row in data.iterrows():
                price_map[i] = row['close']
        
        # Simulate trading for each model
        all_results = {}
        
        for model_type, model_signals in signals.items():
            logger.info(f"Simulating trading based on {model_type} model signals")
            
            # Initialize trading state
            capital = initial_capital
            position = None
            entry_price = 0
            entry_timestamp = None
            trades = []
            
            # Track portfolio value over time
            portfolio_history = [{'timestamp': model_signals[0]['timestamp'], 'value': capital}]
            
            # Process signals sequentially
            for i, signal_data in enumerate(model_signals):
                timestamp = signal_data['timestamp']
                signal = signal_data['signal']
                confidence = signal_data['confidence']
                
                # Lookup current price
                if timestamp not in price_map:
                    continue
                
                current_price = price_map[timestamp]
                
                # Calculate position size based on risk and confidence
                risk_adjusted = risk_per_trade * min(1.0, confidence / 0.8)
                position_size = (capital * risk_adjusted) / current_price
                
                # Apply leverage (proportional to confidence)
                effective_leverage = max_leverage * min(1.0, confidence / 0.9)
                position_size *= effective_leverage
                
                # Execute trading logic
                if position is None:
                    # No position, check for entry signals
                    if signal == 'BUY':
                        # Enter long position
                        position = 'LONG'
                        entry_price = current_price
                        entry_timestamp = timestamp
                        logger.info(f"[{model_type}] LONG entry at {current_price} with size {position_size}")
                    elif signal == 'SELL':
                        # Enter short position
                        position = 'SHORT'
                        entry_price = current_price
                        entry_timestamp = timestamp
                        logger.info(f"[{model_type}] SHORT entry at {current_price} with size {position_size}")
                else:
                    # Already in a position, check for exit signals
                    exit_signal = False
                    
                    if position == 'LONG':
                        if signal == 'SELL':
                            exit_signal = True
                        elif i == len(model_signals) - 1:
                            # Force exit at the end of simulation
                            exit_signal = True
                    elif position == 'SHORT':
                        if signal == 'BUY':
                            exit_signal = True
                        elif i == len(model_signals) - 1:
                            # Force exit at the end of simulation
                            exit_signal = True
                    
                    if exit_signal:
                        # Close the position
                        if position == 'LONG':
                            pnl = (current_price - entry_price) * position_size
                            logger.info(f"[{model_type}] LONG exit at {current_price}, "
                                      f"PnL: ${pnl:.2f} ({(current_price/entry_price-1)*100:.2f}%)")
                        else:
                            pnl = (entry_price - current_price) * position_size
                            logger.info(f"[{model_type}] SHORT exit at {current_price}, "
                                      f"PnL: ${pnl:.2f} ({(entry_price/current_price-1)*100:.2f}%)")
                        
                        capital += pnl
                        trades.append({
                            'entry_timestamp': entry_timestamp,
                            'entry_price': entry_price,
                            'exit_timestamp': timestamp,
                            'exit_price': current_price,
                            'position': position,
                            'pnl': pnl,
                            'pnl_percent': (pnl / initial_capital) * 100
                        })
                        
                        position = None
                        entry_price = 0
                        entry_timestamp = None
                
                # Record portfolio value
                portfolio_value = capital
                if position == 'LONG':
                    portfolio_value += (current_price - entry_price) * position_size
                elif position == 'SHORT':
                    portfolio_value += (entry_price - current_price) * position_size
                
                portfolio_history.append({
                    'timestamp': timestamp,
                    'value': portfolio_value
                })
            
            # Calculate trading statistics
            if trades:
                total_pnl = sum(trade['pnl'] for trade in trades)
                winning_trades = [trade for trade in trades if trade['pnl'] > 0]
                losing_trades = [trade for trade in trades if trade['pnl'] <= 0]
                win_rate = len(winning_trades) / len(trades) if trades else 0
                
                if winning_trades:
                    avg_win = sum(trade['pnl'] for trade in winning_trades) / len(winning_trades)
                else:
                    avg_win = 0
                    
                if losing_trades:
                    avg_loss = sum(trade['pnl'] for trade in losing_trades) / len(losing_trades)
                else:
                    avg_loss = 0
                
                profit_factor = abs(sum(trade['pnl'] for trade in winning_trades) / 
                                   sum(trade['pnl'] for trade in losing_trades)) if losing_trades else float('inf')
            else:
                total_pnl = 0
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
            
            # Store results
            all_results[model_type] = {
                'capital': capital,
                'total_pnl': total_pnl,
                'total_pnl_percent': (total_pnl / initial_capital) * 100,
                'num_trades': len(trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'trades': trades,
                'portfolio_history': portfolio_history
            }
            
            logger.info(f"[{model_type}] Trading simulation results:")
            logger.info(f"  Final capital: ${capital:.2f}")
            logger.info(f"  Total P&L: ${total_pnl:.2f} ({(total_pnl/initial_capital)*100:.2f}%)")
            logger.info(f"  Number of trades: {len(trades)}")
            logger.info(f"  Win rate: {win_rate*100:.2f}%")
            logger.info(f"  Profit factor: {profit_factor:.2f}")
        
        return all_results
    except Exception as e:
        logger.error(f"Error simulating trading: {str(e)}")
        return None

def generate_report(backtest_results, signals, args, output_dir='backtest_results'):
    """
    Generate a backtest report.
    
    Args:
        backtest_results: Dictionary with backtest results
        signals: Dictionary with trading signals
        args: Command line arguments
        output_dir: Directory to save the report
        
    Returns:
        Path to the report file
    """
    logger.info("Generating backtest report")
    
    if not backtest_results:
        logger.error("Cannot generate report without backtest results")
        return None
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Report filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"{args.pair.replace('/', '')}_backtest_{timestamp}.txt"
    report_path = os.path.join(output_dir, report_filename)
    
    try:
        with open(report_path, 'w') as f:
            f.write("==========================================\n")
            f.write("ENHANCED MODEL BACKTESTING REPORT\n")
            f.write("==========================================\n\n")
            
            f.write(f"Trading Pair: {args.pair}\n")
            f.write(f"Backtest Period: {args.days} days\n")
            f.write(f"Models: {args.models}\n")
            f.write(f"Initial Capital: ${args.capital:,.2f}\n")
            f.write(f"Risk Per Trade: {args.risk*100:.1f}%\n")
            f.write(f"Max Leverage: {args.leverage:.1f}x\n")
            f.write(f"Confidence Threshold: {args.confidence:.2f}\n\n")
            
            f.write("==========================================\n")
            f.write("TRADING PERFORMANCE SUMMARY\n")
            f.write("==========================================\n\n")
            
            # Sort results by total P&L (descending)
            sorted_results = sorted(
                backtest_results.items(),
                key=lambda x: x[1]['total_pnl'],
                reverse=True
            )
            
            for model_type, results in sorted_results:
                f.write(f"MODEL: {model_type.upper()}\n")
                f.write(f"Final Capital: ${results['capital']:,.2f}\n")
                f.write(f"Total P&L: ${results['total_pnl']:,.2f} ({results['total_pnl_percent']:.2f}%)\n")
                f.write(f"Number of Trades: {results['num_trades']}\n")
                f.write(f"Win Rate: {results['win_rate']*100:.2f}%\n")
                f.write(f"Average Win: ${results['avg_win']:,.2f}\n")
                f.write(f"Average Loss: ${results['avg_loss']:,.2f}\n")
                f.write(f"Profit Factor: {results['profit_factor']:.2f}\n\n")
            
            f.write("==========================================\n")
            f.write("SIGNAL STATISTICS\n")
            f.write("==========================================\n\n")
            
            for model_type, model_signals in signals.items():
                buy_signals = sum(1 for s in model_signals if s['signal'] == 'BUY')
                sell_signals = sum(1 for s in model_signals if s['signal'] == 'SELL')
                neutral_signals = sum(1 for s in model_signals if s['signal'] == 'NEUTRAL')
                total_signals = len(model_signals)
                
                f.write(f"MODEL: {model_type.upper()}\n")
                f.write(f"Total Signals: {total_signals}\n")
                f.write(f"BUY Signals: {buy_signals} ({buy_signals/total_signals*100:.2f}%)\n")
                f.write(f"SELL Signals: {sell_signals} ({sell_signals/total_signals*100:.2f}%)\n")
                f.write(f"NEUTRAL Signals: {neutral_signals} ({neutral_signals/total_signals*100:.2f}%)\n\n")
            
            f.write("==========================================\n")
            f.write("TOP 10 TRADES BY PROFIT\n")
            f.write("==========================================\n\n")
            
            for model_type, results in sorted_results:
                if results['trades']:
                    f.write(f"MODEL: {model_type.upper()}\n")
                    
                    # Sort trades by P&L (descending)
                    sorted_trades = sorted(
                        results['trades'],
                        key=lambda x: x['pnl'],
                        reverse=True
                    )
                    
                    # Take top 10 trades
                    top_trades = sorted_trades[:10]
                    
                    for i, trade in enumerate(top_trades, 1):
                        entry_time = trade['entry_timestamp']
                        exit_time = trade['exit_timestamp']
                        if isinstance(entry_time, (pd.Timestamp, datetime)):
                            entry_time = entry_time.strftime('%Y-%m-%d %H:%M:%S')
                        if isinstance(exit_time, (pd.Timestamp, datetime)):
                            exit_time = exit_time.strftime('%Y-%m-%d %H:%M:%S')
                        
                        f.write(f"{i}. {trade['position']} {entry_time} to {exit_time}\n")
                        f.write(f"   Entry: ${trade['entry_price']:.2f}, Exit: ${trade['exit_price']:.2f}\n")
                        f.write(f"   P&L: ${trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%)\n\n")
            
            # Generate visualizations if matplotlib is available
            if MATPLOTLIB_AVAILABLE:
                # Save equity curves
                equity_curve_path = os.path.join(output_dir, f"{args.pair.replace('/', '')}_equity_curves_{timestamp}.png")
                generate_equity_curves(backtest_results, equity_curve_path)
                f.write(f"Equity curves chart saved to: {os.path.basename(equity_curve_path)}\n\n")
            
            f.write("==========================================\n")
            f.write("CONCLUSIONS AND RECOMMENDATIONS\n")
            f.write("==========================================\n\n")
            
            # Find best model
            best_model = sorted_results[0][0]
            best_pnl = sorted_results[0][1]['total_pnl']
            best_win_rate = sorted_results[0][1]['win_rate']
            
            f.write(f"BEST PERFORMING MODEL: {best_model.upper()}\n")
            f.write(f"Total P&L: ${best_pnl:,.2f} ({sorted_results[0][1]['total_pnl_percent']:.2f}%)\n")
            f.write(f"Win Rate: {best_win_rate*100:.2f}%\n\n")
            
            # Generate recommendations
            if best_pnl > 0:
                f.write("RECOMMENDATIONS:\n")
                f.write(f"1. Consider using the {best_model.upper()} model for live trading.\n")
                
                if best_win_rate < 0.5:
                    f.write("2. Despite profitability, the win rate is below 50%. Consider adjusting\n")
                    f.write("   confidence thresholds to improve trade quality.\n")
                
                if any(r[1]['total_pnl'] < 0 for r in sorted_results):
                    f.write("3. Some models performed poorly. Consider excluding them from the ensemble.\n")
            else:
                f.write("RECOMMENDATIONS:\n")
                f.write("1. All models need further optimization before live trading.\n")
                f.write("2. Consider adjusting confidence thresholds and risk parameters.\n")
                f.write("3. Retrain models with more recent data to improve performance.\n")
        
        logger.info(f"Backtest report saved to {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return None

def generate_equity_curves(backtest_results, output_path):
    """
    Generate equity curves chart.
    
    Args:
        backtest_results: Dictionary with backtest results
        output_path: Path to save the chart
        
    Returns:
        None
    """
    if not MATPLOTLIB_AVAILABLE or not PANDAS_AVAILABLE:
        logger.warning("Matplotlib or Pandas not available, skipping equity curves visualization")
        return
    
    try:
        plt.figure(figsize=(12, 8))
        
        for model_type, results in backtest_results.items():
            # Extract equity curve data
            equity_data = results['portfolio_history']
            
            # Convert to DataFrame
            df = pd.DataFrame(equity_data)
            
            # Plot equity curve
            plt.plot(range(len(df)), df['value'], label=f"{model_type} (${results['capital']:,.2f})")
        
        plt.title('Equity Curves by Model')
        plt.xlabel('Time (Trading Intervals)')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Equity curves chart saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating equity curves: {str(e)}")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Parse model types
    model_types = [model.strip().lower() for model in args.models.split(',')]
    
    # Load historical data
    data = load_historical_data(args.pair, args.days)
    if data is None:
        logger.error("Failed to load historical data")
        return 1
    
    # Load models
    models = load_models(args.pair, model_types)
    if not models:
        logger.error("Failed to load models")
        return 1
    
    # Load ensemble configuration
    ensemble_config = load_ensemble_config(args.pair)
    
    # Load ML configuration
    ml_config = load_ml_config()
    
    # Generate predictions
    predictions = generate_predictions(data, models, ensemble_config, args.confidence)
    if not predictions:
        logger.error("Failed to generate predictions")
        return 1
    
    # Generate trading signals
    signals = generate_trading_signals(predictions, args.confidence)
    if not signals:
        logger.error("Failed to generate trading signals")
        return 1
    
    # Simulate trading
    backtest_results = simulate_trading(
        data, signals, args.pair, args.capital, args.risk, args.leverage
    )
    if not backtest_results:
        logger.error("Failed to simulate trading")
        return 1
    
    # Generate report
    report_path = generate_report(backtest_results, signals, args, args.output)
    if report_path:
        logger.info(f"Backtest complete. Report saved to {report_path}")
    else:
        logger.error("Failed to generate report")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())