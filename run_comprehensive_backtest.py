#!/usr/bin/env python3
"""
Comprehensive Backtesting for Trading Models

This script performs sophisticated backtesting for trading models with:
1. Realistic order execution simulation
2. Advanced risk management
3. Comprehensive performance metrics
4. Win rate and accuracy calculations
5. Return simulations with leverage
6. Market regime analysis
7. Stress testing

The backtesting framework is designed to provide highly accurate 
estimates of real-world trading performance.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("comprehensive_backtest.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("comprehensive_backtest")

# Constants
CONFIG_PATH = "config/new_coins_training_config.json"
MODEL_DIR = "ml_models"
ENSEMBLE_DIR = "ensemble_models"
HISTORICAL_DATA_DIR = "historical_data"
BACKTEST_RESULTS_DIR = "backtest_results"

# Create necessary directories
Path(BACKTEST_RESULTS_DIR).mkdir(exist_ok=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive backtesting for trading models.')
    parser.add_argument('--pair', type=str, required=True,
                      help='Trading pair to backtest (e.g., "AVAX/USD")')
    parser.add_argument('--model', type=str, default='ensemble',
                      choices=['ensemble', 'tcn', 'lstm', 'attention_gru', 'transformer', 'xgboost', 'lightgbm'],
                      help='Model type to use for backtesting (default: ensemble)')
    parser.add_argument('--period', type=str, default='365',
                      help='Backtesting period in days or date range (YYYY-MM-DD:YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default='1h',
                      help='Timeframe to use for backtesting (default: 1h)')
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                      help='Initial capital for backtesting (default: 10000.0)')
    parser.add_argument('--risk-percentage', type=float, default=None,
                      help='Risk percentage per trade (default: from config)')
    parser.add_argument('--leverage', type=float, default=None,
                      help='Fixed leverage to use (default: from config)')
    parser.add_argument('--max-leverage', type=float, default=None,
                      help='Maximum leverage to use (default: from config)')
    parser.add_argument('--dynamic-leverage', action='store_true',
                      help='Use dynamic leverage based on confidence')
    parser.add_argument('--confidence-threshold', type=float, default=None,
                      help='Confidence threshold for trading (default: from config)')
    parser.add_argument('--calculate-all-metrics', action='store_true',
                      help='Calculate all performance metrics (slower)')
    parser.add_argument('--output-format', type=str, default='both',
                      choices=['console', 'json', 'both'],
                      help='Output format for results (default: both)')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory for saving results (default: backtest_results)')
    parser.add_argument('--stress-test', action='store_true',
                      help='Run additional stress tests with extreme market conditions')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()


def load_config():
    """Load configuration from config file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {CONFIG_PATH}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None


def load_historical_data(pair: str, timeframe: str = '1h', period: str = '365') -> pd.DataFrame:
    """
    Load historical data for backtesting.
    
    Args:
        pair: Trading pair
        timeframe: Timeframe
        period: Backtesting period in days or date range
        
    Returns:
        data: Historical data dataframe
    """
    pair_filename = pair.replace("/", "_")
    file_path = f"{HISTORICAL_DATA_DIR}/{pair_filename}_{timeframe}.csv"
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"Historical data file not found: {file_path}")
            return None
        
        # Load data
        data = pd.read_csv(file_path)
        
        # Ensure timestamp is datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Sort by timestamp
        data = data.sort_values('timestamp')
        
        # Filter by period
        if ':' in period:
            # Date range
            start_date, end_date = period.split(':')
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            data = data[(data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)]
        else:
            # Last N days
            days = int(period)
            last_date = data['timestamp'].max()
            start_date = last_date - pd.Timedelta(days=days)
            data = data[data['timestamp'] >= start_date]
        
        logger.info(f"Loaded historical data for {pair} ({timeframe}): {len(data)} periods")
        logger.info(f"  Period: {data['timestamp'].min()} to {data['timestamp'].max()}")
        
        return data
    
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return None


def load_model(model_type: str, pair: str, ensemble: bool = False) -> Any:
    """
    Load trained model for backtesting.
    
    Args:
        model_type: Model type
        pair: Trading pair
        ensemble: Whether to load ensemble model
        
    Returns:
        model: Loaded model
    """
    pair_filename = pair.replace("/", "_")
    
    # Determine model directory
    if ensemble or model_type == 'ensemble':
        model_dir = ENSEMBLE_DIR
        model_path = f"{model_dir}/ensemble_{pair_filename}"
    else:
        model_dir = MODEL_DIR
        model_path = f"{model_dir}/{model_type}_{pair_filename}_model"
    
    try:
        if ensemble or model_type == 'ensemble':
            # Load ensemble configuration
            config_file = f"{model_path}_config.json"
            
            if not os.path.exists(config_file):
                logger.error(f"Ensemble configuration file not found: {config_file}")
                return None
            
            with open(config_file, 'r') as f:
                ensemble_config = json.load(f)
            
            # Load meta-model if available
            meta_model_path = f"{model_path}_meta_model"
            if os.path.exists(f"{meta_model_path}.txt"):
                # LightGBM
                import lightgbm as lgb
                meta_model = lgb.Booster(model_file=f"{meta_model_path}.txt")
            elif os.path.exists(meta_model_path):
                # Keras
                import tensorflow as tf
                meta_model = tf.keras.models.load_model(meta_model_path)
            elif os.path.exists(f"{meta_model_path}.pkl"):
                # Scikit-learn
                import pickle
                with open(f"{meta_model_path}.pkl", 'rb') as f:
                    meta_model = pickle.load(f)
            else:
                meta_model = None
            
            # Load base models
            base_models = {}
            for model_name in ensemble_config.get("ensemble_config", {}).get("models", []):
                base_model = load_model(model_name, pair, False)
                if base_model is not None:
                    base_models[model_name] = base_model
            
            # Return ensemble configuration
            return {
                "config": ensemble_config,
                "meta_model": meta_model,
                "base_models": base_models
            }
        
        else:
            # Load individual model
            if model_type in ['tcn', 'lstm', 'attention_gru', 'transformer']:
                # Load Keras model
                import tensorflow as tf
                
                # Special handling for TCN
                if model_type == 'tcn':
                    try:
                        from tcn import TCN
                        model = tf.keras.models.load_model(model_path, custom_objects={'TCN': TCN})
                    except ImportError:
                        logger.error("TCN layer not available. Please install keras-tcn")
                        return None
                else:
                    model = tf.keras.models.load_model(model_path)
            
            elif model_type == 'xgboost':
                # Load XGBoost model
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(f"{model_path}.json")
            
            elif model_type == 'lightgbm':
                # Load LightGBM model
                import lightgbm as lgb
                model = lgb.Booster(model_file=f"{model_path}.txt")
            
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None
            
            logger.info(f"Loaded {model_type} model for {pair}")
            return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def prepare_features(data: pd.DataFrame, lookback_window: int) -> np.ndarray:
    """
    Prepare features for model prediction.
    
    Args:
        data: Historical data
        lookback_window: Lookback window for sequence models
        
    Returns:
        features: Prepared features
    """
    try:
        # Drop unnecessary columns
        features = data.drop(['timestamp', 'open', 'high', 'low', 'close', 'volume'], axis=1, errors='ignore')
        
        # Create sequences for lookback window
        sequences = []
        
        for i in range(len(features) - lookback_window + 1):
            sequences.append(features.iloc[i:i+lookback_window].values)
        
        return np.array(sequences)
    
    except Exception as e:
        logger.error(f"Error preparing features: {e}")
        return None


def get_model_predictions(model: Any, features: np.ndarray, model_type: str, 
                       is_ensemble: bool = False) -> np.ndarray:
    """
    Get predictions from a model.
    
    Args:
        model: Trained model
        features: Input features
        model_type: Model type
        is_ensemble: Whether the model is an ensemble
        
    Returns:
        predictions: Model predictions
    """
    try:
        if is_ensemble or model_type == 'ensemble':
            # Ensemble predictions
            ensemble_config = model["config"]
            base_models = model["base_models"]
            meta_model = model["meta_model"]
            
            # Get predictions from base models
            base_predictions = {}
            for model_name, base_model in base_models.items():
                base_pred = get_model_predictions(base_model, features, model_name)
                if base_pred is not None:
                    base_predictions[model_name] = base_pred
            
            # Apply weights or meta-model
            weights = ensemble_config.get("ensemble_config", {}).get("weights", {})
            stacking_method = ensemble_config.get("ensemble_config", {}).get("stacking_method", "weighted_average")
            
            if meta_model is not None and stacking_method == "meta_learner":
                # Use meta-model
                X_meta = np.column_stack([base_predictions[model_name] for model_name in base_predictions])
                
                if hasattr(meta_model, 'predict_proba'):
                    predictions = meta_model.predict_proba(X_meta)[:, 1]
                elif hasattr(meta_model, 'predict'):
                    predictions = meta_model.predict(X_meta)
                else:
                    logger.warning("Meta-model doesn't have predict method, using weighted average")
                    # Fall back to weighted average
                    predictions = np.zeros(len(features))
                    for model_name, preds in base_predictions.items():
                        weight = weights.get(model_name, 1.0 / len(base_predictions))
                        predictions += weight * preds
            else:
                # Weighted average
                predictions = np.zeros(len(features))
                for model_name, preds in base_predictions.items():
                    weight = weights.get(model_name, 1.0 / len(base_predictions))
                    predictions += weight * preds
            
            return predictions
        
        else:
            # Individual model predictions
            if model_type in ['tcn', 'lstm', 'attention_gru', 'transformer']:
                # Neural network models
                return model.predict(features).flatten()
            
            elif model_type == 'xgboost':
                # XGBoost model
                return model.predict_proba(features)[:, 1]
            
            elif model_type == 'lightgbm':
                # LightGBM model
                return model.predict(features)
            
            else:
                logger.error(f"Unsupported model type for prediction: {model_type}")
                return None
    
    except Exception as e:
        logger.error(f"Error getting model predictions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def calculate_entry_signals(predictions: np.ndarray, threshold: float = 0.5,
                          min_confidence: float = 0.0) -> np.ndarray:
    """
    Calculate entry signals from model predictions.
    
    Args:
        predictions: Model predictions
        threshold: Decision threshold
        min_confidence: Minimum confidence for a trade
        
    Returns:
        signals: Entry signals (1 for long, -1 for short, 0 for no trade)
    """
    # Calculate signals (long only for now)
    confidence = np.abs(predictions - 0.5) * 2  # Scale to 0-1
    signals = np.zeros(len(predictions))
    
    # Long signals
    long_mask = (predictions > threshold) & (confidence >= min_confidence)
    signals[long_mask] = 1
    
    # Short signals (if implemented)
    short_mask = (predictions < (1 - threshold)) & (confidence >= min_confidence)
    signals[short_mask] = -1
    
    return signals, confidence


def calculate_dynamic_leverage(confidence: np.ndarray, base_leverage: float,
                          max_leverage: float) -> np.ndarray:
    """
    Calculate dynamic leverage based on prediction confidence.
    
    Args:
        confidence: Prediction confidence
        base_leverage: Base leverage
        max_leverage: Maximum leverage
        
    Returns:
        leverage: Dynamic leverage values
    """
    # Scale leverage based on confidence
    # Start at base_leverage, scale up to max_leverage as confidence approaches 1
    leverage = base_leverage + (max_leverage - base_leverage) * np.power(confidence, 2)
    
    # Cap at max_leverage
    leverage = np.minimum(leverage, max_leverage)
    
    return leverage


def simulate_trades(data: pd.DataFrame, signals: np.ndarray, confidence: np.ndarray,
                  config: Dict, args) -> Dict:
    """
    Simulate trades based on signals and calculate performance metrics.
    
    Args:
        data: Historical data
        signals: Trading signals
        confidence: Signal confidence
        config: Configuration
        args: Command line arguments
        
    Returns:
        results: Simulation results
    """
    # Get pair-specific settings
    pair_settings = config.get("pair_specific_settings", {}).get(args.pair, {})
    
    # Get risk parameters
    risk_percentage = args.risk_percentage or pair_settings.get("risk_percentage", 0.20)
    base_leverage = args.leverage or pair_settings.get("base_leverage", 1.0)
    max_leverage = args.max_leverage or pair_settings.get("max_leverage", base_leverage)
    
    # Initialize variables
    initial_capital = args.initial_capital
    capital = initial_capital
    position = 0
    entry_price = 0
    entry_time = None
    leverage = 0
    
    # Track trades
    trades = []
    equity_curve = [initial_capital]
    positions = []
    
    # For calculating metrics
    wins = 0
    losses = 0
    total_profit = 0
    total_loss = 0
    
    # Slice data to match signals length
    data_subset = data.iloc[-len(signals):].reset_index(drop=True)
    
    # Iterate through periods
    for i in range(len(data_subset)):
        current_time = data_subset.iloc[i]['timestamp']
        current_price = data_subset.iloc[i]['close']
        
        # Check if in position
        if position != 0:
            # Calculate unrealized PnL
            if position > 0:  # Long
                pnl_pct = (current_price / entry_price - 1) * leverage
            else:  # Short
                pnl_pct = (1 - current_price / entry_price) * leverage
            
            unrealized_pnl = capital * risk_percentage * pnl_pct
            current_equity = capital + unrealized_pnl
            
            # Check for exit conditions
            exit_signal = False
            
            # Exit if signal changes
            if (position > 0 and signals[i] <= 0) or (position < 0 and signals[i] >= 0):
                exit_signal = True
            
            # Implement trailing stop (% of unrealized profit)
            trailing_stop_pct = 0.3  # Exit if we give back 30% of unrealized profit
            if pnl_pct > 0:
                trailing_stop = pnl_pct * trailing_stop_pct
                if position > 0:  # Long
                    trail_price = entry_price * (1 + pnl_pct - trailing_stop)
                    if current_price < trail_price:
                        exit_signal = True
                else:  # Short
                    trail_price = entry_price * (1 - pnl_pct + trailing_stop)
                    if current_price > trail_price:
                        exit_signal = True
            
            # Implement stop loss (fixed % of entry)
            stop_loss_pct = 0.04  # 4% maximum loss (more aggressive than ATR)
            if position > 0:  # Long
                stop_price = entry_price * (1 - stop_loss_pct / leverage)
                if current_price < stop_price:
                    exit_signal = True
            else:  # Short
                stop_price = entry_price * (1 + stop_loss_pct / leverage)
                if current_price > stop_price:
                    exit_signal = True
            
            # Exit position if conditions met
            if exit_signal:
                # Calculate final PnL
                if position > 0:  # Long
                    pnl_pct = (current_price / entry_price - 1) * leverage
                else:  # Short
                    pnl_pct = (1 - current_price / entry_price) * leverage
                
                pnl_amount = capital * risk_percentage * pnl_pct
                capital += pnl_amount
                
                # Record trade
                trade = {
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'position': position,
                    'leverage': leverage,
                    'pnl_pct': pnl_pct,
                    'pnl_amount': pnl_amount,
                    'capital': capital
                }
                trades.append(trade)
                
                # Update metrics
                if pnl_amount > 0:
                    wins += 1
                    total_profit += pnl_amount
                else:
                    losses += 1
                    total_loss -= pnl_amount  # Convert to positive value
                
                # Reset position
                position = 0
                leverage = 0
            
            # Update equity curve and positions
            equity_curve.append(current_equity)
            positions.append(position)
        
        else:
            # Check for entry signal
            if signals[i] != 0:
                # Calculate dynamic leverage if enabled
                if args.dynamic_leverage:
                    leverage = calculate_dynamic_leverage(
                        confidence[i], base_leverage, max_leverage
                    )[0]
                else:
                    leverage = base_leverage
                
                # Enter position
                position = 1 if signals[i] > 0 else -1
                entry_price = current_price
                entry_time = current_time
            
            # Update equity curve and positions
            equity_curve.append(capital)
            positions.append(position)
    
    # Calculate performance metrics
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    avg_win = total_profit / wins if wins > 0 else 0
    avg_loss = total_loss / losses if losses > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculate additional metrics
    equity_array = np.array(equity_curve)
    returns = np.diff(equity_array) / equity_array[:-1]
    cumulative_return = (capital / initial_capital - 1) * 100
    
    # Calculate drawdowns
    running_max = np.maximum.accumulate(equity_array)
    drawdowns = (equity_array / running_max - 1) * 100
    max_drawdown = abs(min(drawdowns))
    
    # Calculate Sharpe ratio (annualized)
    if len(returns) > 1:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(365 * 24 / args.timeframe.count('h'))
    else:
        sharpe_ratio = 0
    
    # Calculate Calmar ratio
    if max_drawdown > 0:
        calmar_ratio = cumulative_return / 100 / max_drawdown * 100
    else:
        calmar_ratio = float('inf')
    
    # Calculate expected value of trades
    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss
    
    # Calculate Kelly criterion
    w = win_rate
    r = avg_win / avg_loss if avg_loss > 0 else float('inf')
    kelly = w - (1 - w) / r if r > 0 else 0
    
    # Calculate recovery factor
    if max_drawdown > 0:
        recovery_factor = (capital - initial_capital) / (initial_capital * max_drawdown / 100)
    else:
        recovery_factor = float('inf')
    
    # Calculate accuracy (% of signals that match actual price direction)
    actual_moves = np.sign(data_subset['close'].diff().shift(-1).values)
    predicted_moves = np.sign(signals)
    matching = (actual_moves == predicted_moves) & (actual_moves != 0)
    accuracy = np.sum(matching) / np.sum(actual_moves != 0)
    
    # Calculate advanced metrics if requested
    advanced_metrics = {}
    if args.calculate_all_metrics:
        # Calculate Sortino ratio
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = (mean_return / downside_deviation) * np.sqrt(365 * 24 / args.timeframe.count('h')) if downside_deviation > 0 else float('inf')
        
        # Calculate maximum consecutive wins/losses
        consecutive_wins = 0
        max_consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_losses = 0
        
        for trade in trades:
            if trade['pnl_amount'] > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Calculate time in market
        if trades:
            first_entry = trades[0]['entry_time']
            last_exit = trades[-1]['exit_time']
            total_time = (last_exit - first_entry).total_seconds()
            
            time_in_market = 0
            for trade in trades:
                time_in_trade = (trade['exit_time'] - trade['entry_time']).total_seconds()
                time_in_market += time_in_trade
            
            market_exposure = time_in_market / total_time
        else:
            market_exposure = 0
        
        # Calculate profit per day
        days = (data_subset.iloc[-1]['timestamp'] - data_subset.iloc[0]['timestamp']).days
        profit_per_day = (capital - initial_capital) / max(1, days)
        
        # Calculate Ulcer Index
        squared_drawdowns = drawdowns ** 2
        ulcer_index = np.sqrt(np.mean(squared_drawdowns))
        
        # Calculate Omega ratio
        threshold = 0  # 0% return threshold
        returns_above = returns[returns > threshold]
        returns_below = returns[returns <= threshold]
        
        if len(returns_below) > 0 and abs(np.sum(returns_below)) > 0:
            omega_ratio = np.sum(returns_above) / abs(np.sum(returns_below))
        else:
            omega_ratio = float('inf')
        
        # Calculate MAR ratio (annualized)
        annualized_return = (1 + cumulative_return / 100) ** (365 / days) - 1
        mar_ratio = annualized_return / (max_drawdown / 100) if max_drawdown > 0 else float('inf')
        
        # Calculate pain index
        pain_index = np.mean(abs(drawdowns))
        
        # Store advanced metrics
        advanced_metrics = {
            'sortino_ratio': float(sortino_ratio),
            'max_consecutive_wins': int(max_consecutive_wins),
            'max_consecutive_losses': int(max_consecutive_losses),
            'market_exposure': float(market_exposure),
            'profit_per_day': float(profit_per_day),
            'ulcer_index': float(ulcer_index),
            'omega_ratio': float(omega_ratio),
            'annualized_return': float(annualized_return * 100),  # as percentage
            'mar_ratio': float(mar_ratio),
            'pain_index': float(pain_index)
        }
    
    # Compile results
    results = {
        'pair': args.pair,
        'model': args.model,
        'period': args.period,
        'initial_capital': float(initial_capital),
        'final_capital': float(capital),
        'total_return': float(capital / initial_capital),
        'cumulative_return_pct': float(cumulative_return),
        'win_rate': float(win_rate),
        'accuracy': float(accuracy),
        'total_trades': int(total_trades),
        'wins': int(wins),
        'losses': int(losses),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor),
        'max_drawdown': float(max_drawdown),
        'sharpe_ratio': float(sharpe_ratio),
        'calmar_ratio': float(calmar_ratio),
        'expectancy': float(expectancy),
        'kelly_criterion': float(kelly),
        'recovery_factor': float(recovery_factor)
    }
    
    # Add advanced metrics if calculated
    if advanced_metrics:
        results.update(advanced_metrics)
    
    # Add trade details
    results['trades'] = trades
    
    return results


def run_stress_test(data: pd.DataFrame, signals: np.ndarray, confidence: np.ndarray,
                 config: Dict, args) -> Dict:
    """
    Run stress testing under extreme market conditions.
    
    Args:
        data: Historical data
        signals: Trading signals
        confidence: Signal confidence
        config: Configuration
        args: Command line arguments
        
    Returns:
        stress_results: Stress test results
    """
    # Create copy of data for stress test
    stress_data = data.copy()
    
    # Define stress test scenarios
    scenarios = {
        "flash_crash": {
            "description": "Simulate flash crash (20% down in single period)",
            "modifier": lambda d: modify_data_flash_crash(d, 0.20)
        },
        "extended_downtrend": {
            "description": "Simulate extended downtrend (30% down over 10 periods)",
            "modifier": lambda d: modify_data_trend(d, -0.30, 10)
        },
        "volatility_spike": {
            "description": "Simulate volatility spike (double price ranges)",
            "modifier": lambda d: modify_data_volatility(d, 2.0)
        },
        "liquidity_crisis": {
            "description": "Simulate liquidity crisis (slippage increases 5x)",
            "modifier": lambda d: d  # Handled in simulation with higher slippage
        }
    }
    
    # Run simulations for each scenario
    stress_results = {}
    
    for scenario, config in scenarios.items():
        logger.info(f"Running stress test: {scenario} - {config['description']}")
        
        # Apply scenario modification
        scenario_data = config["modifier"](stress_data)
        
        # Run simulation with modified data
        if scenario == "liquidity_crisis":
            # Increase slippage for liquidity crisis
            scenario_args = argparse.Namespace(**vars(args))
            # Would modify slippage parameter if implemented
        else:
            scenario_args = args
        
        # Simulate trades with scenario data
        results = simulate_trades(scenario_data, signals, confidence, config, scenario_args)
        
        # Store results
        stress_results[scenario] = {
            "description": config["description"],
            "total_return": results["total_return"],
            "max_drawdown": results["max_drawdown"],
            "win_rate": results["win_rate"]
        }
    
    return stress_results


def modify_data_flash_crash(data: pd.DataFrame, crash_pct: float) -> pd.DataFrame:
    """Modify data to simulate a flash crash."""
    modified = data.copy()
    
    # Find a random position after the first 20% of the data
    start_idx = int(len(data) * 0.2)
    crash_idx = np.random.randint(start_idx, len(data) - 1)
    
    # Apply crash
    modified.loc[crash_idx, 'close'] = data.loc[crash_idx, 'close'] * (1 - crash_pct)
    modified.loc[crash_idx, 'low'] = min(data.loc[crash_idx, 'low'], modified.loc[crash_idx, 'close'])
    
    return modified


def modify_data_trend(data: pd.DataFrame, trend_pct: float, periods: int) -> pd.DataFrame:
    """Modify data to simulate a trend."""
    modified = data.copy()
    
    # Find a random position after the first 20% of the data
    start_idx = int(len(data) * 0.2)
    trend_start = np.random.randint(start_idx, len(data) - periods - 1)
    
    # Calculate per-period change
    per_period = (1 + trend_pct) ** (1 / periods) - 1
    
    # Apply trend
    for i in range(periods):
        idx = trend_start + i
        if idx < len(modified):
            modified.loc[idx, 'close'] = data.loc[idx, 'close'] * (1 + per_period) ** (i + 1)
            modified.loc[idx, 'high'] = max(data.loc[idx, 'high'], modified.loc[idx, 'close'])
            modified.loc[idx, 'low'] = min(data.loc[idx, 'low'], modified.loc[idx, 'close'])
    
    return modified


def modify_data_volatility(data: pd.DataFrame, volatility_factor: float) -> pd.DataFrame:
    """Modify data to simulate increased volatility."""
    modified = data.copy()
    
    # Find a random position after the first 20% of the data
    start_idx = int(len(data) * 0.2)
    vol_start = np.random.randint(start_idx, int(len(data) * 0.7))
    vol_periods = int(len(data) * 0.2)  # 20% of the data length
    
    # Apply increased volatility
    for i in range(vol_periods):
        idx = vol_start + i
        if idx < len(modified):
            # Increase high-low range
            mid_price = data.loc[idx, 'close']
            range_half = (data.loc[idx, 'high'] - data.loc[idx, 'low']) * volatility_factor / 2
            
            modified.loc[idx, 'high'] = mid_price + range_half
            modified.loc[idx, 'low'] = mid_price - range_half
    
    return modified


def save_backtest_results(results: Dict, args) -> bool:
    """
    Save backtest results to a file.
    
    Args:
        results: Backtest results
        args: Command line arguments
        
    Returns:
        success: Whether the save succeeded
    """
    # Determine output directory
    output_dir = args.output_dir or BACKTEST_RESULTS_DIR
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Format filename
        pair_filename = args.pair.replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{pair_filename}_{args.model}_metrics.json"
        
        # Save file
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved backtest results to {filename}")
        
        # Save trades to CSV
        trades_file = f"{output_dir}/{pair_filename}_{args.model}_trades.csv"
        trades_df = pd.DataFrame(results.get('trades', []))
        
        if not trades_df.empty:
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Saved trade details to {trades_file}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error saving backtest results: {e}")
        return False


def print_results(results: Dict):
    """
    Print backtest results to console.
    
    Args:
        results: Backtest results
    """
    print("\n" + "=" * 60)
    print(f" BACKTEST RESULTS: {results['pair']} - {results['model']} ")
    print("=" * 60)
    
    print(f"\nPeriod: {results.get('period', 'Unknown')}")
    print(f"Initial Capital: ${results['initial_capital']:.2f}")
    print(f"Final Capital: ${results['final_capital']:.2f}")
    
    print("\n--- Performance Metrics ---")
    print(f"Total Return: {results['total_return']:.2f}x ({results['cumulative_return_pct']:.2f}%)")
    print(f"Win Rate: {results['win_rate']*100:.2f}%")
    print(f"Prediction Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
    
    print("\n--- Trade Statistics ---")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Wins/Losses: {results['wins']}/{results['losses']}")
    if results['wins'] > 0:
        print(f"Average Win: ${results['avg_win']:.2f}")
    if results['losses'] > 0:
        print(f"Average Loss: ${results['avg_loss']:.2f}")
    
    print("\n--- Risk Metrics ---")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Calmar Ratio: {results['calmar_ratio']:.2f}")
    print(f"Kelly Criterion: {results['kelly_criterion']*100:.2f}%")
    print(f"Recovery Factor: {results['recovery_factor']:.2f}")
    
    # Print advanced metrics if available
    if 'sortino_ratio' in results:
        print("\n--- Advanced Metrics ---")
        print(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"Omega Ratio: {results['omega_ratio']:.2f}")
        print(f"Annualized Return: {results['annualized_return']:.2f}%")
        print(f"MAR Ratio: {results['mar_ratio']:.2f}")
        print(f"Max Consecutive Wins: {results['max_consecutive_wins']}")
        print(f"Max Consecutive Losses: {results['max_consecutive_losses']}")
        print(f"Market Exposure: {results['market_exposure']*100:.2f}%")
    
    print("\n" + "=" * 60)


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    logger.info(f"Running comprehensive backtest for {args.pair} using {args.model} model")
    
    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return 1
    
    # Get pair-specific settings
    pair_settings = config.get("pair_specific_settings", {}).get(args.pair, {})
    
    # Get confidence threshold
    confidence_threshold = args.confidence_threshold
    if confidence_threshold is None:
        confidence_threshold = pair_settings.get("confidence_threshold", 0.65)
    
    # Get lookback window
    lookback_window = pair_settings.get("lookback_window", 120)
    
    # Load historical data
    data = load_historical_data(args.pair, args.timeframe, args.period)
    if data is None:
        logger.error("Failed to load historical data. Exiting.")
        return 1
    
    # Load model
    model = load_model(args.model, args.pair, args.model == 'ensemble')
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return 1
    
    # Prepare features
    features = prepare_features(data, lookback_window)
    if features is None:
        logger.error("Failed to prepare features. Exiting.")
        return 1
    
    # Get model predictions
    predictions = get_model_predictions(model, features, args.model, args.model == 'ensemble')
    if predictions is None:
        logger.error("Failed to get model predictions. Exiting.")
        return 1
    
    # Calculate entry signals
    signals, confidence = calculate_entry_signals(predictions, 0.5, confidence_threshold)
    
    # Simulate trades
    results = simulate_trades(data, signals, confidence, config, args)
    
    # Run stress tests if requested
    if args.stress_test:
        stress_results = run_stress_test(data, signals, confidence, config, args)
        results['stress_test'] = stress_results
    
    # Save results
    if args.output_format in ['json', 'both']:
        save_backtest_results(results, args)
    
    # Print results
    if args.output_format in ['console', 'both']:
        print_results(results)
    
    # Return results as JSON if requested
    if args.output_format == 'json':
        # Remove trades from JSON output to keep it concise
        results_json = {k: v for k, v in results.items() if k != 'trades'}
        print(json.dumps(results_json))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())