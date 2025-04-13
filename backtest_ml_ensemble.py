#!/usr/bin/env python3
"""
Comprehensive ML Ensemble Backtesting Script

This script runs a thorough backtest of all ML models across all available 
cryptocurrency pairs at the lowest possible timeframes.

It evaluates:
1. Individual model performance (all 8 models)
2. Ensemble performance with dynamic weighting
3. Performance across different market regimes
4. Performance across different trading pairs
5. Feature importance and sensitivity

The script generates comprehensive reports with key performance metrics.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load ML models and ensemble
from advanced_ensemble_model import DynamicWeightedEnsemble, MODEL_TYPES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("backtest_results/ml_ensemble/backtest.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MLEnsembleBacktest")

# Constants
INITIAL_CAPITAL = 20000.0
TRADING_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD"]
TIMEFRAMES = ["1h"]  # Using 1h as the lowest available timeframe
RESULTS_DIR = "backtest_results/ml_ensemble"

# Create output directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "results"), exist_ok=True)

class MLEnsembleBacktester:
    """Specialized backtester for ML ensemble models"""
    
    def __init__(self, trading_pair, timeframe="1h", start_date=None, end_date=None,
                 initial_capital=INITIAL_CAPITAL, position_size_pct=0.2):
        """Initialize backtester"""
        self.trading_pair = trading_pair
        self.symbol = trading_pair.replace("/", "")
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.position_size_pct = position_size_pct
        
        # Trading state
        self.in_position = False
        self.position_type = None  # 'long' or 'short'
        self.entry_price = 0.0
        self.position_size = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
        # Performance tracking
        self.equity_curve = []
        self.trades = []
        self.model_predictions = {}
        self.model_accuracy = {model_type: [] for model_type in MODEL_TYPES}
        self.ensemble_predictions = []
        self.ensemble_accuracy = []
        
        # Load data
        self.data = self._load_data()
        if self.data is None or len(self.data) == 0:
            raise ValueError(f"No data available for {trading_pair} on {timeframe} timeframe")
        
        logger.info(f"Loaded {len(self.data)} candles for {trading_pair} on {timeframe} timeframe")
        
        # Detect market regimes
        self._detect_market_regimes()
        logger.info(f"Detected market regimes for {trading_pair}")
        
        # Initialize ensemble
        self.ensemble = DynamicWeightedEnsemble(trading_pair=trading_pair, timeframe=timeframe)
        self.models_loaded = len(self.ensemble.models) > 0
        
        if self.models_loaded:
            logger.info(f"Loaded {len(self.ensemble.models)} models for ensemble")
            logger.info(f"Model types: {list(self.ensemble.models.keys())}")
        else:
            logger.warning(f"No models loaded for {trading_pair}")
    
    def _load_data(self):
        """Load historical data"""
        try:
            # Construct path based on symbol (without the slash)
            symbol = self.trading_pair.replace("/", "")
            
            # Check in both the root historical_data directory and the symbol subdirectory
            potential_paths = [
                f"historical_data/{symbol}_{self.timeframe}.csv",
                f"historical_data/{symbol}/{symbol}_{self.timeframe}.csv"
            ]
            
            for data_path in potential_paths:
                if os.path.exists(data_path):
                    logger.info(f"Loading data from {data_path}")
                    df = pd.read_csv(data_path)
                    
                    # Convert timestamp to datetime if needed
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    
                    # Filter by date if specified
                    if self.start_date:
                        df = df[df.index >= pd.to_datetime(self.start_date)]
                    
                    if self.end_date:
                        df = df[df.index <= pd.to_datetime(self.end_date)]
                    
                    # Make sure we have OHLCV data
                    required_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in required_columns:
                        if col not in df.columns:
                            logger.error(f"Required column '{col}' not found in {data_path}")
                            return None
                    
                    return df
            
            # If we get here, no valid data was found
            logger.error(f"Could not find data file for {self.trading_pair} at {self.timeframe} timeframe")
            return None
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def _detect_market_regimes(self):
        """Detect market regimes"""
        df = self.data.copy()
        
        # Calculate returns if not present
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # Calculate EMAs if not present
        if 'ema50' not in df.columns:
            df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        if 'ema100' not in df.columns:
            df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
            
        # Calculate volatility
        if 'volatility_20' not in df.columns:
            df['volatility_20'] = df['returns'].rolling(window=20).std()
            
        # Calculate trend
        df['trend'] = (df['ema50'] > df['ema100']).astype(int)
        
        # Define volatility threshold (75th percentile)
        volatility_threshold = df['volatility_20'].quantile(0.75)
        
        # Determine regimes
        conditions = [
            (df['volatility_20'] <= volatility_threshold) & (df['trend'] == 1),  # Normal trending up
            (df['volatility_20'] > volatility_threshold) & (df['trend'] == 1),   # Volatile trending up
            (df['volatility_20'] <= volatility_threshold) & (df['trend'] == 0),  # Normal trending down
            (df['volatility_20'] > volatility_threshold) & (df['trend'] == 0)    # Volatile trending down
        ]
        
        regimes = ['normal_trending_up', 'volatile_trending_up', 
                  'normal_trending_down', 'volatile_trending_down']
        
        df['market_regime'] = np.select(conditions, regimes, default='normal_trending_up')
        
        # Count regimes
        regime_counts = df['market_regime'].value_counts()
        logger.info(f"Market regimes: {regime_counts.to_dict()}")
        
        self.data = df
    
    def _prepare_prediction_window(self, current_idx, window_size=60):
        """
        Prepare data window for model prediction with comprehensive feature generation
        to ensure all required features are available for the ML models
        """
        if current_idx < window_size:
            logger.warning(f"Not enough historical data for prediction window")
            return None
            
        # Get the data window
        window_start = current_idx - window_size
        window_end = current_idx
        window_data = self.data.iloc[window_start:window_end].copy()
        
        # Log available features for debugging
        if current_idx == window_size:  # Only log once at the beginning
            logger.info(f"Data window columns ({len(window_data.columns)}): {list(window_data.columns)[:10]}...")
        
        try:
            # Ensure all OHLCV columns have standard names
            ohlcv_columns = {
                'open': ['open', f'open_{self.timeframe}', 'Open'],
                'high': ['high', f'high_{self.timeframe}', 'High'],
                'low': ['low', f'low_{self.timeframe}', 'Low'],
                'close': ['close', f'close_{self.timeframe}', 'Close'],
                'volume': ['volume', f'volume_{self.timeframe}', 'Volume']
            }
            
            # Standardize OHLCV column names
            for std_name, variants in ohlcv_columns.items():
                if std_name not in window_data.columns:
                    for variant in variants:
                        if variant in window_data.columns:
                            window_data[std_name] = window_data[variant]
                            if current_idx == window_size:  # Only log once
                                logger.debug(f"Standardized column: {variant} -> {std_name}")
                            break
            
            # Calculate basic indicators if not present (models expect these)
            # Add EMA indicators
            for period in [9, 20, 21, 50, 100]:
                ema_col = f'ema{period}'
                if ema_col not in window_data.columns and 'close' in window_data.columns:
                    window_data[ema_col] = window_data['close'].ewm(span=period, adjust=False).mean()
                    if current_idx == window_size:  # Only log once
                        logger.debug(f"Added calculated {ema_col}")
            
            # Add RSI if not present
            if 'rsi' not in window_data.columns and 'close' in window_data.columns:
                delta = window_data['close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
                window_data['rsi'] = 100 - (100 / (1 + rs))
                window_data['rsi'] = window_data['rsi'].fillna(50)  # Fill NaNs with neutral RSI
                if current_idx == window_size:  # Only log once
                    logger.debug("Added calculated RSI")
            
            # Add MACD if not present
            if 'macd' not in window_data.columns and 'close' in window_data.columns:
                ema12 = window_data['close'].ewm(span=12, adjust=False).mean()
                ema26 = window_data['close'].ewm(span=26, adjust=False).mean()
                window_data['macd'] = ema12 - ema26
                window_data['macd_signal'] = window_data['macd'].ewm(span=9, adjust=False).mean()
                window_data['macd_hist'] = window_data['macd'] - window_data['macd_signal']
                if current_idx == window_size:  # Only log once
                    logger.debug("Added calculated MACD indicators")
            
            # Add Bollinger Bands if not present
            if 'bb_upper' not in window_data.columns and 'close' in window_data.columns:
                window = 20
                std_dev = 2
                sma = window_data['close'].rolling(window=window).mean()
                rolling_std = window_data['close'].rolling(window=window).std()
                
                window_data['bb_upper'] = sma + (rolling_std * std_dev)
                window_data['bb_lower'] = sma - (rolling_std * std_dev)
                window_data['bb_middle'] = sma
                
                # Fill NaN values
                window_data['bb_upper'] = window_data['bb_upper'].fillna(window_data['close'] * 1.05)
                window_data['bb_lower'] = window_data['bb_lower'].fillna(window_data['close'] * 0.95)
                window_data['bb_middle'] = window_data['bb_middle'].fillna(window_data['close'])
                
                if current_idx == window_size:  # Only log once
                    logger.debug("Added calculated Bollinger Bands")
            
            # Fill any remaining NaN values
            window_data = window_data.fillna(method='ffill').fillna(method='bfill')
            
            # Check for any remaining NaNs and report them
            nan_cols = window_data.columns[window_data.isna().any()].tolist()
            if nan_cols and current_idx == window_size:
                logger.warning(f"NaN values remain in columns: {nan_cols}")
                window_data = window_data.fillna(0)  # Last resort fill with zeros
                
        except Exception as e:
            logger.error(f"Error preparing prediction window: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return window_data
    
    def run_backtest(self, sequence_length=60, use_ensemble_pruning=True):
        """Run backtest on historical data"""
        if not self.models_loaded:
            logger.error("No models loaded, cannot run backtest")
            return
        
        logger.info(f"Starting ML ensemble backtest for {self.trading_pair} ({self.timeframe})")
        logger.info(f"Backtest period: {self.data.index[0]} to {self.data.index[-1]}")
        
        num_candles = len(self.data)
        test_start_idx = sequence_length
        
        # Initialize result tracking
        self.equity_curve = [self.initial_capital]
        self.trades = []
        self.model_predictions = {model_type: [] for model_type in self.ensemble.models.keys()}
        self.ensemble_predictions = []
        
        # Ground truth (for evaluation)
        ground_truth = []
        
        # Trading simulation
        for i in range(test_start_idx, num_candles-1):
            current_time = self.data.index[i]
            current_price = self.data.iloc[i]['close']
            next_price = self.data.iloc[i+1]['close']
            current_regime = self.data.iloc[i]['market_regime']
            
            # Prepare feature window for prediction
            window_data = self._prepare_prediction_window(i, sequence_length)
            if window_data is None:
                continue
                
            # Make predictions with each model and ensemble
            try:
                # Log preprocessing for debugging
                if i == test_start_idx:
                    logger.info(f"Sending data to predict method with shape: {window_data.shape if hasattr(window_data, 'shape') else 'unknown'}")
                    if hasattr(window_data, 'columns'):
                        logger.info(f"First few columns: {list(window_data.columns)[:5]}")
                
                # Get ensemble prediction
                prediction, confidence, details = self.ensemble.predict(window_data)
                
                # Debug logging for first prediction
                if i == test_start_idx:
                    logger.info(f"Prediction result: {prediction}, Confidence: {confidence:.4f}")
                    logger.info(f"Available models used: {list(details['model_predictions'].keys())}")
                
                # Save individual model predictions
                for model_type, model_prediction in details['model_predictions'].items():
                    self.model_predictions[model_type].append({
                        'timestamp': current_time,
                        'price': current_price,
                        'prediction': model_prediction,
                        'regime': current_regime,
                        'confidence': details['confidences'].get(model_type, 0.5) if 'confidences' in details else 0.5
                    })
                
                # Save ensemble prediction
                self.ensemble_predictions.append({
                    'timestamp': current_time,
                    'price': current_price,
                    'prediction': prediction,
                    'confidence': confidence,
                    'regime': current_regime,
                    'weights': details['weights']
                })
                
                # Determine ground truth (for evaluation)
                truth = 1 if next_price > current_price else -1 if next_price < current_price else 0
                ground_truth.append(truth)
                
                # Update model performance
                outcomes = {}
                for model_type, model_prediction in details['model_predictions'].items():
                    # Determine outcome (1=correct, -1=incorrect, 0=neutral)
                    if model_prediction * truth > 0:  # Same direction = correct
                        outcome = 1
                    elif model_prediction * truth < 0:  # Opposite direction = incorrect
                        outcome = -1
                    else:  # One or both are neutral
                        outcome = 0
                    
                    outcomes[model_type] = outcome
                    self.model_accuracy[model_type].append(outcome)
                
                # Determine ensemble outcome
                if prediction * truth > 0:
                    ensemble_outcome = 1
                elif prediction * truth < 0:
                    ensemble_outcome = -1
                else:
                    ensemble_outcome = 0
                
                self.ensemble_accuracy.append(ensemble_outcome)
                
                # Update ensemble weights based on performance
                trade_details = {
                    'actual_price_change': next_price - current_price,
                    'predicted_direction': prediction,
                    'market_regime': current_regime,
                    'timestamp': current_time.isoformat()
                }
                
                # Update ensemble weights
                self.ensemble.update_performance(outcomes, trade_details)
                
                # Periodically prune underperforming models
                if use_ensemble_pruning and i % 100 == 0 and i > test_start_idx + 100:
                    pruned, kept = self.ensemble.auto_prune_models(
                        performance_threshold=0.05,  # Models must be more right than wrong
                        min_samples=5,              # Need at least 5 predictions to evaluate
                        keep_minimum=2              # Always keep at least 2 models
                    )
                    if pruned:
                        logger.info(f"Pruned models: {pruned}")
                        logger.info(f"Remaining models: {kept}")
                
                # Execute simulated trades based on prediction
                self._execute_trade_signals(current_time, current_price, prediction, confidence)
                
                # Update portfolio value based on next candle's price
                self._update_portfolio_value(current_time, next_price)
            
            except Exception as e:
                logger.error(f"Error at {current_time}: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Calculate final performance metrics
        self._calculate_performance_metrics(ground_truth)
        
        return self.get_results()
    
    def _execute_trade_signals(self, timestamp, current_price, prediction, confidence):
        """Execute trade signals based on predictions"""
        # Only trade if confidence is above threshold
        confidence_threshold = 0.6
        
        if not self.in_position:
            # Check for entry signals
            if prediction > 0 and confidence >= confidence_threshold:
                # Long entry
                self._enter_position(timestamp, current_price, 'long', prediction, confidence)
            elif prediction < 0 and confidence >= confidence_threshold:
                # Short entry
                self._enter_position(timestamp, current_price, 'short', prediction, confidence)
        else:
            # Check for exit signals
            if self.position_type == 'long' and prediction < 0:
                # Exit long position
                self._exit_position(timestamp, current_price, 'signal reversal')
            elif self.position_type == 'short' and prediction > 0:
                # Exit short position
                self._exit_position(timestamp, current_price, 'signal reversal')
    
    def _enter_position(self, timestamp, price, position_type, prediction, confidence):
        """Enter a new position"""
        position_size = self.current_capital * self.position_size_pct
        
        # Calculate number of units based on price
        units = position_size / price
        
        # Set stop loss and take profit
        stop_pct = 0.03  # 3% stop loss
        profit_pct = 0.05  # 5% take profit
        
        if position_type == 'long':
            stop_loss = price * (1 - stop_pct)
            take_profit = price * (1 + profit_pct)
        else:  # short
            stop_loss = price * (1 + stop_pct)
            take_profit = price * (1 - profit_pct)
        
        # Update position state
        self.in_position = True
        self.position_type = position_type
        self.entry_price = price
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Record trade
        self.trades.append({
            'entry_time': timestamp,
            'entry_price': price,
            'position_type': position_type,
            'position_size': position_size,
            'units': units,
            'prediction': prediction,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        })
        
        logger.info(f"Entered {position_type.upper()} position at {price} ({timestamp})")
    
    def _exit_position(self, timestamp, price, reason):
        """Exit the current position"""
        if not self.in_position:
            return
            
        # Calculate profit/loss
        if self.position_type == 'long':
            profit_pct = (price - self.entry_price) / self.entry_price
        else:  # short
            profit_pct = (self.entry_price - price) / self.entry_price
            
        profit_amount = self.position_size * profit_pct
        
        # Update capital
        self.current_capital += profit_amount
        
        # Update the last trade
        if self.trades:
            last_trade = self.trades[-1]
            last_trade['exit_time'] = timestamp
            last_trade['exit_price'] = price
            last_trade['exit_reason'] = reason
            last_trade['profit_pct'] = profit_pct
            last_trade['profit_amount'] = profit_amount
            
            logger.info(f"Exited {self.position_type.upper()} position at {price} ({timestamp}), Profit: {profit_pct:.2%}")
        
        # Reset position state
        self.in_position = False
        self.position_type = None
        self.entry_price = 0.0
        self.position_size = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
    
    def _update_portfolio_value(self, timestamp, price):
        """Update portfolio value based on current price"""
        if not self.in_position:
            # No position, just record current capital
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': self.current_capital,
                'in_position': False
            })
        else:
            # Calculate unrealized P&L
            if self.position_type == 'long':
                profit_pct = (price - self.entry_price) / self.entry_price
                # Check for stop loss or take profit
                if price <= self.stop_loss:
                    self._exit_position(timestamp, self.stop_loss, 'stop loss')
                elif price >= self.take_profit:
                    self._exit_position(timestamp, self.take_profit, 'take profit')
            else:  # short
                profit_pct = (self.entry_price - price) / self.entry_price
                # Check for stop loss or take profit
                if price >= self.stop_loss:
                    self._exit_position(timestamp, self.stop_loss, 'stop loss')
                elif price <= self.take_profit:
                    self._exit_position(timestamp, self.take_profit, 'take profit')
            
            # Calculate portfolio equity including unrealized P&L
            unrealized_profit = self.position_size * profit_pct
            equity = self.current_capital + unrealized_profit
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'in_position': True,
                'position_type': self.position_type,
                'unrealized_profit': unrealized_profit
            })
    
    def _calculate_performance_metrics(self, ground_truth):
        """Calculate performance metrics for backtesting results"""
        # Trading performance
        self.performance = {}
        
        if not self.equity_curve:
            logger.warning("No equity curve data available")
            return
            
        initial_equity = self.initial_capital
        final_equity = self.equity_curve[-1]['equity'] if isinstance(self.equity_curve[-1], dict) else self.equity_curve[-1]
        
        # Calculate returns
        total_return = final_equity - initial_equity
        total_return_pct = (final_equity / initial_equity) - 1
        
        # Calculate Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            equity_values = [entry['equity'] if isinstance(entry, dict) else entry for entry in self.equity_curve]
            returns = np.diff(equity_values) / equity_values[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Calculate max drawdown
        peak = initial_equity
        max_drawdown = 0
        
        for entry in self.equity_curve:
            equity = entry['equity'] if isinstance(entry, dict) else entry
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate win rate
        wins = sum(1 for trade in self.trades if trade.get('profit_pct', 0) > 0)
        total_trades = len(self.trades)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        # ML model accuracy
        model_accuracies = {}
        for model_type, outcomes in self.model_accuracy.items():
            if outcomes:
                correct = sum(1 for o in outcomes if o == 1)
                incorrect = sum(1 for o in outcomes if o == -1)
                neutral = sum(1 for o in outcomes if o == 0)
                total = len(outcomes)
                
                model_accuracies[model_type] = {
                    'accuracy': correct / total if total > 0 else 0,
                    'correct': correct,
                    'incorrect': incorrect,
                    'neutral': neutral,
                    'total': total
                }
        
        # Ensemble accuracy
        if self.ensemble_accuracy:
            correct = sum(1 for o in self.ensemble_accuracy if o == 1)
            incorrect = sum(1 for o in self.ensemble_accuracy if o == -1)
            neutral = sum(1 for o in self.ensemble_accuracy if o == 0)
            total = len(self.ensemble_accuracy)
            
            ensemble_accuracy = {
                'accuracy': correct / total if total > 0 else 0,
                'correct': correct,
                'incorrect': incorrect,
                'neutral': neutral,
                'total': total
            }
        else:
            ensemble_accuracy = None
        
        # Combine all metrics
        self.performance = {
            'trading_metrics': {
                'initial_capital': initial_equity,
                'final_capital': final_equity,
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades
            },
            'model_accuracy': model_accuracies,
            'ensemble_accuracy': ensemble_accuracy
        }
        
        # Log key metrics
        logger.info(f"Backtest results for {self.trading_pair}:")
        logger.info(f"Total return: {total_return_pct:.2%} (${total_return:.2f})")
        logger.info(f"Sharpe ratio: {sharpe_ratio:.2f}")
        logger.info(f"Max drawdown: {max_drawdown:.2%}")
        logger.info(f"Win rate: {win_rate:.2%} ({wins}/{total_trades})")
        
        if ensemble_accuracy:
            logger.info(f"Ensemble prediction accuracy: {ensemble_accuracy['accuracy']:.2%} ({ensemble_accuracy['correct']}/{ensemble_accuracy['total']})")
        
        for model_type, acc in model_accuracies.items():
            logger.info(f"{model_type.upper()} accuracy: {acc['accuracy']:.2%} ({acc['correct']}/{acc['total']})")
    
    def get_results(self):
        """Get backtest results"""
        return {
            'trading_pair': self.trading_pair,
            'timeframe': self.timeframe,
            'performance': self.performance,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'model_predictions': self.model_predictions,
            'ensemble_predictions': self.ensemble_predictions
        }
    
    def plot_results(self, save_path=None):
        """Plot backtest results"""
        if not self.equity_curve:
            logger.warning("No equity curve data available for plotting")
            return
            
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        
        # Plot equity curve
        plt.subplot(3, 1, 1)
        equity_values = [entry['equity'] if isinstance(entry, dict) else entry for entry in self.equity_curve]
        plt.plot(equity_values)
        plt.title(f'Equity Curve - {self.trading_pair} ({self.timeframe})')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        
        # Plot model accuracy
        plt.subplot(3, 1, 2)
        model_types = []
        accuracies = []
        
        for model_type, acc_data in self.performance.get('model_accuracy', {}).items():
            model_types.append(model_type)
            accuracies.append(acc_data['accuracy'])
        
        if model_types:
            # Add ensemble accuracy
            if self.performance.get('ensemble_accuracy'):
                model_types.append('ensemble')
                accuracies.append(self.performance['ensemble_accuracy']['accuracy'])
                
            plt.bar(model_types, accuracies)
            plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
            plt.title('Model Prediction Accuracy')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.grid(True)
            
        # Plot trades
        if self.trades:
            plt.subplot(3, 1, 3)
            
            # Extract price data
            close_prices = self.data['close'].values
            timestamps = list(range(len(close_prices)))
            
            # Plot price series
            plt.plot(timestamps, close_prices, label='Price', color='black', alpha=0.5)
            
            # Extract trade data
            for trade in self.trades:
                # Find timestamp indices
                entry_idx = self.data.index.get_loc(trade['entry_time']) if isinstance(trade['entry_time'], pd.Timestamp) else 0
                
                if 'exit_time' in trade and trade['exit_time'] is not None:
                    exit_idx = self.data.index.get_loc(trade['exit_time']) if isinstance(trade['exit_time'], pd.Timestamp) else len(close_prices)-1
                else:
                    exit_idx = len(close_prices)-1
                
                # Determine color based on trade outcome
                if 'profit_pct' in trade:
                    color = 'green' if trade['profit_pct'] > 0 else 'red'
                else:
                    color = 'blue'
                
                # Plot trade entries and exits
                if trade['position_type'] == 'long':
                    plt.plot(entry_idx, close_prices[entry_idx], '^', color=color, markersize=8)
                    if 'exit_time' in trade and trade['exit_time'] is not None:
                        plt.plot(exit_idx, close_prices[exit_idx], 'v', color=color, markersize=8)
                else:  # short
                    plt.plot(entry_idx, close_prices[entry_idx], 'v', color=color, markersize=8)
                    if 'exit_time' in trade and trade['exit_time'] is not None:
                        plt.plot(exit_idx, close_prices[exit_idx], '^', color=color, markersize=8)
            
            plt.title('Price Chart with Trades')
            plt.ylabel('Price')
            plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Results plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def run_backtest(trading_pair, timeframe="1h", start_date=None, end_date=None, 
                initial_capital=INITIAL_CAPITAL, position_size_pct=0.2,
                use_ensemble_pruning=True):
    """Run backtest for a single trading pair and timeframe"""
    try:
        backtester = MLEnsembleBacktester(
            trading_pair=trading_pair,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct
        )
        
        results = backtester.run_backtest(use_ensemble_pruning=use_ensemble_pruning)
        
        # Save results to file
        symbol = trading_pair.replace("/", "")
        results_path = os.path.join(RESULTS_DIR, "results", f"{symbol}_{timeframe}_results.json")
        
        # Prepare serializable results
        serializable_results = {
            'trading_pair': trading_pair,
            'timeframe': timeframe,
            'performance': results['performance'],
            'trades': [
                {k: (str(v) if isinstance(v, (pd.Timestamp, datetime)) else v) 
                 for k, v in trade.items()}
                for trade in results['trades']
            ]
        }
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Create and save plots
        plot_path = os.path.join(RESULTS_DIR, "plots", f"{symbol}_{timeframe}_plot.png")
        backtester.plot_results(save_path=plot_path)
        
        return results
        
    except Exception as e:
        logger.error(f"Error running backtest for {trading_pair} ({timeframe}): {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def run_all_backtests():
    """Run backtests for all trading pairs and timeframes"""
    all_results = {}
    
    for trading_pair in TRADING_PAIRS:
        for timeframe in TIMEFRAMES:
            logger.info(f"Running backtest for {trading_pair} ({timeframe})")
            results = run_backtest(trading_pair, timeframe)
            
            if results:
                symbol = trading_pair.replace("/", "")
                all_results[f"{symbol}_{timeframe}"] = results
    
    return all_results


def analyze_overall_results(all_results):
    """Analyze overall results across all backtests"""
    if not all_results:
        logger.error("No results to analyze")
        return
        
    logger.info("=== OVERALL BACKTEST RESULTS ===")
    
    # Calculate aggregate performance
    total_return_pct = []
    sharpe_ratios = []
    max_drawdowns = []
    win_rates = []
    
    # Prepare comparative model accuracy analysis
    model_accuracy = {model_type: [] for model_type in MODEL_TYPES}
    model_accuracy['ensemble'] = []
    
    # Analyze each backtest
    for backtest_key, results in all_results.items():
        # Trading metrics
        trading_metrics = results['performance']['trading_metrics']
        total_return_pct.append(trading_metrics['total_return_pct'])
        sharpe_ratios.append(trading_metrics['sharpe_ratio'])
        max_drawdowns.append(trading_metrics['max_drawdown'])
        win_rates.append(trading_metrics['win_rate'])
        
        # Model accuracy
        for model_type, acc_data in results['performance'].get('model_accuracy', {}).items():
            if model_type in model_accuracy:
                model_accuracy[model_type].append(acc_data['accuracy'])
        
        # Ensemble accuracy
        if results['performance'].get('ensemble_accuracy'):
            model_accuracy['ensemble'].append(results['performance']['ensemble_accuracy']['accuracy'])
    
    # Calculate averages
    avg_return = sum(total_return_pct) / len(total_return_pct) if total_return_pct else 0
    avg_sharpe = sum(sharpe_ratios) / len(sharpe_ratios) if sharpe_ratios else 0
    avg_drawdown = sum(max_drawdowns) / len(max_drawdowns) if max_drawdowns else 0
    avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0
    
    # Print overall trading performance
    logger.info("Trading Performance Across All Backtests:")
    logger.info(f"Average Return: {avg_return:.2%}")
    logger.info(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
    logger.info(f"Average Max Drawdown: {avg_drawdown:.2%}")
    logger.info(f"Average Win Rate: {avg_win_rate:.2%}")
    
    # Print model comparison
    logger.info("\nModel Accuracy Comparison:")
    for model_type, accuracies in model_accuracy.items():
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            logger.info(f"{model_type.upper()}: {avg_accuracy:.2%}")
    
    # Create summary plot
    plt.figure(figsize=(15, 10))
    
    # Plot trading pair returns
    plt.subplot(2, 1, 1)
    backtest_keys = list(all_results.keys())
    returns = [results['performance']['trading_metrics']['total_return_pct'] for results in all_results.values()]
    plt.bar(backtest_keys, returns)
    plt.title('Returns by Trading Pair and Timeframe')
    plt.ylabel('Return (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # Plot model accuracy comparison
    plt.subplot(2, 1, 2)
    model_types = []
    avg_accuracies = []
    
    for model_type, accuracies in model_accuracy.items():
        if accuracies:
            model_types.append(model_type)
            avg_accuracies.append(sum(accuracies) / len(accuracies))
    
    plt.bar(model_types, avg_accuracies)
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
    plt.title('Average Model Accuracy Across All Backtests')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    summary_plot_path = os.path.join(RESULTS_DIR, "plots", "overall_summary.png")
    plt.savefig(summary_plot_path)
    logger.info(f"Overall summary plot saved to {summary_plot_path}")
    
    plt.close()
    
    # Create comparison tables
    summary_data = {
        'backtest': [],
        'return_pct': [],
        'sharpe_ratio': [],
        'max_drawdown': [],
        'win_rate': [],
        'num_trades': []
    }
    
    for model_type in MODEL_TYPES + ['ensemble']:
        summary_data[f'{model_type}_accuracy'] = []
    
    for backtest_key, results in all_results.items():
        trading_metrics = results['performance']['trading_metrics']
        summary_data['backtest'].append(backtest_key)
        summary_data['return_pct'].append(trading_metrics['total_return_pct'])
        summary_data['sharpe_ratio'].append(trading_metrics['sharpe_ratio'])
        summary_data['max_drawdown'].append(trading_metrics['max_drawdown'])
        summary_data['win_rate'].append(trading_metrics['win_rate'])
        summary_data['num_trades'].append(trading_metrics['total_trades'])
        
        # Add model accuracies
        for model_type in MODEL_TYPES:
            if model_type in results['performance'].get('model_accuracy', {}):
                summary_data[f'{model_type}_accuracy'].append(
                    results['performance']['model_accuracy'][model_type]['accuracy']
                )
            else:
                summary_data[f'{model_type}_accuracy'].append(None)
        
        # Add ensemble accuracy
        if results['performance'].get('ensemble_accuracy'):
            summary_data['ensemble_accuracy'].append(
                results['performance']['ensemble_accuracy']['accuracy']
            )
        else:
            summary_data['ensemble_accuracy'].append(None)
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to CSV
    summary_path = os.path.join(RESULTS_DIR, "results", "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Summary data saved to {summary_path}")
    
    # Print summary table
    logger.info("\nSummary Table:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    logger.info(f"\n{summary_df}")


if __name__ == "__main__":
    # Process command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print(f"Usage: {sys.argv[0]} [trading_pair [timeframe]]")
            print("If no arguments are provided, backtests are run for all trading pairs and timeframes.")
            print(f"Available trading pairs: {TRADING_PAIRS}")
            print(f"Available timeframes: {TIMEFRAMES}")
            sys.exit(0)
        
        # Single trading pair specified
        trading_pair = sys.argv[1]
        timeframe = sys.argv[2] if len(sys.argv) > 2 else "1h"
        
        logger.info(f"Running backtest for {trading_pair} ({timeframe})")
        results = run_backtest(trading_pair, timeframe)
        
        all_results = {f"{trading_pair.replace('/', '')}_{timeframe}": results} if results else {}
    else:
        # Run all backtests
        logger.info("Running backtests for all trading pairs and timeframes")
        all_results = run_all_backtests()
    
    # Analyze overall results
    if all_results:
        analyze_overall_results(all_results)
        logger.info("Backtesting complete. Results saved to backtest_results/ml_ensemble/")
    else:
        logger.error("No backtest results were generated. Check logs for errors.")