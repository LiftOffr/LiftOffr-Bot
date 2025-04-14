#!/usr/bin/env python3
"""
Hyper-Optimized ML Ensemble for Maximum Accuracy

This script performs advanced optimization of ensemble models to push prediction
accuracy as close as possible to 100%. It combines:

1. Multi-level feature engineering for enhanced signal detection
2. Advanced hyperparameter optimization with genetic algorithms
3. Market regime-specific optimization for all trading conditions
4. Multi-objective optimization for both accuracy and profit
5. Specialized weight optimization for each cryptocurrency pair

Running this will produce optimized weights and configs for each trading pair.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/hyper_optimization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Import local modules
from market_context import detect_market_regime, MarketRegime, analyze_market_context
from optimize_ensemble_weights import EnsembleWeightOptimizer
from advanced_ml_models import create_tcn_model, create_transformer_model
from advanced_ensemble_model import DynamicWeightedEnsemble
from backtest_ml_ensemble import MLEnsembleBacktester

# Constants
SUPPORTED_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "DOT/USD", "ADA/USD", "LINK/USD"]
DEFAULT_TIMEFRAME = "1h"
MODEL_TYPES = ["lstm", "attention", "tcn", "transformer", "cnn"]
OPTIMIZATION_METRICS = ["accuracy", "f1", "profit", "sharpe"]
MODEL_DIR = "models"
ENSEMBLE_DIR = "models/ensemble"
HISTORICAL_DATA_DIR = "historical_data"
RESULTS_DIR = "optimization_results/enhanced"
TARGET_ACCURACY = 0.99  # Target 99% accuracy 

class HyperOptimizedEnsemble:
    """
    Creates hyper-optimized ensemble models with maximum accuracy and returns
    
    This class implements a multi-step optimization process to maximize both
    prediction accuracy and trading returns through:
    1. Enhanced feature engineering
    2. Optimal model weight selection
    3. Market regime-specific optimizations
    4. Cross-asset signal integration
    """
    
    def __init__(self, trading_pair: str, timeframe: str = "1h",
                 target_accuracy: float = TARGET_ACCURACY,
                 output_dir: str = RESULTS_DIR,
                 epochs: int = 500,
                 batch_size: int = 64,
                 optimization_iterations: int = 50,
                 verbose: bool = True):
        """
        Initialize the hyper-optimized ensemble
        
        Args:
            trading_pair: Trading pair to optimize (e.g., "SOL/USD")
            timeframe: Timeframe to use for data (e.g., "1h")
            target_accuracy: Target accuracy to achieve (default: 0.99)
            output_dir: Directory to save optimization results
            epochs: Maximum epochs for model training
            batch_size: Batch size for model training
            optimization_iterations: Number of optimization iterations
            verbose: Whether to print verbose output
        """
        self.trading_pair = trading_pair
        self.pair_code = trading_pair.replace("/", "")
        self.timeframe = timeframe
        self.target_accuracy = target_accuracy
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimization_iterations = optimization_iterations
        self.verbose = verbose
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize member variables
        self.data = None
        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
        self.models = {}
        self.base_model_accuracy = {}
        self.feature_importance = {}
        self.optimal_weights = {}
        self.market_regime_weights = {}
        self.best_accuracy = 0.0
        self.best_profit_factor = 0.0
        
        logger.info(f"Initializing hyper-optimized ensemble for {trading_pair} on {timeframe} timeframe")
        logger.info(f"Target accuracy: {target_accuracy * 100:.2f}%")
        
    def load_and_prepare_data(self) -> bool:
        """
        Load and prepare data for training and evaluation
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info(f"Loading data for {self.trading_pair}...")
        
        # Load historical data
        data_path = os.path.join(HISTORICAL_DATA_DIR, f"{self.pair_code}_{self.timeframe}.csv")
        if not os.path.exists(data_path):
            logger.error(f"Historical data not found: {data_path}")
            return False
        
        # Read data and ensure it's sorted by time
        data = pd.read_csv(data_path)
        data = data.sort_values('timestamp')
        
        # Add enhanced features
        data = self._add_enhanced_features(data)
        
        # Detect market regimes
        data = self._add_market_regimes(data)
        
        # Split into features (X) and target (y)
        X, y = self._prepare_features_target(data)
        
        # Split into train, validation, and test sets (60%/20%/20%)
        train_size = int(len(X) * 0.6)
        val_size = int(len(X) * 0.2)
        
        self.X_train, self.y_train = X[:train_size], y[:train_size]
        self.X_val, self.y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        self.X_test, self.y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        self.data = data
        
        logger.info(f"Data prepared: {len(self.X_train)} training samples, {len(self.X_val)} validation samples, {len(self.X_test)} test samples")
        return True
        
    def _add_enhanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add enhanced features to the data
        
        Args:
            data: DataFrame with historical data
            
        Returns:
            DataFrame with enhanced features
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure we have OHLCV data
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in data")
                return df
        
        # Calculate price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_1d'] = df['close'].pct_change(24)  # 1-day change (assuming hourly data)
        df['price_change_1w'] = df['close'].pct_change(168)  # 1-week change
        
        # Calculate volatility
        df['volatility'] = df['close'].rolling(window=24).std() / df['close'].rolling(window=24).mean()
        df['volatility_change'] = df['volatility'].pct_change()
        
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma'] = df['volume'].rolling(window=24).mean()
        df['relative_volume'] = df['volume'] / df['volume_ma']
        
        # Price and volume correlation
        df['price_volume_corr'] = df['close'].rolling(window=24).corr(df['volume'])
        
        # Technical indicators
        
        # Moving Averages
        for window in [9, 21, 50, 100, 200]:
            df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
            
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        # Drop any remaining NaN values
        df = df.dropna()
        
        return df
        
    def _add_market_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add market regime detection to the data
        
        Args:
            data: DataFrame with historical data
            
        Returns:
            DataFrame with market regime information
        """
        df = data.copy()
        
        # Detect market regimes
        regimes = []
        for i in range(len(df)):
            if i < 100:  # Need enough data for regime detection
                regimes.append(MarketRegime.RANGING.value)
                continue
                
            window = df.iloc[max(0, i-100):i+1]
            regime = detect_market_regime(
                window['close'].values,
                window['volatility'].values if 'volatility' in window.columns else None,
                window['rsi'].values if 'rsi' in window.columns else None
            )
            regimes.append(regime.value)
            
        df['market_regime'] = regimes
        
        # One-hot encode market regimes
        df['regime_trending'] = (df['market_regime'] == MarketRegime.TRENDING.value).astype(int)
        df['regime_ranging'] = (df['market_regime'] == MarketRegime.RANGING.value).astype(int)
        df['regime_volatile'] = (df['market_regime'] == MarketRegime.VOLATILE.value).astype(int)
        
        return df
        
    def _prepare_features_target(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target variables from data
        
        Args:
            data: DataFrame with all features
            
        Returns:
            Tuple of (features, targets)
        """
        # Drop unnecessary columns
        feature_cols = [col for col in data.columns if col not in [
            'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume',
            'market_regime'  # Keep the numerical regime indicators but drop the string version
        ]]
        
        # Define lookback window (24 hours / 1 day for hourly data)
        lookback = 24
        
        # Create 3D inputs for sequence models (samples, time steps, features)
        X = []
        y = []
        
        # Use price direction as target (1 for up, 0 for down)
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        
        # Create sequences
        for i in range(len(data) - lookback - 1):
            # Get features for the lookback period
            sequence = data[feature_cols].iloc[i:i+lookback].values
            
            # Get target (next period's price direction)
            target = data['target'].iloc[i+lookback]
            
            X.append(sequence)
            y.append(target)
            
        return np.array(X), np.array(y)
        
    def train_base_models(self) -> bool:
        """
        Train all base models for the ensemble
        
        Returns:
            bool: True if all models trained successfully, False otherwise
        """
        logger.info(f"Training base models for {self.trading_pair}...")
        
        # Common callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=10, min_lr=1e-6)
        ]
        
        # Train models
        for model_type in MODEL_TYPES:
            logger.info(f"Training {model_type} model...")
            
            # Create model based on type
            if model_type == 'lstm':
                model = self._create_lstm_model()
            elif model_type == 'attention':
                model = self._create_attention_model()
            elif model_type == 'tcn':
                model = create_tcn_model(self.X_train.shape[1:])
            elif model_type == 'transformer':
                model = create_transformer_model(self.X_train.shape[1:])
            elif model_type == 'cnn':
                model = self._create_cnn_model()
            else:
                logger.warning(f"Unknown model type: {model_type}")
                continue
                
            # Train model
            try:
                history = model.fit(
                    self.X_train, self.y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=(self.X_val, self.y_val),
                    callbacks=callbacks,
                    verbose=0 if not self.verbose else 1
                )
                
                # Get best validation accuracy
                best_val_acc = max(history.history['val_accuracy'])
                
                # Evaluate on test set
                test_loss, test_acc = model.evaluate(self.X_test, self.y_test, verbose=0)
                
                logger.info(f"{model_type.upper()} model: Test accuracy = {test_acc:.4f}")
                
                # Save model
                model_dir = os.path.join(MODEL_DIR, model_type)
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"{self.pair_code}_{self.timeframe}.h5")
                model.save(model_path)
                
                # Store model and accuracy
                self.models[model_type] = model
                self.base_model_accuracy[model_type] = test_acc
                
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
                return False
                
        # Log results
        logger.info("Base model training complete")
        for model_type, acc in self.base_model_accuracy.items():
            logger.info(f"{model_type.upper()}: {acc:.4f}")
            
        return True
        
    def _create_lstm_model(self) -> tf.keras.Model:
        """Create an enhanced LSTM model"""
        input_shape = self.X_train.shape[1:]
        
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def _create_attention_model(self) -> tf.keras.Model:
        """Create an enhanced attention model"""
        input_shape = self.X_train.shape[1:]
        
        # Define input
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # LSTM layer
        lstm_out = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
        
        # Self-attention mechanism
        attention = tf.keras.layers.Dense(1, activation='tanh')(lstm_out)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(128)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # Apply attention to LSTM output
        sent_representation = tf.keras.layers.Multiply()([lstm_out, attention])
        sent_representation = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(sent_representation)
        
        # Dense layers
        x = tf.keras.layers.BatchNormalization()(sent_representation)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def _create_cnn_model(self) -> tf.keras.Model:
        """Create an enhanced CNN model"""
        input_shape = self.X_train.shape[1:]
        
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def optimize_ensemble_weights(self) -> bool:
        """
        Optimize ensemble weights for maximum accuracy
        
        Returns:
            bool: True if optimization successful, False otherwise
        """
        logger.info("Optimizing ensemble weights...")
        
        optimizer = EnsembleWeightOptimizer(
            trading_pair=self.trading_pair,
            timeframe=self.timeframe,
            fitness_metric='f1',  # Use F1 score for balanced accuracy
            population_size=30,
            generations=20,
            mutation_rate=0.3,
            crossover_rate=0.7
        )
        
        # Set higher target for optimization
        optimizer.set_target_accuracy(self.target_accuracy)
        
        # Run optimization
        try:
            best_weights, market_regime_weights, metrics = optimizer.optimize(
                base_models=self.models,
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                X_test=self.X_test,
                y_test=self.y_test,
                iterations=self.optimization_iterations
            )
            
            self.optimal_weights = best_weights
            self.market_regime_weights = market_regime_weights
            self.best_accuracy = metrics['accuracy']
            self.best_profit_factor = metrics.get('profit_factor', 1.0)
            
            logger.info(f"Optimization complete. Best accuracy: {self.best_accuracy:.4f}")
            logger.info(f"Best weights: {self.optimal_weights}")
            
            # Save optimization results
            self._save_optimization_results()
            
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing ensemble weights: {e}")
            return False
            
    def _save_optimization_results(self) -> None:
        """Save optimization results to file"""
        result = {
            "trading_pair": self.trading_pair,
            "timeframe": self.timeframe,
            "base_weights": self.optimal_weights,
            "market_regime_weights": self.market_regime_weights,
            "optimization_results": {
                "max_accuracy": float(self.best_accuracy),
                "max_profit_factor": float(self.best_profit_factor),
                "best_weights": self.optimal_weights,
                "best_regime_weights": self.market_regime_weights
            },
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "version": "2.0.0"
        }
        
        # Save to ensemble directory
        os.makedirs(ENSEMBLE_DIR, exist_ok=True)
        weights_path = os.path.join(ENSEMBLE_DIR, f"{self.pair_code}_weights.json")
        
        with open(weights_path, 'w') as f:
            json.dump(result, f, indent=2)
            
        logger.info(f"Optimization results saved to {weights_path}")
        
    def create_ensemble_model(self) -> bool:
        """
        Create and save the final ensemble model
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Creating ensemble model...")
        
        ensemble_config = {
            "pair": self.trading_pair,
            "timeframe": self.timeframe,
            "base_models": {
                model_type: f"{model_type}/{self.pair_code}_{self.timeframe}.h5"
                for model_type in MODEL_TYPES
            },
            "weights": self.optimal_weights,
            "market_regime_weights": self.market_regime_weights,
            "prediction_threshold": 0.55,  # Higher threshold for increased confidence
            "features": {
                "use_price": True,
                "use_volume": True,
                "use_technical": True,
                "use_sentiment": False,  # Could be enabled if sentiment data available
                "lookback_periods": 24
            }
        }
        
        # Save ensemble configuration
        ensemble_path = os.path.join(ENSEMBLE_DIR, f"{self.pair_code}_ensemble.json")
        with open(ensemble_path, 'w') as f:
            json.dump(ensemble_config, f, indent=2)
            
        logger.info(f"Ensemble model saved to {ensemble_path}")
        
        return True
        
    def backtest_ensemble(self) -> Dict[str, Any]:
        """
        Backtest the ensemble model
        
        Returns:
            Dict: Backtest results
        """
        logger.info("Backtesting ensemble model...")
        
        try:
            # Load the latest data for backtesting
            data_path = os.path.join(HISTORICAL_DATA_DIR, f"{self.pair_code}_{self.timeframe}.csv")
            data = pd.read_csv(data_path)
            
            # Configure backtester
            backtester = MLEnsembleBacktester(
                trading_pair=self.trading_pair,
                timeframe=self.timeframe,
                historical_data=data,
                initial_capital=10000.0,
                leverage=20.0,
                ensemble_weights=self.optimal_weights,
                market_regime_weights=self.market_regime_weights
            )
            
            # Run backtest
            results = backtester.run_backtest()
            
            # Log results
            logger.info(f"Backtest results for {self.trading_pair}:")
            logger.info(f"Total trades: {results['total_trades']}")
            logger.info(f"Win rate: {results['win_rate'] * 100:.2f}%")
            logger.info(f"Profit factor: {results['profit_factor']:.2f}")
            logger.info(f"Return: {results['total_return'] * 100:.2f}%")
            logger.info(f"Max drawdown: {results['max_drawdown'] * 100:.2f}%")
            logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
            
            # Save backtest results
            backtest_dir = os.path.join("backtest_results", "enhanced")
            os.makedirs(backtest_dir, exist_ok=True)
            backtest_path = os.path.join(backtest_dir, f"{self.pair_code}_backtest.json")
            
            with open(backtest_path, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Backtest results saved to {backtest_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error backtesting ensemble model: {e}")
            return {"error": str(e)}
            
    def run_full_optimization(self) -> Dict[str, Any]:
        """
        Run the full optimization process
        
        Returns:
            Dict: Results of the optimization
        """
        logger.info(f"Starting full optimization for {self.trading_pair}...")
        
        # Step 1: Load and prepare data
        if not self.load_and_prepare_data():
            return {"error": "Failed to load and prepare data"}
            
        # Step 2: Train base models
        if not self.train_base_models():
            return {"error": "Failed to train base models"}
            
        # Step 3: Optimize ensemble weights
        if not self.optimize_ensemble_weights():
            return {"error": "Failed to optimize ensemble weights"}
            
        # Step 4: Create ensemble model
        if not self.create_ensemble_model():
            return {"error": "Failed to create ensemble model"}
            
        # Step 5: Backtest ensemble
        backtest_results = self.backtest_ensemble()
        
        # Final results
        results = {
            "trading_pair": self.trading_pair,
            "timeframe": self.timeframe,
            "target_accuracy": self.target_accuracy,
            "achieved_accuracy": self.best_accuracy,
            "achieved_profit_factor": self.best_profit_factor,
            "optimal_weights": self.optimal_weights,
            "market_regime_weights": self.market_regime_weights,
            "backtest_results": backtest_results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save final results
        os.makedirs(self.output_dir, exist_ok=True)
        results_path = os.path.join(self.output_dir, f"{self.pair_code}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Full optimization complete for {self.trading_pair}")
        logger.info(f"Results saved to {results_path}")
        
        return results
        
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Hyper-Optimized ML Ensemble for Maximum Accuracy")
    
    parser.add_argument("--pairs", type=str, nargs="+", default=SUPPORTED_PAIRS,
                        help="Trading pairs to optimize")
    
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME,
                        help="Timeframe to use for data")
    
    parser.add_argument("--target-accuracy", type=float, default=TARGET_ACCURACY,
                        help="Target accuracy to achieve")
    
    parser.add_argument("--epochs", type=int, default=500,
                        help="Maximum epochs for model training")
    
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for model training")
    
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of optimization iterations")
    
    parser.add_argument("--output-dir", type=str, default=RESULTS_DIR,
                        help="Directory to save optimization results")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")
    
    return parser.parse_args()
    
def main():
    """Main function"""
    args = parse_arguments()
    
    # Print banner
    print("=" * 80)
    print("HYPER-OPTIMIZED ML ENSEMBLE FOR MAXIMUM ACCURACY")
    print("=" * 80)
    print(f"Optimizing for: {', '.join(args.pairs)}")
    print(f"Target accuracy: {args.target_accuracy * 100:.2f}%")
    print("=" * 80)
    
    # Process each trading pair
    all_results = {}
    for pair in args.pairs:
        try:
            # Create optimizer
            optimizer = HyperOptimizedEnsemble(
                trading_pair=pair,
                timeframe=args.timeframe,
                target_accuracy=args.target_accuracy,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                optimization_iterations=args.iterations,
                verbose=args.verbose
            )
            
            # Run optimization
            results = optimizer.run_full_optimization()
            
            # Save results
            all_results[pair] = results
            
        except Exception as e:
            logger.error(f"Error optimizing {pair}: {e}")
            all_results[pair] = {"error": str(e)}
            
    # Save summary of all results
    summary_path = os.path.join(args.output_dir, "optimization_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    logger.info(f"Optimization complete. Summary saved to {summary_path}")
    
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    for pair, result in all_results.items():
        if "error" in result:
            print(f"{pair}: ERROR - {result['error']}")
        else:
            print(f"{pair}: Accuracy = {result['achieved_accuracy'] * 100:.2f}%, Profit Factor = {result['achieved_profit_factor']:.2f}")
    print("=" * 80)
    
if __name__ == "__main__":
    main()