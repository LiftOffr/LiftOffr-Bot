#!/usr/bin/env python3
"""
Strategy Ensemble Trainer

This module implements a collaborative training approach for multiple trading strategies,
optimizing how they work together rather than just individually. It combines:

1. Cross-strategy signal coordination
2. Ensemble model training with specialized roles
3. Collaborative reinforcement learning
4. Signal arbitration optimization
5. Market regime-specific strategy specialization

The goal is to create a "team" of strategies that complement each other rather than
competing or duplicating efforts across different market conditions.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from enum import Enum

# Local imports - adjust as needed
from market_context import detect_market_regime
from dynamic_position_sizing_ml import calculate_dynamic_leverage, calculate_dynamic_position_size
from ml_models import prepare_data, train_model, evaluate_model
from advanced_ensemble_model import DynamicWeightedEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('strategy_ensemble_training.log')
    ]
)
logger = logging.getLogger(__name__)

# Strategy roles and specialization
class StrategyRole(Enum):
    TREND_FOLLOWER = "trend_follower"
    COUNTER_TREND = "counter_trend"
    BREAKOUT = "breakout"
    RANGE_BOUND = "range_bound"
    VOLATILITY = "volatility"
    MULTI_TIMEFRAME = "multi_timeframe"
    DEFENSIVE = "defensive"
    AGGRESSIVE = "aggressive"

# Market regime specialization mapping (which strategies work best in which regimes)
REGIME_STRATEGY_MAPPING = {
    "volatile_trending_up": [
        StrategyRole.TREND_FOLLOWER,
        StrategyRole.BREAKOUT,
        StrategyRole.VOLATILITY
    ],
    "volatile_trending_down": [
        StrategyRole.COUNTER_TREND,
        StrategyRole.VOLATILITY,
        StrategyRole.DEFENSIVE
    ],
    "normal_trending_up": [
        StrategyRole.TREND_FOLLOWER,
        StrategyRole.MULTI_TIMEFRAME,
        StrategyRole.AGGRESSIVE
    ],
    "normal_trending_down": [
        StrategyRole.COUNTER_TREND,
        StrategyRole.MULTI_TIMEFRAME,
        StrategyRole.DEFENSIVE
    ],
    "neutral": [
        StrategyRole.RANGE_BOUND,
        StrategyRole.VOLATILITY,
        StrategyRole.DEFENSIVE
    ]
}

# Signal agreement threshold (how much strategies need to agree)
SIGNAL_AGREEMENT_THRESHOLD = 0.7

# Strategy name to role mapping (customize based on your strategies)
STRATEGY_ROLES = {
    "ARIMAStrategy": [StrategyRole.TREND_FOLLOWER, StrategyRole.RANGE_BOUND],
    "AdaptiveStrategy": [StrategyRole.COUNTER_TREND, StrategyRole.DEFENSIVE],
    "IntegratedStrategy": [StrategyRole.VOLATILITY, StrategyRole.MULTI_TIMEFRAME],
    "MLStrategy": [StrategyRole.BREAKOUT, StrategyRole.AGGRESSIVE],
}

class StrategyEnsembleTrainer:
    """
    Trains multiple strategies to work effectively as an ensemble
    by optimizing their collective performance rather than individual performance
    """
    
    def __init__(
        self,
        strategies: List[str],
        assets: List[str] = ["SOL/USD", "ETH/USD", "BTC/USD"],
        data_dir: str = "historical_data",
        timeframes: List[str] = ["5m", "15m", "1h", "4h"],
        training_days: int = 90,
        validation_days: int = 30,
        ensemble_output_dir: str = "models/ensemble"
    ):
        """
        Initialize the ensemble trainer
        
        Args:
            strategies: List of strategy names to train together
            assets: List of trading pairs to train on
            data_dir: Directory with historical data
            timeframes: List of timeframes to use
            training_days: Days of data to use for training
            validation_days: Days of data to use for validation
            ensemble_output_dir: Directory to save ensemble models
        """
        self.strategies = strategies
        self.assets = assets
        self.data_dir = data_dir
        self.timeframes = timeframes
        self.training_days = training_days
        self.validation_days = validation_days
        self.output_dir = ensemble_output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track performance metrics during training
        self.performance_history = {}
        self.regime_performance = {}
        self.strategy_weights = self._initialize_strategy_weights()
        self.collaborative_metrics = []
        
        # Map strategies to their roles
        self.strategy_to_roles = {}
        for strategy in strategies:
            self.strategy_to_roles[strategy] = STRATEGY_ROLES.get(strategy, [])
        
        logger.info(f"Initialized Strategy Ensemble Trainer with {len(strategies)} strategies: {strategies}")
        logger.info(f"Training on assets: {assets}")
    
    def _initialize_strategy_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize strategy weights for each market regime
        
        Returns:
            Dict: Dictionary mapping regimes to strategy weights
        """
        weights = {}
        
        # Initialize weights for each market regime
        for regime in REGIME_STRATEGY_MAPPING.keys():
            weights[regime] = {}
            # Create equal initial weights for all strategies
            for strategy in self.strategies:
                weights[regime][strategy] = 1.0 / len(self.strategies)
        
        return weights
    
    def _load_historical_data(
        self, 
        asset: str, 
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load historical data for a specific asset and timeframe
        
        Args:
            asset: Trading pair to load data for
            timeframe: Timeframe to load
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            DataFrame: Historical price data
        """
        # Determine file path
        asset_clean = asset.replace("/", "")
        file_path = os.path.join(self.data_dir, f"{asset_clean}_{timeframe}.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Historical data file not found: {file_path}")
            return pd.DataFrame()
        
        # Load data
        try:
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            # Filter by date range if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            logger.info(f"Loaded {len(df)} historical data points for {asset} ({timeframe})")
            return df
        
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def _prepare_regime_specific_data(
        self,
        data: pd.DataFrame,
        regime: str
    ) -> pd.DataFrame:
        """
        Prepare data specific to a market regime for better training
        
        Args:
            data: Input data (full dataset)
            regime: Market regime to extract
            
        Returns:
            DataFrame: Data specific to the regime
        """
        # Detect market regime for each data point
        regimes = []
        
        for i in range(len(data)):
            # Use a window of 100 data points (or less if not available)
            start_idx = max(0, i - 99)
            window = data.iloc[start_idx:i+1]
            if len(window) > 20:  # Need minimum data points
                current_regime = detect_market_regime(window)
                regimes.append(current_regime)
            else:
                regimes.append("neutral")  # Default when not enough data
        
        data['detected_regime'] = regimes
        
        # Extract data specific to the requested regime
        regime_data = data[data['detected_regime'] == regime].copy()
        
        # If not enough regime-specific data, use the closest regime
        if len(regime_data) < 100:
            logger.warning(f"Not enough data for regime {regime}, using similar regime")
            
            # Define similar regimes
            similar_regimes = {
                "volatile_trending_up": ["normal_trending_up"],
                "volatile_trending_down": ["normal_trending_down"],
                "normal_trending_up": ["volatile_trending_up"],
                "normal_trending_down": ["volatile_trending_down"],
                "neutral": ["normal_trending_up", "normal_trending_down"]
            }
            
            for similar in similar_regimes.get(regime, []):
                similar_data = data[data['detected_regime'] == similar]
                regime_data = pd.concat([regime_data, similar_data])
                if len(regime_data) >= 100:
                    break
        
        # Clean up
        if 'detected_regime' in regime_data.columns:
            regime_data.drop('detected_regime', axis=1, inplace=True)
        
        return regime_data
    
    def _prepare_strategy_specific_features(
        self,
        data: pd.DataFrame,
        strategy: str
    ) -> pd.DataFrame:
        """
        Add strategy-specific features to the dataset
        
        Args:
            data: Input data
            strategy: Strategy name
            
        Returns:
            DataFrame: Data with strategy-specific features
        """
        df = data.copy()
        
        # Add ARIMA-specific features
        if strategy == "ARIMAStrategy":
            # Add Moving Averages
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['sma50'] = df['close'].rolling(window=50).mean()
            df['sma100'] = df['close'].rolling(window=100).mean()
            df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
            
            # Price change rates
            df['price_change_1'] = df['close'].pct_change(1)
            df['price_change_5'] = df['close'].pct_change(5)
            df['price_change_10'] = df['close'].pct_change(10)
        
        # Add Adaptive-specific features
        elif strategy == "AdaptiveStrategy":
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['stddev'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['sma20'] + (df['stddev'] * 2)
            df['lower_band'] = df['sma20'] - (df['stddev'] * 2)
            
            # MACD
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Add Integrated-specific features
        elif strategy == "IntegratedStrategy":
            # ATR
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
            # Keltner Channel
            df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['kc_upper'] = df['ema20'] + df['atr'] * 2
            df['kc_lower'] = df['ema20'] - df['atr'] * 2
            
            # Volatility
            df['volatility'] = df['close'].pct_change().rolling(20).std()
            
            # Multi-timeframe features (simulated here)
            df['higher_tf_trend'] = df['close'].rolling(50).mean().pct_change(10)
            df['lower_tf_momentum'] = df['close'].pct_change(5).rolling(10).mean()
        
        # Add ML-specific features
        elif strategy == "MLStrategy":
            # Include all of the above features plus:
            
            # Return rates over multiple timeframes
            for period in [1, 3, 5, 10, 20, 50]:
                df[f'return_{period}'] = df['close'].pct_change(period)
            
            # Volume-based features
            if 'volume' in df.columns:
                df['volume_change'] = df['volume'].pct_change()
                df['volume_ma10'] = df['volume'].rolling(10).mean()
                df['relative_volume'] = df['volume'] / df['volume_ma10']
            
            # Price patterns (simple approximation)
            df['hl_range'] = (df['high'] - df['low']) / df['low']
            df['body_size'] = (df['close'] - df['open']).abs() / df['open']
            df['upper_wick'] = (df['high'] - df['close'].clip(lower=df['open'])) / df['close']
            df['lower_wick'] = (df['close'].clip(upper=df['open']) - df['low']) / df['close']
        
        # Drop NAs that result from various calculations
        df.dropna(inplace=True)
        
        return df
    
    def _train_regime_specific_model(
        self,
        data: pd.DataFrame,
        strategy: str,
        regime: str,
        asset: str
    ) -> Dict[str, Any]:
        """
        Train a regime-specific model for a particular strategy
        
        Args:
            data: Training data with features
            strategy: Strategy name
            regime: Market regime
            asset: Trading pair
            
        Returns:
            Dict: Trained model and metrics
        """
        # Prepare features and labels based on strategy
        features, labels = self._prepare_strategy_features_labels(data, strategy)
        
        # Split into training and validation sets (80/20)
        split_idx = int(len(features) * 0.8)
        X_train, X_val = features[:split_idx], features[split_idx:]
        y_train, y_val = labels[:split_idx], labels[split_idx:]
        
        # Train model
        logger.info(f"Training {strategy} model for {regime} regime on {asset}")
        
        model = train_model(
            X_train, y_train, X_val, y_val,
            model_type=strategy,
            epochs=50,
            batch_size=32,
            learning_rate=0.001
        )
        
        # Evaluate model
        metrics = evaluate_model(model, X_val, y_val)
        
        logger.info(f"Model performance: {metrics}")
        
        # Save model
        model_filename = f"{strategy}_{regime}_{asset.replace('/', '')}_model"
        model_path = os.path.join(self.output_dir, model_filename)
        
        # In a real implementation, save the actual model
        # model.save(model_path)
        
        return {
            "model": model,
            "metrics": metrics,
            "model_path": model_path
        }
    
    def _prepare_strategy_features_labels(
        self,
        data: pd.DataFrame,
        strategy: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare strategy-specific features and labels for training
        
        Args:
            data: Input data with features
            strategy: Strategy name
            
        Returns:
            Tuple: (features, labels)
        """
        # Define the features to use for each strategy
        if strategy == "ARIMAStrategy":
            feature_cols = ['open', 'high', 'low', 'close',
                            'sma20', 'sma50', 'sma100', 'ema9', 'ema21',
                            'price_change_1', 'price_change_5', 'price_change_10']
        
        elif strategy == "AdaptiveStrategy":
            feature_cols = ['open', 'high', 'low', 'close',
                            'rsi', 'sma20', 'upper_band', 'lower_band',
                            'macd', 'macd_signal', 'macd_hist']
        
        elif strategy == "IntegratedStrategy":
            feature_cols = ['open', 'high', 'low', 'close',
                            'atr', 'ema20', 'kc_upper', 'kc_lower',
                            'volatility', 'higher_tf_trend', 'lower_tf_momentum']
        
        elif strategy == "MLStrategy":
            # Use all available features for ML strategy
            feature_cols = [col for col in data.columns if col not in ['timestamp', 'detected_regime']]
        
        else:
            # Default features
            feature_cols = ['open', 'high', 'low', 'close']
        
        # Filter columns that exist in the data
        available_cols = [col for col in feature_cols if col in data.columns]
        
        # Create features array
        features = data[available_cols].values
        
        # Create labels based on future returns
        future_returns = data['close'].pct_change(5).shift(-5)  # 5-period forward returns
        labels = np.where(future_returns > 0, 1, 0)  # 1 for up, 0 for down
        
        return features, labels
    
    def _optimize_strategy_weights(
        self,
        individual_performances: Dict[str, Dict],
        collaborative_performance: float,
        regime: str
    ) -> Dict[str, float]:
        """
        Optimize strategy weights based on both individual and collaborative performance
        
        Args:
            individual_performances: Dictionary of individual strategy performances
            collaborative_performance: Performance of the ensemble
            regime: Market regime
            
        Returns:
            Dict: Updated strategy weights
        """
        weights = self.strategy_weights[regime].copy()
        
        # Get current weights
        strategies = list(weights.keys())
        
        # Calculate performance scores
        scores = {}
        for strategy in strategies:
            if strategy in individual_performances:
                # Combine accuracy, precision, and recall
                perf = individual_performances[strategy]
                score = (
                    perf.get('accuracy', 0) * 0.4 +
                    perf.get('precision', 0) * 0.3 +
                    perf.get('recall', 0) * 0.3
                )
                scores[strategy] = score
            else:
                scores[strategy] = 0.0
        
        # Adjust weights based on scores
        total_score = sum(scores.values())
        if total_score > 0:
            new_weights = {strategy: score / total_score for strategy, score in scores.items()}
            
            # Blend with original weights
            for strategy in strategies:
                weights[strategy] = weights[strategy] * 0.3 + new_weights[strategy] * 0.7
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {strategy: w / total_weight for strategy, w in weights.items()}
        
        logger.info(f"Updated weights for {regime} regime: {weights}")
        
        return weights
    
    def _train_collaborative_model(
        self,
        strategies: List[str],
        data: pd.DataFrame,
        regime: str,
        asset: str
    ) -> Dict[str, Any]:
        """
        Train a collaborative ensemble model that combines strategies
        
        Args:
            strategies: List of strategies to include
            data: Training data
            regime: Market regime
            asset: Trading pair
            
        Returns:
            Dict: Ensemble model and metrics
        """
        logger.info(f"Training collaborative ensemble for {regime} regime on {asset}")
        
        # Split data into training and validation
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        # Train individual strategy models
        individual_models = {}
        individual_performances = {}
        
        for strategy in strategies:
            # Prepare strategy-specific data
            strategy_data = self._prepare_strategy_specific_features(data, strategy)
            
            # Train strategy model
            model_info = self._train_regime_specific_model(
                strategy_data, strategy, regime, asset
            )
            
            individual_models[strategy] = model_info['model']
            individual_performances[strategy] = model_info['metrics']
        
        # Create ensemble configuration
        ensemble_config = {
            "asset": asset,
            "regime": regime,
            "strategies": strategies,
            "weights": self.strategy_weights[regime],
            "model_paths": {strategy: f"{strategy}_{regime}_{asset.replace('/', '')}_model" 
                           for strategy in strategies}
        }
        
        # Save ensemble configuration
        config_path = os.path.join(
            self.output_dir, 
            f"ensemble_{regime}_{asset.replace('/', '')}_config.json"
        )
        
        with open(config_path, 'w') as f:
            json.dump(ensemble_config, f, indent=4)
        
        # Evaluate collaborative performance
        collaborative_performance = self._evaluate_collaborative_performance(
            individual_models, val_data, strategies, regime
        )
        
        # Optimize strategy weights
        updated_weights = self._optimize_strategy_weights(
            individual_performances, collaborative_performance['ensemble_accuracy'], regime
        )
        
        # Update weights
        self.strategy_weights[regime] = updated_weights
        
        return {
            "config": ensemble_config,
            "performance": collaborative_performance,
            "config_path": config_path
        }
    
    def _evaluate_collaborative_performance(
        self,
        models: Dict[str, Any],
        data: pd.DataFrame,
        strategies: List[str],
        regime: str
    ) -> Dict[str, float]:
        """
        Evaluate how well strategies work together
        
        Args:
            models: Dictionary of trained models
            data: Validation data
            strategies: List of strategies to evaluate
            regime: Market regime
            
        Returns:
            Dict: Performance metrics
        """
        predictions = {}
        true_direction = np.sign(data['close'].pct_change(5).shift(-5))
        
        # Get predictions from each strategy
        for strategy in strategies:
            # Prepare strategy-specific features
            strategy_data = self._prepare_strategy_specific_features(data, strategy)
            features, _ = self._prepare_strategy_features_labels(strategy_data, strategy)
            
            # Generate predictions
            model = models[strategy]
            raw_preds = model.predict(features)
            
            # Convert to -1, 0, 1 (sell, neutral, buy)
            pred_direction = np.zeros(len(raw_preds))
            pred_direction[raw_preds > 0.6] = 1  # Buy signal with high confidence
            pred_direction[raw_preds < 0.4] = -1  # Sell signal with high confidence
            
            predictions[strategy] = pred_direction
        
        # Create ensemble prediction
        weights = self.strategy_weights[regime]
        ensemble_prediction = np.zeros(len(true_direction))
        
        for strategy in strategies:
            if strategy in weights and strategy in predictions:
                ensemble_prediction += weights[strategy] * predictions[strategy]
        
        # Convert to -1, 0, 1
        ensemble_prediction = np.sign(ensemble_prediction)
        
        # Calculate ensemble accuracy
        ensemble_accuracy = np.mean(ensemble_prediction == true_direction)
        
        # Calculate agreement rate
        agreement_matrix = np.zeros((len(strategies), len(true_direction)))
        for i, strategy in enumerate(strategies):
            if strategy in predictions:
                agreement_matrix[i, :] = predictions[strategy]
        
        # Proportion of data points where strategies agree
        agreement_rate = np.mean(np.std(agreement_matrix, axis=0) == 0)
        
        # Calculate individual accuracies
        individual_accuracies = {}
        for strategy in strategies:
            if strategy in predictions:
                individual_accuracies[strategy] = np.mean(predictions[strategy] == true_direction)
        
        # Calculate collaborative lift (how much better the ensemble is than individuals)
        best_individual = max(individual_accuracies.values()) if individual_accuracies else 0
        collaborative_lift = ensemble_accuracy - best_individual
        
        results = {
            "ensemble_accuracy": ensemble_accuracy,
            "agreement_rate": agreement_rate,
            "collaborative_lift": collaborative_lift,
            "individual_accuracies": individual_accuracies
        }
        
        logger.info(f"Collaborative performance for {regime}: {results}")
        
        return results
    
    def train_strategy_ensemble(self):
        """
        Train the complete strategy ensemble across all assets and regimes
        """
        logger.info("Starting strategy ensemble training")
        
        # Calculate training and validation date ranges
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.training_days + self.validation_days)
        
        # Ensemble results for all assets and regimes
        ensemble_results = {}
        
        # Train for each asset
        for asset in self.assets:
            ensemble_results[asset] = {}
            
            # Load and prepare data
            asset_data = {}
            for timeframe in self.timeframes:
                data = self._load_historical_data(asset, timeframe, start_date, end_date)
                if not data.empty:
                    asset_data[timeframe] = data
            
            if not asset_data:
                logger.warning(f"No data available for {asset}, skipping")
                continue
            
            # Use 1h timeframe as the primary data
            primary_data = asset_data.get('1h')
            if primary_data is None:
                primary_timeframe = list(asset_data.keys())[0]
                primary_data = asset_data[primary_timeframe]
                logger.warning(f"1h data not available for {asset}, using {primary_timeframe} instead")
            
            # Train for each market regime
            for regime in REGIME_STRATEGY_MAPPING.keys():
                # Prepare regime-specific data
                regime_data = self._prepare_regime_specific_data(primary_data, regime)
                
                if len(regime_data) < 100:
                    logger.warning(f"Not enough data for {regime} regime on {asset}, skipping")
                    continue
                
                # Determine which strategies work best for this regime
                regime_roles = REGIME_STRATEGY_MAPPING[regime]
                appropriate_strategies = []
                
                for strategy in self.strategies:
                    strategy_roles = self.strategy_to_roles.get(strategy, [])
                    # Check if strategy has at least one matching role for this regime
                    if any(role in regime_roles for role in strategy_roles):
                        appropriate_strategies.append(strategy)
                
                if not appropriate_strategies:
                    appropriate_strategies = self.strategies
                    logger.warning(f"No strategies specifically suited for {regime}, using all strategies")
                
                # Train collaborative model for this regime
                ensemble = self._train_collaborative_model(
                    appropriate_strategies, regime_data, regime, asset
                )
                
                ensemble_results[asset][regime] = ensemble
        
        # Save final strategy weights
        weights_path = os.path.join(self.output_dir, "strategy_ensemble_weights.json")
        with open(weights_path, 'w') as f:
            json.dump(self.strategy_weights, f, indent=4)
        
        logger.info(f"Strategy ensemble training completed, weights saved to {weights_path}")
        
        return ensemble_results
    
    def visualize_ensemble_performance(self, results: Dict[str, Dict[str, Dict]]):
        """
        Visualize the performance of the strategy ensemble
        
        Args:
            results: Dictionary of ensemble results
        """
        # Create a figure for accuracy comparison
        plt.figure(figsize=(15, 10))
        
        # Plot accuracy by regime and asset
        regimes = list(REGIME_STRATEGY_MAPPING.keys())
        assets = self.assets
        
        # Prepare data for plotting
        ensemble_accuracies = np.zeros((len(assets), len(regimes)))
        individual_accuracies = np.zeros((len(self.strategies), len(assets), len(regimes)))
        
        for i, asset in enumerate(assets):
            if asset not in results:
                continue
                
            for j, regime in enumerate(regimes):
                if regime not in results[asset]:
                    continue
                
                # Get ensemble accuracy
                perf = results[asset][regime]['performance']
                ensemble_accuracies[i, j] = perf.get('ensemble_accuracy', 0)
                
                # Get individual accuracies
                indiv_acc = perf.get('individual_accuracies', {})
                for k, strategy in enumerate(self.strategies):
                    individual_accuracies[k, i, j] = indiv_acc.get(strategy, 0)
        
        # Plot ensemble accuracies
        plt.subplot(2, 1, 1)
        plt.title('Ensemble Accuracy by Regime and Asset')
        plt.imshow(ensemble_accuracies, cmap='viridis', aspect='auto')
        plt.colorbar(label='Accuracy')
        plt.xticks(range(len(regimes)), [r.replace('_', ' ').title() for r in regimes], rotation=45)
        plt.yticks(range(len(assets)), assets)
        plt.xlabel('Market Regime')
        plt.ylabel('Asset')
        
        # Plot strategy weights
        plt.subplot(2, 1, 2)
        plt.title('Strategy Weights by Regime')
        
        # Convert weights to array for plotting
        weight_data = np.zeros((len(self.strategies), len(regimes)))
        for i, regime in enumerate(regimes):
            for j, strategy in enumerate(self.strategies):
                weight_data[j, i] = self.strategy_weights[regime].get(strategy, 0)
        
        plt.imshow(weight_data, cmap='plasma', aspect='auto')
        plt.colorbar(label='Weight')
        plt.xticks(range(len(regimes)), [r.replace('_', ' ').title() for r in regimes], rotation=45)
        plt.yticks(range(len(self.strategies)), self.strategies)
        plt.xlabel('Market Regime')
        plt.ylabel('Strategy')
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = os.path.join(self.output_dir, "ensemble_performance.png")
        plt.savefig(output_path)
        logger.info(f"Performance visualization saved to {output_path}")
        
        # Also create individual plots for each asset
        for asset in assets:
            if asset not in results:
                continue
                
            plt.figure(figsize=(12, 8))
            plt.title(f'Strategy Performance for {asset}')
            
            # Extract data for this asset
            asset_idx = assets.index(asset)
            asset_data = individual_accuracies[:, asset_idx, :]
            
            plt.imshow(asset_data, cmap='viridis', aspect='auto')
            plt.colorbar(label='Accuracy')
            plt.xticks(range(len(regimes)), [r.replace('_', ' ').title() for r in regimes], rotation=45)
            plt.yticks(range(len(self.strategies)), self.strategies)
            plt.xlabel('Market Regime')
            plt.ylabel('Strategy')
            
            plt.tight_layout()
            
            # Save the asset visualization
            asset_output_path = os.path.join(
                self.output_dir, 
                f"ensemble_performance_{asset.replace('/', '')}.png"
            )
            plt.savefig(asset_output_path)

def main():
    """Run the strategy ensemble trainer"""
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Train strategy ensemble')
    parser.add_argument('--strategies', nargs='+', default=["ARIMAStrategy", "AdaptiveStrategy", "IntegratedStrategy", "MLStrategy"],
                      help='List of strategies to train')
    parser.add_argument('--assets', nargs='+', default=["SOL/USD", "ETH/USD", "BTC/USD"],
                      help='List of assets to train on')
    parser.add_argument('--timeframes', nargs='+', default=["5m", "15m", "1h", "4h"],
                      help='List of timeframes to use')
    parser.add_argument('--training-days', type=int, default=90,
                      help='Days of data for training')
    parser.add_argument('--validation-days', type=int, default=30,
                      help='Days of data for validation')
    parser.add_argument('--output-dir', type=str, default='models/ensemble',
                      help='Output directory for ensemble models')
    
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = StrategyEnsembleTrainer(
        strategies=args.strategies,
        assets=args.assets,
        timeframes=args.timeframes,
        training_days=args.training_days,
        validation_days=args.validation_days,
        ensemble_output_dir=args.output_dir
    )
    
    results = trainer.train_strategy_ensemble()
    trainer.visualize_ensemble_performance(results)
    
    logger.info("Strategy ensemble training completed")

if __name__ == "__main__":
    main()