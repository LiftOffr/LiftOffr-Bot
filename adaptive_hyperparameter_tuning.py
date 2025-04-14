#!/usr/bin/env python3
"""
Adaptive Hyperparameter Tuning for Kraken Trading Bot

This module implements a dynamic hyperparameter optimization system
that automatically adjusts model parameters based on recent performance
and market conditions to maximize trading results.
"""

import os
import json
import time
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
TUNING_RESULTS_DIR = "hyperparameter_tuning"
os.makedirs(TUNING_RESULTS_DIR, exist_ok=True)
MAX_TRIALS = 50
EVALUATION_INTERVAL = 24  # Hours between re-evaluations
PERFORMANCE_WINDOW = 100  # Number of trades to evaluate performance

class AdaptiveHyperparameterTuner:
    """
    Adaptive hyperparameter tuning system that automatically
    adjusts model parameters based on performance and market conditions.
    """
    def __init__(self, model_type, asset, strategy_name=None):
        """
        Initialize the adaptive hyperparameter tuner
        
        Args:
            model_type (str): Type of model to tune (e.g., "tcn", "lstm", "attention_gru", "tft")
            asset (str): Trading pair/asset (e.g., "SOL/USD")
            strategy_name (str, optional): Trading strategy name if applicable
        """
        self.model_type = model_type
        self.asset = asset
        self.strategy_name = strategy_name
        self.best_params = {}
        self.tuning_history = []
        self.current_market_regime = "normal"
        self.last_tuning_time = None
        self.performance_history = []
        
        # Create asset-specific directory
        self.results_dir = os.path.join(TUNING_RESULTS_DIR, asset.replace('/', '_'))
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load previous results if available
        self._load_previous_results()
        
        logger.info(f"Initialized hyperparameter tuner for {model_type} model on {asset}")
    
    def _load_previous_results(self):
        """Load previous tuning results if available"""
        results_path = os.path.join(
            self.results_dir, 
            f"{self.model_type}_params.json"
        )
        
        if os.path.exists(results_path):
            try:
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                self.best_params = results.get('best_params', {})
                self.tuning_history = results.get('tuning_history', [])
                self.last_tuning_time = results.get('last_tuning_time')
                self.performance_history = results.get('performance_history', [])
                
                logger.info(f"Loaded previous tuning results for {self.model_type} on {self.asset}")
            except Exception as e:
                logger.error(f"Error loading previous results: {e}")
    
    def _save_results(self):
        """Save tuning results for future use"""
        results_path = os.path.join(
            self.results_dir, 
            f"{self.model_type}_params.json"
        )
        
        results = {
            'model_type': self.model_type,
            'asset': self.asset,
            'best_params': self.best_params,
            'tuning_history': self.tuning_history,
            'last_tuning_time': datetime.now().isoformat(),
            'performance_history': self.performance_history
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved tuning results for {self.model_type} on {self.asset}")
    
    def update_market_regime(self, regime):
        """
        Update the current market regime
        
        Args:
            regime (str): Current market regime (e.g., "trending", "ranging", "volatile")
        """
        if regime != self.current_market_regime:
            logger.info(f"Market regime changed from {self.current_market_regime} to {regime}")
            self.current_market_regime = regime
    
    def update_performance(self, trades, profit_loss):
        """
        Update performance history with recent trading results
        
        Args:
            trades (list): List of recent trades
            profit_loss (float): Recent profit/loss
        """
        performance_entry = {
            'timestamp': datetime.now().isoformat(),
            'profit_loss': profit_loss,
            'trade_count': len(trades),
            'win_rate': sum(1 for t in trades if t.get('profit', 0) > 0) / max(1, len(trades)),
            'market_regime': self.current_market_regime,
            'params': self.best_params.copy()
        }
        
        self.performance_history.append(performance_entry)
        
        # Keep only recent history
        if len(self.performance_history) > PERFORMANCE_WINDOW:
            self.performance_history = self.performance_history[-PERFORMANCE_WINDOW:]
        
        # Save updated history
        self._save_results()
        
        logger.info(f"Updated performance history for {self.model_type} on {self.asset}")
    
    def should_tune_parameters(self):
        """
        Determine if hyperparameters should be re-tuned based on
        time elapsed and recent performance
        
        Returns:
            bool: True if parameters should be tuned, False otherwise
        """
        # Check if we've ever tuned before
        if not self.last_tuning_time:
            logger.info(f"No previous tuning found for {self.model_type} on {self.asset}")
            return True
        
        # Convert last tuning time from string to datetime
        if isinstance(self.last_tuning_time, str):
            last_tuning = datetime.fromisoformat(self.last_tuning_time)
        else:
            last_tuning = self.last_tuning_time
            
        # Check if enough time has elapsed
        hours_elapsed = (datetime.now() - last_tuning).total_seconds() / 3600
        if hours_elapsed >= EVALUATION_INTERVAL:
            logger.info(f"Time for scheduled re-tuning ({hours_elapsed:.1f} hours since last tune)")
            return True
        
        # Check performance degradation
        if len(self.performance_history) >= 10:
            recent_performance = self.performance_history[-5:]
            earlier_performance = self.performance_history[-10:-5]
            
            recent_profit = sum(p.get('profit_loss', 0) for p in recent_performance)
            earlier_profit = sum(p.get('profit_loss', 0) for p in earlier_performance)
            
            # If recent performance is significantly worse than earlier
            if recent_profit < 0 and recent_profit < earlier_profit * 0.7:
                logger.info(f"Performance degradation detected: recent={recent_profit:.2f}, earlier={earlier_profit:.2f}")
                return True
        
        # Check for market regime change requiring adaptation
        # Find the last performance entry that happened after tuning
        regime_at_tuning = None
        for entry in reversed(self.performance_history):
            entry_time = datetime.fromisoformat(entry.get('timestamp', '2000-01-01'))
            if entry_time <= last_tuning:
                regime_at_tuning = entry.get('market_regime')
                break
        
        if regime_at_tuning and regime_at_tuning != self.current_market_regime:
            logger.info(f"Market regime changed from {regime_at_tuning} to {self.current_market_regime}")
            return True
        
        return False
    
    def get_parameter_space(self):
        """
        Define parameter space based on model type
        
        Returns:
            dict: Parameter space definition with ranges
        """
        param_space = {}
        
        # Common parameters for all models
        param_space["learning_rate"] = {
            "type": "float",
            "min": 1e-5, 
            "max": 1e-2,
            "log": True
        }
        param_space["batch_size"] = {
            "type": "int",
            "min": 16, 
            "max": 256,
            "step": 16
        }
        param_space["epochs"] = {
            "type": "int", 
            "min": 50, 
            "max": 500,
            "step": 50
        }
        param_space["dropout_rate"] = {
            "type": "float", 
            "min": 0.1, 
            "max": 0.5
        }
        param_space["sequence_length"] = {
            "type": "int", 
            "min": 30, 
            "max": 120,
            "step": 10
        }
        
        # Model-specific parameters
        if self.model_type == "tcn":
            param_space["nb_filters"] = {
                "type": "int", 
                "min": 32, 
                "max": 128,
                "step": 16
            }
            param_space["kernel_size"] = {
                "type": "int", 
                "min": 2, 
                "max": 8,
                "step": 1
            }
            param_space["nb_stacks"] = {
                "type": "int", 
                "min": 1, 
                "max": 5,
                "step": 1
            }
        
        elif self.model_type == "lstm":
            param_space["lstm_units"] = {
                "type": "int", 
                "min": 32, 
                "max": 256,
                "step": 32
            }
            param_space["num_layers"] = {
                "type": "int", 
                "min": 1, 
                "max": 3,
                "step": 1
            }
            param_space["bidirectional"] = {
                "type": "categorical",
                "values": [True, False]
            }
        
        elif self.model_type == "attention_gru":
            param_space["gru_units"] = {
                "type": "int", 
                "min": 32, 
                "max": 256,
                "step": 32
            }
            param_space["attention_dim"] = {
                "type": "int", 
                "min": 16, 
                "max": 128,
                "step": 16
            }
            param_space["loss_ratio"] = {
                "type": "float", 
                "min": 1.0, 
                "max": 4.0
            }
            param_space["num_layers"] = {
                "type": "int", 
                "min": 1, 
                "max": 3,
                "step": 1
            }
        
        elif self.model_type == "tft":
            param_space["hidden_units"] = {
                "type": "int", 
                "min": 32, 
                "max": 256,
                "step": 32
            }
            param_space["num_heads"] = {
                "type": "int", 
                "min": 1, 
                "max": 8,
                "step": 1
            }
            param_space["loss_ratio"] = {
                "type": "float", 
                "min": 1.0, 
                "max": 4.0
            }
            param_space["forecast_horizon"] = {
                "type": "int", 
                "min": 1, 
                "max": 12,
                "step": 1
            }
        
        elif self.model_type == "transformer":
            param_space["d_model"] = {
                "type": "int", 
                "min": 32, 
                "max": 256,
                "step": 32
            }
            param_space["num_heads"] = {
                "type": "int", 
                "min": 1, 
                "max": 8,
                "step": 1
            }
            param_space["num_layers"] = {
                "type": "int", 
                "min": 1, 
                "max": 6,
                "step": 1
            }
            param_space["dff"] = {
                "type": "int", 
                "min": 64, 
                "max": 512,
                "step": 64
            }
        
        # Add strategy-specific parameters if applicable
        if self.strategy_name == "arima":
            param_space["lookback"] = {
                "type": "int", 
                "min": 10, 
                "max": 60,
                "step": 5
            }
            param_space["stop_multiplier"] = {
                "type": "float", 
                "min": 1.0, 
                "max": 3.0
            }
            param_space["profit_target_multiplier"] = {
                "type": "float", 
                "min": 1.0, 
                "max": 5.0
            }
        
        elif self.strategy_name == "adaptive":
            param_space["rsi_threshold"] = {
                "type": "int", 
                "min": 20, 
                "max": 40,
                "step": 1
            }
            param_space["adx_threshold"] = {
                "type": "int", 
                "min": 15, 
                "max": 30,
                "step": 1
            }
            param_space["stop_multiplier"] = {
                "type": "float", 
                "min": 1.0, 
                "max": 3.0
            }
        
        # Add market regime-specific adjustments
        if self.current_market_regime == "volatile":
            # For volatile markets, increase potential regularization
            param_space["dropout_rate"]["max"] = 0.7
            
            # More aggressive position sizing
            if "loss_ratio" in param_space:
                param_space["loss_ratio"]["max"] = 5.0
            
            # Shorter sequence length for faster adaptation
            param_space["sequence_length"]["min"] = 20
            param_space["sequence_length"]["max"] = 90
        
        elif self.current_market_regime == "trending":
            # For trending markets, favor longer-term signals
            param_space["sequence_length"]["min"] = 60
            
            # Less aggressive risk parameters
            if "stop_multiplier" in param_space:
                param_space["stop_multiplier"]["min"] = 1.5
            
            # Higher profit targets in trends
            if "profit_target_multiplier" in param_space:
                param_space["profit_target_multiplier"]["min"] = 2.0
        
        return param_space
    
    def _define_objective_function(self, X_train, y_train, X_val, y_val, create_model_fn):
        """
        Define objective function for hyperparameter optimization
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            create_model_fn (function): Function to create model with hyperparameters
            
        Returns:
            function: Objective function for optimization
        """
        def objective(trial):
            # Extract parameters based on the parameter space
            param_space = self.get_parameter_space()
            params = {}
            
            for param_name, param_config in param_space.items():
                if param_config["type"] == "float":
                    if param_config.get("log", False):
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_config["min"], 
                            param_config["max"], 
                            log=True
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_config["min"], 
                            param_config["max"]
                        )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name, 
                        param_config["min"], 
                        param_config["max"],
                        step=param_config.get("step", 1)
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name, 
                        param_config["values"]
                    )
            
            # Create and train model
            model = create_model_fn(**params)
            
            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=params.get('epochs', 100),
                batch_size=params.get('batch_size', 32),
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate model
            val_loss = min(history.history['val_loss'])
            
            # Record trial information
            trial.set_user_attr('val_loss', val_loss)
            trial.set_user_attr('params', params)
            trial.set_user_attr('epochs_trained', len(history.history['loss']))
            
            return val_loss
        
        return objective
    
    def tune_hyperparameters(self, X, y, create_model_fn, n_trials=MAX_TRIALS):
        """
        Tune hyperparameters for the model
        
        Args:
            X (np.ndarray): Features for training and validation
            y (np.ndarray): Targets for training and validation
            create_model_fn (function): Function to create model with hyperparameters
            n_trials (int): Number of optimization trials
            
        Returns:
            dict: Optimized hyperparameters
        """
        logger.info(f"Starting hyperparameter tuning for {self.model_type} on {self.asset} ({self.current_market_regime} regime)")
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Define study name
        study_name = f"{self.model_type}_{self.asset.replace('/', '_')}_{self.current_market_regime}"
        
        # Create study
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Define objective function
        objective = self._define_objective_function(X_train, y_train, X_val, y_val, create_model_fn)
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        # Record tuning results
        tuning_result = {
            'timestamp': datetime.now().isoformat(),
            'market_regime': self.current_market_regime,
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': n_trials
        }
        
        self.tuning_history.append(tuning_result)
        self.best_params = best_params
        self.last_tuning_time = datetime.now().isoformat()
        
        # Save results
        self._save_results()
        
        logger.info(f"Completed hyperparameter tuning for {self.model_type}. Best val_loss: {best_value:.6f}")
        
        return best_params
    
    def get_best_parameters(self, market_regime=None):
        """
        Get best hyperparameters, optionally for a specific market regime
        
        Args:
            market_regime (str, optional): Market regime to get parameters for
            
        Returns:
            dict: Best hyperparameters
        """
        # If specific regime requested and we have tuning history
        if market_regime and self.tuning_history:
            # Find most recent tuning for the requested regime
            regime_tunings = [t for t in self.tuning_history if t.get('market_regime') == market_regime]
            if regime_tunings:
                # Sort by timestamp (most recent first)
                sorted_tunings = sorted(regime_tunings, key=lambda x: x.get('timestamp', ''), reverse=True)
                return sorted_tunings[0].get('best_params', {})
        
        # Otherwise return current best parameters
        return self.best_params
    
    def analyze_parameter_importance(self):
        """
        Analyze parameter importance based on tuning history
        
        Returns:
            dict: Parameter importance scores
        """
        if not self.tuning_history:
            logger.warning("No tuning history available for parameter importance analysis")
            return {}
        
        # Collect all parameters and performance metrics
        all_params = []
        val_losses = []
        
        for tuning in self.tuning_history:
            params = tuning.get('best_params', {})
            val_loss = tuning.get('best_value')
            
            if params and val_loss is not None:
                all_params.append(params)
                val_losses.append(val_loss)
        
        if not all_params:
            return {}
        
        # Convert to DataFrame
        params_df = pd.DataFrame(all_params)
        
        # Calculate correlation with performance
        importance = {}
        for column in params_df.columns:
            if len(params_df[column].unique()) > 1:  # Only for parameters with variation
                correlation = np.corrcoef(params_df[column].astype(float), val_losses)[0, 1]
                importance[column] = abs(correlation)
        
        # Normalize importance scores
        max_score = max(importance.values()) if importance else 1.0
        importance = {k: v / max_score for k, v in importance.items()}
        
        return importance
    
    def suggest_parameter_adjustments(self, recent_profit_loss, min_adj=0.05, max_adj=0.3):
        """
        Suggest parameter adjustments based on recent performance
        
        Args:
            recent_profit_loss (float): Recent profit/loss
            min_adj (float): Minimum adjustment factor
            max_adj (float): Maximum adjustment factor
            
        Returns:
            dict: Suggested parameter adjustments
        """
        if not self.best_params:
            logger.warning("No best parameters available for adjustment suggestions")
            return {}
        
        # Calculate adjustment factor based on profitability
        # Lower profits mean larger adjustments
        if recent_profit_loss >= 0:
            adjustment_factor = min_adj + (max_adj - min_adj) * (1 - min(1, recent_profit_loss / 10))
        else:
            # If losing money, use maximum adjustment
            adjustment_factor = max_adj
        
        # Get parameter importance
        importance = self.analyze_parameter_importance()
        
        # Generate suggestions
        suggestions = {}
        param_space = self.get_parameter_space()
        
        for param, value in self.best_params.items():
            if param in param_space:
                space = param_space[param]
                param_importance = importance.get(param, 0.5)  # Default to medium importance
                
                # Calculate adjustment size based on importance
                adj_size = adjustment_factor * param_importance
                
                if space["type"] == "float":
                    min_val = space["min"]
                    max_val = space["max"]
                    range_size = max_val - min_val
                    
                    # Calculate adjustment range
                    lower_adj = max(min_val, value - range_size * adj_size)
                    upper_adj = min(max_val, value + range_size * adj_size)
                    
                    suggestions[param] = (lower_adj, upper_adj)
                
                elif space["type"] == "int":
                    min_val = space["min"]
                    max_val = space["max"]
                    range_size = max_val - min_val
                    
                    # Calculate adjustment range
                    lower_adj = max(min_val, value - int(range_size * adj_size))
                    upper_adj = min(max_val, value + int(range_size * adj_size))
                    
                    suggestions[param] = (lower_adj, upper_adj)
                
                elif space["type"] == "categorical":
                    # For categorical, just suggest keeping current value or trying alternatives
                    alternatives = [v for v in space["values"] if v != value]
                    if alternatives:
                        suggestions[param] = [value] + random.sample(alternatives, min(2, len(alternatives)))
                    else:
                        suggestions[param] = [value]
        
        return suggestions
    
    def adjust_parameters_for_market_regime(self):
        """
        Adjust parameters based on current market regime
        
        Returns:
            dict: Adjusted parameters for current market regime
        """
        # First check if we have specific tuning for this regime
        regime_params = self.get_best_parameters(market_regime=self.current_market_regime)
        if regime_params:
            logger.info(f"Using regime-specific parameters for {self.current_market_regime}")
            return regime_params
        
        # If no specific tuning, adjust based on heuristics
        logger.info(f"Adjusting parameters for {self.current_market_regime} regime based on heuristics")
        
        params = self.best_params.copy()
        
        if self.current_market_regime == "volatile":
            # For volatile markets: 
            # - Increase regularization (dropout)
            # - Shorter sequence length for faster adaptation
            # - Higher learning rate for faster adaptation
            if "dropout_rate" in params:
                params["dropout_rate"] = min(0.7, params["dropout_rate"] * 1.5)
            
            if "sequence_length" in params:
                params["sequence_length"] = max(20, int(params["sequence_length"] * 0.8))
            
            if "learning_rate" in params:
                params["learning_rate"] = min(0.01, params["learning_rate"] * 1.5)
            
            # Increase loss ratio for asymmetric loss to be more conservative
            if "loss_ratio" in params:
                params["loss_ratio"] = min(5.0, params["loss_ratio"] * 1.3)
        
        elif self.current_market_regime == "trending":
            # For trending markets:
            # - Use longer sequences to capture trend
            # - Lower dropout for less regularization
            # - More hidden units for more capacity
            if "dropout_rate" in params:
                params["dropout_rate"] = max(0.1, params["dropout_rate"] * 0.8)
            
            if "sequence_length" in params:
                params["sequence_length"] = min(120, int(params["sequence_length"] * 1.2))
            
            # Increase hidden units if applicable
            for unit_param in ["lstm_units", "gru_units", "hidden_units", "nb_filters", "d_model"]:
                if unit_param in params:
                    params[unit_param] = min(256, int(params[unit_param] * 1.2))
        
        elif self.current_market_regime == "ranging":
            # For ranging markets:
            # - More balanced parameters
            # - Slightly higher dropout to prevent overfitting to noise
            # - Medium sequence length
            if "dropout_rate" in params:
                params["dropout_rate"] = min(0.5, params["dropout_rate"] * 1.1)
            
            # Adjust sequence length to medium range
            if "sequence_length" in params:
                if params["sequence_length"] < 40:
                    params["sequence_length"] = min(60, params["sequence_length"] * 1.2)
                elif params["sequence_length"] > 80:
                    params["sequence_length"] = max(60, params["sequence_length"] * 0.9)
        
        return params
    
    def get_optimal_parameters(self):
        """
        Get optimal parameters considering market regime and recent performance
        
        Returns:
            dict: Optimal parameters for current conditions
        """
        # First check if we should re-tune
        if self.should_tune_parameters():
            logger.info(f"Recommended to re-tune {self.model_type} for {self.asset}")
            # We can't actually run the tuning here as it requires data and model creation function
            # Return current best with note about re-tuning
            return {
                'params': self.best_params,
                'suggestion': 'retune',
                'market_regime': self.current_market_regime
            }
        
        # Otherwise adjust for current market regime
        adjusted_params = self.adjust_parameters_for_market_regime()
        
        return {
            'params': adjusted_params,
            'suggestion': 'use_adjusted',
            'market_regime': self.current_market_regime
        }

class HyperparameterManager:
    """
    Manager class to coordinate hyperparameter tuning across multiple models and assets
    """
    def __init__(self):
        self.tuners = {}
        self.global_market_regime = "normal"
        
        logger.info("Initialized HyperparameterManager")
    
    def get_tuner(self, model_type, asset, strategy_name=None):
        """
        Get or create a tuner for the specified model and asset
        
        Args:
            model_type (str): Type of model to tune
            asset (str): Trading pair/asset
            strategy_name (str, optional): Trading strategy name if applicable
            
        Returns:
            AdaptiveHyperparameterTuner: Tuner instance
        """
        key = f"{model_type}_{asset}_{strategy_name or ''}"
        
        if key not in self.tuners:
            self.tuners[key] = AdaptiveHyperparameterTuner(model_type, asset, strategy_name)
            # Set the current market regime
            self.tuners[key].update_market_regime(self.global_market_regime)
        
        return self.tuners[key]
    
    def update_global_market_regime(self, regime):
        """
        Update global market regime and propagate to all tuners
        
        Args:
            regime (str): Market regime to set
        """
        if regime != self.global_market_regime:
            logger.info(f"Global market regime changed from {self.global_market_regime} to {regime}")
            self.global_market_regime = regime
            
            # Update all tuners
            for tuner in self.tuners.values():
                tuner.update_market_regime(regime)
    
    def get_models_requiring_tuning(self):
        """
        Get list of models that require re-tuning
        
        Returns:
            list: List of (model_type, asset, strategy_name) tuples
        """
        needs_tuning = []
        
        for key, tuner in self.tuners.items():
            if tuner.should_tune_parameters():
                parts = key.split('_')
                if len(parts) >= 2:
                    model_type = parts[0]
                    asset = parts[1]
                    strategy_name = '_'.join(parts[2:]) if len(parts) > 2 else None
                    needs_tuning.append((model_type, asset, strategy_name))
        
        return needs_tuning
    
    def get_optimal_parameters_for_all(self):
        """
        Get optimal parameters for all models
        
        Returns:
            dict: Dictionary of optimal parameters for each model
        """
        optimal_params = {}
        
        for key, tuner in self.tuners.items():
            optimal_params[key] = tuner.get_optimal_parameters()
        
        return optimal_params

# Create a global instance for easy access
hyperparameter_manager = HyperparameterManager()

def main():
    """Test the adaptive hyperparameter tuning system"""
    logger.info("Adaptive Hyperparameter Tuning module imported")

if __name__ == "__main__":
    main()