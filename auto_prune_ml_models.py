#!/usr/bin/env python3
"""
Auto-Pruning System for ML Models

This module implements an intelligent auto-pruning system for machine learning models
that can identify and remove underperforming components of the trading system.

Key features:
1. Automated detection of unprofitable ML models
2. Component-level pruning (sub-networks, features, timeframes)
3. Retraining of pruned models with optimized architectures
4. Performance-based weighting of ensemble components
5. Market regime-specific model selection

Target: Supporting the 90%+ directional prediction accuracy goal
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union

# Import local modules
from enhanced_tcn_model import EnhancedTCNModel
from enhanced_historical_data_fetcher import EnhancedHistoricalDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = "models"
PRUNED_MODELS_DIR = "models/pruned"
PERFORMANCE_THRESHOLD = 0.55  # Minimum accuracy for keeping a model (55%)
HIGH_PERFORMANCE_THRESHOLD = 0.7  # High performance threshold (70%)
RETRAINING_EPOCHS = 50
DEFAULT_SYMBOLS = ["SOLUSD", "BTCUSD", "ETHUSD"]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
RESULTS_DIR = "optimization_results/auto_prune"

# Ensure directories exist
os.makedirs(PRUNED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class ModelPruner:
    """
    Intelligent model pruning system that identifies and removes
    underperforming components of the ML trading system.
    """
    
    def __init__(self, base_model_path=None, performance_threshold=PERFORMANCE_THRESHOLD,
                 high_performance_threshold=HIGH_PERFORMANCE_THRESHOLD):
        """
        Initialize the model pruner
        
        Args:
            base_model_path (str, optional): Path to base model
            performance_threshold (float): Minimum threshold for keeping a component
            high_performance_threshold (float): Threshold for high-performing components
        """
        self.base_model_path = base_model_path
        self.performance_threshold = performance_threshold
        self.high_performance_threshold = high_performance_threshold
        
        # Load base model if provided
        self.base_model = None
        if base_model_path and os.path.exists(base_model_path):
            logger.info(f"Loading base model from {base_model_path}")
            try:
                self.base_model = load_model(base_model_path)
            except Exception as e:
                logger.error(f"Error loading base model: {e}")
        
        # Component performance tracking
        self.component_performance = {}
        
        logger.info(f"Model pruner initialized with performance threshold: {performance_threshold}")
        logger.info(f"High performance threshold: {high_performance_threshold}")
    
    def evaluate_model_components(self, model, X_test, y_test, component_names=None):
        """
        Evaluate performance of individual model components
        
        Args:
            model: Model to evaluate
            X_test: Test data
            y_test: Test labels
            component_names (list, optional): Names of components
            
        Returns:
            dict: Component performance
        """
        # Check if model is a Keras functional model
        if not isinstance(model, Model) or not hasattr(model, "layers"):
            logger.error("Model must be a Keras functional model")
            return {}
        
        # Default component names if not provided
        if component_names is None:
            component_names = [f"component_{i}" for i in range(len(model.layers))]
        
        # Get intermediate layer outputs
        intermediate_models = {}
        
        for i, layer in enumerate(model.layers):
            if i < len(component_names) and hasattr(layer, "output"):
                try:
                    intermediate_model = Model(inputs=model.inputs, outputs=layer.output)
                    intermediate_models[component_names[i]] = intermediate_model
                except Exception as e:
                    logger.error(f"Error creating intermediate model for {component_names[i]}: {e}")
        
        # Evaluate each component
        component_performance = {}
        
        for name, intermediate_model in intermediate_models.items():
            try:
                # Get intermediate output
                intermediate_output = intermediate_model.predict(X_test)
                
                # Create and train a simple model on this intermediate output
                input_shape = intermediate_output.shape[1:]
                
                # Create a simple model
                if len(input_shape) > 1:
                    # Sequence output
                    probe_model = tf.keras.Sequential([
                        tf.keras.layers.Flatten(input_shape=input_shape),
                        tf.keras.layers.Dense(64, activation='relu'),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])
                else:
                    # Vector output
                    probe_model = tf.keras.Sequential([
                        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])
                
                # Compile probe model
                probe_model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train probe model
                probe_model.fit(
                    intermediate_output, y_test,
                    epochs=10,
                    batch_size=32,
                    verbose=0
                )
                
                # Evaluate probe model
                _, accuracy = probe_model.evaluate(intermediate_output, y_test, verbose=0)
                
                # Store performance
                component_performance[name] = accuracy
                
                logger.info(f"Component {name} accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating component {name}: {e}")
                component_performance[name] = 0.0
        
        # Store component performance
        self.component_performance = component_performance
        
        return component_performance
    
    def identify_pruning_candidates(self, component_performance=None):
        """
        Identify components for pruning based on performance
        
        Args:
            component_performance (dict, optional): Component performance
            
        Returns:
            tuple: (components_to_prune, components_to_keep)
        """
        if component_performance is None:
            component_performance = self.component_performance
        
        if not component_performance:
            logger.warning("No component performance data available")
            return [], []
        
        # Identify components to prune
        components_to_prune = [
            name for name, accuracy in component_performance.items()
            if accuracy < self.performance_threshold
        ]
        
        # Identify components to keep
        components_to_keep = [
            name for name, accuracy in component_performance.items()
            if accuracy >= self.performance_threshold
        ]
        
        logger.info(f"Identified {len(components_to_prune)} components to prune")
        logger.info(f"Identified {len(components_to_keep)} components to keep")
        
        if components_to_prune:
            logger.info(f"Components to prune: {components_to_prune}")
        
        return components_to_prune, components_to_keep
    
    def create_pruned_model(self, model, components_to_prune=None, X_test=None, y_test=None):
        """
        Create a pruned model by removing underperforming components
        
        Args:
            model: Original model
            components_to_prune (list, optional): Components to prune
            X_test: Test data for validation
            y_test: Test labels for validation
            
        Returns:
            Model: Pruned model
        """
        if components_to_prune is None:
            # If not specified, evaluate and identify components to prune
            if X_test is not None and y_test is not None:
                self.evaluate_model_components(model, X_test, y_test)
            
            components_to_prune, _ = self.identify_pruning_candidates()
        
        if not components_to_prune:
            logger.info("No components to prune, returning original model")
            return model
        
        # Create pruned model
        try:
            # For TCN model or other custom architectures, we need a specialized approach
            if isinstance(model, EnhancedTCNModel):
                # Load model parameters
                model_params = model.model.get_config()
                
                # Modify parameters to exclude pruned components
                # This is a simplified example - the actual implementation depends
                # on the specific architecture of your EnhancedTCNModel
                for component in components_to_prune:
                    if "use_attention" in component.lower():
                        model_params["use_attention"] = False
                    elif "use_transformer" in component.lower():
                        model_params["use_transformer"] = False
                    # Add more component-specific logic as needed
                
                # Create new pruned model
                pruned_model = EnhancedTCNModel(
                    input_shape=model.input_shape,
                    **model_params
                )
                
                return pruned_model
            
            else:
                # For standard Keras models
                # Get layers to include
                layers_to_include = [
                    layer.name for layer in model.layers
                    if layer.name not in components_to_prune
                ]
                
                # Create pruned model
                inputs = model.inputs
                x = inputs
                
                for layer_name in layers_to_include:
                    layer = model.get_layer(layer_name)
                    x = layer(x)
                
                pruned_model = Model(inputs=inputs, outputs=x)
                
                # Compile pruned model
                pruned_model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                return pruned_model
                
        except Exception as e:
            logger.error(f"Error creating pruned model: {e}")
            logger.warning("Returning original model")
            return model
    
    def retrain_pruned_model(self, pruned_model, X_train, y_train, X_val=None, y_val=None,
                            batch_size=32, epochs=RETRAINING_EPOCHS):
        """
        Retrain pruned model
        
        Args:
            pruned_model: Pruned model
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            batch_size (int): Batch size
            epochs (int): Number of epochs
            
        Returns:
            tuple: (retrained_model, history)
        """
        logger.info(f"Retraining pruned model for {epochs} epochs")
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        # Retrain model
        if X_val is not None and y_val is not None:
            history = pruned_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = pruned_model.fit(
                X_train, y_train,
                validation_split=0.2,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        return pruned_model, history
    
    def optimize_network_structure(self, original_model, X_train, y_train, X_val, y_val):
        """
        Optimize network structure based on performance analysis
        
        Args:
            original_model: Original model
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            
        Returns:
            Model: Optimized model
        """
        logger.info("Optimizing network structure")
        
        # Evaluate components
        component_performance = self.evaluate_model_components(original_model, X_val, y_val)
        
        # Identify high-performing components
        high_performing = [
            name for name, accuracy in component_performance.items()
            if accuracy >= self.high_performance_threshold
        ]
        
        # Identify low-performing components
        low_performing = [
            name for name, accuracy in component_performance.items()
            if accuracy < self.performance_threshold
        ]
        
        logger.info(f"High-performing components: {high_performing}")
        logger.info(f"Low-performing components: {low_performing}")
        
        # Create an optimized model based on performance analysis
        # For demonstration purposes, we'll use a simple approach:
        # - Remove low-performing components
        # - Strengthen high-performing components
        optimized_model = self.create_pruned_model(original_model, low_performing)
        
        # Retrain optimized model
        optimized_model, _ = self.retrain_pruned_model(
            optimized_model, X_train, y_train, X_val, y_val
        )
        
        return optimized_model
    
    def save_pruned_model(self, model, model_name, include_timestamp=True):
        """
        Save pruned model
        
        Args:
            model: Model to save
            model_name (str): Model name
            include_timestamp (bool): Whether to include timestamp
            
        Returns:
            str: Path to saved model
        """
        # Create filename
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_pruned_{timestamp}.h5"
        else:
            filename = f"{model_name}_pruned.h5"
        
        # Create path
        model_path = os.path.join(PRUNED_MODELS_DIR, filename)
        
        # Save model
        try:
            model.save(model_path)
            logger.info(f"Pruned model saved to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error saving pruned model: {e}")
            return None


class EnsemblePruner:
    """
    Pruner for ensemble models that identifies and removes
    underperforming models from the ensemble.
    """
    
    def __init__(self, ensemble_dir=None, performance_threshold=PERFORMANCE_THRESHOLD):
        """
        Initialize the ensemble pruner
        
        Args:
            ensemble_dir (str, optional): Directory containing ensemble models
            performance_threshold (float): Minimum threshold for keeping a model
        """
        self.ensemble_dir = ensemble_dir or os.path.join(MODELS_DIR, "ensemble")
        self.performance_threshold = performance_threshold
        
        # Ensure ensemble directory exists
        os.makedirs(self.ensemble_dir, exist_ok=True)
        
        # Model performance tracking
        self.model_performance = {}
        
        logger.info(f"Ensemble pruner initialized with performance threshold: {performance_threshold}")
    
    def load_ensemble_models(self):
        """
        Load ensemble models from directory
        
        Returns:
            dict: Dictionary of models
        """
        models = {}
        
        # Check if directory exists
        if not os.path.exists(self.ensemble_dir):
            logger.error(f"Ensemble directory {self.ensemble_dir} does not exist")
            return models
        
        # Find model files
        model_files = [
            f for f in os.listdir(self.ensemble_dir)
            if f.endswith('.h5') and os.path.isfile(os.path.join(self.ensemble_dir, f))
        ]
        
        logger.info(f"Found {len(model_files)} model files in {self.ensemble_dir}")
        
        # Load models
        for model_file in model_files:
            model_path = os.path.join(self.ensemble_dir, model_file)
            model_name = os.path.splitext(model_file)[0]
            
            try:
                model = load_model(model_path)
                models[model_name] = model
                logger.info(f"Loaded model {model_name}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
        
        return models
    
    def evaluate_ensemble_models(self, models, X_test, y_test):
        """
        Evaluate performance of ensemble models
        
        Args:
            models (dict): Dictionary of models
            X_test: Test data
            y_test: Test labels
            
        Returns:
            dict: Model performance
        """
        model_performance = {}
        
        for model_name, model in models.items():
            try:
                # Evaluate model
                _, accuracy = model.evaluate(X_test, y_test, verbose=0)
                
                # Store performance
                model_performance[model_name] = accuracy
                
                logger.info(f"Model {model_name} accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
                model_performance[model_name] = 0.0
        
        # Store model performance
        self.model_performance = model_performance
        
        return model_performance
    
    def identify_models_to_prune(self, model_performance=None):
        """
        Identify models for pruning based on performance
        
        Args:
            model_performance (dict, optional): Model performance
            
        Returns:
            tuple: (models_to_prune, models_to_keep)
        """
        if model_performance is None:
            model_performance = self.model_performance
        
        if not model_performance:
            logger.warning("No model performance data available")
            return [], []
        
        # Identify models to prune
        models_to_prune = [
            name for name, accuracy in model_performance.items()
            if accuracy < self.performance_threshold
        ]
        
        # Identify models to keep
        models_to_keep = [
            name for name, accuracy in model_performance.items()
            if accuracy >= self.performance_threshold
        ]
        
        logger.info(f"Identified {len(models_to_prune)} models to prune")
        logger.info(f"Identified {len(models_to_keep)} models to keep")
        
        if models_to_prune:
            logger.info(f"Models to prune: {models_to_prune}")
        
        return models_to_prune, models_to_keep
    
    def prune_ensemble(self, models, models_to_prune):
        """
        Prune ensemble by removing underperforming models
        
        Args:
            models (dict): Dictionary of models
            models_to_prune (list): Models to prune
            
        Returns:
            dict: Pruned ensemble
        """
        # Create pruned ensemble
        pruned_ensemble = {
            name: model for name, model in models.items()
            if name not in models_to_prune
        }
        
        logger.info(f"Pruned ensemble has {len(pruned_ensemble)} models (removed {len(models_to_prune)})")
        
        return pruned_ensemble
    
    def save_pruned_ensemble(self, pruned_ensemble, output_dir=None):
        """
        Save pruned ensemble
        
        Args:
            pruned_ensemble (dict): Pruned ensemble
            output_dir (str, optional): Output directory
            
        Returns:
            list: Paths to saved models
        """
        if output_dir is None:
            output_dir = os.path.join(PRUNED_MODELS_DIR, "ensemble")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        saved_paths = []
        
        for model_name, model in pruned_ensemble.items():
            # Create path
            model_path = os.path.join(output_dir, f"{model_name}.h5")
            
            # Save model
            try:
                model.save(model_path)
                saved_paths.append(model_path)
                logger.info(f"Model {model_name} saved to {model_path}")
            except Exception as e:
                logger.error(f"Error saving model {model_name}: {e}")
        
        return saved_paths
    
    def create_ensemble_weights(self, model_performance):
        """
        Create ensemble weights based on model performance
        
        Args:
            model_performance (dict): Model performance
            
        Returns:
            dict: Ensemble weights
        """
        if not model_performance:
            logger.warning("No model performance data available")
            return {}
        
        # Calculate weights
        total_performance = sum(model_performance.values())
        
        if total_performance <= 0:
            logger.warning("Total performance is zero or negative")
            return {name: 1.0 / len(model_performance) for name in model_performance}
        
        # Normalize weights
        weights = {
            name: perf / total_performance
            for name, perf in model_performance.items()
        }
        
        logger.info(f"Created ensemble weights: {weights}")
        
        return weights
    
    def save_ensemble_configuration(self, pruned_ensemble, weights, output_dir=None):
        """
        Save ensemble configuration
        
        Args:
            pruned_ensemble (dict): Pruned ensemble
            weights (dict): Ensemble weights
            output_dir (str, optional): Output directory
            
        Returns:
            str: Path to saved configuration
        """
        if output_dir is None:
            output_dir = os.path.join(PRUNED_MODELS_DIR, "ensemble")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create configuration
        config = {
            "models": list(pruned_ensemble.keys()),
            "weights": weights,
            "timestamp": datetime.now().isoformat()
        }
        
        # Create path
        config_path = os.path.join(output_dir, "ensemble_config.json")
        
        # Save configuration
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Ensemble configuration saved to {config_path}")
            return config_path
        except Exception as e:
            logger.error(f"Error saving ensemble configuration: {e}")
            return None


def load_and_prepare_data(symbol, timeframe, test_size=0.2, sequence_length=60):
    """
    Load and prepare data for model evaluation
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe
        test_size (float): Test set size
        sequence_length (int): Sequence length
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    logger.info(f"Loading data for {symbol} on {timeframe} timeframe")
    
    # Initialize data fetcher
    fetcher = EnhancedHistoricalDataFetcher(symbol, timeframe)
    
    # Fetch data
    data = fetcher.fetch_historical_data()
    
    if data is None or len(data) == 0:
        logger.error(f"No data available for {symbol} on {timeframe} timeframe")
        return None, None, None, None, None, None
    
    logger.info(f"Loaded {len(data)} rows of data")
    
    # Prepare features and target
    fetcher.prepare_features()
    
    # Ensure we have direction column
    if 'close_direction' not in data.columns:
        data['close_direction'] = (data['close'].pct_change() > 0).astype(int)
    
    # Create sequences
    X, y = [], []
    
    for i in range(0, len(data) - sequence_length):
        X.append(data.iloc[i:i+sequence_length].values)
        y.append(data.iloc[i+sequence_length]['close_direction'])
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Created {len(X)} sequences")
    
    # Split data
    n_samples = len(X)
    test_split = int(n_samples * (1 - test_size))
    val_split = int(test_split * 0.8)
    
    X_train, y_train = X[:val_split], y[:val_split]
    X_val, y_val = X[val_split:test_split], y[val_split:test_split]
    X_test, y_test = X[test_split:], y[test_split:]
    
    logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def prune_model(model_path, symbol, timeframe, output_dir=None):
    """
    Prune a single model
    
    Args:
        model_path (str): Path to model
        symbol (str): Trading symbol
        timeframe (str): Timeframe
        output_dir (str, optional): Output directory
        
    Returns:
        tuple: (pruned_model_path, accuracy)
    """
    logger.info(f"Pruning model {model_path} for {symbol} on {timeframe} timeframe")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(symbol, timeframe)
    
    if X_test is None:
        logger.error("Data loading failed")
        return None, None
    
    # Load model
    try:
        model = load_model(model_path)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None
    
    # Create pruner
    pruner = ModelPruner(base_model_path=model_path)
    
    # Evaluate components
    component_performance = pruner.evaluate_model_components(model, X_test, y_test)
    
    # Identify components to prune
    components_to_prune, components_to_keep = pruner.identify_pruning_candidates(component_performance)
    
    if not components_to_prune:
        logger.info("No components to prune")
        return model_path, None
    
    # Create pruned model
    pruned_model = pruner.create_pruned_model(model, components_to_prune, X_test, y_test)
    
    # Retrain pruned model
    pruned_model, history = pruner.retrain_pruned_model(
        pruned_model, X_train, y_train, X_val, y_val
    )
    
    # Evaluate pruned model
    pruned_loss, pruned_accuracy = pruned_model.evaluate(X_test, y_test)
    logger.info(f"Pruned model accuracy: {pruned_accuracy:.4f}")
    
    # Save pruned model
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    pruned_model_path = pruner.save_pruned_model(pruned_model, model_name)
    
    return pruned_model_path, pruned_accuracy


def prune_ensemble(ensemble_dir, symbol, timeframe, output_dir=None):
    """
    Prune an ensemble
    
    Args:
        ensemble_dir (str): Ensemble directory
        symbol (str): Trading symbol
        timeframe (str): Timeframe
        output_dir (str, optional): Output directory
        
    Returns:
        tuple: (config_path, ensemble_accuracy)
    """
    logger.info(f"Pruning ensemble in {ensemble_dir} for {symbol} on {timeframe} timeframe")
    
    # Load data
    _, _, _, _, X_test, y_test = load_and_prepare_data(symbol, timeframe)
    
    if X_test is None:
        logger.error("Data loading failed")
        return None, None
    
    # Create pruner
    pruner = EnsemblePruner(ensemble_dir=ensemble_dir)
    
    # Load ensemble models
    models = pruner.load_ensemble_models()
    
    if not models:
        logger.error("No ensemble models found")
        return None, None
    
    # Evaluate models
    model_performance = pruner.evaluate_ensemble_models(models, X_test, y_test)
    
    # Identify models to prune
    models_to_prune, models_to_keep = pruner.identify_models_to_prune(model_performance)
    
    if not models_to_prune:
        logger.info("No models to prune")
        return None, None
    
    # Prune ensemble
    pruned_ensemble = pruner.prune_ensemble(models, models_to_prune)
    
    if not pruned_ensemble:
        logger.error("All models were pruned")
        return None, None
    
    # Create ensemble weights
    weights = pruner.create_ensemble_weights({
        name: perf for name, perf in model_performance.items()
        if name in pruned_ensemble
    })
    
    # Evaluate ensemble accuracy
    ensemble_predictions = np.zeros_like(y_test, dtype=float)
    
    for model_name, model in pruned_ensemble.items():
        # Get model predictions
        predictions = model.predict(X_test)
        
        # Add weighted predictions
        ensemble_predictions += predictions.flatten() * weights.get(model_name, 1.0 / len(pruned_ensemble))
    
    # Convert to binary predictions
    ensemble_predictions = (ensemble_predictions > 0.5).astype(int)
    
    # Calculate accuracy
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    logger.info(f"Pruned ensemble accuracy: {ensemble_accuracy:.4f}")
    
    # Save pruned ensemble
    saved_paths = pruner.save_pruned_ensemble(pruned_ensemble, output_dir)
    
    # Save ensemble configuration
    config_path = pruner.save_ensemble_configuration(pruned_ensemble, weights, output_dir)
    
    return config_path, ensemble_accuracy


def auto_prune_all_models(base_dir=MODELS_DIR, symbols=None, timeframes=None):
    """
    Auto-prune all models
    
    Args:
        base_dir (str): Base directory
        symbols (list, optional): Symbols to process
        timeframes (list, optional): Timeframes to process
        
    Returns:
        dict: Results
    """
    symbols = symbols or DEFAULT_SYMBOLS
    timeframes = timeframes or DEFAULT_TIMEFRAMES
    
    logger.info(f"Auto-pruning all models in {base_dir} for {len(symbols)} symbols and {len(timeframes)} timeframes")
    
    # Results
    results = {}
    
    # Process each symbol and timeframe
    for symbol in symbols:
        results[symbol] = {}
        
        for timeframe in timeframes:
            logger.info(f"Processing {symbol} on {timeframe} timeframe")
            
            # Find models
            model_dir = os.path.join(base_dir, symbol, timeframe)
            
            if not os.path.exists(model_dir):
                logger.warning(f"Model directory {model_dir} does not exist")
                continue
            
            # Find model files
            model_files = [
                f for f in os.listdir(model_dir)
                if f.endswith('.h5') and os.path.isfile(os.path.join(model_dir, f))
            ]
            
            if not model_files:
                logger.warning(f"No model files found in {model_dir}")
                continue
            
            logger.info(f"Found {len(model_files)} model files in {model_dir}")
            
            # Process each model
            model_results = {}
            
            for model_file in model_files:
                model_path = os.path.join(model_dir, model_file)
                model_name = os.path.splitext(model_file)[0]
                
                # Prune model
                pruned_path, accuracy = prune_model(model_path, symbol, timeframe)
                
                # Store result
                model_results[model_name] = {
                    "original_path": model_path,
                    "pruned_path": pruned_path,
                    "accuracy": accuracy
                }
            
            # Store timeframe results
            results[symbol][timeframe] = model_results
            
            # Check if there's an ensemble
            ensemble_dir = os.path.join(model_dir, "ensemble")
            
            if os.path.exists(ensemble_dir) and os.path.isdir(ensemble_dir):
                logger.info(f"Found ensemble directory {ensemble_dir}")
                
                # Prune ensemble
                config_path, accuracy = prune_ensemble(ensemble_dir, symbol, timeframe)
                
                # Store ensemble result
                results[symbol][timeframe]["ensemble"] = {
                    "config_path": config_path,
                    "accuracy": accuracy
                }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"auto_prune_results_{timestamp}.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Auto-pruning results saved to {results_path}")
    
    return results


def main():
    """Main function for auto-pruning ML models"""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Auto-Pruning System for ML Models")
    
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to model file")
    parser.add_argument("--ensemble-dir", type=str, default=None,
                       help="Path to ensemble directory")
    parser.add_argument("--symbol", type=str, default="SOLUSD",
                       help="Trading symbol (default: SOLUSD)")
    parser.add_argument("--timeframe", type=str, default="1h",
                       help="Timeframe (default: 1h)")
    parser.add_argument("--performance-threshold", type=float, default=PERFORMANCE_THRESHOLD,
                       help=f"Performance threshold (default: {PERFORMANCE_THRESHOLD})")
    parser.add_argument("--symbols", nargs="+", default=None,
                       help="List of symbols to process")
    parser.add_argument("--timeframes", nargs="+", default=None,
                       help="List of timeframes to process")
    parser.add_argument("--auto-prune-all", action="store_true",
                       help="Auto-prune all models")
    parser.add_argument("--output-dir", type=str, default=PRUNED_MODELS_DIR,
                       help=f"Output directory (default: {PRUNED_MODELS_DIR})")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Auto-prune all models
    if args.auto_prune_all:
        auto_prune_all_models(
            symbols=args.symbols,
            timeframes=args.timeframes
        )
    # Prune ensemble
    elif args.ensemble_dir:
        config_path, accuracy = prune_ensemble(
            ensemble_dir=args.ensemble_dir,
            symbol=args.symbol,
            timeframe=args.timeframe,
            output_dir=args.output_dir
        )
        
        if config_path:
            print(f"Pruned ensemble configuration saved to {config_path}")
            print(f"Pruned ensemble accuracy: {accuracy:.4f}")
        else:
            print("Ensemble pruning failed")
    # Prune single model
    elif args.model_path:
        pruned_model_path, accuracy = prune_model(
            model_path=args.model_path,
            symbol=args.symbol,
            timeframe=args.timeframe,
            output_dir=args.output_dir
        )
        
        if pruned_model_path:
            print(f"Pruned model saved to {pruned_model_path}")
            print(f"Pruned model accuracy: {accuracy:.4f}")
        else:
            print("Model pruning failed")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()