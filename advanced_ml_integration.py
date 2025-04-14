#!/usr/bin/env python3
"""
Advanced ML Integration for Kraken Trading Bot

This module integrates all the advanced machine learning components:
1. Temporal Fusion Transformer models
2. Multi-Asset Feature Fusion
3. Adaptive Hyperparameter Tuning
4. Explainable AI
5. Sentiment Analysis

Together, these components aim to achieve 90% prediction accuracy and 1000%+ returns.
"""

import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import custom modules
import temporal_fusion_transformer as tft
import multi_asset_feature_fusion as maff
import adaptive_hyperparameter_tuning as aht
import explainable_ai_integration as xai
import sentiment_analysis_integration as sai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = "advanced_ml_models"
os.makedirs(MODELS_DIR, exist_ok=True)
TRAINING_RESULTS_DIR = "training_results"
os.makedirs(TRAINING_RESULTS_DIR, exist_ok=True)
PREDICTION_LOG_DIR = "prediction_logs"
os.makedirs(PREDICTION_LOG_DIR, exist_ok=True)

# Default multi-asset feature fusion settings
DEFAULT_RELATED_ASSETS = {
    "SOL/USD": ["ETH/USD", "BTC/USD", "DOT/USD", "LINK/USD"],
    "ETH/USD": ["BTC/USD", "SOL/USD", "LINK/USD", "DOT/USD"],
    "BTC/USD": ["ETH/USD", "SOL/USD", "LINK/USD", "DOT/USD"],
    "DOT/USD": ["SOL/USD", "ETH/USD", "BTC/USD", "LINK/USD"],
    "LINK/USD": ["ETH/USD", "SOL/USD", "BTC/USD", "DOT/USD"]
}

# Default trading pairs
DEFAULT_TRADING_PAIRS = ["SOL/USD", "ETH/USD", "BTC/USD", "DOT/USD", "LINK/USD"]

# Target performance metrics
TARGET_ACCURACY = 0.90  # 90% prediction accuracy
TARGET_RETURN = 10.0    # 1000% return

class AdvancedMLIntegration:
    """
    Integrates all advanced ML components to enhance trading performance.
    """
    def __init__(self, config_path=None):
        """
        Initialize the advanced ML integration
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize component managers
        self.hyperparameter_manager = aht.hyperparameter_manager
        self.explainable_ai_manager = xai.explainable_ai_manager
        self.market_sentiment_integrator = sai.market_sentiment_integrator
        
        # Initialize model containers
        self.tft_models = {}
        self.asset_fusion_modules = {}
        self.performance_metrics = {}
        
        # Initialize related assets mapping
        self.related_assets = self.config.get('related_assets', DEFAULT_RELATED_ASSETS)
        
        # Initialize trading pairs
        self.trading_pairs = self.config.get('trading_pairs', DEFAULT_TRADING_PAIRS)
        
        logger.info("Initialized Advanced ML Integration")
    
    def _load_config(self, config_path):
        """
        Load configuration from file or use defaults
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            dict: Configuration settings
        """
        default_config = {
            'trading_pairs': DEFAULT_TRADING_PAIRS,
            'related_assets': DEFAULT_RELATED_ASSETS,
            'model_settings': {
                'tft': {
                    'hidden_units': 128,
                    'num_heads': 4,
                    'dropout_rate': 0.1,
                    'forecast_horizon': 12
                }
            },
            'training_settings': {
                'batch_size': 64,
                'epochs': 200,
                'learning_rate': 0.001,
                'early_stopping_patience': 20,
                'validation_split': 0.2
            },
            'feature_fusion_settings': {
                'lookback_periods': 90,
                'correlation_threshold': 0.6,
                'use_pca': True,
                'pca_components': 5
            },
            'hyperparameter_tuning': {
                'enabled': True,
                'evaluation_interval': 24,  # hours
                'max_trials': 50
            },
            'sentiment_analysis': {
                'enabled': True,
                'refresh_interval': 12,  # hours
                'weight': 0.2  # 20% sentiment, 80% ML model
            },
            'explainable_ai': {
                'enabled': True,
                'save_explanations': True,
                'explanation_frequency': 24  # hours
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults (keeping user settings where provided)
                merged_config = default_config.copy()
                
                # Update nested dictionaries
                for key, value in loaded_config.items():
                    if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                        merged_config[key].update(value)
                    else:
                        merged_config[key] = value
                
                logger.info(f"Loaded configuration from {config_path}")
                return merged_config
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                return default_config
        else:
            logger.info("Using default configuration")
            return default_config
    
    def initialize_models(self):
        """
        Initialize all advanced ML models for all trading pairs
        
        Returns:
            bool: Success of initialization
        """
        success = True
        
        for asset in self.trading_pairs:
            logger.info(f"Initializing models for {asset}")
            
            try:
                # Initialize TFT model
                self._initialize_tft_model(asset)
                
                # Initialize feature fusion module
                self._initialize_feature_fusion(asset)
                
                # Initialize performance tracking
                self.performance_metrics[asset] = {
                    'accuracy': [],
                    'precision': [],
                    'recall': [],
                    'f1_score': [],
                    'returns': [],
                    'drawdowns': [],
                    'sharpe_ratio': []
                }
                
                logger.info(f"Successfully initialized models for {asset}")
            except Exception as e:
                logger.error(f"Error initializing models for {asset}: {e}")
                success = False
        
        return success
    
    def _initialize_tft_model(self, asset):
        """
        Initialize Temporal Fusion Transformer model for an asset
        
        Args:
            asset (str): Asset to initialize model for
        """
        # Get model settings
        model_settings = self.config.get('model_settings', {}).get('tft', {})
        
        # Try to load existing model if available
        model_path = os.path.join(MODELS_DIR, asset.replace('/', '_'), f"{asset.replace('/', '_')}_tft.h5")
        if os.path.exists(model_path):
            try:
                # Load existing model
                model = tft.load_tft_model(asset.replace('/', '_'), MODELS_DIR)
                self.tft_models[asset] = model
                logger.info(f"Loaded existing TFT model for {asset}")
                return
            except Exception as e:
                logger.error(f"Error loading existing TFT model for {asset}: {e}")
                # Continue to create new model
        
        # Create new TFT model
        try:
            # Default features: close, open, high, low, volume + technical indicators (estimated ~30 features)
            num_features = 30
            
            # Get hyperparameters from tuner if available
            params = {}
            tuner = self.hyperparameter_manager.get_tuner('tft', asset)
            
            if tuner and tuner.best_params:
                # Use tuned parameters
                params = tuner.get_optimal_parameters().get('params', {})
                logger.info(f"Using tuned hyperparameters for {asset} TFT model")
            else:
                # Use default parameters
                params = {
                    'hidden_units': model_settings.get('hidden_units', 128),
                    'num_heads': model_settings.get('num_heads', 4),
                    'dropout_rate': model_settings.get('dropout_rate', 0.1),
                    'forecast_horizon': model_settings.get('forecast_horizon', 12)
                }
                logger.info(f"Using default hyperparameters for {asset} TFT model")
            
            # Create model with asymmetric loss
            model = tft.create_tft_model_with_asymmetric_loss(
                num_features=num_features,
                hidden_units=params.get('hidden_units', 128),
                num_heads=params.get('num_heads', 4),
                dropout_rate=params.get('dropout_rate', 0.1),
                forecast_horizon=params.get('forecast_horizon', 12),
                loss_ratio=params.get('loss_ratio', 2.0)
            )
            
            self.tft_models[asset] = model
            logger.info(f"Created new TFT model for {asset}")
        except Exception as e:
            logger.error(f"Error creating TFT model for {asset}: {e}")
            raise
    
    def _initialize_feature_fusion(self, asset):
        """
        Initialize Multi-Asset Feature Fusion for an asset
        
        Args:
            asset (str): Asset to initialize fusion for
        """
        # Get related assets for this asset
        related_assets = self.related_assets.get(asset, [])
        
        # Get feature fusion settings
        fusion_settings = self.config.get('feature_fusion_settings', {})
        
        # Create feature fusion module
        try:
            fusion_module = maff.MultiAssetFeatureFusion(
                base_asset=asset,
                related_assets=related_assets,
                lookback_periods=fusion_settings.get('lookback_periods', 90)
            )
            
            self.asset_fusion_modules[asset] = fusion_module
            logger.info(f"Initialized feature fusion for {asset} with related assets: {related_assets}")
        except Exception as e:
            logger.error(f"Error initializing feature fusion for {asset}: {e}")
            raise
    
    def train_models(self, data_dict, force_retrain=False, save_models=True):
        """
        Train all models with the provided data
        
        Args:
            data_dict (dict): Dictionary of DataFrames with asset data
            force_retrain (bool): Whether to force retraining even if models exist
            save_models (bool): Whether to save the trained models
            
        Returns:
            dict: Training results
        """
        training_results = {}
        
        for asset in self.trading_pairs:
            if asset not in data_dict:
                logger.warning(f"No data provided for {asset}, skipping training")
                continue
            
            logger.info(f"Training models for {asset}")
            
            try:
                # Prepare data for training
                X, y, feature_names = self._prepare_training_data(asset, data_dict)
                
                # Check if we should tune hyperparameters
                should_tune = self.config.get('hyperparameter_tuning', {}).get('enabled', True)
                tuner = self.hyperparameter_manager.get_tuner('tft', asset)
                
                if should_tune and (force_retrain or tuner.should_tune_parameters()):
                    logger.info(f"Tuning hyperparameters for {asset}")
                    
                    # Define model creation function for tuning
                    def create_model_fn(**params):
                        return tft.create_tft_model_with_asymmetric_loss(
                            num_features=X.shape[2],
                            hidden_units=params.get('hidden_units', 128),
                            num_heads=params.get('num_heads', 4),
                            dropout_rate=params.get('dropout_rate', 0.1),
                            forecast_horizon=params.get('forecast_horizon', 12),
                            loss_ratio=params.get('loss_ratio', 2.0)
                        )
                    
                    # Tune hyperparameters
                    best_params = tuner.tune_hyperparameters(
                        X, y, create_model_fn,
                        n_trials=self.config.get('hyperparameter_tuning', {}).get('max_trials', 50)
                    )
                    
                    # Create model with tuned parameters
                    model = create_model_fn(**best_params)
                    self.tft_models[asset] = model
                    
                    logger.info(f"Created TFT model with tuned parameters for {asset}")
                
                # Train the model
                model = self.tft_models[asset]
                
                # Get training settings
                training_settings = self.config.get('training_settings', {})
                
                # Train model
                train_X, val_X, train_y, val_y = self._split_train_validation(
                    X, y, validation_split=training_settings.get('validation_split', 0.2)
                )
                
                # Train with early stopping
                model, history = tft.train_tft_model(
                    model,
                    train_X, train_y,
                    X_val=val_X, y_val=val_y,
                    batch_size=training_settings.get('batch_size', 64),
                    epochs=training_settings.get('epochs', 200),
                    model_path=None  # Will save later if needed
                )
                
                # Save model if needed
                if save_models:
                    model_path = tft.save_tft_model(model, asset.replace('/', '_'), MODELS_DIR)
                    logger.info(f"Saved TFT model for {asset} to {model_path}")
                
                # Evaluate model
                val_predictions = model.predict(val_X)
                
                # Extract predictions based on model output format
                if isinstance(val_predictions, dict):
                    val_preds = val_predictions['predictions'][:, -1, 0]  # Last timestep, first output feature
                else:
                    val_preds = val_predictions
                
                # Calculate metrics
                mse = np.mean((val_preds - val_y) ** 2)
                mae = np.mean(np.abs(val_preds - val_y))
                
                # Direction accuracy (positive/negative prediction matches actual direction)
                direction_correct = np.sum((val_preds > 0) == (val_y > 0)) / len(val_y)
                
                # Save training results
                training_results[asset] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'direction_accuracy': float(direction_correct),
                    'epochs_trained': len(history.history['loss']),
                    'final_loss': float(history.history['loss'][-1]),
                    'final_val_loss': float(history.history['val_loss'][-1]),
                    'trained_at': datetime.now().isoformat()
                }
                
                logger.info(f"Trained model for {asset} - Direction accuracy: {direction_correct:.4f}")
                
                # Initialize feature fusion with the data
                if asset in self.asset_fusion_modules:
                    logger.info(f"Analyzing cross-asset relationships for {asset}")
                    
                    # Filter data_dict to only include assets we're interested in
                    fusion_data_dict = {
                        k: v for k, v in data_dict.items() 
                        if k == asset or k in self.related_assets.get(asset, [])
                    }
                    
                    # Analyze relationships between assets
                    self.asset_fusion_modules[asset].analyze_relationships(fusion_data_dict)
                    logger.info(f"Completed cross-asset analysis for {asset}")
                
                # Prepare explainable AI
                if self.config.get('explainable_ai', {}).get('enabled', True):
                    # Create or get explainer
                    explainer = self.explainable_ai_manager.get_explainer('tft', asset, model)
                    
                    # Initialize SHAP explainer with background data
                    sample_data = X[:min(10, len(X))]
                    explainer.initialize_shap_explainer(sample_data)
                    logger.info(f"Initialized explainable AI for {asset}")
            
            except Exception as e:
                logger.error(f"Error training models for {asset}: {e}")
                training_results[asset] = {
                    'error': str(e),
                    'success': False
                }
        
        # Save training results
        results_path = os.path.join(TRAINING_RESULTS_DIR, f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"Saved training results to {results_path}")
        
        return training_results
    
    def _prepare_training_data(self, asset, data_dict):
        """
        Prepare data for training models
        
        Args:
            asset (str): Asset to prepare data for
            data_dict (dict): Dictionary of DataFrames with asset data
            
        Returns:
            tuple: (X, y, feature_names)
        """
        # Get data for the asset
        df = data_dict[asset]
        
        # Check if we have feature fusion enabled
        use_fusion = asset in self.asset_fusion_modules
        
        # Extract features and target
        feature_names = []
        
        # Price and volume features
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_names.extend(price_cols)
        
        # Technical indicators (if available)
        tech_indicators = [col for col in df.columns if col not in price_cols and col != 'timestamp']
        feature_names.extend(tech_indicators)
        
        # If feature fusion is enabled, add cross-asset features
        if use_fusion:
            try:
                # Get related assets data
                related_assets = self.related_assets.get(asset, [])
                fusion_data_dict = {
                    k: v for k, v in data_dict.items() 
                    if k == asset or k in related_assets
                }
                
                # Generate cross-asset features
                fusion_module = self.asset_fusion_modules[asset]
                cross_asset_features = fusion_module.create_cross_asset_features_for_model(fusion_data_dict)
                
                # Merge with main features
                if not cross_asset_features.empty:
                    # Align timestamps
                    if 'timestamp' in df.columns and 'timestamp' in cross_asset_features.columns:
                        df = df.set_index('timestamp')
                        cross_asset_features = cross_asset_features.set_index('timestamp')
                        
                        # Join features
                        combined_df = df.join(cross_asset_features, how='inner')
                        
                        # Reset index
                        combined_df = combined_df.reset_index()
                        
                        # Update DataFrame
                        df = combined_df
                        
                        # Add cross-asset feature names
                        fusion_feature_names = [col for col in cross_asset_features.columns if col != 'timestamp']
                        feature_names.extend(fusion_feature_names)
                    
                    logger.info(f"Added {len(cross_asset_features.columns) - 1} cross-asset features for {asset}")
            except Exception as e:
                logger.error(f"Error adding cross-asset features for {asset}: {e}")
        
        # Extract features
        features = df[feature_names].values
        
        # Create target: future price change
        # For simplicity, we'll use next day's close price change as target
        target_horizon = 1  # 1-day ahead
        df['target'] = df['close'].shift(-target_horizon) - df['close']
        
        # Remove rows with NaN targets
        valid_indices = ~df['target'].isna()
        features = features[valid_indices]
        targets = df['target'].values[valid_indices]
        
        # Reshape for sequence prediction
        sequence_length = self.config.get('model_settings', {}).get('tft', {}).get('sequence_length', 60)
        
        X = []
        y = []
        
        for i in range(len(features) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(targets[i+sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Prepared training data for {asset}: {X.shape}, {y.shape}")
        
        return X, y, feature_names
    
    def _split_train_validation(self, X, y, validation_split=0.2):
        """
        Split data into training and validation sets
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Targets
            validation_split (float): Proportion of data to use for validation
            
        Returns:
            tuple: (train_X, val_X, train_y, val_y)
        """
        # Calculate split index (use time-based split for time series)
        split_idx = int(len(X) * (1 - validation_split))
        
        train_X = X[:split_idx]
        val_X = X[split_idx:]
        train_y = y[:split_idx]
        val_y = y[split_idx:]
        
        return train_X, val_X, train_y, val_y
    
    def predict(self, data_dict, include_explanations=True):
        """
        Make predictions for all assets using the trained models
        
        Args:
            data_dict (dict): Dictionary of DataFrames with asset data
            include_explanations (bool): Whether to include explanations with predictions
            
        Returns:
            dict: Predictions for each asset
        """
        predictions = {}
        
        for asset in self.trading_pairs:
            if asset not in data_dict:
                logger.warning(f"No data provided for {asset}, skipping prediction")
                continue
            
            if asset not in self.tft_models:
                logger.warning(f"No trained model for {asset}, skipping prediction")
                continue
            
            logger.info(f"Making predictions for {asset}")
            
            try:
                # Prepare input data
                X, feature_names = self._prepare_prediction_data(asset, data_dict)
                
                # Get model
                model = self.tft_models[asset]
                
                # Make prediction
                raw_prediction = model.predict(X)
                
                # Extract prediction value
                if isinstance(raw_prediction, dict):
                    prediction_value = raw_prediction['predictions'][0, -1, 0]  # First batch, last timestep, first output
                else:
                    prediction_value = raw_prediction[0, 0] if raw_prediction.ndim > 1 else raw_prediction[0]
                
                # Determine trading signal
                signal = 'BUY' if prediction_value > 0 else 'SELL' if prediction_value < 0 else 'NEUTRAL'
                confidence = min(1.0, abs(prediction_value) * 2)  # Scale to 0-1 range
                
                # Incorporate sentiment if enabled
                sentiment_adjustment = None
                adjusted_prediction = prediction_value
                
                if self.config.get('sentiment_analysis', {}).get('enabled', True):
                    # Get sentiment adjustment
                    sentiment_adjustment = self.market_sentiment_integrator.get_trading_sentiment_adjustment(
                        asset,
                        sensitivity=1.0,
                        refresh_cache=False
                    )
                    
                    # Apply sentiment adjustment
                    sentiment_weight = self.config.get('sentiment_analysis', {}).get('weight', 0.2)
                    adjusted_prediction = self.market_sentiment_integrator.apply_sentiment_to_model_prediction(
                        prediction_value,
                        sentiment_adjustment,
                        weight=sentiment_weight
                    )
                    
                    # Update signal based on adjusted prediction
                    adjusted_signal = 'BUY' if adjusted_prediction > 0 else 'SELL' if adjusted_prediction < 0 else 'NEUTRAL'
                    adjusted_confidence = min(1.0, abs(adjusted_prediction) * 2)
                    
                    logger.info(f"{asset} raw signal: {signal} ({confidence:.2f}), adjusted: {adjusted_signal} ({adjusted_confidence:.2f})")
                
                # Prepare prediction result
                prediction_result = {
                    'asset': asset,
                    'timestamp': datetime.now().isoformat(),
                    'raw_prediction': float(prediction_value),
                    'adjusted_prediction': float(adjusted_prediction),
                    'signal': signal,
                    'confidence': float(confidence),
                    'adjusted_signal': adjusted_signal if 'adjusted_signal' in locals() else signal,
                    'adjusted_confidence': float(adjusted_confidence) if 'adjusted_confidence' in locals() else float(confidence)
                }
                
                # Add sentiment information if available
                if sentiment_adjustment:
                    prediction_result['sentiment'] = {
                        'signal': sentiment_adjustment.get('signal'),
                        'strength': sentiment_adjustment.get('strength'),
                        'confidence': sentiment_adjustment.get('confidence'),
                        'classification': sentiment_adjustment.get('sentiment_classification')
                    }
                
                # Add explanation if requested
                if include_explanations and self.config.get('explainable_ai', {}).get('enabled', True):
                    try:
                        # Get explainer
                        explainer = self.explainable_ai_manager.get_explainer('tft', asset, model)
                        
                        # Generate explanation
                        explanation = explainer.explain_prediction(X, feature_names=feature_names)
                        
                        # Add to prediction result
                        prediction_result['explanation'] = {
                            'feature_importance': explanation.get('feature_importance', {}),
                            'explanation_text': explainer.generate_explanation_text(explanation)
                        }
                        
                        # Generate comparison with baseline if we have history
                        comparison = explainer.compare_with_baseline(explanation)
                        if comparison:
                            prediction_result['explanation']['comparison'] = {
                                'feature_deviations': comparison.get('feature_deviations', {}),
                                'comparison_text': explainer.generate_comparison_text(comparison)
                            }
                        
                        logger.info(f"Generated explanation for {asset} prediction")
                    except Exception as e:
                        logger.error(f"Error generating explanation for {asset}: {e}")
                
                # Store prediction
                predictions[asset] = prediction_result
                
                # Log prediction
                log_path = os.path.join(PREDICTION_LOG_DIR, f"{asset.replace('/', '_')}_predictions.jsonl")
                with open(log_path, 'a') as f:
                    f.write(json.dumps(prediction_result) + '\n')
                
                logger.info(f"Made prediction for {asset}: {adjusted_signal} with confidence {adjusted_confidence:.2f}")
            
            except Exception as e:
                logger.error(f"Error making prediction for {asset}: {e}")
                predictions[asset] = {
                    'asset': asset,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'success': False
                }
        
        return predictions
    
    def _prepare_prediction_data(self, asset, data_dict):
        """
        Prepare data for making predictions
        
        Args:
            asset (str): Asset to prepare data for
            data_dict (dict): Dictionary of DataFrames with asset data
            
        Returns:
            tuple: (X, feature_names)
        """
        # Get data for the asset
        df = data_dict[asset]
        
        # Extract features (similar to training data preparation but for a single sequence)
        feature_names = []
        
        # Price and volume features
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_names.extend(price_cols)
        
        # Technical indicators (if available)
        tech_indicators = [col for col in df.columns if col not in price_cols and col != 'timestamp']
        feature_names.extend(tech_indicators)
        
        # If feature fusion is enabled, add cross-asset features
        if asset in self.asset_fusion_modules:
            try:
                # Get related assets data
                related_assets = self.related_assets.get(asset, [])
                fusion_data_dict = {
                    k: v for k, v in data_dict.items() 
                    if k == asset or k in related_assets
                }
                
                # Get current data for calculating cross-asset features
                current_data = {
                    k: v.iloc[-1].to_dict() for k, v in fusion_data_dict.items()
                }
                
                # Generate cross-asset features
                fusion_module = self.asset_fusion_modules[asset]
                cross_asset_features = fusion_module.calculate_cross_asset_features(current_data)
                
                # Add cross-asset features to the DataFrame
                for feature_name, value in cross_asset_features.items():
                    df[feature_name] = np.nan
                    df.iloc[-1, df.columns.get_loc(feature_name)] = value
                    feature_names.append(feature_name)
                
                logger.info(f"Added {len(cross_asset_features)} cross-asset features for {asset} prediction")
            except Exception as e:
                logger.error(f"Error adding cross-asset features for {asset} prediction: {e}")
        
        # Extract features
        features = df[feature_names].values
        
        # Get sequence length
        sequence_length = self.config.get('model_settings', {}).get('tft', {}).get('sequence_length', 60)
        
        # Ensure we have enough data
        if len(features) < sequence_length:
            raise ValueError(f"Not enough data for {asset} prediction. Need at least {sequence_length} rows.")
        
        # Get the most recent sequence
        X = features[-sequence_length:].reshape(1, sequence_length, len(feature_names))
        
        return X, feature_names
    
    def evaluate_performance(self, predictions, actual_outcomes):
        """
        Evaluate model performance based on predictions and actual outcomes
        
        Args:
            predictions (dict): Predictions made by the models
            actual_outcomes (dict): Actual market outcomes
            
        Returns:
            dict: Performance metrics
        """
        performance = {}
        
        for asset in self.trading_pairs:
            if asset not in predictions or asset not in actual_outcomes:
                logger.warning(f"Missing prediction or outcome for {asset}, skipping evaluation")
                continue
            
            logger.info(f"Evaluating performance for {asset}")
            
            try:
                # Get prediction and outcome
                prediction = predictions[asset]
                outcome = actual_outcomes[asset]
                
                # Extract values
                predicted_signal = prediction.get('adjusted_signal', prediction.get('signal'))
                actual_direction = outcome.get('direction')
                profit_loss = outcome.get('profit_loss', 0)
                
                # Calculate metrics
                correct_direction = (predicted_signal == 'BUY' and actual_direction == 'up') or \
                                   (predicted_signal == 'SELL' and actual_direction == 'down') or \
                                   (predicted_signal == 'NEUTRAL' and actual_direction == 'sideways')
                
                # Update performance metrics
                if asset not in self.performance_metrics:
                    self.performance_metrics[asset] = {
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1_score': [],
                        'returns': [],
                        'drawdowns': [],
                        'sharpe_ratio': []
                    }
                
                # Append new metrics
                self.performance_metrics[asset]['returns'].append(profit_loss)
                
                # Update explanation performance if available
                if 'explanation' in prediction and self.config.get('explainable_ai', {}).get('enabled', True):
                    try:
                        # Get explainer
                        explainer = self.explainable_ai_manager.get_explainer('tft', asset)
                        
                        # Track explanation performance
                        explanation_id = prediction.get('explanation_id')
                        if explanation_id:
                            explainer.track_explanation_performance(
                                explanation_id,
                                {'correct': correct_direction, 'profit': profit_loss}
                            )
                    except Exception as e:
                        logger.error(f"Error tracking explanation performance for {asset}: {e}")
                
                # Calculate performance metrics over window
                all_predictions = []
                all_actuals = []
                
                # Get prediction logs for this asset
                log_path = os.path.join(PREDICTION_LOG_DIR, f"{asset.replace('/', '_')}_predictions.jsonl")
                if os.path.exists(log_path):
                    try:
                        with open(log_path, 'r') as f:
                            for line in f:
                                try:
                                    entry = json.loads(line.strip())
                                    if 'adjusted_signal' in entry and 'outcome' in entry:
                                        all_predictions.append(entry['adjusted_signal'])
                                        all_actuals.append(entry['outcome']['direction'])
                                except Exception:
                                    continue
                    except Exception as e:
                        logger.error(f"Error reading prediction logs for {asset}: {e}")
                
                if all_predictions and all_actuals:
                    # Convert to numeric for metrics calculation
                    y_pred = [1 if p == 'BUY' else -1 if p == 'SELL' else 0 for p in all_predictions]
                    y_true = [1 if a == 'up' else -1 if a == 'down' else 0 for a in all_actuals]
                    
                    # Calculate binary metrics (treating non-neutrals as the positive class)
                    y_pred_binary = [1 if p != 0 else 0 for p in y_pred]
                    y_true_binary = [1 if t != 0 else 0 for t in y_true]
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_true, y_pred)
                    
                    # Only calculate other metrics if we have non-neutral predictions
                    if sum(y_pred_binary) > 0 and sum(y_true_binary) > 0:
                        precision = precision_score(y_true_binary, y_pred_binary)
                        recall = recall_score(y_true_binary, y_pred_binary)
                        f1 = f1_score(y_true_binary, y_pred_binary)
                        
                        self.performance_metrics[asset]['precision'].append(precision)
                        self.performance_metrics[asset]['recall'].append(recall)
                        self.performance_metrics[asset]['f1_score'].append(f1)
                    
                    self.performance_metrics[asset]['accuracy'].append(accuracy)
                    
                    # Calculate cumulative return
                    cumulative_return = sum(self.performance_metrics[asset]['returns'])
                    
                    # Calculate Sharpe ratio if we have enough data
                    sharpe_ratio = 0
                    if len(self.performance_metrics[asset]['returns']) > 1:
                        returns_array = np.array(self.performance_metrics[asset]['returns'])
                        sharpe_ratio = np.mean(returns_array) / (np.std(returns_array) + 1e-10) * np.sqrt(252)  # Annualized
                        self.performance_metrics[asset]['sharpe_ratio'].append(sharpe_ratio)
                    
                    # Calculate maximum drawdown
                    returns = np.array(self.performance_metrics[asset]['returns'])
                    cumulative = np.cumsum(returns)
                    drawdown = np.max(np.maximum.accumulate(cumulative) - cumulative)
                    self.performance_metrics[asset]['drawdowns'].append(drawdown)
                    
                    # Get recent accuracy
                    recent_accuracy = np.mean(self.performance_metrics[asset]['accuracy'][-10:]) if len(self.performance_metrics[asset]['accuracy']) >= 10 else accuracy
                    
                    # Compile performance report
                    performance[asset] = {
                        'accuracy': float(accuracy),
                        'recent_accuracy': float(recent_accuracy),
                        'current_prediction': predicted_signal,
                        'last_outcome': actual_direction,
                        'was_correct': correct_direction,
                        'profit_loss': float(profit_loss),
                        'cumulative_return': float(cumulative_return),
                        'sharpe_ratio': float(sharpe_ratio),
                        'max_drawdown': float(drawdown)
                    }
                    
                    logger.info(f"Performance for {asset}: Accuracy={recent_accuracy:.4f}, Return={cumulative_return:.2f}%")
                    
                    # Check if we're meeting performance targets
                    if recent_accuracy >= TARGET_ACCURACY:
                        logger.info(f"✅ {asset} model has reached target accuracy: {recent_accuracy:.4f}")
                    
                    if cumulative_return >= TARGET_RETURN:
                        logger.info(f"✅ {asset} model has reached target return: {cumulative_return:.2f}%")
                else:
                    logger.warning(f"Not enough historical predictions for {asset} to calculate metrics")
            
            except Exception as e:
                logger.error(f"Error evaluating performance for {asset}: {e}")
                performance[asset] = {
                    'error': str(e),
                    'success': False
                }
        
        return performance
    
    def update_sentiment_data(self, force_refresh=False):
        """
        Update sentiment data for all assets
        
        Args:
            force_refresh (bool): Whether to force refresh of sentiment data
            
        Returns:
            dict: Updated sentiment data
        """
        sentiment_data = {}
        
        if not self.config.get('sentiment_analysis', {}).get('enabled', True):
            logger.info("Sentiment analysis disabled, skipping update")
            return sentiment_data
        
        for asset in self.trading_pairs:
            logger.info(f"Updating sentiment data for {asset}")
            
            try:
                # Check if refresh interval has passed
                refresh_interval = self.config.get('sentiment_analysis', {}).get('refresh_interval', 12)
                should_refresh = force_refresh
                
                # Check last update time
                cache_file = os.path.join(sai.SENTIMENT_DATA_DIR, f"{asset.replace('/', '_')}_sentiment.json")
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            cached_data = json.load(f)
                        
                        # Check age of cached data
                        cache_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
                        age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                        
                        should_refresh = should_refresh or age_hours >= refresh_interval
                    except Exception:
                        should_refresh = True
                else:
                    should_refresh = True
                
                # Update sentiment if needed
                if should_refresh:
                    asset_sentiment = self.market_sentiment_integrator.analyze_asset_sentiment(
                        asset,
                        max_age_hours=24,
                        refresh_cache=True
                    )
                    
                    # Update sentiment history
                    self.market_sentiment_integrator.update_sentiment_history(asset, asset_sentiment)
                    
                    logger.info(f"Updated sentiment data for {asset}")
                else:
                    # Use cached data
                    with open(cache_file, 'r') as f:
                        asset_sentiment = json.load(f)
                    
                    logger.info(f"Using cached sentiment data for {asset}")
                
                sentiment_data[asset] = asset_sentiment
            
            except Exception as e:
                logger.error(f"Error updating sentiment data for {asset}: {e}")
                sentiment_data[asset] = {
                    'error': str(e),
                    'success': False
                }
        
        return sentiment_data
    
    def generate_performance_report(self):
        """
        Generate a comprehensive performance report
        
        Returns:
            dict: Performance report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'assets': {},
            'overall': {
                'accuracy': 0,
                'return': 0,
                'sharpe_ratio': 0
            }
        }
        
        total_accuracy = 0
        total_return = 0
        asset_count = 0
        
        for asset, metrics in self.performance_metrics.items():
            if not metrics.get('accuracy'):
                continue
            
            # Calculate average metrics
            avg_accuracy = np.mean(metrics['accuracy']) if metrics['accuracy'] else 0
            avg_return = sum(metrics['returns']) if metrics['returns'] else 0
            avg_sharpe = np.mean(metrics['sharpe_ratio']) if metrics['sharpe_ratio'] else 0
            max_drawdown = max(metrics['drawdowns']) if metrics['drawdowns'] else 0
            
            # Add to overall totals
            total_accuracy += avg_accuracy
            total_return += avg_return
            asset_count += 1
            
            # Add to report
            report['assets'][asset] = {
                'accuracy': float(avg_accuracy),
                'return_percent': float(avg_return),
                'sharpe_ratio': float(avg_sharpe),
                'max_drawdown': float(max_drawdown),
                'prediction_count': len(metrics['accuracy'])
            }
        
        # Calculate overall metrics
        if asset_count > 0:
            report['overall']['accuracy'] = float(total_accuracy / asset_count)
            report['overall']['return'] = float(total_return / asset_count)
            
            # Calculate overall Sharpe ratio
            all_returns = []
            for asset, metrics in self.performance_metrics.items():
                all_returns.extend(metrics['returns'])
            
            if all_returns:
                all_returns_array = np.array(all_returns)
                overall_sharpe = np.mean(all_returns_array) / (np.std(all_returns_array) + 1e-10) * np.sqrt(252)
                report['overall']['sharpe_ratio'] = float(overall_sharpe)
        
        # Add target achievement status
        report['targets'] = {
            'accuracy': {
                'target': TARGET_ACCURACY,
                'current': report['overall']['accuracy'],
                'achieved': report['overall']['accuracy'] >= TARGET_ACCURACY
            },
            'return': {
                'target': TARGET_RETURN,
                'current': report['overall']['return'],
                'achieved': report['overall']['return'] >= TARGET_RETURN
            }
        }
        
        # Save report
        report_path = os.path.join(TRAINING_RESULTS_DIR, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated performance report: Overall accuracy={report['overall']['accuracy']:.4f}, Return={report['overall']['return']:.2f}%")
        
        return report

class MLModelIntegrator:
    """
    Handles integration of ML models with the trading system.
    """
    def __init__(self, advanced_ml=None):
        """
        Initialize the ML model integrator
        
        Args:
            advanced_ml (AdvancedMLIntegration, optional): Advanced ML integration instance
        """
        self.advanced_ml = advanced_ml or AdvancedMLIntegration()
        
        # Ensure models are initialized
        if not hasattr(self.advanced_ml, 'tft_models') or not self.advanced_ml.tft_models:
            self.advanced_ml.initialize_models()
        
        logger.info("Initialized ML Model Integrator")
    
    def get_trading_signals(self, data_dict):
        """
        Get trading signals for all assets
        
        Args:
            data_dict (dict): Dictionary of DataFrames with asset data
            
        Returns:
            dict: Trading signals for each asset
        """
        # Make predictions
        predictions = self.advanced_ml.predict(data_dict)
        
        # Convert predictions to trading signals
        signals = {}
        
        for asset, prediction in predictions.items():
            if 'error' in prediction:
                logger.error(f"Error in prediction for {asset}: {prediction['error']}")
                continue
            
            # Extract signal information
            signal = prediction.get('adjusted_signal', prediction.get('signal'))
            confidence = prediction.get('adjusted_confidence', prediction.get('confidence'))
            
            # Determine signal strength based on confidence
            strength = confidence
            
            # Add to signals
            signals[asset] = {
                'signal': signal,
                'strength': float(strength),
                'confidence': float(confidence),
                'prediction': float(prediction.get('adjusted_prediction', prediction.get('raw_prediction', 0))),
                'timestamp': prediction.get('timestamp')
            }
            
            # Add explanation if available
            if 'explanation' in prediction:
                signals[asset]['explanation'] = prediction['explanation'].get('explanation_text')
            
            # Add sentiment information if available
            if 'sentiment' in prediction:
                signals[asset]['sentiment'] = prediction['sentiment']
            
            logger.info(f"Signal for {asset}: {signal} (strength: {strength:.2f})")
        
        return signals
    
    def update_model_weights(self, performance_data):
        """
        Update model weights based on performance
        
        Args:
            performance_data (dict): Performance metrics for each asset/model
            
        Returns:
            bool: Success of update operation
        """
        # This would update weights in ml_config.json based on performance
        # For now, we'll just log the performance data
        logger.info("Model weights would be updated based on performance metrics")
        return True
    
    def optimize_trading_parameters(self, asset, performance_data):
        """
        Optimize trading parameters for an asset based on performance
        
        Args:
            asset (str): Asset to optimize parameters for
            performance_data (dict): Performance metrics
            
        Returns:
            dict: Optimized parameters
        """
        # Get hyperparameter tuner for this asset
        tuner = self.advanced_ml.hyperparameter_manager.get_tuner('tft', asset)
        
        # Get recent performance
        recent_pl = performance_data.get('profit_loss', 0)
        
        # Get suggested parameter adjustments
        suggestions = tuner.suggest_parameter_adjustments(recent_pl)
        
        # Log suggestions
        logger.info(f"Suggested parameter adjustments for {asset}: {suggestions}")
        
        # In a full implementation, you would use these suggestions to update parameters
        return suggestions
    
    def get_model_explanations(self, asset, prediction):
        """
        Get human-readable explanations for a model prediction
        
        Args:
            asset (str): Asset the prediction is for
            prediction (dict): Prediction data
            
        Returns:
            str: Human-readable explanation
        """
        explanation_text = "No explanation available."
        
        if 'explanation' in prediction and 'explanation_text' in prediction['explanation']:
            explanation_text = prediction['explanation']['explanation_text']
        
        return explanation_text
    
    def generate_summary_report(self):
        """
        Generate a summary report of model performance
        
        Returns:
            str: Summary report
        """
        # Get performance report
        report = self.advanced_ml.generate_performance_report()
        
        # Format as readable text
        summary = [
            "# ML Model Performance Summary",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"## Overall Performance",
            f"Accuracy: {report['overall']['accuracy']:.2%} (Target: {TARGET_ACCURACY:.2%})",
            f"Return: {report['overall']['return']:.2f}% (Target: {TARGET_RETURN:.2f}%)",
            f"Sharpe Ratio: {report['overall']['sharpe_ratio']:.2f}",
            ""
        ]
        
        # Add asset-specific performance
        summary.append("## Asset Performance")
        
        for asset, metrics in report['assets'].items():
            summary.append(f"### {asset}")
            summary.append(f"Accuracy: {metrics['accuracy']:.2%}")
            summary.append(f"Return: {metrics['return_percent']:.2f}%")
            summary.append(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            summary.append(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            summary.append(f"Predictions: {metrics['prediction_count']}")
            summary.append("")
        
        # Add target achievement status
        summary.append("## Target Achievement")
        summary.append(f"Accuracy Target: {'✅ Achieved' if report['targets']['accuracy']['achieved'] else '❌ Not Yet Achieved'}")
        summary.append(f"Return Target: {'✅ Achieved' if report['targets']['return']['achieved'] else '❌ Not Yet Achieved'}")
        
        return "\n".join(summary)

# Create global instances for easy access
advanced_ml = AdvancedMLIntegration()
ml_model_integrator = MLModelIntegrator(advanced_ml)

def main():
    """Test the advanced ML integration module"""
    logger.info("Advanced ML Integration module imported")

if __name__ == "__main__":
    main()