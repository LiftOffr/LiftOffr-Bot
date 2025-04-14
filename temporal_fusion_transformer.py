#!/usr/bin/env python3
"""
Temporal Fusion Transformer Model for Kraken Trading Bot

This module implements the Temporal Fusion Transformer (TFT) architecture
for multi-horizon time series forecasting with interpretability features.

The TFT model is designed to handle complex dependencies in time series data
by using a combination of recurrent layers, attention mechanisms, and interpretable
variable selection networks.

Reference:
Lim et al. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GRU, Dropout, LayerNormalization
from tensorflow.keras.layers import Input, Concatenate, Multiply, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MODELS_DIR = "models/tft"
os.makedirs(MODELS_DIR, exist_ok=True)

class GatedResidualNetwork(tf.keras.layers.Layer):
    """
    Gated Residual Network as described in the TFT paper.
    This is a key component for variable selection and processing.
    """
    def __init__(self, units, dropout_rate=0.1, use_time_distributed=False, **kwargs):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed
        
        # Layers
        self.dense_1 = Dense(units, activation='elu')
        self.dense_2 = Dense(units, activation=None)
        self.dense_3 = Dense(units, activation=None)
        self.gate = Dense(units, activation='sigmoid')
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()
        
    def call(self, inputs, context=None, training=None):
        x = inputs
        
        # If additional context is provided, incorporate it
        if context is not None:
            context_layer = Dense(self.units, activation=None)(context)
            x = Add()([x, context_layer])
        
        # First residual layer
        skip = x
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dropout(x, training=training)
        
        # Gating layer
        gate = self.gate(x)
        
        # Apply gating mechanism
        x = Multiply()([gate, x])
        x = Add()([x, skip])
        x = self.layer_norm(x)
        
        return x
    
    def get_config(self):
        config = super(GatedResidualNetwork, self).get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'use_time_distributed': self.use_time_distributed
        })
        return config

class VariableSelectionNetwork(tf.keras.layers.Layer):
    """
    Variable Selection Network as described in the TFT paper.
    This component selects the most relevant variables at each time step.
    """
    def __init__(self, num_features, units, dropout_rate=0.1, **kwargs):
        super(VariableSelectionNetwork, self).__init__(**kwargs)
        self.num_features = num_features
        self.units = units
        self.dropout_rate = dropout_rate
        
        # Layers
        self.grn_flat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = Dense(num_features, activation='softmax')
        self.feature_grns = [GatedResidualNetwork(units, dropout_rate) for _ in range(num_features)]
        
    def call(self, inputs, context=None, training=None):
        # Process each feature independently
        processed_features = []
        for i in range(self.num_features):
            feature = inputs[:, :, i:i+1]
            processed = self.feature_grns[i](feature, training=training)
            processed_features.append(processed)
        
        # Create a combined feature tensor
        combined = tf.concat([f for f in processed_features], axis=-1)
        
        # Calculate variable selection weights
        if context is not None:
            flat_inputs = tf.reshape(inputs, [-1, inputs.shape[1] * inputs.shape[2]])
            flat_context = tf.concat([flat_inputs, context], axis=-1)
            weights = self.grn_flat(flat_context, training=training)
        else:
            flat_inputs = tf.reshape(inputs, [-1, inputs.shape[1] * inputs.shape[2]])
            weights = self.grn_flat(flat_inputs, training=training)
            
        weights = self.softmax(weights)
        weights = tf.expand_dims(weights, -2)
        
        # Apply weights to features
        weighted_features = tf.multiply(weights, combined)
        return weighted_features, weights
    
    def get_config(self):
        config = super(VariableSelectionNetwork, self).get_config()
        config.update({
            'num_features': self.num_features,
            'units': self.units,
            'dropout_rate': self.dropout_rate
        })
        return config

class TemporalFusionTransformer(tf.keras.Model):
    """
    Temporal Fusion Transformer model for interpretable time series forecasting.
    
    This model combines LSTM/GRU encoders with multi-head attention and
    specialized components for variable selection and processing.
    """
    def __init__(self, 
                 num_features,
                 num_static_features=0,
                 hidden_units=64,
                 dropout_rate=0.1,
                 num_heads=4,
                 forecast_horizon=1,
                 max_sequence_length=252,
                 **kwargs):
        super(TemporalFusionTransformer, self).__init__(**kwargs)
        self.num_features = num_features
        self.num_static_features = num_static_features
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.forecast_horizon = forecast_horizon
        self.max_sequence_length = max_sequence_length
        
        # Variable selection networks
        self.feature_selection = VariableSelectionNetwork(
            num_features=num_features,
            units=hidden_units,
            dropout_rate=dropout_rate
        )
        
        # Static feature processing (if any)
        if num_static_features > 0:
            self.static_selection = VariableSelectionNetwork(
                num_features=num_static_features,
                units=hidden_units,
                dropout_rate=dropout_rate
            )
            self.static_context_grn = GatedResidualNetwork(
                units=hidden_units,
                dropout_rate=dropout_rate
            )
            self.static_enrichment_grn = GatedResidualNetwork(
                units=hidden_units,
                dropout_rate=dropout_rate
            )
        
        # Sequence processing
        self.encoder_lstm = GRU(
            hidden_units, 
            return_sequences=True,
            return_state=True,
            dropout=dropout_rate
        )
        self.decoder_lstm = GRU(
            hidden_units, 
            return_sequences=True,
            return_state=True,
            dropout=dropout_rate
        )
        
        # Attention mechanism
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_units
        )
        self.attention_dropout = Dropout(dropout_rate)
        self.attention_norm = LayerNormalization()
        
        # Output processing
        self.output_grn = GatedResidualNetwork(
            units=hidden_units,
            dropout_rate=dropout_rate
        )
        self.output_layer = Dense(1)
        
        # Additional layers for position-wise processing
        self.post_lstm_norm = LayerNormalization()
        self.post_attention_norm = LayerNormalization()
        
    def call(self, inputs, training=None):
        time_features = inputs
        batch_size = tf.shape(time_features)[0]
        
        # Process static features if available
        static_context = None
        if self.num_static_features > 0:
            static_features = inputs['static_features']
            static_embeddings, static_weights = self.static_selection(static_features, training=training)
            static_context = self.static_context_grn(static_embeddings, training=training)
        
        # Process time-varying features
        processed_features, feature_weights = self.feature_selection(
            time_features, 
            context=static_context, 
            training=training
        )
        
        # Temporal processing - Encoder
        encoder_output, encoder_state = self.encoder_lstm(processed_features, training=training)
        encoder_output = self.post_lstm_norm(encoder_output)
        
        # Enrichment with static features if available
        if self.num_static_features > 0:
            static_enrichment = tf.tile(
                tf.expand_dims(static_context, 1),
                [1, tf.shape(encoder_output)[1], 1]
            )
            encoder_output = self.static_enrichment_grn(
                encoder_output, 
                context=static_enrichment,
                training=training
            )
        
        # Apply attention mechanism
        attention_output = self.attention(
            query=encoder_output,
            key=encoder_output,
            value=encoder_output,
            training=training
        )
        attention_output = self.attention_dropout(attention_output, training=training)
        attention_output = self.attention_norm(attention_output + encoder_output)
        
        # Final processing
        output = self.output_grn(attention_output, training=training)
        output = self.output_layer(output)
        
        # Return model outputs and attention weights for interpretability
        return {
            'predictions': output,
            'attention_weights': attention_output,
            'feature_weights': feature_weights
        }
    
    def get_config(self):
        config = super(TemporalFusionTransformer, self).get_config()
        config.update({
            'num_features': self.num_features,
            'num_static_features': self.num_static_features,
            'hidden_units': self.hidden_units,
            'dropout_rate': self.dropout_rate,
            'num_heads': self.num_heads,
            'forecast_horizon': self.forecast_horizon,
            'max_sequence_length': self.max_sequence_length
        })
        return config

def create_tft_model(num_features, num_static_features=0, hidden_units=64, 
                    dropout_rate=0.1, num_heads=4, forecast_horizon=1):
    """
    Create and compile a Temporal Fusion Transformer model
    
    Args:
        num_features (int): Number of time-varying input features
        num_static_features (int): Number of static input features
        hidden_units (int): Dimension of hidden units
        dropout_rate (float): Dropout rate for regularization
        num_heads (int): Number of heads in multi-head attention
        forecast_horizon (int): Number of time steps to forecast
        
    Returns:
        Model: Compiled TFT model
    """
    logger.info(f"Creating TFT model with {num_features} features, {hidden_units} hidden units")
    
    # Define model inputs based on features
    model = TemporalFusionTransformer(
        num_features=num_features,
        num_static_features=num_static_features,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        num_heads=num_heads,
        forecast_horizon=forecast_horizon
    )
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    logger.info(f"TFT model created successfully")
    return model

def create_tft_model_with_asymmetric_loss(num_features, num_static_features=0, 
                                         hidden_units=64, dropout_rate=0.1, 
                                         num_heads=4, forecast_horizon=1,
                                         loss_ratio=2.0):
    """
    Create a TFT model with asymmetric loss function
    
    Args:
        num_features (int): Number of time-varying input features
        num_static_features (int): Number of static input features
        hidden_units (int): Dimension of hidden units
        dropout_rate (float): Dropout rate for regularization
        num_heads (int): Number of heads in multi-head attention
        forecast_horizon (int): Number of time steps to forecast
        loss_ratio (float): Ratio for asymmetric loss (penalizing errors differently)
        
    Returns:
        Model: Compiled TFT model with asymmetric loss
    """
    # Create TFT model
    model = create_tft_model(
        num_features=num_features,
        num_static_features=num_static_features,
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        num_heads=num_heads,
        forecast_horizon=forecast_horizon
    )
    
    # Define asymmetric loss function
    def asymmetric_mse(y_true, y_pred):
        residual = y_true - y_pred
        positive_error = tf.maximum(0.0, residual)  # Underestimation errors
        negative_error = tf.maximum(0.0, -residual)  # Overestimation errors
        
        # Apply different weights to over/under estimation errors
        weighted_error = loss_ratio * positive_error**2 + negative_error**2
        return tf.reduce_mean(weighted_error)
    
    # Recompile with asymmetric loss
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss=asymmetric_mse,
        metrics=['mae']
    )
    
    logger.info(f"TFT model with asymmetric loss (ratio: {loss_ratio}) created successfully")
    return model

def train_tft_model(model, X_train, y_train, X_val=None, y_val=None, 
                   batch_size=32, epochs=100, model_path=None):
    """
    Train the TFT model
    
    Args:
        model (Model): TFT model to train
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation targets
        batch_size (int): Batch size for training
        epochs (int): Maximum number of training epochs
        model_path (str): Path to save the trained model
        
    Returns:
        tuple: (model, history)
    """
    logger.info(f"Training TFT model with {len(X_train)} samples")
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=20,
            restore_best_weights=True
        )
    ]
    
    # Add model checkpoint callback if model_path is provided
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss' if X_val is not None else 'loss',
                save_best_only=True
            )
        )
    
    # Train the model
    if X_val is not None and y_val is not None:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
    logger.info("TFT model training completed")
    return model, history

def save_tft_model(model, ticker_symbol, model_dir=MODELS_DIR):
    """
    Save TFT model and its configuration
    
    Args:
        model (Model): Trained TFT model
        ticker_symbol (str): Ticker symbol for the asset
        model_dir (str): Directory to save the model
        
    Returns:
        str: Path to the saved model
    """
    # Create model directory for this asset
    asset_model_dir = os.path.join(model_dir, ticker_symbol)
    os.makedirs(asset_model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(asset_model_dir, f"{ticker_symbol}.h5")
    model.save(model_path)
    
    # Save model configuration
    config = model.get_config()
    config_path = os.path.join(asset_model_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    logger.info(f"TFT model for {ticker_symbol} saved to {model_path}")
    return model_path

def load_tft_model(ticker_symbol, model_dir=MODELS_DIR):
    """
    Load TFT model for the specified asset
    
    Args:
        ticker_symbol (str): Ticker symbol for the asset
        model_dir (str): Directory containing the model
        
    Returns:
        Model: Loaded TFT model
    """
    # Construct model path
    model_path = os.path.join(model_dir, ticker_symbol, f"{ticker_symbol}.h5")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {ticker_symbol} not found at {model_path}")
    
    # Define custom objects for model loading
    custom_objects = {
        'TemporalFusionTransformer': TemporalFusionTransformer,
        'GatedResidualNetwork': GatedResidualNetwork,
        'VariableSelectionNetwork': VariableSelectionNetwork
    }
    
    # Load the model with custom objects
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    logger.info(f"TFT model for {ticker_symbol} loaded from {model_path}")
    
    return model

def get_feature_importance(model, X_test):
    """
    Extract feature importance scores from the TFT model
    
    Args:
        model (Model): Trained TFT model
        X_test (np.ndarray): Test data to evaluate feature importance
        
    Returns:
        dict: Feature importance scores
    """
    # Get model predictions which include feature weights
    predictions = model.predict(X_test)
    feature_weights = predictions['feature_weights']
    
    # Average feature weights across all samples
    avg_feature_weights = tf.reduce_mean(feature_weights, axis=0).numpy()
    
    # Create dictionary of feature importance
    importance = {
        'feature_weights': avg_feature_weights.tolist(),
        'attention_score': tf.reduce_mean(predictions['attention_weights'], axis=0).numpy().tolist()
    }
    
    return importance

def get_interpretable_prediction(model, X_sample):
    """
    Get an interpretable prediction with attention scores and feature importance
    
    Args:
        model (Model): Trained TFT model
        X_sample (np.ndarray): Single sample for prediction
        
    Returns:
        dict: Prediction with interpretability information
    """
    # Ensure input is batched (add dimension if needed)
    if len(X_sample.shape) == 2:
        X_sample = np.expand_dims(X_sample, axis=0)
    
    # Get prediction
    result = model.predict(X_sample)
    
    # Extract components
    prediction = result['predictions'][0, -1, 0]  # Last time step, first output
    feature_weights = result['feature_weights'][0].numpy()  # Feature importance
    attention = result['attention_weights'][0].numpy()  # Attention weights
    
    return {
        'prediction': float(prediction),
        'feature_importance': feature_weights.tolist(),
        'temporal_importance': attention.tolist()
    }

if __name__ == "__main__":
    # Example usage
    print("Temporal Fusion Transformer module imported")