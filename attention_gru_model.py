#!/usr/bin/env python3
"""
Attention-based GRU model for improved prediction accuracy

This model combines GRU (Gated Recurrent Unit) cells with a self-attention mechanism
to better capture temporal dependencies in financial time series data.
The attention mechanism helps the model focus on the most relevant parts of the 
input sequence when making predictions.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, GRU, Dropout, BatchNormalization, 
    Activation, concatenate, Bidirectional, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Custom Attention Layer for the GRU model
class AttentionLayer(Layer):
    """
    Attention mechanism to focus on relevant parts of input sequence
    
    This layer calculates attention weights for each timestep in the input sequence,
    allowing the model to focus on more important timeframes when making predictions.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Calculate attention scores
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        
        # Calculate attention weights via softmax
        a = tf.nn.softmax(e, axis=1)
        
        # Apply attention weights to input
        output = x * a
        return tf.reduce_sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def create_attention_gru_model(input_shape, output_units=1, dropout_rate=0.3):
    """
    Create an Attention-GRU model for time series prediction
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features)
        output_units (int): Number of output units (1 for regression, 2+ for classification)
        dropout_rate (float): Dropout rate to prevent overfitting
        
    Returns:
        Model: Compiled Keras model
    """
    logger.info(f"Creating Attention-GRU model with input shape: {input_shape}")
    
    # Input layer
    inputs = Input(shape=input_shape, name="input")
    
    # Bidirectional GRU layers with increasing units
    gru1 = Bidirectional(GRU(64, return_sequences=True, 
                           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(inputs)
    bn1 = BatchNormalization()(gru1)
    drop1 = Dropout(dropout_rate)(bn1)
    
    gru2 = Bidirectional(GRU(128, return_sequences=True,
                           kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)))(drop1)
    bn2 = BatchNormalization()(gru2)
    drop2 = Dropout(dropout_rate)(bn2)
    
    # Attention mechanism
    attention_output = AttentionLayer()(drop2)
    
    # Additional dense layers for feature extraction
    dense1 = Dense(64, activation=None)(attention_output)
    bn3 = BatchNormalization()(dense1)
    act1 = Activation('relu')(bn3)
    drop3 = Dropout(dropout_rate)(act1)
    
    # Output layer - linear for regression, softmax for classification
    if output_units == 1:  # Regression task
        outputs = Dense(output_units, activation='linear', name="output")(drop3)
    else:  # Classification task
        outputs = Dense(output_units, activation='softmax', name="output")(drop3)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Adam(learning_rate=0.001)
    
    # Compile with appropriate loss function
    if output_units == 1:  # Regression task
        model.compile(
            optimizer=optimizer, 
            loss='mse',
            metrics=['mae']
        )
    else:  # Classification task
        model.compile(
            optimizer=optimizer, 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    logger.info(f"Attention-GRU model created with {model.count_params()} parameters")
    return model


def create_asymmetric_attention_gru_model(input_shape, output_units=1, loss_ratio=2.0, dropout_rate=0.3):
    """
    Create an Attention-GRU model with asymmetric loss for time series prediction
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features)
        output_units (int): Number of output units (1 for regression, 2+ for classification)
        loss_ratio (float): Ratio to penalize false positive or negative predictions differently
        dropout_rate (float): Dropout rate to prevent overfitting
        
    Returns:
        Model: Compiled Keras model
    """
    logger.info(f"Creating Asymmetric Attention-GRU model with input shape: {input_shape}")
    
    # Custom asymmetric loss function for regression that penalizes errors differently
    def asymmetric_mse(y_true, y_pred):
        residual = y_true - y_pred
        positive_error = tf.maximum(0.0, residual)  # Underestimation errors
        negative_error = tf.maximum(0.0, -residual)  # Overestimation errors
        
        # Apply different weights to over/under estimation errors
        weighted_error = loss_ratio * positive_error**2 + negative_error**2
        return tf.reduce_mean(weighted_error)
    
    # Build the model architecture
    model = create_attention_gru_model(input_shape, output_units, dropout_rate)
    
    # Re-compile with asymmetric loss for regression
    if output_units == 1:
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer, 
            loss=asymmetric_mse,
            metrics=['mae']
        )
    
    return model


def train_attention_gru_model(X_train, y_train, X_val=None, y_val=None, 
                             batch_size=32, epochs=100, output_units=1,
                             model_path="models/attention_gru_model.h5",
                             use_asymmetric_loss=False, loss_ratio=2.0):
    """
    Train the Attention-GRU model
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation targets
        batch_size (int): Batch size for training
        epochs (int): Maximum number of training epochs
        output_units (int): Number of output units
        model_path (str): Path to save the trained model
        use_asymmetric_loss (bool): Whether to use asymmetric loss function
        loss_ratio (float): Ratio for the asymmetric loss
        
    Returns:
        tuple: (model, history)
    """
    # Create validation data if not provided
    if X_val is None or y_val is None:
        logger.info("No validation data provided, using 20% of training data for validation")
        val_split = 0.2
        split_idx = int(len(X_train) * (1 - val_split))
        X_val, y_val = X_train[split_idx:], y_train[split_idx:]
        X_train, y_train = X_train[:split_idx], y_train[:split_idx]
    
    # Get input shape from training data
    input_shape = X_train.shape[1:]
    logger.info(f"Training Attention-GRU model with {len(X_train)} samples, input shape: {input_shape}")
    
    # Create the model
    if use_asymmetric_loss:
        model = create_asymmetric_attention_gru_model(
            input_shape, output_units=output_units, 
            loss_ratio=loss_ratio, dropout_rate=0.3
        )
    else:
        model = create_attention_gru_model(
            input_shape, output_units=output_units, 
            dropout_rate=0.3
        )
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model weights
    model.load_weights(model_path)
    
    logger.info(f"Model training completed. Best val_loss: {min(history.history['val_loss']):.4f}")
    
    return model, history


def evaluate_attention_gru_model(model, X_test, y_test):
    """
    Evaluate the Attention-GRU model
    
    Args:
        model (Model): Trained model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test targets
        
    Returns:
        dict: Evaluation metrics
    """
    # Evaluate the model
    logger.info(f"Evaluating model on {len(X_test)} test samples")
    results = model.evaluate(X_test, y_test, verbose=0)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Compute additional metrics for regression
    if y_test.shape[1] == 1:  # Regression task
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    else:  # Classification task
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1))
        metrics = {
            'accuracy': accuracy
        }
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics


def save_attention_gru_model(model, model_path):
    """
    Save the Attention-GRU model
    
    Args:
        model (Model): Model to save
        model_path (str): Path to save the model
        
    Returns:
        str: Path to the saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    return model_path


def load_attention_gru_model(model_path):
    """
    Load the Attention-GRU model
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        Model: Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load the model with custom objects
    custom_objects = {'AttentionLayer': AttentionLayer}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    return model


if __name__ == "__main__":
    # Simple test with dummy data
    input_shape = (60, 32)  # 60 timesteps, 32 features
    X_train = np.random.random((100, 60, 32))
    y_train = np.random.random((100, 1))
    
    # Create and train model
    model, history = train_attention_gru_model(
        X_train, y_train, 
        epochs=5, 
        model_path="models/test_attention_gru_model.h5"
    )
    
    # Test prediction
    test_input = np.random.random((1, 60, 32))
    prediction = model.predict(test_input)
    print(f"Test prediction: {prediction[0][0]:.4f}")