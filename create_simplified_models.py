#!/usr/bin/env python3
"""
Create Simplified Models for Testing

This script creates simplified models for testing the ensemble architecture
without requiring the full training process. These models are for testing
purposes only and should not be used for actual trading.

Usage:
    python create_simplified_models.py [--pair PAIR] [--timeframe TIMEFRAME]
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('simplified_models.log')
    ]
)

# Constants
SUPPORTED_PAIRS = [
    'SOL/USD', 'BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD',
    'LINK/USD', 'AVAX/USD', 'MATIC/USD', 'UNI/USD', 'ATOM/USD'
]
TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d']
MODEL_TYPES = ['entry', 'exit', 'cancel', 'sizing']


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create simplified models for testing')
    parser.add_argument('--pair', type=str, choices=SUPPORTED_PAIRS,
                       help='Trading pair (default: all pairs)')
    parser.add_argument('--timeframe', type=str, choices=TIMEFRAMES,
                       help='Timeframe (default: all timeframes)')
    return parser.parse_args()


def create_simple_model(model_type: str, input_shape: tuple = (24, 37)) -> tf.keras.Model:
    """
    Create a simple model for testing
    
    Args:
        model_type: Type of model ('entry', 'exit', 'cancel', 'sizing')
        input_shape: Input shape (sequence_length, features)
        
    Returns:
        Simple TensorFlow model
    """
    # Create a simple feed-forward model
    input_layer = Input(shape=input_shape)
    
    # Flatten input
    flattened = tf.keras.layers.Flatten()(input_layer)
    
    # Hidden layers
    x = Dense(64, activation='relu')(flattened)
    x = Dense(32, activation='relu')(x)
    
    # Output layer
    if model_type == 'sizing':
        output_layer = Dense(1, activation='tanh')(x)  # Regression
    else:
        output_layer = Dense(1, activation='sigmoid')(x)  # Classification
    
    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile model
    if model_type == 'sizing':
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def create_model_info(
    pair: str,
    timeframe: str,
    model_type: str,
    sequence_length: int = 24,
    feature_count: int = 37
) -> Dict:
    """
    Create model info dictionary
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        timeframe: Timeframe (e.g., 1h)
        model_type: Type of model ('entry', 'exit', 'cancel', 'sizing')
        sequence_length: Length of input sequences
        feature_count: Number of features
        
    Returns:
        Model info dictionary
    """
    # Create metrics based on model type
    if model_type == 'sizing':
        metrics = {
            'test_loss': 0.015,
            'test_mae': 0.095,
            'r_squared': 0.65
        }
    else:
        # Different metrics based on model type
        if model_type == 'entry':
            precision = 0.75
            recall = 0.65
        elif model_type == 'exit':
            precision = 0.70
            recall = 0.80
        elif model_type == 'cancel':
            precision = 0.65
            recall = 0.60
        
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        metrics = {
            'test_loss': 0.25,
            'test_accuracy': 0.78,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': 0.82,
            'profit_factor': 1.8,
            'win_rate': 0.75,
            'sharpe_ratio': 1.35
        }
    
    # Create model info
    return {
        'pair': pair,
        'timeframe': timeframe,
        'model_type': model_type,
        'feature_columns': [f"feature_{i}" for i in range(feature_count)],
        'sequence_length': sequence_length,
        'metrics': metrics,
        'training_history': {
            'loss': [0.5, 0.4, 0.35, 0.3, 0.28],
            'val_loss': [0.55, 0.45, 0.4, 0.38, 0.36]
        },
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }


def create_ensemble_info(
    pair: str,
    timeframe: str
) -> Dict:
    """
    Create ensemble info dictionary
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        timeframe: Timeframe (e.g., 1h)
        
    Returns:
        Ensemble info dictionary
    """
    # Create ensemble metrics (better than individual models)
    metrics = {
        'win_rate': 0.78,
        'profit_factor': 2.1,
        'sharpe_ratio': 1.8,
        'entry_precision': 0.82,
        'exit_recall': 0.85,
        'cancel_f1_score': 0.75,
        'sizing_mae': 0.08
    }
    
    # Create ensemble info
    return {
        'pair': pair,
        'timeframe': timeframe,
        'component_models': ['entry', 'exit', 'cancel', 'sizing'],
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metrics': metrics
    }


def create_models_for_pair_and_timeframe(
    pair: str,
    timeframe: str
) -> bool:
    """
    Create all models for a specific pair and timeframe
    
    Args:
        pair: Trading pair (e.g., SOL/USD)
        timeframe: Timeframe (e.g., 1h)
        
    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Creating simplified models for {pair} ({timeframe})")
    
    pair_formatted = pair.replace('/', '_')
    
    # Create directory if it doesn't exist
    os.makedirs('ml_models', exist_ok=True)
    
    try:
        # Create each model type
        for model_type in MODEL_TYPES:
            # Create model
            model = create_simple_model(model_type)
            
            # Save model
            model_path = f"ml_models/{pair_formatted}_{timeframe}_{model_type}_model.h5"
            model.save(model_path)
            logging.info(f"Saved {model_type} model to {model_path}")
            
            # Create and save model info
            model_info = create_model_info(pair, timeframe, model_type)
            info_path = f"ml_models/{pair_formatted}_{timeframe}_{model_type}_info.json"
            
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            logging.info(f"Saved {model_type} model info to {info_path}")
        
        # Create ensemble info
        ensemble_info = create_ensemble_info(pair, timeframe)
        ensemble_info_path = f"ml_models/{pair_formatted}_{timeframe}_ensemble_info.json"
        
        with open(ensemble_info_path, 'w') as f:
            json.dump(ensemble_info, f, indent=2)
        
        logging.info(f"Saved ensemble info to {ensemble_info_path}")
        
        return True
    
    except Exception as e:
        logging.error(f"Error creating models for {pair} ({timeframe}): {e}")
        return False


def main():
    """Main function"""
    args = parse_arguments()
    
    # Determine pairs to process
    pairs = [args.pair] if args.pair else SUPPORTED_PAIRS
    
    # Determine timeframes to process
    timeframes = [args.timeframe] if args.timeframe else TIMEFRAMES
    
    # Process each pair and timeframe
    success_count = 0
    total_count = len(pairs) * len(timeframes)
    
    for pair in pairs:
        for timeframe in timeframes:
            if create_models_for_pair_and_timeframe(pair, timeframe):
                success_count += 1
    
    logging.info(f"Created {success_count}/{total_count} model sets")
    
    return True


if __name__ == "__main__":
    main()