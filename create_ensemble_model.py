#!/usr/bin/env python3
"""
Create Ensemble Model

This script creates an ensemble model by combining multiple trained models
with optimized weights to maximize prediction accuracy.

Usage:
    python create_ensemble_model.py --pair PAIR [--model-types MODEL_TYPES]
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ensemble_creation.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PAIR = "SOLUSD"
DEFAULT_TIMEFRAME = "1h"
DEFAULT_MODEL_TYPES = ["lstm", "gru", "transformer", "tcn", "cnn", "bilstm", "attention", "hybrid"]
TRAINING_DATA_DIR = "training_data"
MODEL_DIRS = {
    "tcn": "models/tcn",
    "lstm": "models/lstm",
    "gru": "models/gru",
    "transformer": "models/transformer",
    "cnn": "models/cnn",
    "bilstm": "models/bilstm",
    "attention": "models/attention",
    "hybrid": "models/hybrid"
}
ENSEMBLE_DIR = "models/ensemble"
VALIDATION_SPLIT = 0.2

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Create ensemble model")
    parser.add_argument("--pair", type=str, default=DEFAULT_PAIR,
                        help=f"Trading pair (default: {DEFAULT_PAIR})")
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME,
                        help=f"Timeframe (default: {DEFAULT_TIMEFRAME})")
    parser.add_argument("--model-types", nargs="+", default=None,
                        help=f"Model types to include in ensemble (default: all available)")
    return parser.parse_args()

def load_dataset(pair: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load dataset for ensemble creation"""
    dataset_path = os.path.join(TRAINING_DATA_DIR, f"{pair}_{timeframe}_enhanced.csv")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return None
    
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with {len(df)} samples and {len(df.columns)} features")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return None

def prepare_dataset(df: pd.DataFrame) -> tuple:
    """Prepare dataset for ensemble creation"""
    # Remove timestamp and other non-numeric columns
    columns_to_drop = ["timestamp"]
    
    # Add string/categorical columns to drop list
    for col in df.columns:
        if df[col].dtype == 'object':
            logger.info(f"Dropping non-numeric column: {col}")
            columns_to_drop.append(col)
    
    # Drop non-numeric columns
    df = df.drop(columns_to_drop, axis=1, errors='ignore')
    
    # Get target variable (short-term direction, 8 hours ahead)
    target_col = "target_direction_8"
    if target_col not in df.columns:
        logger.error(f"Target column not found in dataset: {target_col}")
        return None, None, None, None
    
    # Get features and target
    y = df[target_col].values
    X = df.drop([col for col in df.columns if col.startswith("target_")], axis=1)
    
    # Handle missing values
    X = X.fillna(0)
    
    # Verify all columns are numeric
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            logger.error(f"Non-numeric column found after processing: {col}")
            return None, None, None, None
    
    logger.info(f"Processing {len(X.columns)} numeric features")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=VALIDATION_SPLIT, shuffle=False)
    
    # Don't reshape the data - we'll handle different model input shapes individually
    X_train_original = X_train
    X_test_original = X_test
    
    return X_train, X_test, y_train, y_test

def load_models(pair: str, model_types: List[str]) -> Dict:
    """Load trained models"""
    models = {}
    
    for model_type in model_types:
        if model_type not in MODEL_DIRS:
            logger.warning(f"Unknown model type: {model_type}")
            continue
        
        model_dir = MODEL_DIRS[model_type]
        
        # Try different possible filenames
        model_paths = [
            os.path.join(model_dir, f"{pair}.h5"),
            os.path.join(model_dir, f"{pair}_{model_type}.h5")
        ]
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    model = keras.models.load_model(model_path)
                    models[model_type] = model
                    logger.info(f"Loaded {model_type} model from {model_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.error(f"Error loading {model_type} model from {model_path}: {e}")
        
        if not model_loaded:
            logger.warning(f"No {model_type} model found for {pair}")
    
    return models

def evaluate_models(models: Dict, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate individual models"""
    performances = {}
    
    # Get the input shape for each model
    model_input_shapes = {}
    for model_type, model in models.items():
        try:
            # Get the input shape expected by the model
            input_shape = model.input_shape
            logger.info(f"{model_type} model expects input shape: {input_shape}")
            model_input_shapes[model_type] = input_shape
        except Exception as e:
            logger.error(f"Could not determine input shape for {model_type} model: {e}")
    
    for model_type, model in models.items():
        try:
            # Check if we have input shape information
            if model_type not in model_input_shapes:
                logger.warning(f"Skipping {model_type} model due to unknown input shape")
                continue
            
            # Reshape the data according to the model's requirements
            input_shape = model_input_shapes[model_type]
            X_test_reshaped = X_test
            
            # Check if this is a sequential model (LSTM, GRU, etc.) requiring time steps
            if len(input_shape) == 3:
                # Model expects (batch_size, time_steps, features)
                time_steps = input_shape[1]
                features = input_shape[2]
                
                if features == X_test.shape[1]:
                    # Model expects (batch_size, time_steps, features) where features match our data
                    # Just add the time_steps dimension
                    X_test_reshaped = X_test.reshape((X_test.shape[0], time_steps, X_test.shape[1]))
                elif X_test.shape[1] % features == 0:
                    # Features don't match but might be divisible
                    calculated_time_steps = X_test.shape[1] // features
                    X_test_reshaped = X_test.reshape((X_test.shape[0], calculated_time_steps, features))
                else:
                    # Can't reshape without losing data, so create a dummy input
                    logger.warning(f"Cannot properly reshape input for {model_type} model")
                    X_test_reshaped = np.zeros((X_test.shape[0], time_steps, features))
            
            # Make predictions
            y_pred_proba = model.predict(X_test_reshaped, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate accuracy
            accuracy = np.mean(y_pred == y_test)
            
            # Calculate precision, recall, and F1 score
            true_positives = np.sum((y_pred == 1) & (y_test == 1))
            false_positives = np.sum((y_pred == 1) & (y_test == 0))
            false_negatives = np.sum((y_pred == 0) & (y_test == 1))
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate additional metrics
            model_score = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1_score)
            }
            
            performances[model_type] = model_score
            
            logger.info(f"{model_type} model performance: Accuracy={accuracy:.4f}, F1={f1_score:.4f}")
        
        except Exception as e:
            logger.error(f"Error evaluating {model_type} model: {e}")
    
    return performances

def calculate_ensemble_weights(performances: Dict) -> Dict:
    """Calculate optimal ensemble weights based on model performances"""
    weights = {}
    
    if not performances:
        logger.error("No model performances available for weight calculation")
        return weights
    
    # Calculate weights based on F1 scores
    f1_scores = {model_type: perf["f1_score"] for model_type, perf in performances.items()}
    
    # Filter out models with zero F1 score
    f1_scores = {model_type: score for model_type, score in f1_scores.items() if score > 0}
    
    if not f1_scores:
        logger.error("No models with positive F1 scores")
        return weights
    
    # Normalize weights
    total_score = sum(f1_scores.values())
    weights = {model_type: score / total_score for model_type, score in f1_scores.items()}
    
    logger.info(f"Calculated ensemble weights: {weights}")
    
    return weights

def create_ensemble_config(pair: str, weights: Dict, performances: Dict) -> Dict:
    """Create ensemble configuration"""
    ensemble_config = {
        "pair": pair,
        "weights": weights,
        "performances": performances,
        "model_types": list(weights.keys()),
        "created_at": pd.Timestamp.now().isoformat()
    }
    
    return ensemble_config

def save_ensemble_config(pair: str, ensemble_config: Dict):
    """Save ensemble configuration"""
    # Ensure ensemble directory exists
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    
    # Save ensemble weights
    weights_path = os.path.join(ENSEMBLE_DIR, f"{pair}_weights.json")
    with open(weights_path, 'w') as f:
        json.dump(ensemble_config["weights"], f, indent=4)
    
    # Save complete ensemble configuration
    ensemble_path = os.path.join(ENSEMBLE_DIR, f"{pair}_ensemble.json")
    with open(ensemble_path, 'w') as f:
        json.dump(ensemble_config, f, indent=4)
    
    logger.info(f"Saved ensemble configuration to {ensemble_path}")

def create_position_sizing_config(pair: str, performances: Dict):
    """Create position sizing configuration based on model performances"""
    # Ensure ensemble directory exists
    os.makedirs(ENSEMBLE_DIR, exist_ok=True)
    
    # Create position sizing configuration
    position_sizing = {
        "pair": pair,
        "base_confidence": 0.75,
        "model_confidences": {model_type: perf["f1_score"] for model_type, perf in performances.items()},
        "leverage_scaling": {
            "min_leverage": 1,
            "max_leverage": 5,
            "confidence_threshold": 0.8,
            "scaling_factor": 1.5
        },
        "position_sizing": {
            "base_size": 0.2,
            "min_size": 0.05,
            "max_size": 0.5,
            "confidence_scaling": True
        },
        "created_at": pd.Timestamp.now().isoformat()
    }
    
    # Save position sizing configuration
    position_sizing_path = os.path.join(ENSEMBLE_DIR, f"{pair}_position_sizing.json")
    with open(position_sizing_path, 'w') as f:
        json.dump(position_sizing, f, indent=4)
    
    logger.info(f"Saved position sizing configuration to {position_sizing_path}")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Load dataset
    df = load_dataset(args.pair, args.timeframe)
    if df is None:
        return 1
    
    # Prepare dataset
    X_train, X_test, y_train, y_test = prepare_dataset(df)
    if X_train is None:
        return 1
    
    # Determine model types to include
    model_types = args.model_types if args.model_types else DEFAULT_MODEL_TYPES
    logger.info(f"Creating ensemble with model types: {model_types}")
    
    # Load models
    models = load_models(args.pair, model_types)
    if not models:
        logger.error("No models available for ensemble creation")
        return 1
    
    logger.info(f"Loaded {len(models)} models for ensemble creation")
    
    # Evaluate models
    performances = evaluate_models(models, X_test, y_test)
    if not performances:
        logger.error("No model performances available")
        return 1
    
    # Calculate ensemble weights
    weights = calculate_ensemble_weights(performances)
    if not weights:
        logger.error("Failed to calculate ensemble weights")
        return 1
    
    # Create ensemble configuration
    ensemble_config = create_ensemble_config(args.pair, weights, performances)
    
    # Save ensemble configuration
    save_ensemble_config(args.pair, ensemble_config)
    
    # Create position sizing configuration
    create_position_sizing_config(args.pair, performances)
    
    logger.info("Ensemble model creation completed successfully")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())