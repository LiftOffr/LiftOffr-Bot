#!/usr/bin/env python3
"""
Reshape Dataset Features Script

This script addresses the input shape mismatch between our models (expecting shape=(None, 60, 20))
and our dataset features (shape=(None, 1, 56)). It provides:

1. A dedicated function to reshape dataset features to the correct dimensions
2. A command-line tool to apply the reshaping to existing datasets
3. Options to either overwrite existing files or create new ones

Usage:
    python reshape_dataset_features.py --pair SOL/USD --input training_data/SOLUSD_1h_enhanced.csv
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('reshape_dataset.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
PANDAS_AVAILABLE = False
NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not available, some functionality will be limited")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not available, some functionality will be limited")

# Constants
TARGET_LOOKBACK = 60  # Number of time steps (sequence length)
TARGET_FEATURES = 20  # Number of features per time step

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Reshape dataset features for model compatibility')
    parser.add_argument('--pair', type=str, required=True,
                        help='Trading pair (e.g., SOL/USD)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input dataset file path (CSV)')
    parser.add_argument('--output', type=str,
                        help='Output file path (default: input file with "_reshaped" suffix)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite input file instead of creating a new one')
    parser.add_argument('--lookback', type=int, default=TARGET_LOOKBACK,
                        help=f'Sequence length (default: {TARGET_LOOKBACK})')
    parser.add_argument('--features', type=int, default=TARGET_FEATURES,
                        help=f'Number of features per time step (default: {TARGET_FEATURES})')
    parser.add_argument('--save-metadata', action='store_true',
                        help='Save feature metadata to JSON file')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    return parser.parse_args()

def load_dataset(file_path):
    """
    Load dataset from CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with dataset
    """
    logger.info(f"Loading dataset from {file_path}")
    
    if not PANDAS_AVAILABLE:
        logger.error("Pandas is required to load datasets")
        return None
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return None

def select_features(df, target_features=TARGET_FEATURES, excluded_cols=None):
    """
    Select features for reshaping.
    
    Args:
        df: DataFrame with dataset
        target_features: Number of features to select
        excluded_cols: List of columns to exclude from feature selection
        
    Returns:
        Tuple of (list of selected feature columns, list of excluded columns)
    """
    logger.info(f"Selecting {target_features} features from dataset")
    
    if not PANDAS_AVAILABLE:
        logger.error("Pandas is required for feature selection")
        return None, None
    
    try:
        # Default columns to exclude if not specified
        if excluded_cols is None:
            excluded_cols = [
                'timestamp', 'time', 'date', 'index',
                'open', 'high', 'low', 'volume', 
                'arima_forecast', 'adaptive_prediction', 
                'strategy_agreement', 'strategy_combined_strength',
                'arima_dominance', 'adaptive_dominance', 'dominant_strategy'
            ]
        
        # Add target columns to excluded columns
        target_cols = [col for col in df.columns if col.startswith('target_')]
        excluded_cols.extend(target_cols)
        
        # Get numerical columns that are not in excluded_cols
        feature_cols = []
        for col in df.columns:
            if col not in excluded_cols and df[col].dtype.kind in 'fc':
                feature_cols.append(col)
        
        # Sort features by standard deviation (more variance = more informative)
        if len(feature_cols) > target_features:
            std_values = df[feature_cols].std().to_dict()
            feature_cols.sort(key=lambda col: std_values[col], reverse=True)
            
            # Select top features
            selected_features = feature_cols[:target_features]
            excluded_features = feature_cols[target_features:]
            logger.info(f"Selected top {len(selected_features)} features by variance")
        else:
            selected_features = feature_cols
            excluded_features = []
            logger.info(f"Using all {len(selected_features)} available features")
        
        return selected_features, excluded_features + excluded_cols
    except Exception as e:
        logger.error(f"Error selecting features: {str(e)}")
        return None, None

def create_sequences(df, feature_cols, lookback=TARGET_LOOKBACK, target_col='target_direction_24'):
    """
    Create sequences for model input.
    
    Args:
        df: DataFrame with dataset
        feature_cols: List of feature column names
        lookback: Sequence length
        target_col: Target column name
        
    Returns:
        Tuple of (X, y)
    """
    logger.info(f"Creating sequences with lookback={lookback}")
    
    if not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
        logger.error("Pandas and NumPy are required for sequence creation")
        return None, None
    
    try:
        # Extract features and target
        X_raw = df[feature_cols].values
        
        # Check if target column exists
        if target_col not in df.columns:
            target_cols = [col for col in df.columns if col.startswith('target_')]
            if target_cols:
                target_col = target_cols[0]
                logger.warning(f"Target column {target_col} not found, using {target_col} instead")
            else:
                logger.error("No target column found in the dataset")
                return None, None
        
        y = df[target_col].values
        
        # Create sequences
        X_sequences = []
        y_targets = []
        
        for i in range(len(X_raw) - lookback):
            X_sequences.append(X_raw[i:i+lookback])
            y_targets.append(y[i+lookback])
        
        X = np.array(X_sequences)
        y_final = np.array(y_targets)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        return X, y_final
    except Exception as e:
        logger.error(f"Error creating sequences: {str(e)}")
        return None, None

def reshape_features(X, target_features=TARGET_FEATURES):
    """
    Reshape features to the target dimensions.
    
    Args:
        X: Input features array of shape (samples, time_steps, features)
        target_features: Number of features to reshape to
        
    Returns:
        Reshaped features array
    """
    logger.info(f"Reshaping features from {X.shape} to target {target_features} features")
    
    if not NUMPY_AVAILABLE:
        logger.error("NumPy is required for feature reshaping")
        return None
    
    try:
        n_samples, n_timesteps, n_features = X.shape
        
        if n_features == target_features:
            logger.info("Features already have the correct dimensions")
            return X
        
        # Reshape based on whether we have too many or too few features
        if n_features > target_features:
            # Keep only the first target_features features
            logger.info(f"Truncating from {n_features} to {target_features} features")
            X_reshaped = X[:, :, :target_features]
        else:
            # Pad with zeros to reach target_features
            logger.info(f"Padding from {n_features} to {target_features} features")
            X_reshaped = np.zeros((n_samples, n_timesteps, target_features))
            X_reshaped[:, :, :n_features] = X
        
        logger.info(f"Reshaped features to {X_reshaped.shape}")
        
        return X_reshaped
    except Exception as e:
        logger.error(f"Error reshaping features: {str(e)}")
        return None

def save_dataset(df, feature_cols, X_reshaped, y, output_path, save_metadata=False):
    """
    Save reshaped dataset to CSV file.
    
    Args:
        df: Original DataFrame
        feature_cols: Selected feature columns
        X_reshaped: Reshaped features array
        y: Target values array
        output_path: Output file path
        save_metadata: Whether to save feature metadata
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Saving reshaped dataset to {output_path}")
    
    if not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
        logger.error("Pandas and NumPy are required to save the dataset")
        return False
    
    try:
        # Get original index columns
        index_cols = []
        for col in df.columns:
            if col.lower() in ['timestamp', 'time', 'date', 'index']:
                index_cols.append(col)
        
        # Get target columns
        target_cols = [col for col in df.columns if col.startswith('target_')]
        
        # Create a new DataFrame
        n_samples = X_reshaped.shape[0]
        df_reshaped = pd.DataFrame()
        
        # Add index columns
        for col in index_cols:
            df_reshaped[col] = df[col].values[X_reshaped.shape[1]:X_reshaped.shape[1]+n_samples]
        
        # Add target columns
        for col in target_cols:
            df_reshaped[col] = df[col].values[X_reshaped.shape[1]:X_reshaped.shape[1]+n_samples]
        
        # Add reshaped features
        for i, feature in enumerate(feature_cols[:X_reshaped.shape[2]]):
            df_reshaped[feature] = X_reshaped[:, -1, i]  # Use last time step for feature values
        
        # Add metadata columns about the reshaping
        df_reshaped['reshaped'] = True
        df_reshaped['original_features'] = len(feature_cols)
        df_reshaped['reshaped_features'] = X_reshaped.shape[2]
        df_reshaped['lookback'] = X_reshaped.shape[1]
        
        # Save to CSV
        df_reshaped.to_csv(output_path, index=False)
        logger.info(f"Saved reshaped dataset with {len(df_reshaped)} rows to {output_path}")
        
        # Save metadata if requested
        if save_metadata:
            metadata_path = os.path.splitext(output_path)[0] + '_metadata.json'
            metadata = {
                'original_shape': list(X_reshaped.shape),
                'feature_columns': feature_cols[:X_reshaped.shape[2]],
                'reshaped_at': datetime.now().isoformat(),
                'lookback': int(X_reshaped.shape[1]),
                'features': int(X_reshaped.shape[2])
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved feature metadata to {metadata_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving dataset: {str(e)}")
        return False

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check dependencies
    if not PANDAS_AVAILABLE or not NUMPY_AVAILABLE:
        logger.error("Pandas and NumPy are required for this script")
        return 1
    
    # Determine output path
    if args.overwrite:
        output_path = args.input
    elif args.output:
        output_path = args.output
    else:
        # Add _reshaped suffix to input file name
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_reshaped{ext}"
    
    # Load dataset
    df = load_dataset(args.input)
    if df is None:
        return 1
    
    # Select features
    feature_cols, excluded_cols = select_features(df, args.features)
    if feature_cols is None:
        return 1
    
    # Create sequences
    X, y = create_sequences(df, feature_cols, args.lookback)
    if X is None or y is None:
        return 1
    
    # Reshape features
    X_reshaped = reshape_features(X, args.features)
    if X_reshaped is None:
        return 1
    
    # Save reshaped dataset
    if save_dataset(df, feature_cols, X_reshaped, y, output_path, args.save_metadata):
        logger.info("Dataset reshaping completed successfully")
    else:
        logger.error("Failed to save reshaped dataset")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())