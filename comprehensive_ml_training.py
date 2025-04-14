#!/usr/bin/env python3
"""
Comprehensive ML Training Script for Kraken Trading Bot

This script orchestrates the complete ML training pipeline:
1. Fetches extended historical data (365 days)
2. Prepares enhanced datasets with all features
3. Trains multiple ML model architectures with optimized parameters
4. Creates ensemble models with adaptive weighting
5. Evaluates model performance

Recommended to run with nohup for long-running training sessions:
nohup python comprehensive_ml_training.py > training.log 2>&1 &
"""

import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add root directory to sys.path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
PAIRS = ["SOLUSD", "BTCUSD", "ETHUSD", "ADAUSD", "DOTUSD", "LINKUSD", "MATICUSD"]
BASE_TIMEFRAMES = ["1h", "4h", "1d"]  # Primary timeframes to train on
MODEL_TYPES = ["lstm", "tcn", "gru", "bilstm", "attention", "transformer", "cnn", "hybrid"]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Comprehensive ML model training for trading bot')
    parser.add_argument('--fetch-data', action='store_true',
                        help='Fetch extended historical data (365 days)')
    parser.add_argument('--prepare-datasets', action='store_true',
                        help='Prepare enhanced datasets with all features')
    parser.add_argument('--days', type=int, default=365,
                        help='Number of days of historical data to fetch (default: 365)')
    parser.add_argument('--pairs', nargs='+', default=PAIRS,
                        help=f'Trading pairs to process (default: {", ".join(PAIRS)})')
    parser.add_argument('--timeframes', nargs='+', default=BASE_TIMEFRAMES,
                        help=f'Timeframes to train on (default: {", ".join(BASE_TIMEFRAMES)})')
    parser.add_argument('--models', nargs='+', default=MODEL_TYPES,
                        help=f'Model types to train (default: {", ".join(MODEL_TYPES)})')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate existing models without training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for model training (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for model training (default: 32)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size as fraction (default: 0.2)')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Validate models after training (default: True)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training if available')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization')
    parser.add_argument('--ensemble', action='store_true', default=True,
                        help='Create ensemble models (default: True)')
    return parser.parse_args()

def fetch_extended_data(pairs, days=365):
    """
    Fetch extended historical data for all specified pairs
    
    Args:
        pairs (list): List of trading pairs
        days (int): Number of days of historical data to fetch
    """
    logger.info(f"Fetching {days} days of historical data for {len(pairs)} pairs")
    
    try:
        # Import the fetch function to avoid circular imports
        from fetch_extended_historical_data import fetch_data_for_all_pairs
        
        # Fetch all timeframes for maximum flexibility
        fetch_data_for_all_pairs(pairs, list(["1h", "4h", "1d"]), days_back=days, merge=True)
        
        logger.info("Extended historical data fetch completed")
        return True
    except Exception as e:
        logger.error(f"Error fetching extended historical data: {e}")
        return False

def prepare_enhanced_datasets(pairs, timeframes):
    """
    Prepare enhanced datasets with all features for ML training
    
    Args:
        pairs (list): List of trading pairs
        timeframes (list): List of timeframes
    """
    logger.info(f"Preparing enhanced datasets for {len(pairs)} pairs")
    
    try:
        # Import the dataset preparation function
        from prepare_enhanced_dataset import prepare_dataset_for_pair
        
        successful_pairs = []
        
        for pair in pairs:
            logger.info(f"Preparing enhanced datasets for {pair}")
            for timeframe in timeframes:
                logger.info(f"Processing {timeframe} timeframe for {pair}")
                try:
                    # Prepare dataset with extended features
                    df = prepare_dataset_for_pair(pair, timeframe, min_samples=200)
                    if df is not None and not df.empty:
                        successful_pairs.append((pair, timeframe))
                        logger.info(f"Successfully prepared dataset for {pair} {timeframe}")
                    else:
                        logger.warning(f"Failed to prepare dataset for {pair} {timeframe} - not enough data")
                except Exception as e:
                    logger.error(f"Error preparing dataset for {pair} {timeframe}: {e}")
        
        logger.info(f"Enhanced datasets prepared for {len(successful_pairs)} pair/timeframe combinations")
        return successful_pairs
    except Exception as e:
        logger.error(f"Error preparing enhanced datasets: {e}")
        return []

def train_models(pairs, timeframes, model_types, args):
    """
    Train multiple ML model architectures for all specified pairs and timeframes
    
    Args:
        pairs (list): List of trading pairs
        timeframes (list): List of timeframes
        model_types (list): List of model types to train
        args (argparse.Namespace): Command line arguments
    """
    logger.info(f"Training {len(model_types)} model types for {len(pairs)} pairs")
    
    try:
        # Import training modules
        from advanced_ml_training import train_model
        from adaptive_hyperparameter_tuning import AdaptiveHyperparameterTuner
        
        successful_models = []
        
        for pair in pairs:
            for timeframe in timeframes:
                logger.info(f"Training models for {pair} {timeframe}")
                
                # Check if dataset exists
                dataset_file = f"training_data/{pair}_{timeframe}_enhanced.csv"
                if not os.path.exists(dataset_file):
                    logger.warning(f"Dataset file not found: {dataset_file}")
                    continue
                
                # Load dataset
                try:
                    df = pd.read_csv(dataset_file)
                    if len(df) < 500:
                        logger.warning(f"Dataset too small for {pair} {timeframe}: {len(df)} samples")
                        if len(df) < 200:
                            logger.error(f"Insufficient data for {pair} {timeframe}, skipping")
                            continue
                        else:
                            logger.warning(f"Proceeding with limited data for {pair} {timeframe}")
                    
                    logger.info(f"Dataset loaded for {pair} {timeframe}: {len(df)} samples")
                except Exception as e:
                    logger.error(f"Error loading dataset for {pair} {timeframe}: {e}")
                    continue
                
                # Train each model type
                for model_type in model_types:
                    logger.info(f"Training {model_type} model for {pair} {timeframe}")
                    
                    # Get optimized hyperparameters if requested
                    hyperparams = {}
                    if args.optimize:
                        try:
                            tuner = AdaptiveHyperparameterTuner(model_type, pair)
                            hyperparams = tuner.get_optimal_parameters()
                            logger.info(f"Using optimized hyperparameters for {model_type} {pair}")
                        except Exception as e:
                            logger.error(f"Error getting optimized hyperparameters: {e}")
                    
                    # Train the model
                    try:
                        success = train_model(
                            pair=pair,
                            timeframe=timeframe,
                            model_type=model_type,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            patience=args.patience,
                            test_size=args.test_size,
                            use_gpu=args.gpu,
                            hyperparams=hyperparams
                        )
                        
                        if success:
                            successful_models.append((pair, timeframe, model_type))
                            logger.info(f"Successfully trained {model_type} model for {pair} {timeframe}")
                        else:
                            logger.warning(f"Failed to train {model_type} model for {pair} {timeframe}")
                    except Exception as e:
                        logger.error(f"Error training {model_type} model for {pair} {timeframe}: {e}")
        
        logger.info(f"Model training completed: {len(successful_models)} models trained successfully")
        return successful_models
    except Exception as e:
        logger.error(f"Error in model training process: {e}")
        return []

def create_ensembles(pairs, timeframes, model_types):
    """
    Create ensemble models with adaptive weighting for all trained models
    
    Args:
        pairs (list): List of trading pairs
        timeframes (list): List of timeframes
        model_types (list): List of model types in the ensemble
    """
    logger.info(f"Creating ensemble models for {len(pairs)} pairs")
    
    try:
        # Import ensemble creation module
        from advanced_ensemble_model import DynamicWeightedEnsemble
        
        successful_ensembles = []
        
        for pair in pairs:
            logger.info(f"Creating ensemble for {pair}")
            
            # Prioritize certain timeframes for each pair
            primary_timeframe = "1h"
            if pair in ["BTCUSD", "ETHUSD"]:
                primary_timeframe = "4h"  # Less volatile pairs benefit from longer timeframes
            
            # Check if we have trained models for this pair
            all_timeframe_models = []
            for timeframe in timeframes:
                timeframe_models = []
                for model_type in model_types:
                    model_file = f"models/{model_type}/{pair}_{model_type}.h5"
                    if os.path.exists(model_file):
                        timeframe_models.append(model_type)
                
                if timeframe_models:
                    all_timeframe_models.append((timeframe, timeframe_models))
            
            if not all_timeframe_models:
                logger.warning(f"No trained models found for {pair}, skipping ensemble creation")
                continue
            
            # Create ensemble for primary timeframe
            try:
                ensemble = DynamicWeightedEnsemble(trading_pair=pair, timeframe=primary_timeframe)
                
                # Assign initial weights - start with equal distribution
                models_loaded = ensemble.get_loaded_models()
                if models_loaded and len(models_loaded["loaded_models"]) > 0:
                    logger.info(f"Creating ensemble with {len(models_loaded['loaded_models'])} models for {pair}")
                    
                    # Save ensemble configuration
                    ensemble_config = {
                        "models": {},
                        "parameters": {
                            "confidence_threshold": 0.65,
                            "voting_method": "weighted",
                            "trained_date": datetime.now().isoformat()
                        }
                    }
                    
                    # Set weights based on observed performance
                    # We prioritize Transformer, TCN, and LSTM based on their typical performance
                    weights = {}
                    total_weight = 0
                    
                    # Prioritize certain models
                    if "transformer" in models_loaded["loaded_models"]:
                        weights["transformer"] = 0.4
                        total_weight += 0.4
                    
                    if "tcn" in models_loaded["loaded_models"]:
                        weights["tcn"] = 0.35
                        total_weight += 0.35
                    
                    if "lstm" in models_loaded["loaded_models"]:
                        weights["lstm"] = 0.25
                        total_weight += 0.25
                    
                    # Distribute remaining weight evenly
                    remaining_models = [m for m in models_loaded["loaded_models"] 
                                        if m not in ["transformer", "tcn", "lstm"]]
                    remaining_weight = 1.0 - total_weight
                    
                    if remaining_models and remaining_weight > 0:
                        per_model_weight = remaining_weight / len(remaining_models)
                        for model in remaining_models:
                            weights[model] = per_model_weight
                    
                    # Normalize weights to ensure they sum to 1
                    weight_sum = sum(weights.values())
                    if weight_sum > 0:
                        weights = {k: v/weight_sum for k, v in weights.items()}
                    
                    # Update ensemble configuration
                    for model_type in weights:
                        ensemble_config["models"][model_type] = {
                            "path": f"models/{model_type}/{pair}_{model_type}.h5",
                            "weight": weights[model_type]
                        }
                    
                    # Save the updated weights
                    weights_file = f"models/ensemble/{pair}_weights.json"
                    with open(weights_file, 'w') as f:
                        json.dump(weights, f, indent=2)
                    
                    # Save ensemble configuration
                    ensemble_file = f"models/ensemble/{pair}_ensemble.json"
                    with open(ensemble_file, 'w') as f:
                        json.dump(ensemble_config, f, indent=2)
                    
                    # Create position sizing configuration
                    position_sizing_config = {
                        "max_leverage": 125,
                        "min_leverage": 20,
                        "confidence_scaling": {
                            "min_confidence": 0.5,
                            "max_confidence": 0.95
                        },
                        "regime_adjustments": {
                            "trending_up": 1.2,
                            "trending_down": 1.0,
                            "volatile": 1.5,
                            "sideways": 0.7,
                            "uncertain": 0.4
                        },
                        "risk_limits": {
                            "max_capital_allocation": 0.5,
                            "max_drawdown_percentage": 0.2,
                            "profit_taking_threshold": 0.1
                        },
                        "trained_date": datetime.now().isoformat()
                    }
                    
                    # Save position sizing configuration
                    position_sizing_file = f"models/ensemble/{pair}_position_sizing.json"
                    with open(position_sizing_file, 'w') as f:
                        json.dump(position_sizing_config, f, indent=2)
                    
                    successful_ensembles.append(pair)
                    logger.info(f"Successfully created ensemble for {pair}")
                else:
                    logger.warning(f"No models loaded for {pair} ensemble")
            except Exception as e:
                logger.error(f"Error creating ensemble for {pair}: {e}")
        
        logger.info(f"Ensemble creation completed: {len(successful_ensembles)} ensembles created")
        return successful_ensembles
    except Exception as e:
        logger.error(f"Error in ensemble creation process: {e}")
        return []

def validate_models(pairs, timeframes, model_types):
    """
    Validate trained models against test data
    
    Args:
        pairs (list): List of trading pairs
        timeframes (list): List of timeframes
        model_types (list): List of model types to validate
    """
    logger.info(f"Validating models for {len(pairs)} pairs")
    
    try:
        # Import validation modules
        from backtest_ml_ensemble import validate_model
        
        validation_results = {}
        
        for pair in pairs:
            pair_results = {}
            
            for timeframe in timeframes:
                timeframe_results = {}
                
                # Skip if dataset doesn't exist
                dataset_file = f"training_data/{pair}_{timeframe}_enhanced.csv"
                if not os.path.exists(dataset_file):
                    continue
                
                for model_type in model_types:
                    # Skip if model doesn't exist
                    model_file = f"models/{model_type}/{pair}_{model_type}.h5"
                    if not os.path.exists(model_file):
                        continue
                    
                    try:
                        # Validate individual model
                        logger.info(f"Validating {model_type} model for {pair} {timeframe}")
                        result = validate_model(pair, timeframe, model_type)
                        timeframe_results[model_type] = result
                        logger.info(f"Validation results for {model_type} {pair} {timeframe}: {result}")
                    except Exception as e:
                        logger.error(f"Error validating {model_type} model for {pair} {timeframe}: {e}")
                
                if timeframe_results:
                    pair_results[timeframe] = timeframe_results
            
            if pair_results:
                validation_results[pair] = pair_results
        
        # Save validation results
        with open("validation_results.json", 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info("Model validation completed")
        return validation_results
    except Exception as e:
        logger.error(f"Error in model validation process: {e}")
        return {}

def main():
    """Main function"""
    args = parse_arguments()
    
    start_time = time.time()
    logger.info("Starting comprehensive ML training pipeline")
    
    # Step 1: Fetch extended historical data if requested
    if args.fetch_data:
        success = fetch_extended_data(args.pairs, args.days)
        if not success:
            logger.error("Failed to fetch extended historical data, exiting")
            return
    
    # Step 2: Prepare enhanced datasets if requested
    if args.prepare_datasets:
        successful_pairs = prepare_enhanced_datasets(args.pairs, args.timeframes)
        if not successful_pairs:
            logger.error("Failed to prepare any datasets, exiting")
            return
    
    # Step 3: Train models if not eval_only
    if not args.eval_only:
        successful_models = train_models(args.pairs, args.timeframes, args.models, args)
        if not successful_models:
            logger.warning("No models were trained successfully")
    
    # Step 4: Create ensemble models if requested
    if args.ensemble:
        successful_ensembles = create_ensembles(args.pairs, args.timeframes, args.models)
        if not successful_ensembles:
            logger.warning("No ensemble models were created successfully")
    
    # Step 5: Validate models if requested
    if args.validate:
        validation_results = validate_models(args.pairs, args.timeframes, args.models)
        if not validation_results:
            logger.warning("No models were validated successfully")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Comprehensive ML training pipeline completed in {elapsed_time:.2f} seconds")
    logger.info("You can now run the trading bot with ML-enhanced strategies")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)
        sys.exit(1)