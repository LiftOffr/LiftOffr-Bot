#!/usr/bin/env python3
"""
Test script for ML backtesting enhancements

This script tests the fixes we've made to the ML backtesting system:
1. Improved feature standardization
2. Enhanced data preprocessing
3. Better error handling for model compatibility
"""

import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Import our backtesting components
from backtest_ml_ensemble import MLEnsembleBacktester
from advanced_ensemble_model import DynamicWeightedEnsemble

def load_test_data(trading_pair='SOL/USD', timeframe='1h'):
    """Load historical data for testing"""
    try:
        # Define the file path
        symbol = trading_pair.replace("/", "")
        # Check both uppercase and lowercase file names
        data_path = f"./historical_data/{symbol.upper()}_{timeframe}.csv"
        if not os.path.exists(data_path):
            # Try lowercase version
            data_path = f"./historical_data/{symbol.lower()}_{timeframe}.csv"
        
        if not os.path.exists(data_path):
            logger.error(f"Test data file not found: {data_path}")
            return None
            
        # Load the CSV data
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Display information about the data
        logger.info(f"Loaded {len(df)} data points for {trading_pair} ({timeframe})")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Columns available: {list(df.columns)}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        logger.error(traceback.format_exc())
        return None

def test_feature_standardization():
    """Test feature standardization in _prepare_prediction_window"""
    logger.info("Starting feature standardization test...")
    
    # Load test data
    data = load_test_data()
    if data is None:
        logger.error("Cannot run test without test data")
        return False
    
    # Create a backtester
    backtester = MLEnsembleBacktester('SOL/USD', timeframe='1h')
    
    # Manually assign the data
    backtester.data = data
    
    # Test the _prepare_prediction_window method
    window_size = 60
    if len(data) > window_size:
        window_data = backtester._prepare_prediction_window(window_size, window_size)
        
        if window_data is not None:
            logger.info(f"Window data shape: {window_data.shape}")
            logger.info(f"Window data columns: {list(window_data.columns)}")
            
            # Check if the necessary features were created
            required_features = ['open', 'high', 'low', 'close', 'volume', 
                                'ema9', 'ema20', 'ema21', 'ema50', 'ema100', 
                                'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']
            
            missing_features = [f for f in required_features if f not in window_data.columns]
            
            if missing_features:
                logger.error(f"Missing features in window data: {missing_features}")
                return False
            else:
                logger.info("All required features present in window data")
                return True
        else:
            logger.error("Window data is None")
            return False
    else:
        logger.error(f"Not enough data for testing, need at least {window_size+1} rows")
        return False

def test_ensemble_prediction():
    """Test the ensemble prediction with preprocessed data"""
    logger.info("Starting ensemble prediction test...")
    
    # Load test data
    data = load_test_data()
    if data is None:
        logger.error("Cannot run test without test data")
        return False
    
    # Create an ensemble model
    ensemble = DynamicWeightedEnsemble(trading_pair='SOL/USD', timeframe='1h')
    
    # Create a backtester
    backtester = MLEnsembleBacktester('SOL/USD', timeframe='1h')
    
    # Manually assign the data
    backtester.data = data
    
    # Track model loading status
    loaded_models = ensemble.get_loaded_models()
    if not loaded_models['models']:
        logger.warning("No models were loaded, cannot test prediction")
        return False
    
    # Test the preprocessing and prediction
    window_size = 60
    if len(data) > window_size:
        # Prepare window data
        window_data = backtester._prepare_prediction_window(window_size, window_size)
        
        if window_data is not None:
            logger.info(f"Prepared window data with shape: {window_data.shape}")
            
            try:
                # Try to make a prediction
                prediction, confidence, details = ensemble.predict(window_data)
                
                logger.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
                logger.info(f"Models used: {list(details['model_predictions'].keys())}")
                
                # Print model predictions
                for model_type, pred in details['model_predictions'].items():
                    logger.info(f"{model_type}: {pred}")
                
                return True
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                logger.error(traceback.format_exc())
                return False
        else:
            logger.error("Window data is None")
            return False
    else:
        logger.error(f"Not enough data for testing, need at least {window_size+1} rows")
        return False

def main():
    """Run all tests"""
    logger.info("Starting ML backtesting tests...")
    
    # Test feature standardization
    feature_test_result = test_feature_standardization()
    logger.info(f"Feature standardization test {'PASSED' if feature_test_result else 'FAILED'}")
    
    # Test ensemble prediction
    prediction_test_result = test_ensemble_prediction()
    logger.info(f"Ensemble prediction test {'PASSED' if prediction_test_result else 'FAILED'}")
    
    # Overall result
    if feature_test_result and prediction_test_result:
        logger.info("All tests PASSED")
        return 0
    else:
        logger.error("Some tests FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())