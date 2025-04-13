#!/usr/bin/env python3
"""
Fix and debug ML-enhanced backtesting

This script is a focused version of run_enhanced_backtesting.py to debug
and fix issues with ML-enhanced strategy backtesting.
"""

import os
import sys
import logging
import traceback
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import enhanced backtesting system
try:
    from enhanced_backtesting import (
        EnhancedBacktester,
        load_historical_data
    )
    logger.info("Successfully imported enhanced_backtesting module")
except Exception as e:
    logger.error(f"Error importing enhanced_backtesting: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Import trading strategies
try:
    from trading_strategy import TradingStrategy
    from arima_strategy import ARIMAStrategy
    logger.info("Successfully imported base strategies")
except Exception as e:
    logger.error(f"Error importing base strategies: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Import ML-enhanced strategy
try:
    from ml_enhanced_strategy import MLEnhancedStrategy
    from ml_strategy_integrator import MLStrategyIntegrator
    logger.info("Successfully imported ML enhanced strategy modules")
except Exception as e:
    logger.error(f"Error importing ML enhanced strategy: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def main():
    """Main function"""
    # Parameters
    symbol = "SOLUSD"
    timeframe = "1h"
    capital = 20000
    
    # Load data
    logger.info(f"Loading historical data for {symbol} on {timeframe} timeframe")
    data = load_historical_data(symbol, timeframe)
    if data is None or len(data) == 0:
        logger.error("No historical data found")
        return
    
    logger.info(f"Loaded {len(data)} data points")
    
    # Create backtester
    logger.info("Creating backtester")
    backtester = EnhancedBacktester(
        data=data,
        trading_pair=symbol,
        initial_capital=capital,
        timeframe=timeframe,
        include_fees=True,
        enable_slippage=True
    )
    
    # Create base strategy
    logger.info("Creating base ARIMA strategy")
    base_strategy = ARIMAStrategy(symbol)
    
    # Run base strategy backtest as a reference
    logger.info("Running backtest with base strategy")
    base_result = {
        'total_return_pct': 0,
        'sharpe_ratio': 0,
        'max_drawdown_pct': 0,
        'win_rate': 0
    }  # Initialize with default values
    
    try:
        base_result = backtester.run_backtest({"arima": base_strategy})
        logger.info(f"Base strategy result: {base_result['total_return_pct']:.2f}%")
    except Exception as e:
        logger.error(f"Error in base strategy backtest: {e}")
        logger.error(traceback.format_exc())
    
    # Create ML-enhanced strategy - with debug output
    logger.info("Creating ML-enhanced strategy")
    try:
        # Create ML strategy integrator first (for debugging)
        ml_integrator = MLStrategyIntegrator(
            trading_pair=symbol,
            timeframe=timeframe,
            influence_weight=0.5,
            confidence_threshold=0.6
        )
        
        # Check if models are loaded
        loaded_models = ml_integrator.ensemble.get_loaded_models()
        logger.info(f"Loaded ML models: {loaded_models}")
        
        # Check ML predictions functionality
        sample_data = pd.DataFrame(data[:100])  # Use first 100 points as a sample
        
        # Add timestamps for ML processing
        sample_data['timestamp'] = [datetime.now() - timedelta(hours=i) for i in range(len(sample_data), 0, -1)]
        
        # Check ML prediction
        try:
            prediction, confidence, details = ml_integrator.get_ml_prediction(sample_data)
            logger.info(f"ML prediction test: prediction={prediction}, confidence={confidence}")
        except Exception as e:
            logger.error(f"Error testing ML prediction: {e}")
            logger.error(traceback.format_exc())
        
        # Now create ML-enhanced strategy
        ml_enhanced_strategy = MLEnhancedStrategy(
            trading_pair=symbol,
            base_strategy=base_strategy,
            timeframe=timeframe,
            ml_influence=0.5,
            confidence_threshold=0.6
        )
        
        logger.info("Successfully created ML-enhanced strategy")
    except Exception as e:
        logger.error(f"Error creating ML-enhanced strategy: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Add diagnostic test before running the full backtest
    logger.info("Running diagnostic test on ML prediction functionality")
    test_sample = data[-100:].copy()  # Use most recent 100 data points for testing
    
    try:
        # Prepare a detailed diagnostic report
        logger.info("====== DIAGNOSTIC ML PREDICTION TEST ======")
        
        # Log available columns for debugging
        logger.info(f"Available columns in test data: {list(test_sample.columns)}")
        
        # Get the ensemble model directly for detailed diagnostics
        ensemble = ml_enhanced_strategy.ml_integrator.ensemble
        loaded_models_info = ensemble.get_status()
        
        logger.info(f"Model count: {loaded_models_info['model_count']}")
        logger.info(f"Current market regime detected: {loaded_models_info['current_regime']}")
        
        # Get information about available preprocessed data
        preprocessed_data = ensemble._preprocess_data(test_sample)
        logger.info(f"Successfully preprocessed data for {len(preprocessed_data)}/{loaded_models_info['model_count']} models")
        
        # Log which models were successfully preprocessed
        if preprocessed_data:
            logger.info(f"Models with preprocessed data: {list(preprocessed_data.keys())}")
        
        # Test ML prediction directly
        prediction, confidence, details = ml_enhanced_strategy.ml_integrator.get_ml_prediction(test_sample)
        logger.info(f"Prediction result: {prediction:.4f} (confidence: {confidence:.4f})")
        
        # Log prediction details
        if details:
            if 'model_predictions' in details:
                logger.info(f"Individual model predictions: {details['model_predictions']}")
            
            if 'model_confidences' in details:
                logger.info(f"Individual model confidences: {details['model_confidences']}")
            
            if 'adjusted_weights' in details:
                logger.info(f"Adjusted weights: {details['adjusted_weights']}")
            
            if 'models_used' in details and 'models_loaded' in details:
                logger.info(f"Models used in prediction: {details['models_used']}/{details['models_loaded']}")
            
            if 'errors' in details and details['errors']:
                logger.warning(f"Errors encountered during prediction: {details['errors']}")
                
        logger.info("=========================================")
        
    except Exception as e:
        logger.error(f"Error in diagnostic ML prediction test: {e}")
        logger.error(traceback.format_exc())
    
    # Run ML-enhanced backtest
    logger.info("Running backtest with ML-enhanced strategy")
    ml_result = {
        'total_return_pct': 0,
        'sharpe_ratio': 0,
        'max_drawdown_pct': 0,
        'win_rate': 0
    }  # Initialize with default values
    
    try:
        ml_result = backtester.run_backtest({"ml_enhanced": ml_enhanced_strategy})
        logger.info(f"ML-enhanced strategy result: {ml_result['total_return_pct']:.2f}%")
        
        # Print comparative summary
        print("\nComparative Results:")
        print("-" * 60)
        print(f"{'Strategy':<20} {'Return %':<10} {'Sharpe':<10} {'Max DD %':<10} {'Win Rate':<10}")
        print("-" * 60)
        
        base_return = base_result['total_return_pct'] if 'total_return_pct' in base_result else 0
        base_sharpe = base_result['sharpe_ratio'] if 'sharpe_ratio' in base_result else 0
        base_dd = base_result['max_drawdown_pct'] if 'max_drawdown_pct' in base_result else 0
        base_wr = base_result['win_rate'] * 100 if 'win_rate' in base_result else 0
        
        ml_return = ml_result['total_return_pct'] if 'total_return_pct' in ml_result else 0
        ml_sharpe = ml_result['sharpe_ratio'] if 'sharpe_ratio' in ml_result else 0
        ml_dd = ml_result['max_drawdown_pct'] if 'max_drawdown_pct' in ml_result else 0
        ml_wr = ml_result['win_rate'] * 100 if 'win_rate' in ml_result else 0
        
        print(f"{'Base Strategy':<20} {base_return:<10.2f} {base_sharpe:<10.2f} {base_dd:<10.2f} {base_wr:<10.2f}")
        print(f"{'ML-Enhanced':<20} {ml_return:<10.2f} {ml_sharpe:<10.2f} {ml_dd:<10.2f} {ml_wr:<10.2f}")
        
        # Plot results
        backtester.plot_results(None)
        
    except Exception as e:
        logger.error(f"Error in ML-enhanced strategy backtest: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()