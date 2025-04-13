#!/usr/bin/env python3
"""
Run ML Ensemble Backtesting

This script runs the ML backtesting with our fixed code for preprocessing
and normalization handling.

Usage:
    python run_ml_backtest.py [trading_pair] [timeframe]

Example:
    python run_ml_backtest.py SOL/USD 1h
"""

import os
import sys
import time
import logging
import argparse
import traceback
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("backtest_results/ml_ensemble_backtest.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Import our backtesting components
from backtest_ml_ensemble import MLEnsembleBacktester, run_backtest, run_all_backtests

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run ML Ensemble Backtesting')
    parser.add_argument('trading_pair', nargs='?', default='SOL/USD',
                       help='Trading pair to backtest (default: SOL/USD)')
    parser.add_argument('timeframe', nargs='?', default='1h',
                       help='Timeframe to backtest (default: 1h)')
    return parser.parse_args()

def ensure_directories():
    """Ensure necessary directories exist"""
    dirs = ['backtest_results', 'backtest_results/ml_ensemble']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            logger.info(f"Created directory: {d}")

def main():
    """Main entry point"""
    args = parse_args()
    ensure_directories()
    
    logger.info(f"Starting ML Ensemble Backtester for {args.trading_pair} on {args.timeframe} timeframe")
    start_time = time.time()
    
    try:
        # Run single backtest
        results = run_backtest(
            trading_pair=args.trading_pair,
            timeframe=args.timeframe,
            use_ensemble_pruning=True
        )
        
        if results:
            # Extract performance metrics
            metrics = results.get('performance', {}).get('trading_metrics', {})
            model_accuracies = results.get('performance', {}).get('model_accuracy', {})
            ensemble_accuracy = results.get('performance', {}).get('ensemble_accuracy', {})
            
            # Display summary results
            if metrics:
                logger.info(f"===== BACKTEST RESULTS FOR {args.trading_pair} ({args.timeframe}) =====")
                logger.info(f"Initial Capital: ${metrics.get('initial_capital', 0):.2f}")
                logger.info(f"Final Capital:   ${metrics.get('final_capital', 0):.2f}")
                logger.info(f"Total Return:    {metrics.get('total_return_pct', 0) * 100:.2f}% (${metrics.get('total_return', 0):.2f})")
                logger.info(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.2f}")
                logger.info(f"Max Drawdown:    {metrics.get('max_drawdown', 0) * 100:.2f}%")
                logger.info(f"Win Rate:        {metrics.get('win_rate', 0) * 100:.2f}% ({metrics.get('win_count', 0)}/{metrics.get('total_trades', 0)})")
            
            # Display model accuracies
            if ensemble_accuracy:
                logger.info(f"Ensemble Accuracy: {ensemble_accuracy.get('accuracy', 0) * 100:.2f}% ({ensemble_accuracy.get('correct', 0)}/{ensemble_accuracy.get('total', 0)})")
            
            if model_accuracies:
                logger.info("Individual Model Accuracies:")
                for model_type, acc in model_accuracies.items():
                    logger.info(f"  {model_type.upper()}: {acc.get('accuracy', 0) * 100:.2f}% ({acc.get('correct', 0)}/{acc.get('total', 0)})")
            
            # Save results to file
            result_file = f"backtest_results/ml_ensemble/{args.trading_pair.replace('/', '')}_backtest_{args.timeframe}.txt"
            with open(result_file, 'w') as f:
                f.write(f"===== BACKTEST RESULTS FOR {args.trading_pair} ({args.timeframe}) =====\n")
                f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Initial Capital: ${metrics.get('initial_capital', 0):.2f}\n")
                f.write(f"Final Capital:   ${metrics.get('final_capital', 0):.2f}\n")
                f.write(f"Total Return:    {metrics.get('total_return_pct', 0) * 100:.2f}% (${metrics.get('total_return', 0):.2f})\n")
                f.write(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.2f}\n")
                f.write(f"Max Drawdown:    {metrics.get('max_drawdown', 0) * 100:.2f}%\n")
                f.write(f"Win Rate:        {metrics.get('win_rate', 0) * 100:.2f}% ({metrics.get('win_count', 0)}/{metrics.get('total_trades', 0)})\n\n")
                
                if ensemble_accuracy:
                    f.write(f"Ensemble Accuracy: {ensemble_accuracy.get('accuracy', 0) * 100:.2f}% ({ensemble_accuracy.get('correct', 0)}/{ensemble_accuracy.get('total', 0)})\n")
                
                if model_accuracies:
                    f.write("Individual Model Accuracies:\n")
                    for model_type, acc in model_accuracies.items():
                        f.write(f"  {model_type.upper()}: {acc.get('accuracy', 0) * 100:.2f}% ({acc.get('correct', 0)}/{acc.get('total', 0)})\n")
            
            logger.info(f"Saved results to {result_file}")
        else:
            logger.error("Backtest returned no results")
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        logger.error(traceback.format_exc())
    
    elapsed_time = time.time() - start_time
    logger.info(f"Backtest completed in {elapsed_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())