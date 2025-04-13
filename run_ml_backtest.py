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

def run_comprehensive_backtests():
    """Run backtests for all available trading pairs and timeframes"""
    ensure_directories()
    
    # Define trading pairs and timeframes to test
    trading_pairs = ['SOL/USD', 'BTC/USD', 'ETH/USD']
    timeframes = ['1h', '4h']
    
    all_results = {}
    overall_start_time = time.time()
    
    for pair in trading_pairs:
        pair_results = {}
        for timeframe in timeframes:
            symbol = pair.replace("/", "")
            logger.info(f"Starting backtest for {pair} on {timeframe} timeframe")
            start_time = time.time()
            
            try:
                # Check if data exists for this pair/timeframe combination
                data_file = f"./historical_data/{symbol.upper()}_{timeframe}.csv"
                if not os.path.exists(data_file):
                    logger.warning(f"No historical data available for {pair} ({timeframe}). Skipping.")
                    continue
                
                # Run backtest
                results = run_backtest(
                    trading_pair=pair,
                    timeframe=timeframe,
                    use_ensemble_pruning=True
                )
                
                if results:
                    pair_results[timeframe] = results
                    
                    # Log performance metrics
                    metrics = results.get('performance', {}).get('trading_metrics', {})
                    model_accuracies = results.get('performance', {}).get('model_accuracy', {})
                    ensemble_accuracy = results.get('performance', {}).get('ensemble_accuracy', {})
                    
                    if metrics:
                        logger.info(f"===== BACKTEST RESULTS FOR {pair} ({timeframe}) =====")
                        logger.info(f"Initial Capital: ${metrics.get('initial_capital', 0):.2f}")
                        logger.info(f"Final Capital:   ${metrics.get('final_capital', 0):.2f}")
                        logger.info(f"Total Return:    {metrics.get('total_return_pct', 0) * 100:.2f}% (${metrics.get('total_return', 0):.2f})")
                        logger.info(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.2f}")
                        logger.info(f"Max Drawdown:    {metrics.get('max_drawdown', 0) * 100:.2f}%")
                        logger.info(f"Win Rate:        {metrics.get('win_rate', 0) * 100:.2f}% ({metrics.get('win_count', 0)}/{metrics.get('total_trades', 0)})")
                    
                    # Save results to file
                    result_file = f"backtest_results/ml_ensemble/{symbol}_backtest_{timeframe}.txt"
                    with open(result_file, 'w') as f:
                        f.write(f"===== BACKTEST RESULTS FOR {pair} ({timeframe}) =====\n")
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
                    logger.error(f"Backtest for {pair} ({timeframe}) returned no results")
            
            except Exception as e:
                logger.error(f"Error running backtest for {pair} ({timeframe}): {e}")
                logger.error(traceback.format_exc())
            
            elapsed_time = time.time() - start_time
            logger.info(f"Backtest for {pair} ({timeframe}) completed in {elapsed_time:.2f} seconds")
        
        all_results[pair] = pair_results
    
    # Generate comparative summary
    generate_comparative_summary(all_results)
    
    total_elapsed_time = time.time() - overall_start_time
    logger.info(f"All backtests completed in {total_elapsed_time:.2f} seconds")
    
    return all_results

def generate_comparative_summary(all_results):
    """Generate a summary comparing performance across pairs and timeframes"""
    logger.info("Generating comparative summary...")
    
    summary_file = "backtest_results/ml_ensemble/comparative_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("===== ML ENSEMBLE BACKTEST COMPARATIVE SUMMARY =====\n")
        f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Performance by Trading Pair and Timeframe:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Pair':<10} {'Timeframe':<10} {'Return %':<10} {'Sharpe':<10} {'Drawdown':<10} {'Win Rate':<10} {'Ensemble Acc':<15}\n")
        f.write("-" * 80 + "\n")
        
        for pair, pair_results in all_results.items():
            for timeframe, results in pair_results.items():
                metrics = results.get('performance', {}).get('trading_metrics', {})
                ensemble_acc = results.get('performance', {}).get('ensemble_accuracy', {})
                
                return_pct = metrics.get('total_return_pct', 0) * 100
                sharpe = metrics.get('sharpe_ratio', 0)
                drawdown = metrics.get('max_drawdown', 0) * 100
                win_rate = metrics.get('win_rate', 0) * 100
                ensemble_accuracy = ensemble_acc.get('accuracy', 0) * 100 if ensemble_acc else 0
                
                f.write(f"{pair:<10} {timeframe:<10} {return_pct:<10.2f} {sharpe:<10.2f} {drawdown:<10.2f} {win_rate:<10.2f} {ensemble_accuracy:<15.2f}\n")
        
        f.write("-" * 80 + "\n\n")
        
        # Add model performance comparison
        f.write("Model Accuracy Comparison Across All Tests:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<15} {'Avg Accuracy':<15} {'Best Pair/TF':<15} {'Best Accuracy':<15}\n")
        f.write("-" * 80 + "\n")
        
        model_performances = {}
        
        for pair, pair_results in all_results.items():
            for timeframe, results in pair_results.items():
                model_accuracies = results.get('performance', {}).get('model_accuracy', {})
                
                for model_type, acc in model_accuracies.items():
                    accuracy = acc.get('accuracy', 0) * 100
                    
                    if model_type not in model_performances:
                        model_performances[model_type] = {
                            'accuracies': [],
                            'best_accuracy': 0,
                            'best_pair_tf': ''
                        }
                    
                    model_performances[model_type]['accuracies'].append(accuracy)
                    
                    if accuracy > model_performances[model_type]['best_accuracy']:
                        model_performances[model_type]['best_accuracy'] = accuracy
                        model_performances[model_type]['best_pair_tf'] = f"{pair}/{timeframe}"
        
        for model_type, perf in model_performances.items():
            avg_accuracy = sum(perf['accuracies']) / len(perf['accuracies']) if perf['accuracies'] else 0
            f.write(f"{model_type:<15} {avg_accuracy:<15.2f} {perf['best_pair_tf']:<15} {perf['best_accuracy']:<15.2f}\n")
    
    logger.info(f"Comparative summary saved to {summary_file}")

def main():
    """Main entry point"""
    args = parse_args()
    ensure_directories()
    
    # Check if we should run comprehensive tests or a single test
    if args.trading_pair == 'ALL' or args.timeframe == 'ALL':
        logger.info("Starting comprehensive ML Ensemble Backtests across all pairs and timeframes")
        results = run_comprehensive_backtests()
    else:
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