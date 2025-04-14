#!/usr/bin/env python3
"""
Automated ML Model Retraining Scheduler

This script schedules regular retraining of ML models to maintain high accuracy:
1. Sets up a recurring schedule for model retraining (daily, weekly, etc.)
2. Fetches the latest historical data before retraining
3. Performs model optimization and retraining
4. Updates the existing models with new optimized versions
5. Logs performance metrics for tracking improvements
6. Can be run as a cron job or standalone service

Usage:
    python auto_retraining_scheduler.py [--interval daily/weekly] [--time 00:00] [--pairs ALL]
"""

import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('auto_retraining.log')
    ]
)

logger = logging.getLogger(__name__)

# Default values
DEFAULT_INTERVAL = 'daily'
DEFAULT_TIME = '00:00'
DEFAULT_PAIRS = 'ALL'
DEFAULT_RETRAINING_WINDOW = 30  # days

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Automated ML Model Retraining Scheduler')
    
    parser.add_argument(
        '--interval',
        type=str,
        choices=['hourly', 'daily', 'weekly', 'monthly'],
        default=DEFAULT_INTERVAL,
        help=f'Retraining interval (default: {DEFAULT_INTERVAL})'
    )
    
    parser.add_argument(
        '--time',
        type=str,
        default=DEFAULT_TIME,
        help=f'Time for scheduled retraining in HH:MM format (default: {DEFAULT_TIME})'
    )
    
    parser.add_argument(
        '--pairs',
        type=str,
        default=DEFAULT_PAIRS,
        help=f'Trading pairs to retrain (comma-separated or ALL) (default: {DEFAULT_PAIRS})'
    )
    
    parser.add_argument(
        '--window',
        type=int,
        default=DEFAULT_RETRAINING_WINDOW,
        help=f'Data window in days for retraining (default: {DEFAULT_RETRAINING_WINDOW})'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force immediate retraining regardless of schedule'
    )
    
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Skip hyperparameter optimization'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=20,
        help='Number of optimization trials (default: 20)'
    )
    
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as a daemon process'
    )
    
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip fetching new historical data'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()

def run_command(cmd: List[str], description: str = None) -> Optional[subprocess.CompletedProcess]:
    """
    Run a shell command and log output.
    
    Args:
        cmd: List of command and arguments
        description: Description of the command
        
    Returns:
        CompletedProcess if successful, None if failed
    """
    if description:
        logger.info(f"Running: {description}")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        process = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Log stdout if available
        if process.stdout:
            for line in process.stdout.strip().split('\n'):
                if line:
                    logger.info(f"Output: {line}")
        
        return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        
        # Log stderr if available
        if e.stderr:
            for line in e.stderr.strip().split('\n'):
                if line:
                    logger.error(f"Error: {line}")
        
        return None

def check_previous_retraining() -> Dict[str, Any]:
    """
    Check previous retraining timestamp from log file.
    
    Returns:
        Dictionary with previous retraining information
    """
    retraining_log_path = 'retraining_history.json'
    
    if os.path.exists(retraining_log_path):
        try:
            with open(retraining_log_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {retraining_log_path}")
    
    return {
        'last_retraining': None,
        'pairs': {},
        'metrics': {}
    }

def save_retraining_history(history: Dict[str, Any]) -> bool:
    """
    Save retraining history to file.
    
    Args:
        history: Dictionary with retraining history
        
    Returns:
        True if successful, False otherwise
    """
    retraining_log_path = 'retraining_history.json'
    
    try:
        with open(retraining_log_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Retraining history saved to {retraining_log_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save retraining history: {str(e)}")
        return False

def get_all_trading_pairs() -> List[str]:
    """
    Get all available trading pairs from configuration.
    
    Returns:
        List of trading pairs
    """
    # Try to get pairs from ML config
    if os.path.exists('ml_config.json'):
        try:
            with open('ml_config.json', 'r') as f:
                config = json.load(f)
                
                if 'trading_pairs' in config:
                    return config['trading_pairs']
        except json.JSONDecodeError:
            logger.error("Invalid JSON in ml_config.json")
    
    # Default pairs if config not available
    return ['SOL/USD', 'BTC/USD', 'ETH/USD', 'DOT/USD']

def fetch_historical_data(pairs: List[str], window: int = 30) -> bool:
    """
    Fetch latest historical data for the specified pairs.
    
    Args:
        pairs: List of trading pairs
        window: Number of days to fetch
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Fetching historical data for {len(pairs)} pairs...")
    
    # Check if fetch_extended_historical_data.py exists
    if os.path.exists("fetch_extended_historical_data.py"):
        cmd = [
            "python",
            "fetch_extended_historical_data.py",
            "--pairs", ",".join(pairs),
            "--timeframe", "1h",
            "--days", str(window)
        ]
        
        result = run_command(cmd, f"Fetching {window} days of historical data for {len(pairs)} pairs")
        if result is not None:
            return True
    
    # Fall back to historical_data_fetcher.py
    elif os.path.exists("historical_data_fetcher.py"):
        all_successful = True
        
        for pair in pairs:
            cmd = [
                "python",
                "historical_data_fetcher.py",
                "--pair", pair,
                "--timeframe", "1h",
                "--days", str(window)
            ]
            
            result = run_command(cmd, f"Fetching historical data for {pair}")
            if result is None:
                logger.error(f"Failed to fetch historical data for {pair}")
                all_successful = False
        
        return all_successful
    
    # No suitable script found
    else:
        logger.error("No suitable script found for fetching historical data")
        return False

def prepare_datasets(pairs: List[str]) -> bool:
    """
    Prepare datasets for ML training.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Preparing datasets for {len(pairs)} pairs...")
    
    # Check if prepare_all_datasets.py exists
    if os.path.exists("prepare_all_datasets.py"):
        cmd = [
            "python",
            "prepare_all_datasets.py",
            "--pairs", ",".join(pairs),
            "--timeframe", "1h"
        ]
        
        result = run_command(cmd, f"Preparing datasets for {len(pairs)} pairs")
        if result is not None:
            return True
    
    # Fall back to prepare_enhanced_dataset.py
    elif os.path.exists("prepare_enhanced_dataset.py"):
        all_successful = True
        
        for pair in pairs:
            cmd = [
                "python",
                "prepare_enhanced_dataset.py",
                "--pair", pair,
                "--timeframe", "1h"
            ]
            
            result = run_command(cmd, f"Preparing enhanced dataset for {pair}")
            if result is None:
                logger.error(f"Failed to prepare dataset for {pair}")
                all_successful = False
        
        return all_successful
    
    # No suitable script found
    else:
        logger.error("No suitable script found for preparing datasets")
        return False

def optimize_and_train_models(pairs: List[str], optimize: bool = True, trials: int = 20) -> bool:
    """
    Optimize hyperparameters and train ML models.
    
    Args:
        pairs: List of trading pairs
        optimize: Whether to optimize hyperparameters
        trials: Number of optimization trials
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Optimizing and training models for {len(pairs)} pairs...")
    
    if optimize and os.path.exists("optimize_ml_hyperparameters.py"):
        cmd = [
            "python",
            "optimize_ml_hyperparameters.py",
            "--pairs", ",".join(pairs),
            "--epochs", "200",
            "--trials", str(trials)
        ]
        
        result = run_command(cmd, f"Optimizing ML hyperparameters for {len(pairs)} pairs")
        if result is None:
            logger.error(f"Failed to optimize ML hyperparameters")
            return False
    
    # Fall back to enhance_ml_model_accuracy.py
    elif os.path.exists("enhance_ml_model_accuracy.py"):
        cmd = [
            "python",
            "enhance_ml_model_accuracy.py",
            "--pairs", ",".join(pairs),
            "--epochs", "100"
        ]
        
        result = run_command(cmd, f"Enhancing ML model accuracy for {len(pairs)} pairs")
        if result is None:
            logger.error(f"Failed to enhance ML model accuracy")
            return False
    
    # Fall back to basic training if optimization fails or is skipped
    elif os.path.exists("advanced_ml_training.py"):
        cmd = [
            "python",
            "advanced_ml_training.py",
            "--pairs", ",".join(pairs)
        ]
        
        result = run_command(cmd, f"Training advanced ML models for {len(pairs)} pairs")
        if result is None:
            logger.error(f"Failed to train advanced ML models")
            return False
    
    # No suitable script found
    else:
        logger.error("No suitable script found for optimizing and training models")
        return False
    
    return True

def create_ensemble_models(pairs: List[str]) -> bool:
    """
    Create ensemble models for each pair.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Creating ensemble models for {len(pairs)} pairs...")
    
    # Check if create_ensemble_model.py exists
    if os.path.exists("create_ensemble_model.py"):
        all_successful = True
        
        for pair in pairs:
            cmd = [
                "python",
                "create_ensemble_model.py",
                "--pair", pair
            ]
            
            result = run_command(cmd, f"Creating ensemble model for {pair}")
            if result is None:
                # Try alternative argument format
                cmd = [
                    "python",
                    "create_ensemble_model.py",
                    "--trading-pair", pair
                ]
                
                result = run_command(cmd, f"Creating ensemble model for {pair} (alternative format)")
                if result is None:
                    logger.error(f"Failed to create ensemble model for {pair}")
                    all_successful = False
        
        return all_successful
    
    # No suitable script found
    else:
        logger.error("create_ensemble_model.py not found")
        return False

def evaluate_model_performance(pairs: List[str]) -> Dict[str, Any]:
    """
    Evaluate model performance for the specified pairs.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        Dictionary with performance metrics
    """
    logger.info(f"Evaluating model performance for {len(pairs)} pairs...")
    
    metrics = {}
    
    # Check if evaluate_model_performance.py exists
    if os.path.exists("evaluate_model_performance.py"):
        for pair in pairs:
            cmd = [
                "python",
                "evaluate_model_performance.py",
                "--pair", pair,
                "--json-output"
            ]
            
            result = run_command(cmd, f"Evaluating model performance for {pair}")
            if result is not None and result.stdout:
                try:
                    pair_metrics = json.loads(result.stdout)
                    metrics[pair] = pair_metrics
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON output for {pair}")
    
    # Fall back to manual evaluation
    elif os.path.exists("enhanced_backtesting.py"):
        for pair in pairs:
            cmd = [
                "python",
                "enhanced_backtesting.py",
                "--pair", pair,
                "--ml-only",
                "--json-output"
            ]
            
            result = run_command(cmd, f"Backtesting ML model for {pair}")
            if result is not None and result.stdout:
                try:
                    pair_metrics = json.loads(result.stdout)
                    metrics[pair] = pair_metrics
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON output for {pair}")
    
    return metrics

def backup_existing_models(pairs: List[str]) -> bool:
    """
    Backup existing models before updating.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Backing up existing models for {len(pairs)} pairs...")
    
    backup_dir = f"models_backup_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    model_dirs = ['models/lstm', 'models/ensemble', 'models/cnn', 'models/transformer', 'models/attention']
    all_successful = True
    
    for pair in pairs:
        pair_path = pair.replace('/', '')
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                # Check for model files
                model_files = [
                    f"{pair_path}.h5",
                    f"{pair_path}.json",
                    f"{pair_path}_ensemble.json",
                    f"{pair_path}_weights.json"
                ]
                
                for model_file in model_files:
                    model_path = os.path.join(model_dir, model_file)
                    
                    if os.path.exists(model_path):
                        backup_subdir = os.path.join(backup_dir, os.path.basename(model_dir))
                        os.makedirs(backup_subdir, exist_ok=True)
                        
                        backup_path = os.path.join(backup_subdir, model_file)
                        
                        try:
                            import shutil
                            shutil.copy2(model_path, backup_path)
                            logger.info(f"Backed up {model_path} to {backup_path}")
                        except Exception as e:
                            logger.error(f"Failed to backup {model_path}: {str(e)}")
                            all_successful = False
    
    if all_successful:
        logger.info(f"Successfully backed up all models to {backup_dir}")
    else:
        logger.warning(f"Some model backups failed, but continuing with retraining")
    
    return True

def is_retraining_due(retraining_history: Dict[str, Any], interval: str, scheduled_time: str) -> bool:
    """
    Check if retraining is due based on schedule.
    
    Args:
        retraining_history: Dictionary with retraining history
        interval: Retraining interval (hourly, daily, weekly, monthly)
        scheduled_time: Scheduled time for retraining (HH:MM)
        
    Returns:
        True if retraining is due, False otherwise
    """
    if retraining_history.get('last_retraining') is None:
        logger.info("No previous retraining found, retraining is due")
        return True
    
    last_retraining = datetime.datetime.fromisoformat(retraining_history['last_retraining'])
    now = datetime.datetime.now()
    
    # Parse scheduled time
    hour, minute = map(int, scheduled_time.split(':'))
    scheduled_datetime = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    # If current time is before scheduled time, adjust scheduled_datetime to yesterday
    if now.time() < scheduled_datetime.time():
        scheduled_datetime = scheduled_datetime - datetime.timedelta(days=1)
    
    if interval == 'hourly':
        # Check if an hour has passed since last retraining
        return (now - last_retraining).total_seconds() >= 3600
    
    elif interval == 'daily':
        # Check if scheduled time has passed since last retraining
        return last_retraining < scheduled_datetime
    
    elif interval == 'weekly':
        # Check if a week has passed and scheduled time has passed
        week_ago = now - datetime.timedelta(days=7)
        return last_retraining < week_ago and last_retraining < scheduled_datetime
    
    elif interval == 'monthly':
        # Check if a month has passed and scheduled time has passed
        # Approximate a month as 30 days
        month_ago = now - datetime.timedelta(days=30)
        return last_retraining < month_ago and last_retraining < scheduled_datetime
    
    return False

def activate_retrained_models(pairs: List[str]) -> bool:
    """
    Activate the retrained models for trading.
    
    Args:
        pairs: List of trading pairs
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Activating retrained models for {len(pairs)} pairs...")
    
    # Check if quick_activate_ml.py exists
    if os.path.exists("quick_activate_ml.py"):
        cmd = [
            "python",
            "quick_activate_ml.py",
            "--pairs", ",".join(pairs),
            "--sandbox"  # Always use sandbox for safety
        ]
        
        result = run_command(cmd, f"Activating ML trading for {len(pairs)} pairs")
        if result is not None:
            return True
    
    # Fall back to activate_ml_with_ensembles.py
    elif os.path.exists("activate_ml_with_ensembles.py"):
        cmd = [
            "python",
            "activate_ml_with_ensembles.py",
            "--pairs", ",".join(pairs),
            "--sandbox"  # Always use sandbox for safety
        ]
        
        result = run_command(cmd, f"Activating ML trading with ensembles for {len(pairs)} pairs")
        if result is not None:
            return True
    
    # No suitable script found
    else:
        logger.error("No suitable script found for activating models")
        return False

def perform_retraining(pairs: List[str], optimize: bool = True, trials: int = 20, window: int = 30, skip_fetch: bool = False) -> Dict[str, Any]:
    """
    Perform the complete retraining process.
    
    Args:
        pairs: List of trading pairs
        optimize: Whether to optimize hyperparameters
        trials: Number of optimization trials
        window: Data window in days for retraining
        skip_fetch: Skip fetching new historical data
        
    Returns:
        Dictionary with retraining results
    """
    logger.info("Starting ML model retraining process...")
    
    # Initialize retraining results
    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'pairs': pairs,
        'success': False,
        'metrics': {},
        'steps_completed': []
    }
    
    # Step 1: Backup existing models
    if not backup_existing_models(pairs):
        logger.error("Failed to backup existing models")
        return results
    
    results['steps_completed'].append('backup')
    
    # Step 2: Fetch latest historical data
    if not skip_fetch:
        if not fetch_historical_data(pairs, window):
            logger.error("Failed to fetch historical data")
            return results
        
        results['steps_completed'].append('fetch_data')
    else:
        logger.info("Skipping historical data fetch as requested")
    
    # Step 3: Prepare datasets
    if not prepare_datasets(pairs):
        logger.error("Failed to prepare datasets")
        return results
    
    results['steps_completed'].append('prepare_datasets')
    
    # Step 4: Optimize and train models
    if not optimize_and_train_models(pairs, optimize, trials):
        logger.error("Failed to optimize and train models")
        return results
    
    results['steps_completed'].append('train_models')
    
    # Step 5: Create ensemble models
    if not create_ensemble_models(pairs):
        logger.warning("Failed to create ensemble models, but continuing")
    else:
        results['steps_completed'].append('create_ensembles')
    
    # Step 6: Evaluate model performance
    metrics = evaluate_model_performance(pairs)
    results['metrics'] = metrics
    results['steps_completed'].append('evaluate')
    
    # Step 7: Activate retrained models
    if not activate_retrained_models(pairs):
        logger.error("Failed to activate retrained models")
        return results
    
    results['steps_completed'].append('activate')
    
    # Mark retraining as successful
    results['success'] = True
    logger.info("Retraining process completed successfully")
    
    return results

def run_scheduler(args):
    """
    Run the retraining scheduler.
    
    Args:
        args: Command line arguments
    """
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Get pairs to retrain
    if args.pairs == 'ALL':
        pairs = get_all_trading_pairs()
        logger.info(f"Using all available trading pairs: {', '.join(pairs)}")
    else:
        pairs = [pair.strip() for pair in args.pairs.split(',')]
        logger.info(f"Using specified trading pairs: {', '.join(pairs)}")
    
    # Load retraining history
    retraining_history = check_previous_retraining()
    
    # Check if immediate retraining is requested
    if args.force:
        logger.info("Forced retraining requested, starting immediately")
        
        # Perform retraining
        results = perform_retraining(
            pairs,
            optimize=not args.no_optimize,
            trials=args.trials,
            window=args.window,
            skip_fetch=args.skip_fetch
        )
        
        # Update retraining history
        retraining_history['last_retraining'] = results['timestamp']
        retraining_history['pairs'].update({pair: results['timestamp'] for pair in pairs})
        
        if results['metrics']:
            for pair, metrics in results['metrics'].items():
                if pair not in retraining_history['metrics']:
                    retraining_history['metrics'][pair] = []
                
                retraining_history['metrics'][pair].append({
                    'timestamp': results['timestamp'],
                    'metrics': metrics
                })
        
        # Save updated history
        save_retraining_history(retraining_history)
        
        return 0
    
    # Run as daemon if requested
    if args.daemon:
        logger.info(f"Running as daemon with {args.interval} retraining at {args.time}")
        
        while True:
            # Check if retraining is due
            if is_retraining_due(retraining_history, args.interval, args.time):
                logger.info(f"Retraining is due, starting {args.interval} retraining")
                
                # Perform retraining
                results = perform_retraining(
                    pairs,
                    optimize=not args.no_optimize,
                    trials=args.trials,
                    window=args.window,
                    skip_fetch=args.skip_fetch
                )
                
                # Update retraining history
                retraining_history['last_retraining'] = results['timestamp']
                retraining_history['pairs'].update({pair: results['timestamp'] for pair in pairs})
                
                if results['metrics']:
                    for pair, metrics in results['metrics'].items():
                        if pair not in retraining_history['metrics']:
                            retraining_history['metrics'][pair] = []
                        
                        retraining_history['metrics'][pair].append({
                            'timestamp': results['timestamp'],
                            'metrics': metrics
                        })
                
                # Save updated history
                save_retraining_history(retraining_history)
            
            # Sleep for a while before checking again
            # For hourly interval, check every 15 minutes
            # For other intervals, check every hour
            sleep_time = 15 * 60 if args.interval == 'hourly' else 60 * 60
            logger.info(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)
    
    else:
        # Single check for scheduled run
        if is_retraining_due(retraining_history, args.interval, args.time):
            logger.info(f"Retraining is due, starting {args.interval} retraining")
            
            # Perform retraining
            results = perform_retraining(
                pairs,
                optimize=not args.no_optimize,
                trials=args.trials,
                window=args.window,
                skip_fetch=args.skip_fetch
            )
            
            # Update retraining history
            retraining_history['last_retraining'] = results['timestamp']
            retraining_history['pairs'].update({pair: results['timestamp'] for pair in pairs})
            
            if results['metrics']:
                for pair, metrics in results['metrics'].items():
                    if pair not in retraining_history['metrics']:
                        retraining_history['metrics'][pair] = []
                    
                    retraining_history['metrics'][pair].append({
                        'timestamp': results['timestamp'],
                        'metrics': metrics
                    })
            
            # Save updated history
            save_retraining_history(retraining_history)
        else:
            logger.info(f"Retraining is not due yet, next retraining scheduled at {args.time}")
    
    return 0

def main():
    """Main function"""
    args = parse_arguments()
    
    logger.info("=" * 80)
    logger.info("AUTOMATED ML MODEL RETRAINING SCHEDULER")
    logger.info("=" * 80)
    logger.info(f"Retraining interval: {args.interval}")
    logger.info(f"Scheduled time: {args.time}")
    logger.info(f"Trading pairs: {args.pairs}")
    logger.info(f"Data window: {args.window} days")
    logger.info(f"Optimization: {not args.no_optimize}")
    logger.info(f"Optimization trials: {args.trials}")
    logger.info(f"Force retraining: {args.force}")
    logger.info(f"Run as daemon: {args.daemon}")
    logger.info(f"Skip fetch: {args.skip_fetch}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info("=" * 80)
    
    return run_scheduler(args)

if __name__ == "__main__":
    sys.exit(main())