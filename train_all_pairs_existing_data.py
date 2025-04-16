#!/usr/bin/env python3
"""
Train Improved Models for All Pairs Using Existing Data

This script trains improved models for all cryptocurrency pairs using
existing historical data at various timeframes and generates comprehensive
reports including PnL, win rate, and Sharpe ratio metrics.

Usage:
    python train_all_pairs_existing_data.py
"""

import os
import sys
import json
import logging
import argparse
import time
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_all_pairs.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
HISTORICAL_DATA_DIR = "historical_data"
MODEL_WEIGHTS_DIR = "model_weights"
RESULTS_DIR = "training_results"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
ALL_TIMEFRAMES = ['1h', '4h', '1d']
DEFAULT_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD"
]
EXTENDED_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Create required directories
for directory in [HISTORICAL_DATA_DIR, MODEL_WEIGHTS_DIR, RESULTS_DIR, CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train improved models for all pairs using existing data")
    parser.add_argument("--pairs", type=str, default="default",
                        help=f"Trading pairs to train (options: default, extended, or comma-separated list)")
    parser.add_argument("--timeframes", type=str, default="all",
                        help=f"Comma-separated list of timeframes (options: {', '.join(ALL_TIMEFRAMES)} or 'all')")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--sequence_length", type=int, default=60,
                        help="Sequence length for time series (default: 60)")
    parser.add_argument("--summary_report", action="store_true", default=True,
                        help="Generate summary report (default: True)")
    parser.add_argument("--only_pairs_with_data", action="store_true", default=True,
                        help="Only train pairs with existing data (default: True)")
    return parser.parse_args()

def discover_available_data() -> Dict[str, List[str]]:
    """Discover available data for training"""
    available_data = {}
    
    # Check historical_data directory
    if not os.path.exists(HISTORICAL_DATA_DIR):
        logger.warning(f"Historical data directory not found: {HISTORICAL_DATA_DIR}")
        return available_data
    
    # List all files in historical_data directory
    files = os.listdir(HISTORICAL_DATA_DIR)
    
    # Parse filenames to identify pairs and timeframes
    for file in files:
        if file.endswith('.csv'):
            # Try different filename formats
            # Format 1: pair_timeframe.csv (e.g., btc_usd_1h.csv)
            # Format 2: PAIR_timeframe.csv (e.g., BTCUSD_1h.csv)
            
            parts = file.replace('.csv', '').split('_')
            
            if len(parts) >= 2:
                # Try to identify timeframe
                timeframe = None
                for tf in ALL_TIMEFRAMES:
                    if tf in parts[-1]:
                        timeframe = tf
                        break
                
                if timeframe:
                    # Reconstruct pair from remaining parts
                    pair_parts = parts[:-1]
                    
                    # Try different formats
                    potential_pairs = [
                        f"{'/'.join(pair_parts)}",
                        f"{pair_parts[0]}/{pair_parts[1]}" if len(pair_parts) >= 2 else None,
                        f"{pair_parts[0].upper()}/{pair_parts[1].upper()}" if len(pair_parts) >= 2 else None
                    ]
                    
                    # Map to standard pair format
                    pair = None
                    for pp in potential_pairs:
                        if pp and (pp.upper() in [p.upper() for p in EXTENDED_PAIRS] or 
                                  pp.replace('/', '').upper() in [p.replace('/', '').upper() for p in EXTENDED_PAIRS]):
                            # Find the correct case from EXTENDED_PAIRS
                            for extended_pair in EXTENDED_PAIRS:
                                if extended_pair.upper() == pp.upper() or extended_pair.replace('/', '').upper() == pp.replace('/', '').upper():
                                    pair = extended_pair
                                    break
                            break
                    
                    if pair:
                        if pair not in available_data:
                            available_data[pair] = []
                        
                        if timeframe not in available_data[pair]:
                            available_data[pair].append(timeframe)
                            logger.info(f"Found data for {pair} ({timeframe}): {file}")
    
    return available_data

def run_command(command: List[str], description: str = None) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command and log output"""
    if description:
        logger.info(description)
    
    logger.info(f"Running command: {' '.join(command)}")
    
    try:
        process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Log first few lines of output
        output_lines = process.stdout.split('\n')
        if len(output_lines) > 10:
            logger.info(f"Command output (truncated):\n{os.linesep.join(output_lines[:10])}\n[truncated]")
        else:
            logger.info(f"Command output:\n{process.stdout}")
        
        if process.stderr:
            logger.warning(f"Command stderr:\n{process.stderr}")
        
        return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output:\n{e.stderr}")
        return None

def train_model_for_pair_timeframe(pair: str, timeframe: str, args) -> bool:
    """Train model for a specific pair and timeframe"""
    logger.info(f"Training model for {pair} ({timeframe})...")
    
    # Build command
    command = [
        "python", "train_with_existing_data.py",
        "--pair", pair,
        "--timeframe", timeframe,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--sequence_length", str(args.sequence_length)
    ]
    
    # Run command
    start_time = time.time()
    result = run_command(command, f"Training {pair} ({timeframe}) model with {args.epochs} epochs")
    training_time = time.time() - start_time
    
    # Format training time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    training_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    if result is None:
        logger.error(f"Failed to train model for {pair} ({timeframe})")
        return False
    
    logger.info(f"Successfully trained model for {pair} ({timeframe}) in {training_time_str}")
    return True

def load_training_reports() -> Dict:
    """Load all training reports"""
    reports = {}
    
    # Check if results directory exists
    if not os.path.exists(RESULTS_DIR):
        logger.warning(f"Results directory not found: {RESULTS_DIR}")
        return reports
    
    # List all files in results directory
    files = os.listdir(RESULTS_DIR)
    
    # Load training reports
    for file in files:
        if file.startswith('training_report_') and file.endswith('.json'):
            try:
                with open(f"{RESULTS_DIR}/{file}", 'r') as f:
                    report = json.load(f)
                
                pair = report.get('pair')
                timeframe = report.get('timeframe')
                
                if pair and timeframe:
                    key = f"{pair}_{timeframe}"
                    reports[key] = report
                    logger.info(f"Loaded training report for {pair} ({timeframe})")
            except Exception as e:
                logger.error(f"Error loading training report {file}: {e}")
    
    return reports

def generate_summary_report(reports: Dict) -> str:
    """Generate summary report for all trained models"""
    if not reports:
        logger.warning("No training reports found")
        return "No training reports found"
    
    # Create report filename
    report_file = f"{RESULTS_DIR}/summary_report_all_pairs.md"
    
    # Create report
    report = f"# Training Summary Report for All Pairs\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add overview table
    report += "## Performance Overview\n\n"
    report += "| Pair | Timeframe | Accuracy | Direction Accuracy | Win Rate | Total Return | Sharpe Ratio |\n"
    report += "|------|-----------|----------|-------------------|----------|--------------|-------------|\n"
    
    for key, data in sorted(reports.items()):
        pair = data.get('pair', 'Unknown')
        timeframe = data.get('timeframe', 'Unknown')
        metrics = data.get("metrics", {})
        model_acc = metrics.get("model_accuracy", 0)
        dir_acc = metrics.get("direction_accuracy", 0)
        trading = metrics.get("trading_performance", {})
        win_rate = trading.get("win_rate", 0)
        total_return = trading.get("total_return", 0)
        sharpe = trading.get("sharpe_ratio", 0)
        
        report += f"| {pair} | {timeframe} | {model_acc:.2%} | {dir_acc:.2%} | {win_rate:.2%} | {total_return:.2%} | {sharpe:.2f} |\n"
    
    # Add detailed performance section
    report += "\n## Detailed Performance by Pair\n\n"
    
    # Group by pair
    pair_reports = {}
    for key, data in reports.items():
        pair = data.get('pair', 'Unknown')
        if pair not in pair_reports:
            pair_reports[pair] = []
        
        pair_reports[pair].append(data)
    
    for pair, pair_data in sorted(pair_reports.items()):
        report += f"### {pair}\n\n"
        
        # Sort by timeframe
        pair_data.sort(key=lambda x: x.get('timeframe', ''))
        
        for data in pair_data:
            timeframe = data.get('timeframe', 'Unknown')
            metrics = data.get("metrics", {})
            trading = metrics.get("trading_performance", {})
            
            report += f"#### {timeframe} Timeframe\n\n"
            
            report += "**Model Metrics**:\n\n"
            report += f"- **Model Accuracy**: {metrics.get('model_accuracy', 0):.2%}\n"
            report += f"- **Direction Accuracy**: {metrics.get('direction_accuracy', 0):.2%}\n"
            
            # Add class accuracy
            report += "\n**Class-wise Accuracy**:\n\n"
            class_acc = metrics.get("class_accuracy", {})
            for class_name, acc in class_acc.items():
                report += f"- {class_name}: {acc:.2%}\n"
            
            report += "\n**Trading Performance**:\n\n"
            report += f"- **Total Return**: {trading.get('total_return', 0):.2%}\n"
            report += f"- **Annualized Return**: {trading.get('annualized_return', 0):.2%}\n"
            report += f"- **Win Rate**: {trading.get('win_rate', 0):.2%}\n"
            report += f"- **Sharpe Ratio**: {trading.get('sharpe_ratio', 0):.2f}\n"
            report += f"- **Max Drawdown**: {trading.get('max_drawdown', 0):.2%}\n"
            report += f"- **Trades**: {trading.get('trades', 0)}\n"
            report += f"- **Profit Factor**: {trading.get('profit_factor', 0):.2f}\n"
            
            report += "\n"
    
    # Add best performers section
    report += "## Best Performers\n\n"
    
    # Best by accuracy
    best_accuracy = max(reports.values(), key=lambda x: x.get("metrics", {}).get("model_accuracy", 0))
    report += f"**Best Model Accuracy**: {best_accuracy.get('pair')} ({best_accuracy.get('timeframe')}) - {best_accuracy.get('metrics', {}).get('model_accuracy', 0):.2%}\n\n"
    
    # Best by direction accuracy
    best_direction = max(reports.values(), key=lambda x: x.get("metrics", {}).get("direction_accuracy", 0))
    report += f"**Best Direction Accuracy**: {best_direction.get('pair')} ({best_direction.get('timeframe')}) - {best_direction.get('metrics', {}).get('direction_accuracy', 0):.2%}\n\n"
    
    # Best by win rate
    best_win_rate = max(reports.values(), key=lambda x: x.get("metrics", {}).get("trading_performance", {}).get("win_rate", 0))
    report += f"**Best Win Rate**: {best_win_rate.get('pair')} ({best_win_rate.get('timeframe')}) - {best_win_rate.get('metrics', {}).get('trading_performance', {}).get('win_rate', 0):.2%}\n\n"
    
    # Best by return
    best_return = max(reports.values(), key=lambda x: x.get("metrics", {}).get("trading_performance", {}).get("total_return", 0))
    report += f"**Best Total Return**: {best_return.get('pair')} ({best_return.get('timeframe')}) - {best_return.get('metrics', {}).get('trading_performance', {}).get('total_return', 0):.2%}\n\n"
    
    # Best by Sharpe ratio
    best_sharpe = max(reports.values(), key=lambda x: x.get("metrics", {}).get("trading_performance", {}).get("sharpe_ratio", 0))
    report += f"**Best Sharpe Ratio**: {best_sharpe.get('pair')} ({best_sharpe.get('timeframe')}) - {best_sharpe.get('metrics', {}).get('trading_performance', {}).get('sharpe_ratio', 0):.2f}\n\n"
    
    # Add recommendations
    report += "## Recommendations\n\n"
    report += "1. **Best Overall Model**: " + best_sharpe.get('pair') + " (" + best_sharpe.get('timeframe') + ") with Sharpe ratio of " + f"{best_sharpe.get('metrics', {}).get('trading_performance', {}).get('sharpe_ratio', 0):.2f}\n"
    report += "2. Focus on timeframes with higher Sharpe ratios for actual trading\n"
    report += "3. Consider ensemble approaches combining signals from multiple timeframes\n"
    report += "4. Regularly retrain models as more market data becomes available\n"
    
    # Save report
    try:
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved summary report to {report_file}")
        return report_file
    except Exception as e:
        logger.error(f"Error saving summary report: {e}")
        return ""

def update_ml_config_with_best_models() -> bool:
    """Update ML config to mark the best models as preferred"""
    # Load training reports
    reports = load_training_reports()
    if not reports:
        logger.warning("No training reports found")
        return False
    
    # Find best model for each pair
    best_models = {}
    
    for key, report in reports.items():
        pair = report.get('pair')
        if not pair:
            continue
        
        sharpe_ratio = report.get('metrics', {}).get('trading_performance', {}).get('sharpe_ratio', 0)
        
        if pair not in best_models or sharpe_ratio > best_models[pair]['sharpe_ratio']:
            best_models[pair] = {
                'timeframe': report.get('timeframe'),
                'sharpe_ratio': sharpe_ratio,
                'model_path': report.get('model_path')
            }
    
    # Load ML config
    if os.path.exists(ML_CONFIG_PATH):
        try:
            with open(ML_CONFIG_PATH, 'r') as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error loading ML configuration: {e}")
            return False
    else:
        config = {"models": {}, "global_settings": {}}
    
    # Update global settings if not present
    if "global_settings" not in config:
        config["global_settings"] = {}
    
    # Set global settings
    config["global_settings"]["base_leverage"] = 5.0
    config["global_settings"]["max_leverage"] = 75.0
    config["global_settings"]["confidence_threshold"] = 0.65
    config["global_settings"]["risk_percentage"] = 0.20
    config["global_settings"]["max_portfolio_risk"] = 0.25
    
    # Ensure models section exists
    if "models" not in config:
        config["models"] = {}
    
    # Update preferred models
    for pair, best_model in best_models.items():
        timeframe = best_model['timeframe']
        model_key = f"{pair}_{timeframe}"
        
        # Mark all models for this pair as not preferred
        for key in config.get("models", {}):
            if key.startswith(f"{pair}_"):
                config["models"][key]["preferred"] = False
        
        # Mark the best model as preferred
        if model_key in config.get("models", {}):
            config["models"][model_key]["preferred"] = True
            logger.info(f"Marked {pair} ({timeframe}) as preferred model")
    
    # Save config
    try:
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Updated ML config with preferred models")
        return True
    except Exception as e:
        logger.error(f"Error saving ML config: {e}")
        return False

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Determine pairs to train
    if args.pairs.lower() == "default":
        pairs = DEFAULT_PAIRS
    elif args.pairs.lower() == "extended":
        pairs = EXTENDED_PAIRS
    else:
        pairs = [p.strip() for p in args.pairs.split(",")]
    
    # Determine timeframes to train
    if args.timeframes.lower() == "all":
        timeframes = ALL_TIMEFRAMES
    else:
        timeframes = [t.strip() for t in args.timeframes.split(",")]
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("TRAIN IMPROVED MODELS FOR ALL PAIRS USING EXISTING DATA")
    logger.info("=" * 80)
    logger.info(f"Pairs: {', '.join(pairs)}")
    logger.info(f"Timeframes: {', '.join(timeframes)}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Sequence Length: {args.sequence_length}")
    logger.info(f"Only Train Pairs with Data: {args.only_pairs_with_data}")
    logger.info(f"Summary Report: {args.summary_report}")
    logger.info("=" * 80 + "\n")
    
    # Discover available data
    available_data = discover_available_data()
    
    if not available_data:
        logger.error("No historical data found. Exiting.")
        return 1
    
    logger.info(f"Found data for {len(available_data)} pairs:")
    for pair, available_timeframes in available_data.items():
        logger.info(f"  {pair}: {', '.join(available_timeframes)}")
    
    # Filter pairs based on available data if requested
    if args.only_pairs_with_data:
        pairs = [p for p in pairs if p in available_data]
        
        if not pairs:
            logger.error("No requested pairs have available data. Exiting.")
            return 1
    
    # Train models for each pair and timeframe
    successful_trainings = 0
    failed_trainings = 0
    
    for pair in pairs:
        # Get available timeframes for this pair
        pair_timeframes = available_data.get(pair, [])
        
        if not pair_timeframes and args.only_pairs_with_data:
            logger.warning(f"No data available for {pair}, skipping")
            continue
        
        # Filter timeframes based on available data and requested timeframes
        selected_timeframes = [tf for tf in timeframes if tf in pair_timeframes] if args.only_pairs_with_data else timeframes
        
        if not selected_timeframes:
            logger.warning(f"No requested timeframes available for {pair}, skipping")
            continue
        
        logger.info(f"\nTraining models for {pair} with timeframes: {', '.join(selected_timeframes)}")
        
        for timeframe in selected_timeframes:
            if not args.only_pairs_with_data or timeframe in pair_timeframes:
                # Train model
                success = train_model_for_pair_timeframe(pair, timeframe, args)
                
                if success:
                    successful_trainings += 1
                else:
                    failed_trainings += 1
                
                # Add delay to avoid overwhelming system
                if timeframe != selected_timeframes[-1]:
                    logger.info("Waiting 5 seconds before training next timeframe...")
                    time.sleep(5)
        
        # Add delay between pairs
        if pair != pairs[-1]:
            logger.info("Waiting 10 seconds before training next pair...")
            time.sleep(10)
    
    # Generate summary report
    if args.summary_report and successful_trainings > 0:
        logger.info("\nGenerating summary report...")
        reports = load_training_reports()
        
        if reports:
            report_path = generate_summary_report(reports)
            if report_path:
                logger.info(f"Generated summary report: {report_path}")
            
            # Update ML config with best models
            update_ml_config_with_best_models()
    
    # Print success/failure summary
    logger.info("\nTraining Summary:")
    logger.info(f"  Successfully trained: {successful_trainings} models")
    logger.info(f"  Failed trainings: {failed_trainings} models")
    
    return 0 if successful_trainings > 0 else 1

if __name__ == "__main__":
    sys.exit(main())