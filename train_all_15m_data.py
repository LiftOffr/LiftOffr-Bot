#!/usr/bin/env python3
"""
Train Improved Models for All Pairs Using 15-Minute Data

This script trains improved models for all cryptocurrency pairs using
15-minute historical data from Kraken API and generates comprehensive
reports including PnL, win rate, and Sharpe ratio metrics.

Usage:
    python train_all_15m_data.py
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
        logging.FileHandler("train_all_15m.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
HISTORICAL_DATA_DIR = "historical_data"
MODEL_WEIGHTS_DIR = "model_weights"
RESULTS_DIR = "training_results"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
TIMEFRAME = '15m'
DEFAULT_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD"
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
    parser = argparse.ArgumentParser(description="Train improved models for all pairs using 15-minute data")
    parser.add_argument("--pairs", type=str, default="default",
                        help=f"Trading pairs to train (options: default, extended, or comma-separated list)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--sequence_length", type=int, default=96,
                        help="Sequence length for time series (default: 96 = 24 hours of 15-min data)")
    parser.add_argument("--predict_horizon", type=int, default=4,
                        help="Number of intervals to predict ahead (default: 4 = 1 hour)")
    parser.add_argument("--summary_report", action="store_true", default=True,
                        help="Generate summary report (default: True)")
    parser.add_argument("--fetch_data", action="store_true", default=False,
                        help="Fetch 15-minute data for pairs that don't have it (default: False)")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days of data to fetch if needed (default: 7)")
    return parser.parse_args()

def discover_available_data() -> Dict[str, List[str]]:
    """Discover available 15-minute data for training"""
    available_data = {}
    
    # Check historical_data directory
    if not os.path.exists(HISTORICAL_DATA_DIR):
        logger.warning(f"Historical data directory not found: {HISTORICAL_DATA_DIR}")
        return available_data
    
    # List all files in historical_data directory
    files = os.listdir(HISTORICAL_DATA_DIR)
    
    # Find 15-minute data files
    for file in files:
        if file.endswith('.csv') and TIMEFRAME in file:
            parts = file.replace('.csv', '').split('_')
            if len(parts) >= 2:
                # Try to reconstruct pair
                pair_parts = parts[:-1]  # Remove timeframe part
                potential_pair = '/'.join(pair_parts) if len(pair_parts) == 2 else None
                
                # Check if it's a supported pair
                if potential_pair and potential_pair.upper() in [p.upper() for p in EXTENDED_PAIRS]:
                    # Find the correct case from EXTENDED_PAIRS
                    for extended_pair in EXTENDED_PAIRS:
                        if extended_pair.upper() == potential_pair.upper():
                            pair = extended_pair
                            if pair not in available_data:
                                available_data[pair] = []
                            if TIMEFRAME not in available_data[pair]:
                                available_data[pair].append(TIMEFRAME)
                                logger.info(f"Found 15-minute data for {pair}: {file}")
    
    return available_data

def fetch_missing_data(pairs: List[str], days: int) -> Dict[str, bool]:
    """Fetch 15-minute data for pairs that don't have it"""
    fetch_results = {}
    
    for pair in pairs:
        # Check if data exists
        pair_clean = pair.replace("/", "_")
        file_path = f"{HISTORICAL_DATA_DIR}/{pair_clean}_{TIMEFRAME}.csv"
        
        if os.path.exists(file_path):
            logger.info(f"15-minute data already exists for {pair}")
            fetch_results[pair] = True
            continue
        
        # Fetch data
        logger.info(f"Fetching 15-minute data for {pair}...")
        try:
            command = [
                "python", "fetch_kraken_15m_data.py",
                "--pair", pair,
                "--timeframe", TIMEFRAME,
                "--days", str(days)
            ]
            
            process = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Check if file was created
            if os.path.exists(file_path):
                logger.info(f"Successfully fetched 15-minute data for {pair}")
                fetch_results[pair] = True
            else:
                logger.error(f"Failed to fetch 15-minute data for {pair}")
                fetch_results[pair] = False
            
            # Add delay to avoid rate limiting
            time.sleep(5)
        
        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {e}")
            fetch_results[pair] = False
    
    return fetch_results

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

def train_model_for_pair(pair: str, args) -> bool:
    """Train model for a specific pair using 15-minute data"""
    logger.info(f"Training model for {pair} using 15-minute data...")
    
    # Build command
    command = [
        "python", "train_with_15m_data.py",
        "--pair", pair,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--sequence_length", str(args.sequence_length),
        "--predict_horizon", str(args.predict_horizon)
    ]
    
    # Run command
    start_time = time.time()
    result = run_command(command, f"Training {pair} model with {args.epochs} epochs")
    training_time = time.time() - start_time
    
    # Format training time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    training_time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    if result is None:
        logger.error(f"Failed to train model for {pair}")
        return False
    
    logger.info(f"Successfully trained model for {pair} in {training_time_str}")
    return True

def load_training_reports() -> Dict:
    """Load all 15-minute training reports"""
    reports = {}
    
    # Check if results directory exists
    if not os.path.exists(RESULTS_DIR):
        logger.warning(f"Results directory not found: {RESULTS_DIR}")
        return reports
    
    # List all files in results directory
    files = os.listdir(RESULTS_DIR)
    
    # Load training reports
    for file in files:
        if file.startswith('training_report_') and file.endswith('.json') and TIMEFRAME in file:
            try:
                with open(f"{RESULTS_DIR}/{file}", 'r') as f:
                    report = json.load(f)
                
                pair = report.get('pair')
                timeframe = report.get('timeframe')
                
                if pair and timeframe and timeframe == TIMEFRAME:
                    key = f"{pair}_{timeframe}"
                    reports[key] = report
                    logger.info(f"Loaded 15-minute training report for {pair}")
            except Exception as e:
                logger.error(f"Error loading training report {file}: {e}")
    
    return reports

def generate_summary_report(reports: Dict) -> str:
    """Generate summary report for all trained 15-minute models"""
    if not reports:
        logger.warning("No 15-minute training reports found")
        return "No 15-minute training reports found"
    
    # Create report filename
    report_file = f"{RESULTS_DIR}/summary_report_15m_data.md"
    
    # Create report
    report = f"# 15-Minute Data Training Summary Report\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add overview table
    report += "## Performance Overview\n\n"
    report += "| Pair | Accuracy | Direction Accuracy | Win Rate | Total Return | Sharpe Ratio |\n"
    report += "|------|----------|-------------------|----------|--------------|-------------|\n"
    
    for key, data in sorted(reports.items()):
        pair = data.get('pair', 'Unknown')
        metrics = data.get("metrics", {})
        model_acc = metrics.get("model_accuracy", 0)
        dir_acc = metrics.get("direction_accuracy", 0)
        trading = metrics.get("trading_performance", {})
        win_rate = trading.get("win_rate", 0)
        total_return = trading.get("total_return", 0)
        sharpe = trading.get("sharpe_ratio", 0)
        
        report += f"| {pair} | {model_acc:.2%} | {dir_acc:.2%} | {win_rate:.2%} | {total_return:.2%} | {sharpe:.2f} |\n"
    
    # Add detailed performance section
    report += "\n## Detailed Performance by Pair\n\n"
    
    for key, data in sorted(reports.items()):
        pair = data.get('pair', 'Unknown')
        report += f"### {pair}\n\n"
        
        metrics = data.get("metrics", {})
        trading = metrics.get("trading_performance", {})
        
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
    report += f"**Best Model Accuracy**: {best_accuracy.get('pair')} - {best_accuracy.get('metrics', {}).get('model_accuracy', 0):.2%}\n\n"
    
    # Best by direction accuracy
    best_direction = max(reports.values(), key=lambda x: x.get("metrics", {}).get("direction_accuracy", 0))
    report += f"**Best Direction Accuracy**: {best_direction.get('pair')} - {best_direction.get('metrics', {}).get('direction_accuracy', 0):.2%}\n\n"
    
    # Best by win rate
    best_win_rate = max(reports.values(), key=lambda x: x.get("metrics", {}).get("trading_performance", {}).get("win_rate", 0))
    report += f"**Best Win Rate**: {best_win_rate.get('pair')} - {best_win_rate.get('metrics', {}).get('trading_performance', {}).get('win_rate', 0):.2%}\n\n"
    
    # Best by return
    best_return = max(reports.values(), key=lambda x: x.get("metrics", {}).get("trading_performance", {}).get("total_return", 0))
    report += f"**Best Total Return**: {best_return.get('pair')} - {best_return.get('metrics', {}).get('trading_performance', {}).get('total_return', 0):.2%}\n\n"
    
    # Best by Sharpe ratio
    best_sharpe = max(reports.values(), key=lambda x: x.get("metrics", {}).get("trading_performance", {}).get("sharpe_ratio", 0))
    report += f"**Best Sharpe Ratio**: {best_sharpe.get('pair')} - {best_sharpe.get('metrics', {}).get('trading_performance', {}).get('sharpe_ratio', 0):.2f}\n\n"
    
    # Add recommendations
    report += "## 15-Minute Trading Recommendations\n\n"
    report += "1. **Best Overall Model**: " + best_sharpe.get('pair') + " with Sharpe ratio of " + f"{best_sharpe.get('metrics', {}).get('trading_performance', {}).get('sharpe_ratio', 0):.2f}\n"
    report += "2. **More Frequent Trading**: 15-minute data allows for more frequent trading opportunities\n"
    report += "3. **Short-Term Profit Taking**: Use shorter take-profit levels for 15-minute trading\n"
    report += "4. **Dynamic Trade Management**: Consider using trailing stops for rapidly changing short-term trends\n"
    report += "5. **Indicator Sensitivity**: Use more sensitive indicator settings for short-term trading\n"
    
    # Save report
    try:
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved 15-minute summary report to {report_file}")
        return report_file
    except Exception as e:
        logger.error(f"Error saving summary report: {e}")
        return ""

def update_ml_config_with_15m_models() -> bool:
    """Update ML config with 15-minute models"""
    # Load training reports
    reports = load_training_reports()
    if not reports:
        logger.warning("No 15-minute training reports found")
        return False
    
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
    
    # Ensure models section exists
    if "models" not in config:
        config["models"] = {}
    
    # Update or add 15-minute models
    for key, report in reports.items():
        pair = report.get('pair')
        timeframe = report.get('timeframe')
        if not pair or not timeframe or timeframe != TIMEFRAME:
            continue
        
        metrics = report.get("metrics", {})
        trading = metrics.get("trading_performance", {})
        model_path = report.get("model_path", "")
        
        # Skip if model path doesn't exist
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}")
            continue
        
        # Add or update model config
        model_key = f"{pair}_{timeframe}"
        config["models"][model_key] = {
            "pair": pair,
            "timeframe": timeframe,
            "model_type": "enhanced_hybrid",
            "model_path": model_path,
            "accuracy": metrics.get("model_accuracy", 0),
            "direction_accuracy": metrics.get("direction_accuracy", 0),
            "win_rate": trading.get("win_rate", 0),
            "sharpe_ratio": trading.get("sharpe_ratio", 0),
            "total_return": trading.get("total_return", 0),
            "base_leverage": 5.0,
            "max_leverage": 75.0,
            "confidence_threshold": 0.65,
            "risk_percentage": 0.20,
            "active": True,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"Added 15-minute model for {pair} to ML config")
    
    # Save config
    try:
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Updated ML config with 15-minute models")
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
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("TRAIN IMPROVED MODELS FOR ALL PAIRS USING 15-MINUTE DATA")
    logger.info("=" * 80)
    logger.info(f"Pairs: {', '.join(pairs)}")
    logger.info(f"Timeframe: {TIMEFRAME}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Sequence Length: {args.sequence_length}")
    logger.info(f"Prediction Horizon: {args.predict_horizon} intervals ({args.predict_horizon*15} minutes)")
    logger.info(f"Fetch Missing Data: {args.fetch_data}")
    logger.info(f"Generate Summary Report: {args.summary_report}")
    logger.info("=" * 80 + "\n")
    
    # Discover available data
    available_data = discover_available_data()
    
    logger.info(f"Found 15-minute data for {len(available_data)} pairs:")
    for pair in available_data:
        logger.info(f"  {pair}")
    
    # Fetch missing data if requested
    if args.fetch_data:
        missing_pairs = [p for p in pairs if p not in available_data]
        if missing_pairs:
            logger.info(f"Fetching 15-minute data for {len(missing_pairs)} missing pairs...")
            fetch_results = fetch_missing_data(missing_pairs, args.days)
            
            # Update available data
            for pair, success in fetch_results.items():
                if success and pair not in available_data:
                    available_data[pair] = [TIMEFRAME]
    
    # Filter pairs based on available data
    training_pairs = [p for p in pairs if p in available_data]
    
    if not training_pairs:
        logger.error("No requested pairs have available 15-minute data. Exiting.")
        logger.error("Use --fetch_data flag to fetch missing data.")
        return 1
    
    logger.info(f"Will train models for {len(training_pairs)} pairs with 15-minute data:")
    for pair in training_pairs:
        logger.info(f"  {pair}")
    
    # Train models for each pair
    successful_trainings = 0
    failed_trainings = 0
    
    for i, pair in enumerate(training_pairs):
        logger.info(f"\nTraining model {i+1}/{len(training_pairs)} for {pair}...")
        
        # Train model
        success = train_model_for_pair(pair, args)
        
        if success:
            successful_trainings += 1
        else:
            failed_trainings += 1
        
        # Add delay between trainings
        if i < len(training_pairs) - 1:
            delay = 5
            logger.info(f"Waiting {delay} seconds before next training...")
            time.sleep(delay)
    
    # Generate summary report
    if args.summary_report and successful_trainings > 0:
        logger.info("\nGenerating 15-minute summary report...")
        reports = load_training_reports()
        
        if reports:
            report_path = generate_summary_report(reports)
            if report_path:
                logger.info(f"Generated 15-minute summary report: {report_path}")
            
            # Update ML config with 15-minute models
            update_ml_config_with_15m_models()
    
    # Print success/failure summary
    logger.info("\nTraining Summary:")
    logger.info(f"  Successfully trained: {successful_trainings} models")
    logger.info(f"  Failed trainings: {failed_trainings} models")
    
    return 0 if successful_trainings > 0 else 1

if __name__ == "__main__":
    sys.exit(main())