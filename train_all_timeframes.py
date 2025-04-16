#!/usr/bin/env python3
"""
Train Models for All Timeframes

This script trains models for multiple timeframes (15m, 1h, 4h, 1d) using Amberdata OHLCV data.
It orchestrates the training process and generates a comprehensive summary report with PnL,
win rate, and Sharpe ratio metrics for each trained model.

Usage:
    python train_all_timeframes.py --pair BTC/USD
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
import subprocess
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("train_all_timeframes.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
RESULTS_DIR = "training_results"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
ALL_TIMEFRAMES = ['15m', '1h', '4h', '1d']
ALL_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Create required directories
for directory in [RESULTS_DIR, CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train models for all timeframes")
    parser.add_argument("--pair", type=str, default="BTC/USD",
                        help=f"Trading pair to train model for (options: {', '.join(ALL_PAIRS)})")
    parser.add_argument("--timeframes", type=str, default="all",
                        help=f"Comma-separated list of timeframes (options: {', '.join(ALL_TIMEFRAMES)} or 'all')")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Amberdata API key (if not provided, will try AMBERDATA_API_KEY environment variable)")
    parser.add_argument("--days", type=int, default=365,
                        help="Number of days of historical data to fetch (default: 365)")
    parser.add_argument("--summary_report", action="store_true", default=True,
                        help="Generate summary report (default: True)")
    return parser.parse_args()

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
        if len(output_lines) > 20:
            logger.info(f"Command output (truncated):\n{' '.join(output_lines[:20])}\n[truncated]")
        else:
            logger.info(f"Command output:\n{process.stdout}")
        
        if process.stderr:
            logger.warning(f"Command stderr:\n{process.stderr}")
        
        return process
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"Error output:\n{e.stderr}")
        return None

def train_timeframe(pair: str, timeframe: str, epochs: int, batch_size: int, api_key: str, days: int) -> bool:
    """Train model for a specific timeframe"""
    logger.info(f"Training model for {pair} ({timeframe})...")
    
    # Build command
    command = [
        "python", "train_with_amberdata.py",
        "--pair", pair,
        "--timeframe", timeframe,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--days", str(days)
    ]
    
    # Add API key if provided
    if api_key:
        command.extend(["--api_key", api_key])
    
    # Run command
    start_time = time.time()
    result = run_command(command, f"Training {pair} ({timeframe}) model with {epochs} epochs")
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

def load_training_reports(pair: str, timeframes: List[str]) -> Dict:
    """Load training reports for all timeframes"""
    reports = {}
    pair_clean = pair.replace("/", "_").lower()
    
    for timeframe in timeframes:
        report_path = f"{RESULTS_DIR}/training_report_{pair_clean}_{timeframe}.json"
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r') as f:
                    reports[timeframe] = json.load(f)
                logger.info(f"Loaded training report for {pair} ({timeframe})")
            except Exception as e:
                logger.error(f"Error loading training report for {pair} ({timeframe}): {e}")
    
    return reports

def generate_summary_report(pair: str, reports: Dict) -> str:
    """Generate summary report for all timeframes"""
    if not reports:
        logger.warning("No training reports found")
        return "No training reports found"
    
    # Create report filename
    pair_clean = pair.replace("/", "_").lower()
    report_file = f"{RESULTS_DIR}/summary_report_{pair_clean}.md"
    
    # Create report
    report = f"# Training Summary Report for {pair}\n\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add overview table
    report += "## Performance Overview\n\n"
    report += "| Timeframe | Accuracy | Direction Accuracy | Win Rate | Total Return | Sharpe Ratio |\n"
    report += "|-----------|----------|-------------------|----------|--------------|-------------|\n"
    
    for timeframe, data in sorted(reports.items()):
        metrics = data.get("metrics", {})
        model_acc = metrics.get("model_accuracy", 0)
        dir_acc = metrics.get("direction_accuracy", 0)
        trading = metrics.get("trading_performance", {})
        win_rate = trading.get("win_rate", 0)
        total_return = trading.get("total_return", 0)
        sharpe = trading.get("sharpe_ratio", 0)
        
        report += f"| {timeframe} | {model_acc:.2%} | {dir_acc:.2%} | {win_rate:.2%} | {total_return:.2%} | {sharpe:.2f} |\n"
    
    # Add detailed performance section
    report += "\n## Detailed Performance Metrics\n\n"
    
    for timeframe, data in sorted(reports.items()):
        report += f"### {timeframe} Timeframe\n\n"
        
        metrics = data.get("metrics", {})
        trading = metrics.get("trading_performance", {})
        
        report += "#### Model Metrics\n\n"
        report += f"- **Model Accuracy**: {metrics.get('model_accuracy', 0):.2%}\n"
        report += f"- **Direction Accuracy**: {metrics.get('direction_accuracy', 0):.2%}\n"
        
        # Add class accuracy
        report += "\n**Class-wise Accuracy**:\n\n"
        class_acc = metrics.get("class_accuracy", {})
        for class_name, acc in class_acc.items():
            report += f"- {class_name}: {acc:.2%}\n"
        
        report += "\n#### Trading Performance\n\n"
        report += f"- **Total Return**: {trading.get('total_return', 0):.2%}\n"
        report += f"- **Annualized Return**: {trading.get('annualized_return', 0):.2%}\n"
        report += f"- **Win Rate**: {trading.get('win_rate', 0):.2%}\n"
        report += f"- **Sharpe Ratio**: {trading.get('sharpe_ratio', 0):.2f}\n"
        report += f"- **Max Drawdown**: {trading.get('max_drawdown', 0):.2%}\n"
        report += f"- **Trades**: {trading.get('trades', 0)}\n"
        report += f"- **Profit Factor**: {trading.get('profit_factor', 0):.2f}\n"
        report += f"- **Average Win**: ${trading.get('avg_win', 0):.2f}\n"
        report += f"- **Average Loss**: ${trading.get('avg_loss', 0):.2f}\n"
        
        report += "\n"
    
    # Add conclusion
    report += "## Conclusion\n\n"
    
    # Find best timeframe
    best_timeframe = None
    best_sharpe = -999
    
    for timeframe, data in reports.items():
        trading = data.get("metrics", {}).get("trading_performance", {})
        sharpe = trading.get("sharpe_ratio", 0)
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_timeframe = timeframe
    
    if best_timeframe:
        report += f"Based on Sharpe Ratio, the best performing timeframe is **{best_timeframe}** with a value of {best_sharpe:.2f}.\n\n"
    
    # Add final notes
    report += "### Recommendations\n\n"
    report += "1. Focus on timeframes with higher Sharpe ratios for actual trading\n"
    report += "2. Consider ensemble approaches combining signals from multiple timeframes\n"
    report += "3. Regularly retrain models as market conditions evolve\n"
    
    # Save report
    try:
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved summary report to {report_file}")
        return report_file
    except Exception as e:
        logger.error(f"Error saving summary report: {e}")
        return ""

def update_ml_config_with_best_timeframe(pair: str, reports: Dict) -> bool:
    """Update ML config to mark the best timeframe as preferred"""
    if not reports:
        logger.warning("No training reports found")
        return False
    
    # Find best timeframe
    best_timeframe = None
    best_sharpe = -999
    
    for timeframe, data in reports.items():
        trading = data.get("metrics", {}).get("trading_performance", {})
        sharpe = trading.get("sharpe_ratio", 0)
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_timeframe = timeframe
    
    if not best_timeframe:
        logger.warning("Could not determine best timeframe")
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
        logger.warning(f"ML configuration not found: {ML_CONFIG_PATH}")
        return False
    
    # Update preferred timeframe
    for key in config.get("models", {}):
        if key.startswith(f"{pair}_"):
            timeframe = key.split("_")[-1]
            config["models"][key]["preferred"] = (timeframe == best_timeframe)
    
    # Save config
    try:
        with open(ML_CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Updated ML config with preferred timeframe {best_timeframe} for {pair}")
        return True
    except Exception as e:
        logger.error(f"Error saving ML config: {e}")
        return False

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Determine timeframes to train
    if args.timeframes.lower() == "all":
        timeframes = ALL_TIMEFRAMES
    else:
        timeframes = [t.strip() for t in args.timeframes.split(",")]
    
    # Print banner
    logger.info("\n" + "=" * 80)
    logger.info("TRAIN MODELS FOR ALL TIMEFRAMES")
    logger.info("=" * 80)
    logger.info(f"Pair: {args.pair}")
    logger.info(f"Timeframes: {', '.join(timeframes)}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Days of Data: {args.days}")
    logger.info(f"Summary Report: {args.summary_report}")
    logger.info("=" * 80 + "\n")
    
    # Get API key
    api_key = args.api_key or os.environ.get('AMBERDATA_API_KEY')
    if not api_key:
        logger.error("No Amberdata API key provided. Please set AMBERDATA_API_KEY environment variable or use --api_key")
        return 1
    
    # Train models for each timeframe
    successful_timeframes = []
    
    for timeframe in timeframes:
        logger.info(f"\nTraining for timeframe: {timeframe}")
        success = train_timeframe(
            args.pair, 
            timeframe, 
            args.epochs, 
            args.batch_size, 
            api_key, 
            args.days
        )
        
        if success:
            successful_timeframes.append(timeframe)
        
        # Add delay to avoid API rate limiting
        if timeframe != timeframes[-1]:
            logger.info("Waiting 10 seconds before training next timeframe...")
            time.sleep(10)
    
    # Generate summary report
    if args.summary_report and successful_timeframes:
        logger.info("\nGenerating summary report...")
        reports = load_training_reports(args.pair, successful_timeframes)
        
        if reports:
            report_path = generate_summary_report(args.pair, reports)
            if report_path:
                logger.info(f"Generated summary report: {report_path}")
            
            # Update ML config with best timeframe
            update_ml_config_with_best_timeframe(args.pair, reports)
    
    # Print success/failure summary
    logger.info("\nTraining Summary:")
    logger.info(f"  Successfully trained: {len(successful_timeframes)}/{len(timeframes)} timeframes")
    
    if successful_timeframes:
        logger.info(f"  Successful timeframes: {', '.join(successful_timeframes)}")
    
    failed_timeframes = [t for t in timeframes if t not in successful_timeframes]
    if failed_timeframes:
        logger.info(f"  Failed timeframes: {', '.join(failed_timeframes)}")
    
    return 0 if len(successful_timeframes) > 0 else 1

if __name__ == "__main__":
    sys.exit(main())