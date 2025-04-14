#!/usr/bin/env python3
"""
Optimize All Trading Pairs

This script orchestrates the optimization of all trading pairs:
1. Creates required directories
2. Runs the optimization pipeline for all configured pairs
3. Applies optimized parameters to the trading system
4. Generates a summary report of the optimization results

Usage:
    python optimize_all_pairs.py [--pairs PAIRS] [--days DAYS]
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Import the optimization pipeline
from run_optimization_pipeline import OptimizationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("optimization_all_pairs.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create required directories for optimization outputs"""
    dirs = [
        "optimization_results",
        "pipeline_results",
        "models/ensemble",
        "config"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def create_config_if_missing():
    """Create ML config file if it doesn't exist"""
    config_path = "config/ml_config.json"
    
    if not os.path.exists(config_path):
        default_config = {
            "pairs": ["SOL/USD", "BTC/USD", "ETH/USD", "DOT/USD", "ADA/USD", "LINK/USD"],
            "base_leverage": 20.0,
            "max_leverage": 125.0,
            "risk_percentage": 0.20,
            "confidence_threshold": 0.65,
            "time_scales": ["1h", "4h", "1d"],
            "use_ensemble": True,
            "ensemble_models": ["tcn", "lstm", "attention_gru"],
            "optimization": {
                "strategy_weight": {
                    "arima": 0.3,
                    "adaptive": 0.7
                }
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created default ML config at {config_path}")

def optimize_all_pairs(pairs: Optional[List[str]] = None, days: int = 365):
    """
    Run optimization for all trading pairs.
    
    Args:
        pairs: List of pairs to optimize (None = use config file)
        days: Number of days of historical data to use
    """
    # Initialize the optimization pipeline
    pipeline = OptimizationPipeline()
    
    # Run the pipeline
    try:
        results = pipeline.run_pipeline(pairs=pairs, days=days)
        logger.info("Optimization pipeline completed successfully")
        
        # Generate summary report
        generate_summary_report(results)
        
        # Apply optimized parameters
        apply_optimized_parameters()
        
        return results
    except Exception as e:
        logger.error(f"Error running optimization pipeline: {e}")
        raise

def generate_summary_report(results: Dict[str, Any]):
    """
    Generate a summary report of optimization results.
    
    Args:
        results: Results from the optimization pipeline
    """
    report_path = "optimization_summary.md"
    
    try:
        with open(report_path, 'w') as f:
            f.write("# Optimization Summary Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall summary
            f.write("## Overall Performance\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            
            overall_return = results.get("summary", {}).get("overall_return", 0)
            overall_win_rate = results.get("summary", {}).get("overall_win_rate", 0)
            overall_sharpe = results.get("summary", {}).get("overall_sharpe_ratio", 0)
            
            f.write(f"| Average Return | {overall_return:.2f} |\n")
            f.write(f"| Average Win Rate | {overall_win_rate:.2f} |\n")
            f.write(f"| Average Sharpe Ratio | {overall_sharpe:.2f} |\n")
            f.write(f"| Optimized Pairs | {results.get('summary', {}).get('optimized_pairs', 0)} |\n\n")
            
            # Detailed results by pair
            f.write("## Trading Pair Details\n\n")
            
            for pair, pair_results in results.get("detailed_results", {}).items():
                f.write(f"### {pair}\n\n")
                
                # Market regime
                f.write(f"**Market Regime:** {pair_results.get('market_regime', 'Unknown')}\n\n")
                
                # Optimized parameters
                f.write("**Optimized Parameters:**\n\n")
                f.write("| Parameter | Value |\n")
                f.write("|-----------|-------|\n")
                
                params = pair_results.get("optimized_parameters", {})
                f.write(f"| Risk Percentage | {params.get('risk_percentage', 0):.2f} |\n")
                f.write(f"| Base Leverage | {params.get('base_leverage', 0):.1f} |\n")
                f.write(f"| Max Leverage | {params.get('max_leverage', 0):.1f} |\n")
                f.write(f"| Confidence Threshold | {params.get('confidence_threshold', 0):.2f} |\n")
                
                if "strategy_weights" in params:
                    weights = params["strategy_weights"]
                    f.write(f"| ARIMA Weight | {weights.get('arima', 0):.2f} |\n")
                    f.write(f"| Adaptive Weight | {weights.get('adaptive', 0):.2f} |\n")
                
                f.write("\n**Validation Results:**\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                
                validation = pair_results.get("validation_results", {})
                f.write(f"| Total Return | {validation.get('total_return', 0):.2f} |\n")
                f.write(f"| Win Rate | {validation.get('win_rate', 0):.2f} |\n")
                f.write(f"| Sharpe Ratio | {validation.get('sharpe_ratio', 0):.2f} |\n")
                f.write(f"| Profit Factor | {validation.get('profit_factor', 0):.2f} |\n")
                f.write(f"| Max Drawdown | {validation.get('max_drawdown', 0):.2f} |\n\n")
                
                # Add separator between pairs
                f.write("---\n\n")
            
            # Implementation notes
            f.write("## Implementation Notes\n\n")
            f.write("- These optimized parameters are configured for dynamic adjustment based on confidence levels.\n")
            f.write("- Parameters will be adjusted for each trade according to:\n")
            f.write("  - Signal strength and model confidence\n")
            f.write("  - Current market volatility and regime\n")
            f.write("  - Historical performance patterns\n")
            f.write("- ML models should be retrained periodically (e.g., weekly) to maintain accuracy.\n")
        
        logger.info(f"Generated summary report at {report_path}")
    except Exception as e:
        logger.error(f"Error generating summary report: {e}")

def apply_optimized_parameters():
    """Apply optimized parameters to the trading system"""
    try:
        # Load optimized parameters
        optimized_params_path = "config/optimized_params.json"
        
        if not os.path.exists(optimized_params_path):
            logger.warning(f"No optimized parameters found at {optimized_params_path}")
            return
        
        with open(optimized_params_path, 'r') as f:
            optimized_params = json.load(f)
        
        # Update .env file with optimized parameters
        update_env_with_optimized_params(optimized_params)
        
        # Create a backup of the current config
        ml_config_path = "config/ml_config.json"
        backup_path = f"config/ml_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        if os.path.exists(ml_config_path):
            with open(ml_config_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            logger.info(f"Created backup of ML config at {backup_path}")
        
        # Update ML config with optimized parameters
        with open(ml_config_path, 'r') as f:
            ml_config = json.load(f)
        
        # Update the config with optimized values
        ml_config["optimization_applied"] = datetime.now().isoformat()
        ml_config["optimized_pairs"] = list(optimized_params.keys())
        
        # Save updated config
        with open(ml_config_path, 'w') as f:
            json.dump(ml_config, f, indent=2)
        
        logger.info(f"Updated ML config with optimization results at {ml_config_path}")
    except Exception as e:
        logger.error(f"Error applying optimized parameters: {e}")

def update_env_with_optimized_params(optimized_params: Dict[str, Any]):
    """
    Update .env file with optimized parameters.
    
    Args:
        optimized_params: Dictionary of optimized parameters by pair
    """
    try:
        env_path = ".env"
        env_lines = []
        
        # Read existing .env file
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                env_lines = f.readlines()
        
        # Process existing lines and remove old optimized parameters
        processed_lines = []
        skip_patterns = ["OPTIMIZED_", "RISK_PERCENTAGE_", "LEVERAGE_", "CONFIDENCE_THRESHOLD_"]
        
        for line in env_lines:
            skip = False
            for pattern in skip_patterns:
                if pattern in line and "=" in line:
                    skip = True
                    break
            
            if not skip:
                processed_lines.append(line)
        
        # Add header for optimized parameters
        processed_lines.append("\n# Optimized parameters - generated on {}\n".format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        # Add optimization date
        processed_lines.append("OPTIMIZED_DATE=\"{}\"\n".format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        # Add parameters for each pair
        for pair, params in optimized_params.items():
            pair_key = pair.replace("/", "")
            
            processed_lines.append(f"# Optimized parameters for {pair}\n")
            processed_lines.append(f"RISK_PERCENTAGE_{pair_key}={params.get('risk_percentage', 0.2)}\n")
            processed_lines.append(f"BASE_LEVERAGE_{pair_key}={params.get('base_leverage', 20.0)}\n")
            processed_lines.append(f"MAX_LEVERAGE_{pair_key}={params.get('max_leverage', 125.0)}\n")
            processed_lines.append(f"CONFIDENCE_THRESHOLD_{pair_key}={params.get('confidence_threshold', 0.65)}\n")
            
            # Add strategy weights if available
            if "strategy_weights" in params:
                weights = params["strategy_weights"]
                processed_lines.append(f"ARIMA_WEIGHT_{pair_key}={weights.get('arima', 0.3)}\n")
                processed_lines.append(f"ADAPTIVE_WEIGHT_{pair_key}={weights.get('adaptive', 0.7)}\n")
            
            processed_lines.append("\n")
        
        # Write updated .env file
        with open(env_path, 'w') as f:
            f.writelines(processed_lines)
        
        logger.info(f"Updated .env file with optimized parameters at {env_path}")
    except Exception as e:
        logger.error(f"Error updating .env file: {e}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Optimize all trading pairs")
    parser.add_argument("--pairs", type=str, help="Comma-separated list of trading pairs to optimize")
    parser.add_argument("--days", type=int, default=365, help="Number of days of historical data to use")
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create necessary directories
    create_directories()
    
    # Create default config if missing
    create_config_if_missing()
    
    # Get pairs from args or use default from config
    pairs = args.pairs.split(",") if args.pairs else None
    
    # Run optimization
    logger.info(f"Starting optimization for pairs: {pairs if pairs else 'from config'}")
    
    # Log start
    start_time = time.time()
    
    try:
        # Optimize all pairs
        optimize_all_pairs(pairs=pairs, days=args.days)
        
        # Log completion
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Optimization completed in {execution_time:.2f} seconds")
        
        print("\n" + "=" * 80)
        print("Optimization completed successfully!")
        print("See 'optimization_summary.md' for detailed results.")
        print("=" * 80 + "\n")
        
        return 0
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        print("\n" + "=" * 80)
        print(f"Optimization failed: {e}")
        print("See 'optimization_all_pairs.log' for details.")
        print("=" * 80 + "\n")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())