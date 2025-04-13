#!/usr/bin/env python3
"""
Run Strategy Optimization

This script runs the complete strategy optimization and implementation pipeline
to improve trading strategies based on performance data.
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime
import traceback
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
OPTIMIZATION_DIR = "optimization_results"
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_TRADES_FILE = "trades.csv"
os.makedirs(OPTIMIZATION_DIR, exist_ok=True)

def validate_environment():
    """
    Validate the environment for optimization
    
    Returns:
        bool: True if environment is valid
    """
    # Check if required Python packages are installed
    try:
        import numpy
        import pandas
        import matplotlib
        import sklearn
    except ImportError as e:
        logger.error(f"Required package not installed: {e}")
        logger.error("Please install required packages: numpy, pandas, matplotlib, sklearn")
        return False
    
    # Check if trades file exists
    if not os.path.exists(DEFAULT_TRADES_FILE):
        logger.warning(f"Trades file not found: {DEFAULT_TRADES_FILE}")
        logger.warning("Performance analysis will be limited without trade history")
        
        # Create sample trades file for testing
        create_sample_trades()
    
    # Check if strategy files exist
    strategy_files = [
        "trading_strategy.py", 
        "arima_strategy.py",
        "fixed_strategy.py", 
        "integrated_strategy.py"
    ]
    
    missing_files = [f for f in strategy_files if not os.path.exists(f)]
    if missing_files:
        logger.warning(f"Some strategy files not found: {missing_files}")
    
    return True

def create_sample_trades():
    """Create a sample trades file for testing optimization"""
    
    logger.info("Creating sample trades file for testing")
    
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Create sample trades data
        today = datetime.now()
        strategies = ["ARIMAStrategy", "AdaptiveStrategy", "IntegratedStrategy"]
        trade_types = ["long", "short"]
        symbols = ["SOL/USD", "BTC/USD", "ETH/USD"]
        
        # Generate random trades
        rows = []
        for i in range(100):
            strategy = np.random.choice(strategies)
            trade_type = np.random.choice(trade_types)
            symbol = np.random.choice(symbols)
            
            # Set timestamp within lookback period
            days_ago = np.random.randint(1, DEFAULT_LOOKBACK_DAYS)
            timestamp = today - timedelta(days=days_ago)
            
            # Generate entry and exit prices
            base_price = 100 + np.random.randn() * 10
            
            if trade_type == "long":
                entry_price = base_price
                # Long trades usually have exit > entry for profit
                exit_modifier = np.random.normal(1.02, 0.05)  # Slight bias toward profit
                exit_price = entry_price * exit_modifier
            else:  # short
                entry_price = base_price
                # Short trades profit when exit < entry
                exit_modifier = np.random.normal(0.98, 0.05)  # Slight bias toward profit
                exit_price = entry_price * exit_modifier
            
            # Calculate profit/loss
            if trade_type == "long":
                profit_loss = exit_price - entry_price
            else:
                profit_loss = entry_price - exit_price
            
            # Add some transaction costs
            profit_loss -= 0.1
            
            rows.append({
                "strategy": strategy,
                "symbol": symbol,
                "type": trade_type,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "profit_loss": round(profit_loss, 2),
                "timestamp": timestamp,
                "quantity": round(np.random.uniform(0.5, 2.0), 2),
                "trade_id": f"sample_{i}",
                "status": "closed",
            })
        
        # Create DataFrame and save to CSV
        trades_df = pd.DataFrame(rows)
        trades_df.to_csv(DEFAULT_TRADES_FILE, index=False)
        logger.info(f"Created sample trades file with {len(rows)} trades")
        return True
    
    except Exception as e:
        logger.error(f"Error creating sample trades: {e}")
        traceback.print_exc()
        return False

def run_optimizer(lookback_days):
    """
    Run the strategy optimizer
    
    Args:
        lookback_days (int): Lookback period in days
        
    Returns:
        str: Path to recommendations file
    """
    try:
        logger.info(f"Running strategy optimizer with {lookback_days} days lookback")
        
        # Generate timestamped output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OPTIMIZATION_DIR, f"strategy_improvements_{timestamp}.json")
        
        # Command to run optimizer
        cmd = [
            sys.executable,
            "strategy_optimizer.py",
            "--lookback", str(lookback_days),
            "--trades-file", DEFAULT_TRADES_FILE
        ]
        
        # Run optimizer process
        logger.info(f"Executing: {' '.join(cmd)}")
        process = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        # Log output
        logger.info(process.stdout)
        if process.stderr:
            logger.warning(process.stderr)
        
        logger.info(f"Optimizer completed successfully")
        
        # Check if output file was created
        latest_file = find_latest_improvement_file()
        if latest_file:
            logger.info(f"Recommendations saved to: {latest_file}")
            return latest_file
        else:
            logger.error("No recommendations file found after optimization")
            return None
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running optimizer: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return None
    
    except Exception as e:
        logger.error(f"Unexpected error running optimizer: {e}")
        traceback.print_exc()
        return None

def find_latest_improvement_file():
    """
    Find the latest strategy improvement file
    
    Returns:
        str: Path to the latest file
    """
    files = [f for f in os.listdir(OPTIMIZATION_DIR) 
            if f.startswith("strategy_improvements_") and f.endswith(".json")]
    
    if not files:
        return None
        
    # Sort by timestamp in filename
    files.sort(reverse=True)
    return os.path.join(OPTIMIZATION_DIR, files[0])

def run_implementor(recommendations_file):
    """
    Run the strategy implementor
    
    Args:
        recommendations_file (str): Path to recommendations file
        
    Returns:
        bool: Success status
    """
    try:
        logger.info(f"Running strategy implementor with recommendations: {recommendations_file}")
        
        # Command to run implementor
        cmd = [
            sys.executable,
            "strategy_implementor.py",
            "--recommendations", recommendations_file
        ]
        
        # Run implementor process
        logger.info(f"Executing: {' '.join(cmd)}")
        process = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        # Log output
        logger.info(process.stdout)
        if process.stderr:
            logger.warning(process.stderr)
        
        logger.info(f"Implementor completed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running implementor: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False
    
    except Exception as e:
        logger.error(f"Unexpected error running implementor: {e}")
        traceback.print_exc()
        return False

def main():
    """
    Main function to run the complete optimization pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run strategy optimization pipeline")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS, 
                       help=f"Lookback period in days (default: {DEFAULT_LOOKBACK_DAYS})")
    parser.add_argument("--analyze-only", action="store_true", 
                       help="Only analyze strategies, don't implement changes")
    parser.add_argument("--implementation-only", action="store_true", 
                       help="Only implement latest recommendations, don't analyze")
    args = parser.parse_args()
    
    # Display banner
    print("\n" + "=" * 80)
    print(" STRATEGY OPTIMIZATION PIPELINE ".center(80, "="))
    print("=" * 80 + "\n")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        return 1
    
    try:
        # If implementation only, skip analysis
        if args.implementation_only:
            latest_file = find_latest_improvement_file()
            if not latest_file:
                logger.error("No recommendations file found for implementation")
                return 1
            
            logger.info(f"Using latest recommendations: {latest_file}")
            if run_implementor(latest_file):
                logger.info("Implementation completed successfully")
                return 0
            else:
                logger.error("Implementation failed")
                return 1
        
        # Run the optimizer
        recommendations_file = run_optimizer(args.lookback)
        if not recommendations_file:
            logger.error("Optimization failed")
            return 1
        
        # If analyze only, skip implementation
        if args.analyze_only:
            logger.info("Analysis completed successfully (implementation skipped)")
            return 0
        
        # Run the implementor
        if run_implementor(recommendations_file):
            logger.info("Optimization pipeline completed successfully")
            return 0
        else:
            logger.error("Implementation failed")
            return 1
    
    except KeyboardInterrupt:
        logger.info("Optimization pipeline interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())