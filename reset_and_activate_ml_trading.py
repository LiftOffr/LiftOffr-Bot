#!/usr/bin/env python3
"""
Reset Sandbox Portfolio and Activate ML Trading

This script:
1. Resets the sandbox portfolio to starting capital
2. Updates ML configuration for optimized trading
3. Activates ML trading across all supported pairs
4. Enables continuous self-optimization of trading parameters
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_trading_activation.log')
    ]
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_CAPITAL = 20000.0
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD"]
DEFAULT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_BASE_LEVERAGE = 20.0
DEFAULT_MAX_LEVERAGE = 125.0
DEFAULT_MAX_RISK_PER_TRADE = 0.20  # 20% of available capital
DEFAULT_OPTIMIZATION_INTERVAL = 4  # hours

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Reset portfolio and activate ML trading with continuous optimization')
    
    parser.add_argument(
        '--capital',
        type=float,
        default=DEFAULT_CAPITAL,
        help=f'Starting capital for sandbox portfolio (default: {DEFAULT_CAPITAL})'
    )
    
    parser.add_argument(
        '--pairs',
        type=str,
        default=','.join(DEFAULT_PAIRS),
        help=f'Comma-separated list of trading pairs (default: {",".join(DEFAULT_PAIRS)})'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f'Confidence threshold for ML signals (default: {DEFAULT_CONFIDENCE_THRESHOLD})'
    )
    
    parser.add_argument(
        '--base-leverage',
        type=float,
        default=DEFAULT_BASE_LEVERAGE,
        help=f'Base leverage for trades (default: {DEFAULT_BASE_LEVERAGE})'
    )
    
    parser.add_argument(
        '--max-leverage',
        type=float,
        default=DEFAULT_MAX_LEVERAGE,
        help=f'Maximum leverage for high-confidence trades (default: {DEFAULT_MAX_LEVERAGE})'
    )
    
    parser.add_argument(
        '--max-risk',
        type=float,
        default=DEFAULT_MAX_RISK_PER_TRADE,
        help=f'Maximum risk per trade as fraction of capital (default: {DEFAULT_MAX_RISK_PER_TRADE})'
    )
    
    parser.add_argument(
        '--optimization-interval',
        type=int,
        default=DEFAULT_OPTIMIZATION_INTERVAL,
        help=f'Interval in hours between optimization runs (default: {DEFAULT_OPTIMIZATION_INTERVAL})'
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Use live trading instead of sandbox (USE WITH CAUTION)'
    )
    
    return parser.parse_args()

def reset_sandbox_portfolio() -> bool:
    """
    Reset the sandbox portfolio to starting state.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # First, close any open positions
        logger.info("Closing any open positions...")
        subprocess.run(
            ["python", "close_all_positions.py", "--sandbox"],
            check=True
        )
        logger.info("All positions closed successfully")
        
        # Wait for positions to be closed
        time.sleep(5)
        
        # Attempt to stop any running bots
        try:
            subprocess.run(
                ["python", "stop_bot.py"],
                check=True
            )
            logger.info("Stopped any running bots")
        except subprocess.CalledProcessError:
            logger.info("No running bots to stop")
        
        # Wait for bot processes to terminate
        time.sleep(3)
        
        logger.info("Portfolio reset completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to reset portfolio: {str(e)}")
        return False

def update_ml_configuration(
    pairs: List[str],
    confidence_threshold: float,
    base_leverage: float,
    max_leverage: float,
    max_risk: float
) -> bool:
    """
    Update ML configuration with optimized settings.
    
    Args:
        pairs: List of trading pairs
        confidence_threshold: Confidence threshold for ML signals
        base_leverage: Base leverage for trades
        max_leverage: Maximum leverage for high-confidence trades
        max_risk: Maximum risk per trade as fraction of capital
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Updating ML configuration for pairs: {pairs}")
    
    try:
        # Build command for updating ML configuration
        cmd = [
            "python", "update_ml_config.py",
            "--pairs", ",".join(pairs),
            "--confidence", str(confidence_threshold),
            "--base-leverage", str(base_leverage),
            "--max-leverage", str(max_leverage),
            "--max-risk", str(max_risk),
            "--update-strategy", "--update-ensemble"
        ]
        
        subprocess.run(cmd, check=True)
        logger.info("ML configuration updated successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to update ML configuration: {str(e)}")
        return False

def activate_ml_trading(
    pairs: List[str],
    capital: float,
    live_mode: bool = False
) -> bool:
    """
    Activate ML trading for specified pairs.
    
    Args:
        pairs: List of trading pairs
        capital: Starting capital for trading
        live_mode: Whether to use live trading
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Activating ML trading for pairs: {pairs}")
    
    try:
        # Build command for activating ML trading
        cmd = [
            "python", "activate_ml_with_ensembles.py",
            "--pairs", ",".join(pairs),
            "--capital", str(capital)
        ]
        
        if not live_mode:
            cmd.append("--sandbox")
        
        subprocess.run(cmd, check=True)
        logger.info("ML trading activated successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to activate ML trading: {str(e)}")
        return False

def create_optimization_script() -> bool:
    """
    Create a script for continuous optimization of trading parameters.
    
    Returns:
        True if successful, False otherwise
    """
    script_path = "continuous_optimization.py"
    
    try:
        script_content = """#!/usr/bin/env python3
'''
Continuous Optimization Script

This script continuously optimizes trading parameters based on performance data.
It runs periodically and updates the ML configuration based on results.
'''

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
import random
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('optimization.log')
    ]
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_INTERVAL = 4  # hours
DEFAULT_MIN_TRADES = 10
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD"]

def parse_arguments():
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(description='Continuously optimize trading parameters')
    
    parser.add_argument(
        '--interval',
        type=int,
        default=DEFAULT_INTERVAL,
        help=f'Interval in hours between optimization runs (default: {DEFAULT_INTERVAL})'
    )
    
    parser.add_argument(
        '--min-trades',
        type=int,
        default=DEFAULT_MIN_TRADES,
        help=f'Minimum number of trades required for optimization (default: {DEFAULT_MIN_TRADES})'
    )
    
    parser.add_argument(
        '--pairs',
        type=str,
        default=','.join(DEFAULT_PAIRS),
        help=f'Comma-separated list of trading pairs (default: {",".join(DEFAULT_PAIRS)})'
    )
    
    return parser.parse_args()

def get_trading_performance():
    '''Get performance data from trading logs'''
    try:
        # Run get_current_status.py to get current performance
        result = subprocess.run(
            ["python", "get_current_status_new.py", "--detailed"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract performance metrics
        lines = result.stdout.strip().split('\\n')
        
        # Process the output to extract performance metrics
        metrics = {}
        
        for line in lines:
            if "Win Rate:" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    value = parts[1].strip().split("%")[0].strip()
                    metrics["win_rate"] = float(value) / 100.0
            
            if "Total P&L:" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    value_parts = parts[1].strip().split(" ")
                    metrics["pnl"] = float(value_parts[0].replace("$", "").replace(",", ""))
                    metrics["pnl_percent"] = float(value_parts[1].replace("(", "").replace("%)", "")) / 100.0
            
            if "Total Trades:" in line:
                parts = line.split(":")
                if len(parts) > 1:
                    metrics["total_trades"] = int(parts[1].strip())
        
        return metrics
    except Exception as e:
        logger.error(f"Failed to get trading performance: {str(e)}")
        return None

def optimize_parameters(performance, pairs):
    '''Optimize trading parameters based on performance'''
    if not performance:
        logger.error("Cannot optimize without performance data")
        return None
    
    try:
        # Load current ML configuration
        with open("ml_config.json", "r") as f:
            ml_config = json.load(f)
        
        # Get current parameters
        current_confidence = ml_config.get("confidence_threshold", 0.65)
        current_base_leverage = ml_config.get("base_leverage", 20.0)
        current_max_leverage = ml_config.get("max_leverage", 125.0)
        
        # Calculate optimization adjustments based on performance
        win_rate = performance.get("win_rate", 0.5)
        pnl_percent = performance.get("pnl_percent", 0.0)
        total_trades = performance.get("total_trades", 0)
        
        # Adjust confidence threshold based on win rate
        # Higher win rate allows lower confidence threshold
        new_confidence = current_confidence
        if win_rate > 0.6:
            # Good win rate, can be more aggressive
            new_confidence = max(0.6, current_confidence - 0.02)
        elif win_rate < 0.4:
            # Poor win rate, be more conservative
            new_confidence = min(0.75, current_confidence + 0.02)
        
        # Adjust leverage based on P&L
        new_base_leverage = current_base_leverage
        new_max_leverage = current_max_leverage
        
        if pnl_percent > 0.05:  # >5% return
            # Good performance, can increase leverage
            new_base_leverage = min(50.0, current_base_leverage * 1.1)
            new_max_leverage = min(125.0, current_max_leverage * 1.05)
        elif pnl_percent < -0.05:  # <-5% return
            # Poor performance, reduce leverage
            new_base_leverage = max(5.0, current_base_leverage * 0.9)
            new_max_leverage = max(20.0, current_max_leverage * 0.95)
        
        # Apply small random adjustments for exploration
        new_confidence += random.uniform(-0.01, 0.01)
        new_confidence = max(0.55, min(0.8, new_confidence))
        
        new_base_leverage += random.uniform(-1.0, 1.0)
        new_base_leverage = max(5.0, min(50.0, new_base_leverage))
        
        new_max_leverage += random.uniform(-2.0, 2.0)
        new_max_leverage = max(20.0, min(125.0, new_max_leverage))
        
        # Ensure max leverage is greater than base leverage
        new_max_leverage = max(new_max_leverage, new_base_leverage * 1.5)
        
        # Build optimization result
        optimized = {
            "confidence_threshold": round(new_confidence, 2),
            "base_leverage": round(new_base_leverage, 1),
            "max_leverage": round(new_max_leverage, 1),
            "max_risk_per_trade": 0.2,  # Keep this constant for now
            "reason": f"Optimized based on win rate {win_rate:.2f}, P&L {pnl_percent:.2f}, trades {total_trades}"
        }
        
        logger.info(f"Optimization result: {json.dumps(optimized, indent=2)}")
        return optimized
    except Exception as e:
        logger.error(f"Failed to optimize parameters: {str(e)}")
        return None

def apply_optimization(optimization, pairs):
    '''Apply optimized parameters to ML configuration'''
    if not optimization:
        logger.error("Cannot apply empty optimization")
        return False
    
    try:
        # Build command for updating ML configuration
        cmd = [
            "python", "update_ml_config.py",
            "--pairs", ",".join(pairs),
            "--confidence", str(optimization["confidence_threshold"]),
            "--base-leverage", str(optimization["base_leverage"]),
            "--max-leverage", str(optimization["max_leverage"]),
            "--max-risk", str(optimization["max_risk_per_trade"]),
            "--update-strategy", "--update-ensemble"
        ]
        
        subprocess.run(cmd, check=True)
        
        # Save optimization history
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        history_file = f"optimization_history/{timestamp}_optimization.json"
        
        os.makedirs("optimization_history", exist_ok=True)
        
        with open(history_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "parameters": optimization,
                "pairs": pairs
            }, f, indent=2)
        
        logger.info(f"Applied optimization and saved to {history_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to apply optimization: {str(e)}")
        return False

def main():
    '''Main function'''
    args = parse_arguments()
    
    interval_seconds = args.interval * 60 * 60
    pairs = args.pairs.split(",")
    min_trades = args.min_trades
    
    logger.info(f"Starting continuous optimization with interval {args.interval} hours")
    logger.info(f"Monitoring pairs: {pairs}")
    
    # Create optimization history directory
    os.makedirs("optimization_history", exist_ok=True)
    
    while True:
        try:
            logger.info("Running optimization cycle...")
            
            # Get current performance
            performance = get_trading_performance()
            
            if performance and performance.get("total_trades", 0) >= min_trades:
                logger.info(f"Current performance: {json.dumps(performance, indent=2)}")
                
                # Optimize parameters
                optimization = optimize_parameters(performance, pairs)
                
                if optimization:
                    # Apply optimized parameters
                    apply_optimization(optimization, pairs)
            else:
                if not performance:
                    logger.info("Could not retrieve performance data")
                else:
                    logger.info(f"Not enough trades ({performance.get('total_trades', 0)}/{min_trades}) for optimization")
            
            # Sleep until next optimization cycle
            logger.info(f"Sleeping for {args.interval} hours until next optimization...")
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            logger.info("Optimization stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in optimization cycle: {str(e)}")
            # Sleep for a shorter period on error
            time.sleep(60 * 10)  # 10 minutes

if __name__ == "__main__":
    main()
"""
        
        with open(script_path, "w") as f:
            f.write(script_content)
        
        # Make the script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Created optimization script at {script_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create optimization script: {str(e)}")
        return False

def start_continuous_optimization(interval_hours: int) -> bool:
    """
    Start continuous optimization in the background.
    
    Args:
        interval_hours: Interval in hours between optimization runs
        
    Returns:
        True if successfully started, False otherwise
    """
    try:
        # Create directory for optimization history
        os.makedirs("optimization_history", exist_ok=True)
        
        # Start optimization process in the background
        logger.info(f"Starting continuous optimization with interval {interval_hours} hours")
        
        # Use subprocess to start the optimization script
        process = subprocess.Popen(
            ["python", "continuous_optimization.py", "--interval", str(interval_hours)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Check if process started successfully
        if process.poll() is None:
            logger.info(f"Continuous optimization started with PID {process.pid}")
            
            # Save PID to file for later reference
            with open("optimization.pid", "w") as f:
                f.write(str(process.pid))
            
            return True
        else:
            stdout, stderr = process.communicate()
            logger.error(f"Failed to start optimization: {stderr.decode('utf-8')}")
            return False
    except Exception as e:
        logger.error(f"Error starting continuous optimization: {str(e)}")
        return False

def main():
    """Main function."""
    args = parse_arguments()
    
    # Parse pairs list
    pairs = [pair.strip() for pair in args.pairs.split(',')]
    
    logger.info("=" * 80)
    logger.info("STARTING TRADING BOT RESET AND ML ACTIVATION")
    logger.info("=" * 80)
    logger.info(f"Trading pairs: {', '.join(pairs)}")
    logger.info(f"Starting capital: ${args.capital:.2f}")
    logger.info(f"ML confidence threshold: {args.confidence:.2f}")
    logger.info(f"Leverage settings: Base={args.base_leverage:.1f}x, Max={args.max_leverage:.1f}x")
    logger.info(f"Max risk per trade: {args.max_risk * 100:.1f}%")
    logger.info(f"Trading mode: {'LIVE' if args.live else 'SANDBOX'}")
    logger.info(f"Optimization interval: {args.optimization_interval} hours")
    logger.info("=" * 80)
    
    # Confirmation for live mode
    if args.live:
        confirm = input("WARNING: You are about to start LIVE trading. Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            logger.info("Live trading not confirmed. Exiting.")
            return 1
    
    # 1. Reset sandbox portfolio
    logger.info("Step 1: Resetting sandbox portfolio...")
    if not reset_sandbox_portfolio():
        logger.error("Failed to reset portfolio. Exiting.")
        return 1
    
    # 2. Update ML configuration
    logger.info("Step 2: Updating ML configuration...")
    if not update_ml_configuration(
        pairs,
        args.confidence,
        args.base_leverage,
        args.max_leverage,
        args.max_risk
    ):
        logger.error("Failed to update ML configuration. Exiting.")
        return 1
    
    # 3. Create optimization script
    logger.info("Step 3: Creating continuous optimization script...")
    if not create_optimization_script():
        logger.error("Failed to create optimization script. Continuing anyway...")
    
    # 4. Activate ML trading
    logger.info("Step 4: Activating ML trading...")
    if not activate_ml_trading(pairs, args.capital, args.live):
        logger.error("Failed to activate ML trading. Exiting.")
        return 1
    
    # 5. Start continuous optimization
    logger.info("Step 5: Starting continuous optimization...")
    if not start_continuous_optimization(args.optimization_interval):
        logger.error("Failed to start continuous optimization. Trading will continue without optimization.")
    
    logger.info("=" * 80)
    logger.info("TRADING BOT RESET AND ML ACTIVATION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("The bot is now trading with ML models and will continuously optimize its parameters.")
    logger.info("Check the logs and portfolio status regularly to monitor performance.")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())