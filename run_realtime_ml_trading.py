#!/usr/bin/env python3
"""
Run Realtime ML Trading

This script activates the ML-driven real-time trading system in sandbox mode.
It manages connections to Kraken's WebSocket API and ML models to optimize
trading decisions in real-time.
"""
import argparse
import os
import sys
import logging
from typing import List

# Import our modules
from realtime_ml_manager import RealtimeMLManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default trading pairs
DEFAULT_PAIRS = [
    "SOL/USD", 
    "BTC/USD", 
    "ETH/USD", 
    "ADA/USD", 
    "DOT/USD",
    "LINK/USD",
    "AVAX/USD",
    "MATIC/USD",
    "UNI/USD",
    "ATOM/USD"
]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run Realtime ML Trading")
    
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        help="Trading pairs to use"
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=20000.0,
        help="Initial capital in USD"
    )
    
    parser.add_argument(
        "--max-positions",
        type=int,
        default=7,
        help="Maximum number of open positions"
    )
    
    parser.add_argument(
        "--allocation",
        type=float,
        default=0.85,
        help="Maximum portfolio allocation percentage (0-1)"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live mode (no sandbox)"
    )
    
    return parser.parse_args()

def create_data_directory():
    """Create data directory if it doesn't exist"""
    os.makedirs("data", exist_ok=True)

def check_api_keys():
    """Check if required API keys are set"""
    required_keys = ["KRAKEN_API_KEY", "KRAKEN_API_SECRET"]
    missing_keys = [key for key in required_keys if not os.environ.get(key)]
    
    if missing_keys:
        logger.error(f"Missing required environment variables: {', '.join(missing_keys)}")
        logger.error("Please set the required API keys in your environment or .env file")
        return False
    
    return True

def reset_portfolio(initial_capital: float):
    """Reset portfolio for sandbox mode"""
    import json
    
    # Create portfolio data
    portfolio = {
        "initial_capital": initial_capital,
        "available_capital": initial_capital,
        "total_value": initial_capital,
        "last_updated": "2023-01-01T00:00:00"
    }
    
    # Save to file
    with open("data/sandbox_portfolio.json", "w") as f:
        json.dump(portfolio, f, indent=2)
    
    # Create empty positions and trades files
    with open("data/sandbox_positions.json", "w") as f:
        json.dump([], f, indent=2)
    
    with open("data/sandbox_trades.json", "w") as f:
        json.dump([], f, indent=2)
    
    # Create empty portfolio history file
    with open("data/sandbox_portfolio_history.json", "w") as f:
        json.dump([], f, indent=2)
    
    logger.info(f"Reset portfolio to ${initial_capital:.2f}")

def run_realtime_ml_trading(
    pairs: List[str],
    initial_capital: float = 20000.0,
    max_open_positions: int = 7,
    max_allocation_pct: float = 0.85,
    sandbox: bool = True
):
    """
    Run the realtime ML trading system
    
    Args:
        pairs: List of trading pairs
        initial_capital: Initial capital in USD
        max_open_positions: Maximum number of open positions
        max_allocation_pct: Maximum portfolio allocation percentage
        sandbox: Whether to run in sandbox mode
    """
    # Create the Realtime ML Manager
    ml_manager = RealtimeMLManager(
        trading_pairs=pairs,
        initial_capital=initial_capital,
        max_open_positions=max_open_positions,
        max_allocation_pct=max_allocation_pct,
        sandbox=sandbox
    )
    
    # Start the manager
    ml_manager.start()
    
    logger.info(f"Started Realtime ML Trading for {len(pairs)} pairs")
    
    try:
        # Keep running until interrupted
        import time
        while True:
            # Print status periodically
            status = ml_manager.get_status()
            logger.info(f"Current status: {len(status.get('positions', []))} "
                       f"open positions, ${status.get('portfolio_value', 0):.2f} "
                       f"portfolio value")
            
            # Sleep for 5 minutes
            time.sleep(300)
    
    except KeyboardInterrupt:
        logger.info("Stopping Realtime ML Trading due to user interrupt")
    
    finally:
        # Stop the manager
        ml_manager.stop()
        logger.info("Realtime ML Trading stopped")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create data directory
    create_data_directory()
    
    # Check API keys
    if not check_api_keys():
        sys.exit(1)
    
    # Reset portfolio for sandbox mode
    if not args.live:
        reset_portfolio(args.capital)
    
    # Run the realtime ML trading system
    run_realtime_ml_trading(
        pairs=args.pairs,
        initial_capital=args.capital,
        max_open_positions=args.max_positions,
        max_allocation_pct=args.allocation,
        sandbox=not args.live
    )

if __name__ == "__main__":
    main()