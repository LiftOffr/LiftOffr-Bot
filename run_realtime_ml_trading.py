#!/usr/bin/env python3
"""
Run Realtime ML Trading Bot

This script starts the real-time ML trading bot with WebSocket integration
for Kraken, ensuring trades are managed with the most current market data.
"""
import os
import sys
import time
import json
import logging
import argparse
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import our modules only after logging is configured
from realtime_ml_manager import RealtimeMLManager

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run real-time ML trading bot with WebSockets")
    
    parser.add_argument("--pairs", type=str, default="SOL/USD,BTC/USD,ETH/USD,ADA/USD,DOT/USD,LINK/USD,AVAX/USD,MATIC/USD,UNI/USD,ATOM/USD",
                        help="Comma-separated list of trading pairs")
    
    parser.add_argument("--capital", type=float, default=20000.0,
                        help="Initial capital in USD")
    
    parser.add_argument("--max-positions", type=int, default=7,
                        help="Maximum number of open positions")
    
    parser.add_argument("--max-allocation", type=float, default=0.85,
                        help="Maximum portfolio allocation (0.0-1.0)")
    
    parser.add_argument("--model-weights", type=str, default=None,
                        help="Path to model weights file (optional)")
    
    parser.add_argument("--sandbox", action="store_true", default=True,
                        help="Run in sandbox mode (default: True)")
    
    parser.add_argument("--no-sandbox", action="store_false", dest="sandbox",
                        help="Run in live mode (not sandbox)")
    
    parser.add_argument("--reset-portfolio", action="store_true",
                        help="Reset portfolio before starting")
    
    return parser.parse_args()

def reset_portfolio(capital: float):
    """Reset portfolio to initial state"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Create fresh portfolio
    portfolio = {
        "initial_capital": capital,
        "available_capital": capital,
        "balance": capital,
        "equity": capital,
        "total_value": capital,
        "total_pnl": 0.0,
        "total_pnl_pct": 0.0,
        "unrealized_pnl": 0.0,
        "unrealized_pnl_pct": 0.0,
        "unrealized_pnl_usd": 0.0,
        "win_rate": 0.0,
        "total_trades": 0,
        "profitable_trades": 0,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "open_positions_count": 0,
        "margin_used_pct": 0.0,
        "available_margin": capital
    }
    
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)
    
    # Reset positions to empty list
    with open(POSITIONS_FILE, 'w') as f:
        json.dump([], f, indent=2)
    
    # Reset trades to empty list
    with open(TRADES_FILE, 'w') as f:
        json.dump([], f, indent=2)
    
    # Initialize portfolio history
    with open(PORTFOLIO_HISTORY_FILE, 'w') as f:
        json.dump([{
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "portfolio_value": capital
        }], f, indent=2)
    
    logger.info(f"Portfolio reset to ${capital:,.2f}")
    return True

def validate_websocket_connection():
    """Validate WebSocket connection to Kraken"""
    try:
        import kraken_websocket_test
        
        logger.info("Testing WebSocket connection to Kraken...")
        result = kraken_websocket_test.test_connection()
        
        if result:
            logger.info("✓ WebSocket connection successful")
            return True
        else:
            logger.error("× WebSocket connection failed")
            return False
    
    except Exception as e:
        logger.error(f"Error testing WebSocket connection: {e}")
        return False

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Parse trading pairs
    trading_pairs = args.pairs.split(',')
    logger.info(f"Starting real-time ML trading bot for {len(trading_pairs)} pairs: {args.pairs}")
    
    # Reset portfolio if requested
    if args.reset_portfolio:
        reset_portfolio(args.capital)
    
    # Validate WebSocket connection
    if not validate_websocket_connection():
        logger.error("Cannot proceed without WebSocket connection. Please check your network and try again.")
        return 1
    
    # Create Realtime ML Manager
    manager = RealtimeMLManager(
        trading_pairs=trading_pairs,
        initial_capital=args.capital,
        max_open_positions=args.max_positions,
        max_allocation_pct=args.max_allocation,
        model_weight_path=args.model_weights,
        sandbox=args.sandbox
    )
    
    # Start the manager
    logger.info("Starting Realtime ML Manager...")
    if not manager.start():
        logger.error("Failed to start Realtime ML Manager")
        return 1
    
    logger.info("=" * 70)
    logger.info("REALTIME ML TRADING BOT IS RUNNING")
    logger.info("=" * 70)
    logger.info(f"Mode: {'Sandbox' if args.sandbox else 'Live'}")
    logger.info(f"Trading pairs: {', '.join(trading_pairs)}")
    logger.info(f"Initial capital: ${args.capital:,.2f}")
    logger.info(f"Maximum positions: {args.max_positions}")
    logger.info(f"Maximum allocation: {args.max_allocation * 100:.1f}%")
    logger.info("=" * 70)
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 70)
    
    try:
        # Main loop
        while True:
            # Get status every 5 minutes
            status = manager.get_status()
            
            # Display current status
            logger.info("-" * 50)
            logger.info("CURRENT STATUS:")
            logger.info(f"Open positions: {status['open_positions']}")
            logger.info(f"Closed trades: {status['closed_trades']}")
            logger.info(f"Available capital: ${status['portfolio']['available_capital']:,.2f}")
            logger.info(f"Total portfolio value: ${status['portfolio']['total_value']:,.2f}")
            logger.info(f"Unrealized PnL: ${status['portfolio']['unrealized_pnl']:,.2f} ({status['portfolio']['unrealized_pnl_pct']:.2f}%)")
            logger.info("-" * 50)
            
            # Sleep for 5 minutes
            time.sleep(300)
    
    except KeyboardInterrupt:
        logger.info("User interrupted. Stopping Realtime ML Manager...")
    
    finally:
        # Stop the manager
        manager.stop()
        logger.info("Realtime ML trading bot stopped")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())