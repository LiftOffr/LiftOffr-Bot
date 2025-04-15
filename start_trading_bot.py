#!/usr/bin/env python3
"""
Simple script to start the trading bot without Flask dependencies
"""
import os
import sys
import time
import logging
import argparse
import integrated_trading_bot as bot

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start the trading bot")
    
    parser.add_argument("--sandbox", action="store_true", default=True,
                        help="Run in sandbox mode (default: True)")
    
    parser.add_argument("--pairs", type=str, nargs="+", 
                        default=["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
                                 "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"],
                        help="Trading pairs to use")
    
    return parser.parse_args()

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print(" KRAKEN TRADING BOT WITH ML-BASED RISK MANAGEMENT")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Check for Kraken API credentials
    if not args.sandbox and (not os.environ.get("KRAKEN_API_KEY") or not os.environ.get("KRAKEN_API_SECRET")):
        logger.warning("Kraken API credentials not found, forcing sandbox mode")
        args.sandbox = True
    
    # Start trading bot
    print(f"\nStarting trading bot in {'sandbox' if args.sandbox else 'live'} mode")
    print(f"Trading pairs: {', '.join(args.pairs)}")
    print("\nPress Ctrl+C to stop the bot at any time\n")
    
    trading_bot = bot.IntegratedTradingBot(
        trading_pairs=args.pairs,
        sandbox=args.sandbox
    )
    
    try:
        trading_bot.start()
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user")
    except Exception as e:
        logger.error(f"Error running trading bot: {e}", exc_info=True)
        return 1
    finally:
        if trading_bot:
            trading_bot.cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())