#!/usr/bin/env python3
"""
Start ML-Enhanced Trading Bot

This script starts the Kraken trading bot with ML-enhanced strategies.
It initializes the bot with both original and ML-enhanced strategies,
allowing for side-by-side comparison and trading.

Usage:
    python start_ml_enhanced_trading.py [--sandbox] [--capital AMOUNT] [--leverage LEVEL]
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
import signal
import json

# Import bot and strategy components
from kraken_trading_bot import KrakenTradingBot
from bot_manager import BotManager
from arima_strategy import ARIMAStrategy
from fixed_strategy import AdaptiveStrategy
from trading_strategy import TradingStrategy

# Import ML-enhanced strategy wrapper
from ml_enhanced_strategy import enhance_strategy, MLEnhancedStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default configuration
TRADING_PAIR = "SOL/USD"
INITIAL_CAPITAL = 20000.0
DEFAULT_LEVERAGE = 25
MARGIN_PERCENT = 20.0  # Use 20% of capital per trade
ML_INFLUENCE = 0.5     # 50% influence from ML predictions
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for ML to influence decisions

def create_ml_enhanced_bot(trading_pair=TRADING_PAIR, sandbox_mode=True, 
                          initial_capital=INITIAL_CAPITAL, leverage=DEFAULT_LEVERAGE, 
                          margin_percent=MARGIN_PERCENT, ml_influence=ML_INFLUENCE,
                          confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Create ML-enhanced trading bot
    
    Args:
        trading_pair (str): Trading pair
        sandbox_mode (bool): Whether to run in sandbox mode
        initial_capital (float): Initial capital
        leverage (int): Leverage level
        margin_percent (float): Margin percentage per trade
        ml_influence (float): Influence of ML predictions
        confidence_threshold (float): Confidence threshold for ML
        
    Returns:
        KrakenTradingBot: Configured trading bot
    """
    logger.info(f"Creating ML-Enhanced Trading Bot for {trading_pair}")
    logger.info(f"Configuration: sandbox={sandbox_mode}, capital=${initial_capital}, "
               f"leverage={leverage}x, margin={margin_percent}%, "
               f"ML influence={ml_influence}, confidence threshold={confidence_threshold}")
    
    # Create original strategies
    arima_strategy = ARIMAStrategy(symbol=trading_pair.replace('/', ''))
    adaptive_strategy = AdaptiveStrategy(symbol=trading_pair.replace('/', ''))
    
    # Create ML-enhanced versions
    ml_arima = enhance_strategy(
        strategy=arima_strategy,
        trading_pair=trading_pair,
        ml_influence=ml_influence,
        confidence_threshold=confidence_threshold
    )
    
    ml_adaptive = enhance_strategy(
        strategy=adaptive_strategy,
        trading_pair=trading_pair,
        ml_influence=ml_influence,
        confidence_threshold=confidence_threshold
    )
    
    # Create bot manager
    manager = BotManager()
    
    # Create trading bot
    bot = KrakenTradingBot(
        trading_pair=trading_pair,
        sandbox_mode=sandbox_mode,
        initial_capital=initial_capital,
        leverage=leverage,
        margin_percent=margin_percent,
        bot_manager=manager
    )
    
    # Add original strategies (for comparison)
    bot.add_strategy(arima_strategy, name="arima")
    bot.add_strategy(adaptive_strategy, name="adaptive")
    
    # Add ML-enhanced strategies
    bot.add_strategy(ml_arima, name="ml_arima")
    bot.add_strategy(ml_adaptive, name="ml_adaptive")
    
    logger.info(f"Trading bot created with {len(bot.strategies)} strategies")
    return bot

def handle_shutdown(signum, frame):
    """Handle shutdown signals"""
    logger.info("Shutdown signal received, stopping bot...")
    sys.exit(0)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start ML-Enhanced Kraken Trading Bot")
    parser.add_argument("--pair", type=str, default=TRADING_PAIR,
                        help=f"Trading pair to trade (default: {TRADING_PAIR})")
    parser.add_argument("--sandbox", action="store_true",
                        help="Run in sandbox mode (no real trades)")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL,
                        help=f"Initial capital (default: ${INITIAL_CAPITAL})")
    parser.add_argument("--leverage", type=int, default=DEFAULT_LEVERAGE,
                        help=f"Leverage level (default: {DEFAULT_LEVERAGE}x)")
    parser.add_argument("--margin", type=float, default=MARGIN_PERCENT,
                        help=f"Margin percent per trade (default: {MARGIN_PERCENT}%)")
    parser.add_argument("--ml-influence", type=float, default=ML_INFLUENCE,
                        help=f"ML influence weight (default: {ML_INFLUENCE})")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"ML confidence threshold (default: {CONFIDENCE_THRESHOLD})")
    
    return parser.parse_args()

def main():
    """Main function to start ML-enhanced trading"""
    # Parse arguments
    args = parse_arguments()
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Create and start bot
    bot = create_ml_enhanced_bot(
        trading_pair=args.pair,
        sandbox_mode=args.sandbox,
        initial_capital=args.capital,
        leverage=args.leverage,
        margin_percent=args.margin,
        ml_influence=args.ml_influence,
        confidence_threshold=args.confidence
    )
    
    try:
        # Start the bot
        logger.info(f"Starting ML-Enhanced Trading Bot for {args.pair}")
        bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save bot state
        logger.info("Saving bot state and shutting down...")
        if hasattr(bot, 'save_state'):
            bot.save_state()

if __name__ == "__main__":
    main()