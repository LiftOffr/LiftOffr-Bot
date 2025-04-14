#!/usr/bin/env python3
"""
Run ML Live Trading Bot

This script initializes and runs the ML-enhanced trading bot with
real-time market data from Kraken.
"""

import os
import sys
import json
import time
import logging
import argparse
import signal
import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_bot.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run ML-enhanced trading bot')
    
    parser.add_argument('--pairs', nargs='+', default=["SOL/USD"],
                      help='Trading pairs to trade')
    
    parser.add_argument('--live', action='store_true',
                      help='Run in live trading mode (use with caution)')
    
    parser.add_argument('--sandbox', action='store_true', default=True,
                      help='Run in sandbox mode (default)')
    
    parser.add_argument('--extreme-leverage', action='store_true', default=True,
                      help='Use extreme leverage settings')
    
    parser.add_argument('--ml-position-sizing', action='store_true', default=True,
                      help='Use ML for position sizing')
    
    parser.add_argument('--interval', type=int, default=60,
                      help='Trading interval in seconds')
    
    parser.add_argument('--initial-capital', type=float, default=20000.0,
                      help='Initial trading capital')
    
    parser.add_argument('--max-iterations', type=int, default=None,
                      help='Maximum number of iterations')
    
    args = parser.parse_args()
    
    # Override sandbox if live is specified
    if args.live:
        args.sandbox = False
    
    return args

def initialize_ml_integration(trading_pairs, extreme_leverage=True):
    """
    Initialize the ML trading integration
    
    Args:
        trading_pairs: List of trading pairs
        extreme_leverage: Whether to use extreme leverage settings
        
    Returns:
        object: Initialized ML integration
    """
    try:
        # Import here to avoid circular imports
        from ml_live_trading_integration import MLLiveTradingIntegration
        
        logger.info(f"Initializing ML integration for {len(trading_pairs)} pairs")
        
        ml_integration = MLLiveTradingIntegration(
            trading_pairs=trading_pairs,
            use_extreme_leverage=extreme_leverage
        )
        
        return ml_integration
        
    except Exception as e:
        logger.error(f"Error initializing ML integration: {e}")
        return None

def initialize_model_collaboration(trading_pairs):
    """
    Initialize the model collaboration integrator
    
    Args:
        trading_pairs: List of trading pairs
        
    Returns:
        object: Initialized model collaboration integrator
    """
    try:
        # Import here to avoid circular imports
        from model_collaboration_integrator import ModelCollaborationIntegrator
        
        logger.info(f"Initializing model collaboration for {len(trading_pairs)} pairs")
        
        model_collaboration = ModelCollaborationIntegrator(
            trading_pairs=trading_pairs
        )
        
        return model_collaboration
        
    except Exception as e:
        logger.error(f"Error initializing model collaboration: {e}")
        return None

def reset_portfolio(initial_capital=20000.0):
    """
    Reset the portfolio to its initial state
    
    Args:
        initial_capital: Initial trading capital
        
    Returns:
        bool: Whether reset was successful
    """
    try:
        from bot_manager import BotManager
        
        logger.info(f"Resetting portfolio to initial capital of ${initial_capital:.2f}")
        
        # Create a new bot manager
        bot_manager = BotManager(initial_capital=initial_capital)
        
        # Save the bot manager state
        bot_manager.save_state()
        
        logger.info("Portfolio reset successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error resetting portfolio: {e}")
        return False

def initialize_trading_bot(args):
    """
    Initialize the ML-enhanced trading bot
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (bot_manager, ml_integration, model_collaboration)
    """
    try:
        from bot_manager import BotManager
        from bot_manager_integration import MLTradingBotManager
        
        # Reset portfolio if requested
        reset_portfolio(args.initial_capital)
        
        # Initialize ML integration
        ml_integration = initialize_ml_integration(
            args.pairs, 
            extreme_leverage=args.extreme_leverage
        )
        
        # Initialize model collaboration
        model_collaboration = initialize_model_collaboration(args.pairs)
        
        # Create ML-enhanced bot manager
        bot_manager = MLTradingBotManager(
            trading_pairs=args.pairs,
            initial_capital=args.initial_capital,
            ml_integration=ml_integration,
            model_collaboration=model_collaboration,
            sandbox_mode=args.sandbox,
            use_ml_position_sizing=args.ml_position_sizing
        )
        
        return bot_manager, ml_integration, model_collaboration
        
    except Exception as e:
        logger.error(f"Error initializing trading bot: {e}")
        return None, None, None

def run_trading_bot(args):
    """
    Run the ML-enhanced trading bot
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: Whether execution was successful
    """
    try:
        logger.info(f"Starting ML trading bot in {'SANDBOX' if args.sandbox else 'LIVE'} mode")
        logger.info(f"Trading pairs: {args.pairs}")
        logger.info(f"Interval: {args.interval} seconds")
        logger.info(f"Extreme leverage: {args.extreme_leverage}")
        logger.info(f"ML position sizing: {args.ml_position_sizing}")
        
        # Initialize bot
        bot_manager, ml_integration, model_collaboration = initialize_trading_bot(args)
        
        if bot_manager is None:
            logger.error("Failed to initialize trading bot")
            return False
        
        # Main trading loop
        iteration = 0
        running = True
        
        def signal_handler(sig, frame):
            nonlocal running
            logger.info("Received shutdown signal, exiting gracefully...")
            running = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        while running:
            try:
                # Check if we've reached maximum iterations
                if args.max_iterations is not None and iteration >= args.max_iterations:
                    logger.info(f"Reached maximum iterations ({args.max_iterations}), exiting")
                    break
                
                # Get current time
                now = datetime.datetime.now()
                
                # Update market data
                logger.info(f"Iteration {iteration}: Updating market data at {now}")
                bot_manager.update_market_data()
                
                # Update ML models
                if model_collaboration:
                    model_collaboration.update_market_regime(bot_manager.get_market_data())
                
                # Generate ML signals
                if ml_integration:
                    ml_signals = ml_integration.generate_trading_signals(bot_manager.get_market_data())
                    for pair, signal in ml_signals.items():
                        bot_manager.process_ml_signal(pair, signal)
                
                # Execute trading logic
                bot_manager.evaluate_strategies()
                
                # Display status
                if iteration % 5 == 0:
                    bot_manager.display_status()
                
                # Wait for next interval
                time.sleep(args.interval)
                
                # Increment iteration counter
                iteration += 1
                
            except Exception as e:
                logger.error(f"Error in trading iteration {iteration}: {e}")
                time.sleep(5)  # Wait a bit before retry
        
        # Final status display
        bot_manager.display_status()
        
        logger.info("Trading bot execution completed")
        return True
        
    except Exception as e:
        logger.error(f"Error running trading bot: {e}")
        return False

def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Print banner
        print("=" * 80)
        print("ML-ENHANCED TRADING BOT")
        print("=" * 80)
        print(f"Trading pairs: {args.pairs}")
        print(f"Mode: {'LIVE' if args.live else 'SANDBOX'}")
        print(f"Interval: {args.interval} seconds")
        print(f"Initial capital: ${args.initial_capital:.2f}")
        print(f"Extreme leverage: {args.extreme_leverage}")
        print(f"ML position sizing: {args.ml_position_sizing}")
        if args.max_iterations:
            print(f"Max iterations: {args.max_iterations}")
        print("=" * 80)
        
        # Confirm live trading
        if args.live:
            confirm = input("You are about to start LIVE trading. Type 'CONFIRM' to proceed: ")
            if confirm != "CONFIRM":
                print("Live trading not confirmed. Exiting.")
                return
        
        # Run the trading bot
        success = run_trading_bot(args)
        if not success:
            print("Trading bot execution failed.")
            return
        
    except KeyboardInterrupt:
        print("\nTrading bot interrupted by user.")
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    main()