#!/usr/bin/env python3
"""
Start ML Trading Bot

This script provides a simple entry point for starting the ML-enhanced trading bot
with the specified configuration.
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('start_ml_trading.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Start ML trading bot')
    
    parser.add_argument('--assets', nargs='+', default=["SOL/USD"],
                       help='Assets to trade (default: SOL/USD)')
    
    parser.add_argument('--live', action='store_true',
                       help='Enable live trading (use with caution)')
    
    parser.add_argument('--reset-portfolio', action='store_true',
                       help='Reset portfolio to initial state')
    
    parser.add_argument('--initial-capital', type=float, default=20000.0,
                       help='Initial capital in USD (default: 20000.0)')
    
    parser.add_argument('--optimize-first', action='store_true',
                       help='Run optimization before trading')
    
    parser.add_argument('--interval', type=int, default=60,
                       help='Trading interval in seconds (default: 60)')
    
    parser.add_argument('--max-iterations', type=int, default=None,
                       help='Maximum number of trading iterations')
    
    parser.add_argument('--extreme-leverage', action='store_true', default=True,
                       help='Use extreme leverage settings (default: True)')
    
    parser.add_argument('--ml-position-sizing', action='store_true', default=True,
                       help='Use ML for position sizing (default: True)')
    
    return parser.parse_args()

def reset_portfolio(initial_capital: float) -> bool:
    """Reset portfolio to initial state"""
    try:
        logger.info(f"Resetting portfolio to ${initial_capital:.2f}")
        
        # Import necessary modules
        from bot_manager import BotManager
        
        # Create a new bot manager
        bot_manager = BotManager(
            initial_capital=initial_capital,
            sandbox_mode=True
        )
        
        # Save state
        logger.info("Portfolio reset successful")
        return True
        
    except Exception as e:
        logger.error(f"Failed to reset portfolio: {e}")
        return False

def optimize_ml_models(assets: List[str]) -> bool:
    """Run ML model optimization"""
    try:
        logger.info(f"Optimizing ML models for {assets}")
        
        # Import optimization module
        from ml_live_training_optimizer import optimize_all_models
        
        # Run optimization
        success = optimize_all_models(assets)
        
        if success:
            logger.info("ML model optimization completed successfully")
        else:
            logger.error("ML model optimization failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error optimizing ML models: {e}")
        return False

def create_model_directories() -> bool:
    """Create necessary directories for ML models"""
    try:
        # Create model directories
        directories = [
            "models",
            "models/ensemble",
            "models/transformer",
            "models/tcn",
            "models/lstm"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating model directories: {e}")
        return False

def start_ml_trading(args) -> bool:
    """Start ML trading bot"""
    try:
        # Import necessary modules
        from ml_live_trading_integration import MLLiveTradingIntegration
        from model_collaboration_integrator import ModelCollaborationIntegrator
        from bot_manager_integration import MLTradingBotManager
        
        # Create ML integration
        ml_integration = MLLiveTradingIntegration(
            trading_pairs=args.assets,
            use_extreme_leverage=args.extreme_leverage
        )
        
        # Create model collaboration
        model_collaboration = ModelCollaborationIntegrator(
            trading_pairs=args.assets,
            enable_adaptive_weights=True
        )
        
        # Create ML trading bot manager
        bot_manager = MLTradingBotManager(
            trading_pairs=args.assets,
            initial_capital=args.initial_capital,
            ml_integration=ml_integration,
            model_collaboration=model_collaboration,
            sandbox_mode=not args.live,
            use_ml_position_sizing=args.ml_position_sizing
        )
        
        # Main trading loop
        iteration = 0
        running = True
        
        logger.info("Starting trading loop")
        
        try:
            while running:
                # Check if max iterations reached
                if args.max_iterations and iteration >= args.max_iterations:
                    logger.info(f"Reached maximum iterations ({args.max_iterations})")
                    break
                
                # Log iteration
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"Iteration {iteration} at {current_time}")
                
                # Update market data
                bot_manager.update_market_data()
                
                # Evaluate strategies
                bot_manager.evaluate_strategies()
                
                # Execute trades
                bot_manager.execute_trades()
                
                # Display status every 5 iterations
                if iteration % 5 == 0:
                    bot_manager.display_status()
                
                # Pause between iterations
                time.sleep(args.interval)
                
                # Increment iteration counter
                iteration += 1
                
        except KeyboardInterrupt:
            logger.info("Trading interrupted by user")
            running = False
        
        # Final status
        bot_manager.display_status()
        
        # Save model collaboration weights
        if model_collaboration:
            model_collaboration.save_weights()
        
        logger.info("Trading completed")
        return True
        
    except Exception as e:
        logger.error(f"Error starting ML trading: {e}")
        return False

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Print banner
    print("=" * 80)
    print(f"ML TRADING BOT - {'LIVE' if args.live else 'SANDBOX'} MODE")
    print("=" * 80)
    print(f"Trading assets: {args.assets}")
    print(f"Initial capital: ${args.initial_capital:.2f}")
    print(f"Extreme leverage: {args.extreme_leverage}")
    print(f"ML position sizing: {args.ml_position_sizing}")
    print(f"Trading interval: {args.interval} seconds")
    if args.max_iterations:
        print(f"Maximum iterations: {args.max_iterations}")
    print("=" * 80)
    
    # Confirm live trading
    if args.live:
        confirm = input("You are about to start LIVE trading. Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            print("Live trading not confirmed. Exiting.")
            return
    
    # Create model directories
    if not create_model_directories():
        print("Failed to create model directories. Exiting.")
        return
    
    # Reset portfolio if requested
    if args.reset_portfolio:
        if not reset_portfolio(args.initial_capital):
            print("Failed to reset portfolio. Exiting.")
            return
    
    # Optimize ML models if requested
    if args.optimize_first:
        if not optimize_ml_models(args.assets):
            print("Failed to optimize ML models. Exiting.")
            return
    
    # Start ML trading
    if not start_ml_trading(args):
        print("Failed to start ML trading. Exiting.")
        return
    
    print("ML trading completed.")

if __name__ == "__main__":
    main()