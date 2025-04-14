#!/usr/bin/env python3
"""
ML Live Bot Runner

This script runs the ML-enhanced trading bot in live mode with the collaborative
strategy system. It combines:

1. Advanced ML models for prediction
2. Strategy ensemble for collaborative decision making
3. Dynamic position sizing with extreme leverage settings
4. Multi-asset trading across SOL/USD, ETH/USD, and BTC/USD

Usage:
    python run_ml_live_bot.py --sandbox --pairs "SOL/USD" "ETH/USD" "BTC/USD" --extreme-leverage
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

from ml_live_trading_integration import MLTradingBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_live_bot.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run the ML Live Trading Bot"""
    parser = argparse.ArgumentParser(description='Run ML-enhanced trading bot')
    
    parser.add_argument('--pairs', nargs='+', default=["SOL/USD"],
                      help='Trading pairs to trade (default: SOL/USD)')
    
    parser.add_argument('--data-dir', type=str, default='historical_data',
                      help='Directory containing historical data')
    
    parser.add_argument('--model-dir', type=str, default='models',
                      help='Directory containing trained ML models')
    
    parser.add_argument('--config', type=str, default='config/trading_config.json',
                      help='Path to trading configuration')
    
    parser.add_argument('--interval', type=int, default=60,
                      help='Seconds between trading iterations')
    
    parser.add_argument('--extreme-leverage', action='store_true',
                      help='Use extreme leverage settings (20-125x)')
    
    parser.add_argument('--ml-position-sizing', action='store_true',
                      help='Use ML-based position sizing')
    
    parser.add_argument('--live', action='store_true',
                      help='Run in live trading mode (not sandbox)')
    
    parser.add_argument('--train-first', action='store_true',
                      help='Train models before starting trading')
    
    args = parser.parse_args()
    
    # Print run configuration
    logger.info("Starting ML Live Trading Bot with configuration:")
    logger.info(f"  Trading pairs: {args.pairs}")
    logger.info(f"  Mode: {'LIVE' if args.live else 'SANDBOX'}")
    logger.info(f"  Extreme leverage: {args.extreme_leverage}")
    logger.info(f"  ML position sizing: {args.ml_position_sizing}")
    logger.info(f"  Checking interval: {args.interval} seconds")
    
    # Training first if requested
    if args.train_first:
        logger.info("Training models before starting trading")
        
        from train_ml_live_integration import MLLiveTrainer
        
        trainer = MLLiveTrainer(
            assets=args.pairs,
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            use_extreme_leverage=args.extreme_leverage
        )
        
        trainer.run_full_training_pipeline(
            optimize=True,
            visualize=True
        )
    
    # Create directories if not exist
    os.makedirs(os.path.dirname(args.config), exist_ok=True)
    
    # Create config file if not exists
    if not os.path.exists(args.config):
        config = {
            "assets": args.pairs,
            "sandbox_mode": not args.live,
            "use_extreme_leverage": args.extreme_leverage,
            "use_ml_position_sizing": args.ml_position_sizing,
            "trading_interval": args.interval,
            "max_leverage": {
                "SOL/USD": 125.0,
                "ETH/USD": 100.0,
                "BTC/USD": 85.0
            },
            "min_leverage": {
                "SOL/USD": 20.0,
                "ETH/USD": 15.0,
                "BTC/USD": 12.0
            },
            "capital_allocation": {
                "SOL/USD": 0.4,
                "ETH/USD": 0.35,
                "BTC/USD": 0.25
            }
        }
        
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Created default configuration at {args.config}")
    
    # Initialize and run the bot
    bot = MLTradingBot(
        trading_pairs=args.pairs,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        config_path=args.config,
        use_extreme_leverage=args.extreme_leverage,
        use_ml_position_sizing=args.ml_position_sizing,
        sandbox_mode=not args.live
    )
    
    # Run the bot
    try:
        bot.run(interval=args.interval)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}")
    finally:
        logger.info("Bot shutting down")

if __name__ == "__main__":
    main()