#!/usr/bin/env python3
"""
Ensemble Training Script

This script runs the strategy ensemble trainer to improve how multiple models
work together, optimizing their collective intelligence rather than individual performance.

Usage:
    python run_ensemble_training.py --assets "SOL/USD" "ETH/USD" "BTC/USD" --training-days 60
"""

import os
import sys
import logging
import argparse
from datetime import datetime

from strategy_ensemble_trainer import StrategyEnsembleTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ensemble_training.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train strategy ensemble')
    
    parser.add_argument('--assets', nargs='+', default=["SOL/USD", "ETH/USD", "BTC/USD"],
                      help='Assets to train on (default: SOL/USD ETH/USD BTC/USD)')
    
    parser.add_argument('--strategies', nargs='+', 
                      default=["ARIMAStrategy", "AdaptiveStrategy", "IntegratedStrategy", "MLStrategy"],
                      help='Strategies to include in ensemble (default: ARIMAStrategy AdaptiveStrategy IntegratedStrategy MLStrategy)')
    
    parser.add_argument('--timeframes', nargs='+', default=["5m", "15m", "1h", "4h"],
                      help='Timeframes to use (default: 5m 15m 1h 4h)')
    
    parser.add_argument('--training-days', type=int, default=90,
                      help='Days of data for training (default: 90)')
    
    parser.add_argument('--validation-days', type=int, default=30,
                      help='Days of data for validation (default: 30)')
    
    parser.add_argument('--output-dir', type=str, default='models/ensemble',
                      help='Output directory (default: models/ensemble)')
    
    parser.add_argument('--data-dir', type=str, default='historical_data',
                      help='Directory with historical data (default: historical_data)')
    
    parser.add_argument('--visualize', action='store_true',
                      help='Generate visualization of ensemble performance')
    
    return parser.parse_args()

def main():
    """Run the strategy ensemble trainer"""
    args = parse_arguments()
    
    logger.info("Starting ensemble training with arguments:")
    logger.info(f"  Assets: {args.assets}")
    logger.info(f"  Strategies: {args.strategies}")
    logger.info(f"  Timeframes: {args.timeframes}")
    logger.info(f"  Training days: {args.training_days}")
    logger.info(f"  Validation days: {args.validation_days}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Data directory: {args.data_dir}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = StrategyEnsembleTrainer(
        strategies=args.strategies,
        assets=args.assets,
        data_dir=args.data_dir,
        timeframes=args.timeframes,
        training_days=args.training_days,
        validation_days=args.validation_days,
        ensemble_output_dir=args.output_dir
    )
    
    # Run training
    start_time = datetime.now()
    logger.info(f"Training started at {start_time}")
    
    results = trainer.train_strategy_ensemble()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Training completed at {end_time} (duration: {duration})")
    
    # Generate visualization if requested
    if args.visualize:
        logger.info("Generating ensemble performance visualization")
        trainer.visualize_ensemble_performance(results)
    
    # Output summary
    logger.info("Ensemble training summary:")
    
    for asset in args.assets:
        if asset in results:
            logger.info(f"  {asset}:")
            for regime, ensemble in results[asset].items():
                perf = ensemble['performance']
                acc = perf.get('ensemble_accuracy', 0)
                lift = perf.get('collaborative_lift', 0)
                logger.info(f"    {regime}: accuracy={acc:.2f}, lift={lift:.2f}")
    
    logger.info("Ensemble training completed successfully")
    
    # Output the path to the weights file
    weights_path = os.path.join(args.output_dir, "strategy_ensemble_weights.json")
    logger.info(f"Strategy weights saved to: {weights_path}")
    logger.info("Use these weights with the model_collaboration_integrator.py to improve strategy collaboration")

if __name__ == "__main__":
    main()