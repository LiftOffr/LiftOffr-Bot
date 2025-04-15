#!/usr/bin/env python3
"""
Start Trading Bot

This script starts the trading bot with enhanced simulation features.
"""

import os
import sys
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run trading bot with enhanced simulation')
    parser.add_argument('--sandbox', action='store_true', help='Run in sandbox mode')
    parser.add_argument('--flash-crash', action='store_true', help='Enable flash crash simulation')
    parser.add_argument('--latency', action='store_true', help='Enable latency simulation')
    parser.add_argument('--stress-test', action='store_true', help='Enable stress testing')
    parser.add_argument('--interval', type=int, default=1, help='Trading cycle interval in minutes')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()
    
    # Import here to avoid circular imports
    from run_enhanced_simulation import run_trading_simulation
    
    # Log the startup configuration
    logger.info("Starting trading bot with the following configuration:")
    logger.info(f"  Sandbox mode: {args.sandbox}")
    logger.info(f"  Flash crash simulation: {args.flash_crash}")
    logger.info(f"  Latency simulation: {args.latency}")
    logger.info(f"  Stress testing: {args.stress_test}")
    logger.info(f"  Trading interval: {args.interval} minute(s)")
    
    # Run the trading simulation
    run_trading_simulation(args)