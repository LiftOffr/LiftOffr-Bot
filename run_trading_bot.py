#!/usr/bin/env python3
"""
Trading Bot Runner

This script runs our main trading bot on port 5001 (different from our dashboard port)
"""
import os
import sys
import logging
import argparse
import subprocess

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
    parser.add_argument('--interval', type=int, default=5, help='Trading cycle interval in minutes')
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Prepare command for running the enhanced simulation
    cmd = [
        "python", "run_enhanced_simulation.py",
        "--interval", str(args.interval)
    ]
    
    # Add optional flags
    if args.sandbox:
        cmd.append("--sandbox")
    if args.flash_crash:
        cmd.append("--flash-crash")
    if args.latency:
        cmd.append("--latency")
    if args.stress_test:
        cmd.append("--stress-test")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Execute the command
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Display output in real-time
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
            
        # Wait for completion
        process.wait()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
        process.terminate()
    except Exception as e:
        logger.error(f"Error running trading bot: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())