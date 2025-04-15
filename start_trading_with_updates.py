#!/usr/bin/env python3
"""
Start Trading Bot with Auto Dashboard Updates

This script starts both the trading bot and the dashboard auto-updater
to ensure the dashboard continuously displays the latest information.
"""

import os
import sys
import time
import subprocess
import logging
import threading
import signal
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Start trading bot with dashboard updates')
    parser.add_argument('--pairs', type=str, default="BTC/USD,ETH/USD,SOL/USD",
                      help='Comma-separated list of pairs to trade')
    parser.add_argument('--sandbox', action='store_true',
                      help='Run in sandbox mode')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()

def run_command(cmd, name, stop_event):
    """Run a command in a subprocess and wait for the stop event."""
    logger.info(f"Starting {name}: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Monitor for output
        for line in process.stdout:
            line = line.strip()
            if line:
                logger.info(f"[{name}] {line}")
            
            if stop_event.is_set():
                process.terminate()
                break
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode != 0 and not stop_event.is_set():
            logger.error(f"{name} exited with code {process.returncode}")
        else:
            logger.info(f"{name} completed")
        
    except Exception as e:
        logger.error(f"Error running {name}: {e}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Create stop event
    stop_event = threading.Event()
    
    # Prepare commands
    trading_bot_cmd = [
        sys.executable, "run_enhanced_trading_bot.py",
        "--pairs", args.pairs
    ]
    
    if args.sandbox:
        trading_bot_cmd.append("--sandbox")
    
    if args.verbose:
        trading_bot_cmd.append("--verbose")
    
    dashboard_updater_cmd = [
        sys.executable, "auto_update_dashboard.py"
    ]
    
    # Start processes in threads
    trading_thread = threading.Thread(
        target=run_command, 
        args=(trading_bot_cmd, "Trading Bot", stop_event)
    )
    trading_thread.daemon = True
    trading_thread.start()
    
    dashboard_thread = threading.Thread(
        target=run_command, 
        args=(dashboard_updater_cmd, "Dashboard Updater", stop_event)
    )
    dashboard_thread.daemon = True
    dashboard_thread.start()
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        logger.info("Stopping all processes...")
        stop_event.set()
        trading_thread.join(timeout=5)
        dashboard_thread.join(timeout=5)
        logger.info("All processes stopped")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Wait for threads to complete
    logger.info("Trading system started. Press Ctrl+C to stop.")
    
    try:
        while trading_thread.is_alive() or dashboard_thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())