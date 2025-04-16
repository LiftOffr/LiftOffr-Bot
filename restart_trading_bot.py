#!/usr/bin/env python3
"""
Restart Trading Bot

This script restarts the trading bot by:
1. Stopping any running trading bot process
2. Updating the configuration to use the specified mode (sandbox or live)
3. Starting the trading bot in a new process

Usage:
    python restart_trading_bot.py [--sandbox]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot_restart.log')
    ]
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Restart trading bot')
    parser.add_argument('--sandbox', action='store_true',
                       help='Run in sandbox mode')
    return parser.parse_args()


def get_bot_pid() -> Optional[int]:
    """
    Get the PID of the currently running trading bot
    
    Returns:
        PID of the trading bot process, or None if not running
    """
    pid_file = '.bot_pid.txt'
    
    if not os.path.exists(pid_file):
        return None
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # Check if process is still running
        try:
            os.kill(pid, 0)
            return pid
        except OSError:
            # Process does not exist
            return None
    except:
        return None


def stop_trading_bot() -> bool:
    """
    Stop the currently running trading bot
    
    Returns:
        True if successful, False otherwise
    """
    pid = get_bot_pid()
    
    if pid is None:
        logging.info("No running trading bot found")
        return True
    
    try:
        # Try to kill the process
        os.kill(pid, 15)  # SIGTERM
        
        # Wait for process to terminate
        for _ in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(0.5)
            except OSError:
                # Process has terminated
                logging.info(f"Trading bot process (PID {pid}) terminated")
                return True
        
        # If process still running, force kill
        os.kill(pid, 9)  # SIGKILL
        logging.info(f"Trading bot process (PID {pid}) forcibly terminated")
        return True
    except Exception as e:
        logging.error(f"Error stopping trading bot: {e}")
        return False


def update_trading_config(sandbox: bool = True) -> bool:
    """
    Update trading configuration
    
    Args:
        sandbox: Whether to run in sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    config_path = "config/trading_config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    try:
        # Load existing config if it exists
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Update mode
        config["sandbox_mode"] = sandbox
        
        # Write config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"Updated trading config, sandbox_mode={sandbox}")
        return True
    except Exception as e:
        logging.error(f"Error updating trading config: {e}")
        return False


def start_trading_bot(sandbox: bool = True) -> bool:
    """
    Start the trading bot
    
    Args:
        sandbox: Whether to run in sandbox mode
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Start the trading bot in a new process
        command = ["python", "main.py"]
        if sandbox:
            command.append("--sandbox")
        
        # Start the process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        
        # Save PID to file
        with open('.bot_pid.txt', 'w') as f:
            f.write(str(process.pid))
        
        logging.info(f"Started trading bot with PID {process.pid} in {'sandbox' if sandbox else 'live'} mode")
        return True
    except Exception as e:
        logging.error(f"Error starting trading bot: {e}")
        return False


def main():
    """Main function"""
    args = parse_arguments()
    
    # Stop any running trading bot
    if not stop_trading_bot():
        logging.warning("Failed to stop running trading bot")
    
    # Update trading config
    if not update_trading_config(args.sandbox):
        logging.error("Failed to update trading config")
        return False
    
    # Start trading bot
    if not start_trading_bot(args.sandbox):
        logging.error("Failed to start trading bot")
        return False
    
    logging.info(f"Trading bot restarted in {'sandbox' if args.sandbox else 'live'} mode")
    return True


if __name__ == "__main__":
    main()