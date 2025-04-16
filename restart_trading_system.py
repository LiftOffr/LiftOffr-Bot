#!/usr/bin/env python3
"""
Restart Trading System

This script restarts all components of the trading system:
1. Restarts the trading bot
2. Starts the dashboard server
3. Initializes real-time data connections

Usage:
    python restart_trading_system.py [--reset-portfolio]
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
PORTFOLIO_FILE = "config/sandbox_portfolio.json"
ML_CONFIG_FILE = "config/ml_config.json"
DASHBOARD_PROCESS = None
TRADING_BOT_PID_FILE = ".bot_pid.txt"
TRADING_BOT_RESTART_TRIGGER = ".trading_bot_restart_trigger"
DASHBOARD_WORKFLOW_FILE = ".dashboard_workflow.json"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Restart Trading System")
    parser.add_argument('--reset-portfolio', action='store_true',
                      help='Reset the sandbox portfolio to starting capital')
    return parser.parse_args()


def run_command(command, description=None, check=True, capture_output=True):
    """Run a shell command and log output"""
    if description:
        logger.info(description)
    
    try:
        result = subprocess.run(
            command,
            shell=isinstance(command, str),
            check=check,
            capture_output=capture_output,
            text=True
        )
        
        if capture_output:
            if result.stdout:
                for line in result.stdout.splitlines():
                    logger.info(f"STDOUT: {line}")
            
            if result.stderr:
                for line in result.stderr.splitlines():
                    logger.warning(f"STDERR: {line}")
        
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        
        if capture_output:
            if e.stdout:
                for line in e.stdout.splitlines():
                    logger.info(f"STDOUT: {line}")
            
            if e.stderr:
                for line in e.stderr.splitlines():
                    logger.error(f"STDERR: {line}")
        
        if check:
            raise
        return e


def reset_sandbox_portfolio(starting_capital=20000.0):
    """Reset sandbox portfolio to starting capital"""
    try:
        # Create portfolio with starting capital
        portfolio = {
            "base_currency": "USD",
            "starting_capital": starting_capital,
            "current_capital": starting_capital,
            "equity": starting_capital,
            "positions": {},
            "completed_trades": [],
            "last_updated": datetime.now().isoformat()
        }
        
        # Save portfolio
        os.makedirs(os.path.dirname(PORTFOLIO_FILE), exist_ok=True)
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio, f, indent=2)
        
        logger.info(f"Sandbox portfolio reset to ${starting_capital:.2f}")
        return True
    
    except Exception as e:
        logger.error(f"Error resetting sandbox portfolio: {e}")
        return False


def ensure_ml_config_enabled():
    """Ensure ML configuration is enabled"""
    try:
        if os.path.exists(ML_CONFIG_FILE):
            with open(ML_CONFIG_FILE, 'r') as f:
                config = json.load(f)
            
            # Ensure ML trading is enabled
            if not config.get("enabled", False):
                logger.info("Enabling ML trading in configuration")
                config["enabled"] = True
                
                with open(ML_CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=2)
        
        return True
    
    except Exception as e:
        logger.error(f"Error ensuring ML config is enabled: {e}")
        return False


def restart_trading_bot():
    """Restart the trading bot"""
    try:
        # Create restart trigger file
        with open(TRADING_BOT_RESTART_TRIGGER, 'w') as f:
            f.write(str(datetime.now().timestamp()))
        
        logger.info("Created trading bot restart trigger")
        
        # Kill existing bot process if running
        if os.path.exists(TRADING_BOT_PID_FILE):
            try:
                with open(TRADING_BOT_PID_FILE, 'r') as f:
                    pid = int(f.read().strip())
                
                # Check if process exists
                if pid > 0:
                    try:
                        os.kill(pid, 0)  # Check if process exists
                        # Kill the process
                        run_command(f"kill {pid}", f"Killing existing bot process (PID: {pid})", check=False)
                        time.sleep(2)  # Wait for process to terminate
                    except OSError:
                        # Process doesn't exist
                        pass
            except (ValueError, IOError) as e:
                logger.warning(f"Error reading PID file: {e}")
        
        # Start the bot using the workflow system
        run_command(["python", "restart_trading_bot.py", "--sandbox"], "Restarting trading bot")
        
        # Wait for bot to start
        for _ in range(30):  # Wait up to 30 seconds
            if os.path.exists(TRADING_BOT_PID_FILE):
                logger.info("Trading bot started successfully")
                return True
            time.sleep(1)
        
        logger.warning("Trading bot restart timed out")
        return False
    
    except Exception as e:
        logger.error(f"Error restarting trading bot: {e}")
        return False


def start_dashboard():
    """Start the dashboard server"""
    global DASHBOARD_PROCESS
    
    try:
        # Configure dashboard workflow
        dashboard_config = {
            "command": "python dashboard_app.py",
            "port": 5001,
            "restart_on_change": True
        }
        
        # Save dashboard workflow config
        with open(DASHBOARD_WORKFLOW_FILE, 'w') as f:
            json.dump(dashboard_config, f, indent=2)
        
        # Start dashboard server
        DASHBOARD_PROCESS = subprocess.Popen(
            ["python", "dashboard_app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        logger.info(f"Started dashboard server (PID: {DASHBOARD_PROCESS.pid})")
        
        # Wait for dashboard to start
        time.sleep(3)
        
        # Check if process is still running
        if DASHBOARD_PROCESS.poll() is not None:
            stdout, stderr = DASHBOARD_PROCESS.communicate()
            logger.error(f"Dashboard process exited unexpectedly: {stderr.decode() if stderr else 'No error output'}")
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        return False


def restart_workflow(workflow_name):
    """Restart a workflow by name"""
    try:
        run_command(["restart_workflow", workflow_name], f"Restarting workflow: {workflow_name}")
        return True
    except Exception as e:
        logger.error(f"Error restarting workflow {workflow_name}: {e}")
        return False


def main():
    """Main function"""
    args = parse_arguments()
    
    # Reset portfolio if requested
    if args.reset_portfolio:
        if not reset_sandbox_portfolio():
            logger.error("Failed to reset portfolio")
            return 1
    
    # Ensure ML config is enabled
    if not ensure_ml_config_enabled():
        logger.warning("Failed to ensure ML config is enabled")
    
    # Restart trading bot
    if not restart_trading_bot():
        logger.error("Failed to restart trading bot")
        return 1
    
    # Start dashboard
    if not start_dashboard():
        logger.error("Failed to start dashboard")
        return 1
    
    # Restart trading_bot workflow
    if not restart_workflow("trading_bot"):
        logger.error("Failed to restart trading_bot workflow")
        return 1
    
    # Restart dashboard workflow
    if not restart_workflow("Start application"):
        logger.error("Failed to restart dashboard workflow")
        # Not critical, continue
    
    logger.info("Trading system restarted successfully!")
    logger.info("Dashboard available at: http://localhost:5001")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())