#!/usr/bin/env python3
"""
Activate Live Trading in Sandbox Mode

This script activates the trained ML models for live trading in sandbox mode,
connecting to Kraken's WebSocket API for real-time data and starting the dashboard
for monitoring.

Usage:
    python activate_live_trading.py [--pairs ALL|PAIR1,PAIR2,...] [--timeframe TIMEFRAME]
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import threading

# Import Kraken WebSocket client
from kraken_websocket_client import KrakenWebSocketClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
PORTFOLIO_FILE = f"{CONFIG_DIR}/sandbox_portfolio.json"
PREDICTIONS_FILE = f"{CONFIG_DIR}/current_predictions.json"
DASHBOARD_PROCESS = None
SUPPORTED_PAIRS = [
    'SOL/USD', 'BTC/USD', 'ETH/USD', 'ADA/USD', 'DOT/USD',
    'LINK/USD', 'AVAX/USD', 'MATIC/USD', 'UNI/USD'
]


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Activate Live Trading in Sandbox Mode")
    parser.add_argument('--pairs', type=str, default="ALL",
                      help='Comma-separated list of pairs to trade (e.g., SOL/USD,BTC/USD) or ALL')
    parser.add_argument('--timeframe', type=str, default="1m",
                      help='Timeframe to use for trading (e.g., 1m, 5m, 15m, 1h, 4h, 1d)')
    return parser.parse_args()


def run_command(command: List[str], description: Optional[str] = None) -> Optional[subprocess.CompletedProcess]:
    """Run a shell command and log output"""
    if description:
        logger.info(description)
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"STDOUT: {line}")
        
        if result.stderr:
            for line in result.stderr.splitlines():
                logger.warning(f"STDERR: {line}")
        
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        
        if e.stdout:
            for line in e.stdout.splitlines():
                logger.info(f"STDOUT: {line}")
        
        if e.stderr:
            for line in e.stderr.splitlines():
                logger.error(f"STDERR: {line}")
        
        return None


def load_ml_config() -> Dict[str, Any]:
    """Load ML configuration from file"""
    try:
        if os.path.exists(ML_CONFIG_FILE):
            with open(ML_CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"ML config file {ML_CONFIG_FILE} not found")
            return {}
    except Exception as e:
        logger.error(f"Error loading ML config: {e}")
        return {}


def save_ml_config(config: Dict[str, Any]) -> bool:
    """Save ML configuration to file"""
    try:
        os.makedirs(os.path.dirname(ML_CONFIG_FILE), exist_ok=True)
        with open(ML_CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"ML configuration saved to {ML_CONFIG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving ML config: {e}")
        return False


def save_predictions(predictions: Dict[str, Any]) -> bool:
    """Save current predictions to file"""
    try:
        os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
        with open(PREDICTIONS_FILE, 'w') as f:
            json.dump(predictions, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving predictions: {e}")
        return False


def reset_sandbox_portfolio(starting_capital=20000.0) -> bool:
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


def get_available_models(timeframe: str) -> List[str]:
    """Get list of available models for a specific timeframe"""
    pairs = []
    
    for pair in SUPPORTED_PAIRS:
        pair_formatted = pair.replace('/', '_')
        
        # Check if model files exist
        if os.path.exists(f"ml_models/{pair_formatted}_{timeframe}_entry_model.h5"):
            pairs.append(pair)
    
    return pairs


def update_ml_config(pairs: List[str], timeframe: str) -> bool:
    """Update ML configuration for the specified pairs and timeframe"""
    try:
        # Load existing config or create new one
        ml_config = load_ml_config()
        
        if not ml_config:
            ml_config = {
                "enabled": True,
                "sandbox": True,
                "models": {}
            }
        
        # Enable ML trading
        ml_config["enabled"] = True
        ml_config["sandbox"] = True
        
        # Update configuration for each pair
        for pair in pairs:
            pair_formatted = pair.replace('/', '_')
            
            # Check if model files exist
            entry_model = f"ml_models/{pair_formatted}_{timeframe}_entry_model.h5"
            exit_model = f"ml_models/{pair_formatted}_{timeframe}_exit_model.h5"
            cancel_model = f"ml_models/{pair_formatted}_{timeframe}_cancel_model.h5"
            sizing_model = f"ml_models/{pair_formatted}_{timeframe}_sizing_model.h5"
            
            if os.path.exists(entry_model) and os.path.exists(exit_model) and \
               os.path.exists(cancel_model) and os.path.exists(sizing_model):
                
                # Configure models for this pair
                ml_config["models"][pair] = {
                    "enabled": True,
                    "timeframe": timeframe,
                    "model_path": "ml_models",
                    "use_ensemble": True,
                    "specialized_models": {
                        "entry": {
                            "enabled": True,
                            "model_name": f"{pair_formatted}_{timeframe}_entry_model.h5",
                            "threshold": 0.62
                        },
                        "exit": {
                            "enabled": True,
                            "model_name": f"{pair_formatted}_{timeframe}_exit_model.h5",
                            "threshold": 0.52
                        },
                        "cancel": {
                            "enabled": True,
                            "model_name": f"{pair_formatted}_{timeframe}_cancel_model.h5",
                            "threshold": 0.72
                        },
                        "sizing": {
                            "enabled": True,
                            "model_name": f"{pair_formatted}_{timeframe}_sizing_model.h5",
                            "min_size": 0.25,
                            "max_size": 0.9
                        }
                    },
                    "risk_management": {
                        "max_position_size": 0.15,
                        "max_positions": 3,
                        "confidence_scaling": True,
                        "max_leverage": 50,
                        "min_leverage": 1,
                        "confidence_thresholds": {
                            "low": 0.62,
                            "medium": 0.77,
                            "high": 0.87,
                            "very_high": 0.96
                        },
                        "leverage_tiers": {
                            "low": 1,
                            "medium": 3,
                            "high": 12,
                            "very_high": 50
                        }
                    }
                }
                
                logger.info(f"Configured models for {pair} ({timeframe})")
            else:
                logger.warning(f"Some model files missing for {pair} ({timeframe})")
        
        # Save updated configuration
        if save_ml_config(ml_config):
            logger.info(f"ML configuration updated for {len(pairs)} pairs")
            return True
        else:
            logger.error("Failed to save ML configuration")
            return False
            
    except Exception as e:
        logger.error(f"Error updating ML configuration: {e}")
        return False


def init_kraken_websocket(pairs: List[str]) -> Optional[KrakenWebSocketClient]:
    """Initialize Kraken WebSocket client"""
    try:
        # Create WebSocket client with price update callback
        client = KrakenWebSocketClient(pairs=pairs)
        
        # Connect to WebSocket
        if client.connect():
            logger.info(f"Connected to Kraken WebSocket for {len(pairs)} pairs")
            
            # Wait for initial data
            timeout = 10
            start_time = time.time()
            while time.time() - start_time < timeout:
                prices = client.get_all_prices()
                if prices and len(prices) > 0:
                    logger.info(f"Received initial prices: {prices}")
                    break
                time.sleep(0.5)
            
            return client
        else:
            logger.error("Failed to connect to Kraken WebSocket")
            return None
    
    except Exception as e:
        logger.error(f"Error initializing Kraken WebSocket: {e}")
        return None


def start_dashboard_server():
    """Start the dashboard server"""
    global DASHBOARD_PROCESS
    
    try:
        # Check if dashboard is already running
        if DASHBOARD_PROCESS and DASHBOARD_PROCESS.poll() is None:
            logger.info("Dashboard server already running")
            return True
        
        # Start dashboard server in a separate process
        DASHBOARD_PROCESS = subprocess.Popen(
            ["python", "dashboard_app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        logger.info(f"Started dashboard server (PID: {DASHBOARD_PROCESS.pid})")
        return True
    
    except Exception as e:
        logger.error(f"Error starting dashboard server: {e}")
        return False


def restart_trading_bot() -> bool:
    """Restart the trading bot"""
    try:
        result = run_command(
            ["python", "restart_trading_bot.py"],
            "Restarting trading bot"
        )
        
        if result:
            logger.info("Trading bot restarted successfully")
            return True
        else:
            logger.error("Failed to restart trading bot")
            return False
    
    except Exception as e:
        logger.error(f"Error restarting trading bot: {e}")
        return False


def check_api_keys() -> bool:
    """Check if Kraken API keys are configured"""
    api_key = os.environ.get("KRAKEN_API_KEY")
    api_secret = os.environ.get("KRAKEN_API_SECRET")
    
    if not api_key or not api_secret:
        logger.warning("Kraken API keys not found in environment variables")
        return False
    
    logger.info("Kraken API keys found in environment variables")
    return True


def price_update_thread(ws_client: KrakenWebSocketClient):
    """Thread function to periodically update prices and predictions"""
    try:
        while True:
            # Get current prices
            prices = ws_client.get_all_prices()
            
            if prices:
                # Update predictions file with current prices and timestamps
                predictions = {
                    "timestamp": datetime.now().isoformat(),
                    "prices": prices,
                    "predictions": {}
                }
                
                # Add some mock predictions for demonstration
                for pair, price in prices.items():
                    predictions["predictions"][pair] = {
                        "timestamp": datetime.now().isoformat(),
                        "current_price": price,
                        "signal": "neutral",
                        "confidence": 0.65,
                        "timeframe": "1m"
                    }
                
                # Save predictions to file
                save_predictions(predictions)
            
            # Sleep for a while
            time.sleep(5)
    
    except Exception as e:
        logger.error(f"Error in price update thread: {e}")


def main():
    """Main function"""
    args = parse_arguments()
    
    # Parse pairs to trade
    if args.pairs == "ALL":
        pairs_to_trade = SUPPORTED_PAIRS
    else:
        pairs_to_trade = [p.strip() for p in args.pairs.split(",")]
        
        # Validate pairs
        for pair in pairs_to_trade:
            if pair not in SUPPORTED_PAIRS:
                logger.error(f"Unsupported pair: {pair}")
                return 1
    
    # Check API keys
    check_api_keys()
    
    # Get available models for specified timeframe
    available_pairs = get_available_models(args.timeframe)
    
    # Filter pairs based on available models
    pairs_to_trade = [p for p in pairs_to_trade if p in available_pairs]
    
    if not pairs_to_trade:
        logger.error(f"No models available for the specified pairs and timeframe {args.timeframe}")
        return 1
    
    logger.info(f"Activating ML trading for {len(pairs_to_trade)} pairs: {', '.join(pairs_to_trade)}")
    
    # Reset sandbox portfolio
    if not reset_sandbox_portfolio():
        logger.error("Failed to reset sandbox portfolio")
        return 1
    
    # Update ML configuration
    if not update_ml_config(pairs_to_trade, args.timeframe):
        logger.error("Failed to update ML configuration")
        return 1
    
    # Initialize Kraken WebSocket client
    ws_client = init_kraken_websocket(pairs_to_trade)
    if not ws_client:
        logger.error("Failed to initialize Kraken WebSocket client")
        return 1
    
    # Start price update thread
    update_thread = threading.Thread(target=price_update_thread, args=(ws_client,))
    update_thread.daemon = True
    update_thread.start()
    
    # Start dashboard server
    if not start_dashboard_server():
        logger.error("Failed to start dashboard server")
        return 1
    
    # Restart trading bot
    if not restart_trading_bot():
        logger.error("Failed to restart trading bot")
        return 1
    
    logger.info("ML trading system activated successfully")
    logger.info(f"Dashboard available at http://localhost:5001")
    
    try:
        # Keep the script running
        while True:
            time.sleep(5)
            
            # Check if dashboard is still running
            if DASHBOARD_PROCESS and DASHBOARD_PROCESS.poll() is not None:
                logger.warning("Dashboard server stopped unexpectedly, restarting...")
                start_dashboard_server()
    
    except KeyboardInterrupt:
        logger.info("ML trading system stopped by user")
    finally:
        # Cleanup
        if ws_client:
            ws_client.disconnect()
        
        if DASHBOARD_PROCESS and DASHBOARD_PROCESS.poll() is None:
            DASHBOARD_PROCESS.terminate()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())