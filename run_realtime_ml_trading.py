#!/usr/bin/env python3
"""
Run Realtime ML Trading

This script runs the realtime ML trading system:
1. Loads the ML configuration
2. Initializes the Kraken API client
3. Sets up the ML trading environment
4. Starts the trading loop with configured pairs

Usage:
    python run_realtime_ml_trading.py [--sandbox] [--pairs PAIR1 PAIR2 ...]
"""
import os
import sys
import json
import time
import logging
import argparse
import datetime
import traceback
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MAX_POSITIONS_PER_PAIR = 1
SLEEP_SECONDS = 60  # Check for new trades every minute
SAVE_PID = True  # Save process ID to file

# Directories and files
DATA_DIR = "data"
MODEL_WEIGHTS_DIR = "model_weights"
CONFIG_DIR = "config"
ML_CONFIG_PATH = f"{CONFIG_DIR}/ml_config.json"
PORTFOLIO_PATH = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_PATH = f"{DATA_DIR}/sandbox_positions.json"
TRADES_PATH = f"{DATA_DIR}/sandbox_trades.json"
PID_FILE = ".bot_pid.txt"

class MLTradingBot:
    """ML-powered trading bot for cryptocurrency trading"""
    
    def __init__(self, sandbox=True, pairs=None):
        """
        Initialize the ML trading bot
        
        Args:
            sandbox (bool): Whether to run in sandbox mode
            pairs (list): List of pairs to trade, or None for all configured pairs
        """
        self.sandbox = sandbox
        self.user_pairs = pairs
        self.models = {}
        self.ml_config = None
        self.portfolio = None
        self.positions = {}
        self.trades = {}
        self.features = {}
        self.predictions = {}
        self.last_prices = {}
        self.running = True
        
        # Set up directories
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(CONFIG_DIR, exist_ok=True)
        
        # Initialize components
        self._load_ml_config()
        self._load_portfolio()
        self._load_positions()
        self._load_trades()
        self._load_models()
        
        # Save process ID
        if SAVE_PID:
            with open(PID_FILE, 'w') as f:
                f.write(str(os.getpid()))
    
    def _load_ml_config(self):
        """Load ML configuration"""
        try:
            if os.path.exists(ML_CONFIG_PATH):
                with open(ML_CONFIG_PATH, 'r') as f:
                    self.ml_config = json.load(f)
                logger.info(f"Loaded ML configuration with {len(self.ml_config.get('models', {}))} models")
            else:
                logger.warning(f"ML configuration file not found: {ML_CONFIG_PATH}")
                self.ml_config = {"models": {}, "global_settings": {}}
        except Exception as e:
            logger.error(f"Error loading ML configuration: {e}")
            self.ml_config = {"models": {}, "global_settings": {}}
    
    def _load_portfolio(self):
        """Load portfolio data"""
        try:
            if os.path.exists(PORTFOLIO_PATH):
                with open(PORTFOLIO_PATH, 'r') as f:
                    self.portfolio = json.load(f)
                logger.info(f"Loaded portfolio with balance: ${self.portfolio.get('balance', 0):.2f}")
            else:
                logger.warning(f"Portfolio file not found: {PORTFOLIO_PATH}")
                # Create a default portfolio
                self.portfolio = {
                    "balance": 20000.0,
                    "initial_balance": 20000.0,
                    "last_updated": datetime.datetime.now().isoformat()
                }
                self._save_portfolio()
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            self.portfolio = {
                "balance": 20000.0,
                "initial_balance": 20000.0,
                "last_updated": datetime.datetime.now().isoformat()
            }
            self._save_portfolio()
    
    def _save_portfolio(self):
        """Save portfolio data"""
        try:
            self.portfolio["last_updated"] = datetime.datetime.now().isoformat()
            with open(PORTFOLIO_PATH, 'w') as f:
                json.dump(self.portfolio, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
    
    def _load_positions(self):
        """Load position data"""
        try:
            if os.path.exists(POSITIONS_PATH):
                with open(POSITIONS_PATH, 'r') as f:
                    self.positions = json.load(f)
                logger.info(f"Loaded {len(self.positions)} positions")
            else:
                logger.warning(f"Positions file not found: {POSITIONS_PATH}")
                self.positions = {}
                self._save_positions()
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
            self.positions = {}
            self._save_positions()
    
    def _save_positions(self):
        """Save position data"""
        try:
            with open(POSITIONS_PATH, 'w') as f:
                json.dump(self.positions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
    
    def _load_trades(self):
        """Load trade data"""
        try:
            if os.path.exists(TRADES_PATH):
                with open(TRADES_PATH, 'r') as f:
                    self.trades = json.load(f)
                logger.info(f"Loaded {len(self.trades)} trades")
            else:
                logger.warning(f"Trades file not found: {TRADES_PATH}")
                self.trades = {}
                self._save_trades()
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            self.trades = {}
            self._save_trades()
    
    def _save_trades(self):
        """Save trade data"""
        try:
            with open(TRADES_PATH, 'w') as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trades: {e}")
    
    def _load_models(self):
        """Load ML models for all pairs"""
        if not self.ml_config:
            logger.error("No ML configuration loaded")
            return
        
        models_config = self.ml_config.get('models', {})
        if not models_config:
            logger.error("No models configured in ML configuration")
            return
        
        # Determine which pairs to load
        if self.user_pairs:
            pairs_to_load = [p for p in self.user_pairs if p in models_config]
            if not pairs_to_load:
                logger.warning(f"None of the specified pairs {self.user_pairs} are configured in ML configuration")
                pairs_to_load = list(models_config.keys())
        else:
            pairs_to_load = list(models_config.keys())
        
        logger.info(f"Loading models for pairs: {pairs_to_load}")
        
        # Load models for each pair
        for pair in pairs_to_load:
            try:
                model_info = models_config.get(pair, {})
                model_path = model_info.get('model_path', '')
                
                if not model_path or not os.path.exists(model_path):
                    logger.warning(f"Model file not found for {pair}: {model_path}")
                    continue
                
                # Load the model
                model = load_model(model_path)
                self.models[pair] = {
                    'model': model,
                    'config': model_info
                }
                
                logger.info(f"Loaded model for {pair}: {model_path}")
            except Exception as e:
                logger.error(f"Error loading model for {pair}: {e}")
    
    def _get_current_price(self, pair):
        """
        Get current price for a pair (simulated for now)
        
        In a real implementation, this would use the Kraken API or websocket
        """
        # Extract symbol from pair (e.g., "BTC" from "BTC/USD")
        symbol = pair.split('/')[0].lower()
        
        # For simulation, generate a somewhat realistic price based on the pair
        if pair == "BTC/USD":
            return 57000 + (np.random.randn() * 500)
        elif pair == "ETH/USD":
            return 3500 + (np.random.randn() * 50)
        elif pair == "SOL/USD":
            return 150 + (np.random.randn() * 5)
        elif pair == "ADA/USD":
            return 0.45 + (np.random.randn() * 0.01)
        elif pair == "DOT/USD":
            return 7.5 + (np.random.randn() * 0.2)
        elif pair == "LINK/USD":
            return 16 + (np.random.randn() * 0.5)
        elif pair == "AVAX/USD":
            return 35 + (np.random.randn() * 1)
        elif pair == "MATIC/USD":
            return 0.75 + (np.random.randn() * 0.02)
        elif pair == "UNI/USD":
            return 10 + (np.random.randn() * 0.3)
        elif pair == "ATOM/USD":
            return 9 + (np.random.randn() * 0.25)
        else:
            # Default case for unknown pairs
            return 100 + (np.random.randn() * 2)
    
    def _generate_features(self, pair):
        """
        Generate features for prediction (simulated for now)
        
        In a real implementation, this would use historical price data
        """
        # Create a random feature vector of the right shape
        # Assuming our model expects (1, 24, 40) shape
        return np.random.randn(1, 24, 40)
    
    def _predict(self, pair):
        """
        Make a prediction for a pair
        
        Returns:
            dict: Prediction results including signal, confidence, etc.
        """
        if pair not in self.models:
            logger.warning(f"No model loaded for {pair}")
            return None
        
        try:
            # Get model and config
            model_data = self.models[pair]
            model = model_data['model']
            config = model_data['config']
            
            # Generate features
            features = self._generate_features(pair)
            
            # Make prediction
            raw_prediction = model.predict(features, verbose=0)
            
            # Process prediction - ensure proper handling of array shapes
            # The raw prediction might be a multidimensional array
            if hasattr(raw_prediction, 'shape') and len(raw_prediction.shape) > 1:
                signal_value = float(raw_prediction[0][0])  # Convert to Python float from numpy
            else:
                signal_value = float(raw_prediction[0])  # Handle 1D array case
            
            # Determine signal type and confidence
            if signal_value > 0:
                signal = "buy"
                confidence = min(abs(signal_value), 1.0)
            elif signal_value < 0:
                signal = "sell"
                confidence = min(abs(signal_value), 1.0)
            else:
                signal = "neutral"
                confidence = 0.0
            
            # Get confidence threshold
            confidence_threshold = config.get('confidence_threshold', 
                                            self.ml_config.get('global_settings', {}).get('confidence_threshold', 0.65))
            
            # Determine if signal is strong enough
            execute_signal = confidence >= confidence_threshold
            
            # Calculate dynamic leverage based on confidence
            min_leverage = config.get('min_leverage', 5.0)
            max_leverage = config.get('max_leverage', 125.0)
            
            if confidence > 0:
                # Scale leverage based on confidence
                leverage = min_leverage + ((max_leverage - min_leverage) * confidence)
            else:
                leverage = min_leverage
            
            # Cap leverage at max
            leverage = min(leverage, max_leverage)
            
            # Format the prediction result
            prediction = {
                'pair': pair,
                'signal': signal,
                'confidence': confidence,
                'execute_signal': execute_signal,
                'leverage': leverage,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            return prediction
        except Exception as e:
            logger.error(f"Error making prediction for {pair}: {e}")
            return None
    
    def _can_open_position(self, pair):
        """Check if we can open a position for this pair"""
        # Count existing positions for this pair
        pair_positions = [p for p_id, p in self.positions.items() if p.get('symbol', '') == pair]
        
        # Get max positions per pair from config
        max_positions = self.ml_config.get('global_settings', {}).get(
            'max_positions_per_pair', MAX_POSITIONS_PER_PAIR)
        
        return len(pair_positions) < max_positions
    
    def _calculate_position_size(self, pair, leverage):
        """
        Calculate position size based on account balance and risk
        
        Args:
            pair (str): Trading pair
            leverage (float): Leverage to use
            
        Returns:
            float: Position size
        """
        # Get balance
        balance = float(self.portfolio.get('balance', 0))
        
        # Get risk percentage from config or use default
        risk_percentage = self.models[pair]['config'].get(
            'risk_percentage', 
            self.ml_config.get('global_settings', {}).get('risk_percentage', 0.2)
        )
        
        # Calculate amount to risk (balance * risk_percentage)
        risk_amount = balance * risk_percentage
        
        # Get current price
        current_price = self._get_current_price(pair)
        
        # Calculate position size (amount / price)
        # For leveraged trading, we actually use less margin but control more
        position_size = risk_amount / current_price * leverage
        
        return position_size
    
    def _open_position(self, pair, signal, confidence, leverage):
        """
        Open a new position
        
        Args:
            pair (str): Trading pair
            signal (str): Signal type ('buy' or 'sell')
            confidence (float): Prediction confidence
            leverage (float): Leverage to use
            
        Returns:
            str: Position ID or None if failed
        """
        try:
            # Check if we can open a position
            if not self._can_open_position(pair):
                logger.warning(f"Maximum positions reached for {pair}")
                return None
            
            # Determine if long or short
            long = signal == "buy"
            
            # Get current price
            entry_price = self._get_current_price(pair)
            
            # Calculate position size
            size = self._calculate_position_size(pair, leverage)
            
            # Calculate liquidation price (simple approximation)
            # For longs: liquidation when price falls by (100/leverage)%
            # For shorts: liquidation when price rises by (100/leverage)%
            liquidation_threshold = 1.0 / leverage
            
            if long:
                liquidation_price = entry_price * (1.0 - liquidation_threshold)
            else:
                liquidation_price = entry_price * (1.0 + liquidation_threshold)
            
            # Generate position ID
            position_id = f"{pair.replace('/', '_').lower()}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Create position object
            position = {
                'symbol': pair,
                'entry_price': entry_price,
                'current_price': entry_price,
                'size': size,
                'long': long,
                'confidence': confidence,
                'leverage': leverage,
                'liquidation_price': liquidation_price,
                'entry_time': datetime.datetime.now().isoformat(),
                'last_updated': datetime.datetime.now().isoformat()
            }
            
            # Add to positions
            self.positions[position_id] = position
            
            # Save positions
            self._save_positions()
            
            # Log the position
            direction = "LONG" if long else "SHORT"
            logger.info(f"Opened {direction} position for {pair}: Size={size:.6f}, Entry=${entry_price:.2f}, Leverage={leverage:.1f}x, Confidence={confidence:.2f}")
            
            # Also create a trade entry
            trade = {
                'symbol': pair,
                'entry_price': entry_price,
                'exit_price': 0,
                'size': size,
                'long': long,
                'confidence': confidence,
                'leverage': leverage,
                'liquidation_price': liquidation_price,
                'entry_time': datetime.datetime.now().isoformat(),
                'exit_time': '',
                'profit_loss': 0.0,
                'status': 'open'
            }
            
            # Add to trades
            trade_id = f"trade_{position_id}"
            self.trades[trade_id] = trade
            
            # Save trades
            self._save_trades()
            
            return position_id
        except Exception as e:
            logger.error(f"Error opening position for {pair}: {e}")
            return None
    
    def _update_positions(self):
        """Update all open positions with current prices"""
        if not self.positions:
            return
        
        for position_id, position in list(self.positions.items()):
            try:
                # Get current price
                pair = position.get('symbol', '')
                current_price = self._get_current_price(pair)
                
                # Update current price
                position['current_price'] = current_price
                position['last_updated'] = datetime.datetime.now().isoformat()
                
                # Check for liquidation
                liquidation_price = position.get('liquidation_price', 0)
                long = position.get('long', True)
                
                if (long and current_price <= liquidation_price) or (not long and current_price >= liquidation_price):
                    logger.warning(f"Position {position_id} liquidated at ${current_price:.2f}")
                    
                    # Close the position at liquidation price
                    self._close_position(position_id, liquidation=True)
            except Exception as e:
                logger.error(f"Error updating position {position_id}: {e}")
        
        # Save positions
        self._save_positions()
    
    def _close_position(self, position_id, liquidation=False):
        """
        Close a position
        
        Args:
            position_id (str): Position ID to close
            liquidation (bool): Whether this is a liquidation
        """
        try:
            # Get position
            if position_id not in self.positions:
                logger.warning(f"Position {position_id} not found")
                return
            
            position = self.positions[position_id]
            
            # Get relevant data
            pair = position.get('symbol', '')
            entry_price = position.get('entry_price', 0)
            size = position.get('size', 0)
            long = position.get('long', True)
            leverage = position.get('leverage', 1.0)
            
            # Determine exit price
            if liquidation:
                exit_price = position.get('liquidation_price', 0)
            else:
                exit_price = self._get_current_price(pair)
            
            # Calculate profit/loss
            if long:
                profit_loss = (exit_price - entry_price) * size
            else:
                profit_loss = (entry_price - exit_price) * size
            
            # If liquidated, loss is fixed at the risk amount
            if liquidation:
                # In leveraged trading, we lose the margin amount
                risk_amount = (size * entry_price) / leverage
                profit_loss = -risk_amount
            
            # Update portfolio balance
            self.portfolio['balance'] = self.portfolio.get('balance', 0) + profit_loss
            
            # Save portfolio
            self._save_portfolio()
            
            # Log the trade
            direction = "LONG" if long else "SHORT"
            status = "LIQUIDATED" if liquidation else "CLOSED"
            logger.info(f"{status} {direction} position for {pair}: Entry=${entry_price:.2f}, Exit=${exit_price:.2f}, P/L=${profit_loss:.2f}")
            
            # Update trade record if it exists
            trade_id = f"trade_{position_id}"
            if trade_id in self.trades:
                trade = self.trades[trade_id]
                trade['exit_price'] = exit_price
                trade['exit_time'] = datetime.datetime.now().isoformat()
                trade['profit_loss'] = profit_loss
                trade['status'] = 'liquidated' if liquidation else 'closed'
                
                # Save trades
                self._save_trades()
            
            # Remove position
            del self.positions[position_id]
            
            # Save positions
            self._save_positions()
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
    
    def _check_exit_signals(self):
        """Check for exit signals for all open positions"""
        if not self.positions:
            return
        
        for position_id, position in list(self.positions.items()):
            try:
                # Get position data
                pair = position.get('symbol', '')
                long = position.get('long', True)
                
                # Skip if no model for this pair
                if pair not in self.models:
                    continue
                
                # Get prediction
                prediction = self._predict(pair)
                if not prediction:
                    continue
                
                # Get signal and confidence
                signal = prediction.get('signal', 'neutral')
                confidence = prediction.get('confidence', 0)
                execute_signal = prediction.get('execute_signal', False)
                
                # Check for opposing signal with sufficient confidence
                if execute_signal:
                    if (long and signal == 'sell') or (not long and signal == 'buy'):
                        logger.info(f"Exit signal for {pair} with confidence {confidence:.2f}")
                        self._close_position(position_id)
            except Exception as e:
                logger.error(f"Error checking exit signals for {position_id}: {e}")
    
    def _check_entry_signals(self):
        """Check for entry signals for all pairs"""
        # Get pairs from config
        if not self.ml_config:
            return
        
        models_config = self.ml_config.get('models', {})
        if not models_config:
            return
        
        # Determine which pairs to check
        if self.user_pairs:
            pairs_to_check = [p for p in self.user_pairs if p in models_config]
        else:
            pairs_to_check = list(models_config.keys())
        
        for pair in pairs_to_check:
            try:
                # Skip if we already have max positions for this pair
                if not self._can_open_position(pair):
                    continue
                
                # Skip if no model for this pair
                if pair not in self.models:
                    continue
                
                # Get prediction
                prediction = self._predict(pair)
                if not prediction:
                    continue
                
                # Get signal and confidence
                signal = prediction.get('signal', 'neutral')
                confidence = prediction.get('confidence', 0)
                execute_signal = prediction.get('execute_signal', False)
                leverage = prediction.get('leverage', 1.0)
                
                # Check for entry signal with sufficient confidence
                if execute_signal and signal in ['buy', 'sell']:
                    logger.info(f"Entry signal for {pair}: {signal} with confidence {confidence:.2f}")
                    self._open_position(pair, signal, confidence, leverage)
            except Exception as e:
                logger.error(f"Error checking entry signals for {pair}: {e}")
    
    def run(self):
        """Run the trading bot"""
        logger.info(f"Starting ML trading bot in {'sandbox' if self.sandbox else 'live'} mode")
        
        try:
            while self.running:
                # Update positions with current prices
                self._update_positions()
                
                # Check for exit signals
                self._check_exit_signals()
                
                # Check for entry signals
                self._check_entry_signals()
                
                # Sleep for a bit
                logger.info(f"Sleeping for {SLEEP_SECONDS} seconds...")
                time.sleep(SLEEP_SECONDS)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info("ML trading bot stopped")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run realtime ML trading system")
    parser.add_argument("--sandbox", action="store_true", default=True, 
                        help="Run in sandbox mode (default: True)")
    parser.add_argument("--pairs", nargs="+", type=str,
                        help="Specific pairs to trade (e.g., BTC/USD ETH/USD)")
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Start the ML trading bot
    bot = MLTradingBot(sandbox=args.sandbox, pairs=args.pairs)
    bot.run()

if __name__ == "__main__":
    main()