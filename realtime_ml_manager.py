#!/usr/bin/env python3
"""
Realtime ML Manager for Kraken Trading Bot

This module connects ML models with real-time data streams from Kraken's
WebSocket API to manage trades for maximum profitability.
"""
import os
import json
import time
import logging
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Import ML modules
from ml_risk_manager import MLRiskManager
from ml_websocket_integration import MLWebSocketIntegration

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
ML_CONFIG_FILE = f"{DATA_DIR}/ml_config.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class RealtimeMLManager:
    """
    Realtime ML Manager for trade optimization and risk management.
    
    This class manages ML models and real-time data to optimize trading
    decisions and risk parameters for maximum profitability.
    """
    
    def __init__(
        self,
        trading_pairs: List[str],
        initial_capital: float = 20000.0,
        max_open_positions: int = 7,
        max_allocation_pct: float = 0.85,
        model_weight_path: Optional[str] = None,
        sandbox: bool = True
    ):
        """
        Initialize the Realtime ML Manager
        
        Args:
            trading_pairs: List of trading pairs to trade
            initial_capital: Initial capital in USD
            max_open_positions: Maximum number of open positions
            max_allocation_pct: Maximum portfolio allocation percentage
            model_weight_path: Path to model weights (optional)
            sandbox: Whether to run in sandbox mode
        """
        self.trading_pairs = trading_pairs
        self.initial_capital = initial_capital
        self.max_open_positions = max_open_positions
        self.max_allocation_pct = max_allocation_pct
        self.model_weight_path = model_weight_path
        self.sandbox = sandbox
        
        # Create ML components
        self.risk_manager = MLRiskManager(model_weight_path=model_weight_path)
        self.ws_integration = MLWebSocketIntegration(
            trading_pairs=trading_pairs,
            ml_risk_manager=self.risk_manager,
            sandbox=sandbox
        )
        
        # Internal state
        self.portfolio = {}
        self.positions = []
        self.trades = []
        self.ml_config = {}
        self.model_performance = {}
        self.running = False
        self.update_thread = None
        
        # Load ML configuration
        self._load_ml_config()
        
        # Register callbacks
        self.ws_integration.register_callback(self._ml_prediction_callback)
        
        logger.info(f"Initialized Realtime ML Manager for {len(trading_pairs)} pairs")
    
    def _load_ml_config(self):
        """Load ML configuration from file"""
        try:
            if os.path.exists(ML_CONFIG_FILE):
                with open(ML_CONFIG_FILE, 'r') as f:
                    self.ml_config = json.load(f)
            else:
                # Create default configuration
                self.ml_config = {
                    "base_leverage": 50.0,
                    "max_leverage": 125.0,
                    "confidence_threshold": 0.65,
                    "risk_percentage": 0.20,
                    "pairs": {}
                }
                
                # Add pair-specific configurations
                for pair in self.trading_pairs:
                    self.ml_config["pairs"][pair] = {
                        "enabled": True,
                        "max_allocation": 0.2,
                        "model_weights": {
                            "arima": 0.3,
                            "adaptive": 0.7
                        }
                    }
                
                # Save default configuration
                self._save_ml_config()
        
        except Exception as e:
            logger.error(f"Error loading ML configuration: {e}")
    
    def _save_ml_config(self):
        """Save ML configuration to file"""
        try:
            os.makedirs(os.path.dirname(ML_CONFIG_FILE), exist_ok=True)
            with open(ML_CONFIG_FILE, 'w') as f:
                json.dump(self.ml_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving ML configuration: {e}")
    
    def _load_data(self):
        """Load portfolio, positions, and trades data"""
        try:
            if os.path.exists(PORTFOLIO_FILE):
                with open(PORTFOLIO_FILE, 'r') as f:
                    self.portfolio = json.load(f)
            
            if os.path.exists(POSITIONS_FILE):
                with open(POSITIONS_FILE, 'r') as f:
                    self.positions = json.load(f)
            
            if os.path.exists(TRADES_FILE):
                with open(TRADES_FILE, 'r') as f:
                    self.trades = json.load(f)
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _ml_prediction_callback(self, pair: str, prediction: Dict[str, Any], market_data: Dict[str, Any]):
        """
        Handle ML predictions from WebSocket integration
        
        Args:
            pair: Trading pair
            prediction: ML prediction
            market_data: Market data
        """
        logger.info(f"ML prediction for {pair}: {prediction.get('direction', '').upper()} "
                   f"with {prediction.get('confidence', 0):.2f} confidence")
        
        # Check for strong signals that we should act on
        self._check_action_signals(pair, prediction, market_data)
        
        # Update model performance tracking
        self._update_model_performance(pair, prediction)
    
    def _check_action_signals(self, pair: str, prediction: Dict[str, Any], market_data: Dict[str, Any]):
        """
        Check if we should take action based on ML predictions
        
        Args:
            pair: Trading pair
            prediction: ML prediction
            market_data: Market data
        """
        # Check if there's an existing position for this pair
        existing_position = next((p for p in self.positions if p.get('pair') == pair), None)
        
        # If we have a position, the WebSocket integration will handle it
        if existing_position:
            return
        
        # If we don't have a position, check if we should enter one
        direction = prediction.get('direction', '').lower()
        confidence = prediction.get('confidence', 0)
        
        # Only enter if confidence is high enough
        min_confidence = self.ml_config.get('confidence_threshold', 0.65)
        if confidence < min_confidence:
            logger.debug(f"Confidence too low for {pair}: {confidence:.2f} < {min_confidence:.2f}")
            return
        
        # Check if we can open a new position (max positions limit)
        if len(self.positions) >= self.max_open_positions:
            logger.debug(f"Maximum positions reached: {len(self.positions)} >= {self.max_open_positions}")
            return
        
        # Get available capital
        available_capital = self.portfolio.get('available_capital', self.initial_capital)
        if available_capital <= 0:
            logger.debug(f"No available capital: {available_capital}")
            return
        
        # Get optimal position size and leverage based on confidence
        current_price = prediction.get('price', market_data.get('c', [0])[0])
        if not current_price:
            return
        
        # Calculate the optimal risk parameters
        entry_params = self.risk_manager.get_entry_parameters(
            pair=pair,
            confidence=confidence,
            account_balance=available_capital,
            current_price=current_price
        )
        
        # Create the new position
        position = {
            'pair': pair,
            'direction': direction,
            'entry_price': current_price,
            'current_price': current_price,
            'position_size': entry_params['position_size'],
            'leverage': entry_params['leverage'],
            'entry_time': datetime.now().isoformat(),
            'unrealized_pnl_pct': 0.0,
            'unrealized_pnl_amount': 0.0,
            'confidence': confidence,
            'model': prediction.get('model', 'Adaptive'),
            'category': prediction.get('category', 'those dudes'),
            'stop_loss_pct': entry_params.get('stop_loss_pct', 4.0),
            'take_profit_pct': entry_params.get('take_profit_pct', confidence * 20),
            'liquidation_price': entry_params.get(
                'liquidation_price_long' if direction == 'long' else 'liquidation_price_short'
            )
        }
        
        # Add trade to trades list
        trade = {
            'id': f"trade_{len(self.trades) + 1}",
            'pair': pair,
            'direction': direction,
            'entry_price': current_price,
            'position_size': entry_params['position_size'],
            'leverage': entry_params['leverage'],
            'entry_time': datetime.now().isoformat(),
            'status': 'open',
            'pnl': 0.0,
            'pnl_pct': 0.0,
            'exit_price': None,
            'exit_time': None,
            'confidence': confidence,
            'model': prediction.get('model', 'Adaptive'),
            'category': prediction.get('category', 'those dudes'),
            'stop_loss_pct': entry_params.get('stop_loss_pct', 4.0),
            'take_profit_pct': entry_params.get('take_profit_pct', confidence * 20),
            'liquidation_price': entry_params.get(
                'liquidation_price_long' if direction == 'long' else 'liquidation_price_short'
            )
        }
        
        # Get open trade ID to link position with trade
        position['open_trade_id'] = trade['id']
        
        # Update portfolio and data files
        try:
            # Update positions
            self.positions.append(position)
            with open(POSITIONS_FILE, 'w') as f:
                json.dump(self.positions, f, indent=2)
            
            # Update trades
            self.trades.append(trade)
            with open(TRADES_FILE, 'w') as f:
                json.dump(self.trades, f, indent=2)
            
            # Update portfolio
            available_capital -= entry_params['position_size']
            self.portfolio['available_capital'] = available_capital
            with open(PORTFOLIO_FILE, 'w') as f:
                json.dump(self.portfolio, f, indent=2)
            
            logger.info(f"Opened new position: {pair} {direction} with size ${entry_params['position_size']} "
                       f"and leverage {entry_params['leverage']}x (confidence: {confidence:.2f})")
        
        except Exception as e:
            logger.error(f"Error updating data files: {e}")
    
    def _update_model_performance(self, pair: str, prediction: Dict[str, Any]):
        """
        Update model performance tracking
        
        Args:
            pair: Trading pair
            prediction: ML prediction
        """
        # Initialize performance tracking for this pair if needed
        if pair not in self.model_performance:
            self.model_performance[pair] = {
                'predictions': [],
                'accuracy': 0.0,
                'confidence': 0.0,
                'win_rate': 0.0,
                'last_updated': datetime.now().isoformat()
            }
        
        # Add this prediction to history
        self.model_performance[pair]['predictions'].append({
            'timestamp': datetime.now().isoformat(),
            'direction': prediction.get('direction', ''),
            'confidence': prediction.get('confidence', 0),
            'price': prediction.get('price', 0)
        })
        
        # Keep only the last 100 predictions
        if len(self.model_performance[pair]['predictions']) > 100:
            self.model_performance[pair]['predictions'] = self.model_performance[pair]['predictions'][-100:]
        
        # Update average confidence
        confidences = [p.get('confidence', 0) for p in self.model_performance[pair]['predictions']]
        self.model_performance[pair]['confidence'] = sum(confidences) / len(confidences) if confidences else 0
        
        # Update last updated timestamp
        self.model_performance[pair]['last_updated'] = datetime.now().isoformat()
    
    def _update_model_weights(self):
        """Update model weights based on performance"""
        # For each pair, analyze completed trades to determine which models perform best
        pair_models = {}
        
        for trade in self.trades:
            if trade.get('status') != 'closed':
                continue
            
            pair = trade.get('pair')
            model = trade.get('model')
            pnl = trade.get('pnl', 0)
            
            if not pair or not model:
                continue
            
            if pair not in pair_models:
                pair_models[pair] = {}
            
            if model not in pair_models[pair]:
                pair_models[pair][model] = {
                    'trades': 0,
                    'wins': 0,
                    'pnl': 0.0
                }
            
            pair_models[pair][model]['trades'] += 1
            pair_models[pair][model]['pnl'] += pnl
            if pnl > 0:
                pair_models[pair][model]['wins'] += 1
        
        # Update model weights in ML config
        for pair, models in pair_models.items():
            if pair not in self.ml_config['pairs']:
                self.ml_config['pairs'][pair] = {
                    'enabled': True,
                    'max_allocation': 0.2,
                    'model_weights': {}
                }
            
            # Calculate new weights based on performance
            weights = {}
            total_pnl = sum(m.get('pnl', 0) for m in models.values())
            
            if total_pnl > 0:
                # Weight by positive PnL
                for model, stats in models.items():
                    if stats.get('pnl', 0) > 0:
                        weights[model] = stats.get('pnl', 0) / total_pnl
                    else:
                        weights[model] = 0.1  # Minimum weight for models with negative PnL
            else:
                # Weight by win rate if total PnL is negative
                total_wins = sum(m.get('wins', 0) for m in models.values())
                if total_wins > 0:
                    for model, stats in models.items():
                        weights[model] = stats.get('wins', 0) / total_wins if stats.get('trades', 0) > 0 else 0.1
                else:
                    # Equal weights if no wins
                    for model in models:
                        weights[model] = 1.0 / len(models)
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {m: w / total_weight for m, w in weights.items()}
            
            # Update config with new weights
            self.ml_config['pairs'][pair]['model_weights'] = weights
        
        # Save updated config
        self._save_ml_config()
        logger.info("Updated model weights based on performance")
    
    def _update_thread_function(self):
        """Background thread for periodic updates"""
        while self.running:
            try:
                # Load latest data
                self._load_data()
                
                # Update model weights weekly
                now = datetime.now()
                if now.weekday() == 0 and now.hour == 0 and now.minute < 5:
                    self._update_model_weights()
                
                # Sleep for 1 hour
                time.sleep(3600)
            
            except Exception as e:
                logger.error(f"Error in update thread: {e}")
                time.sleep(60)  # Short sleep on error
    
    def start(self):
        """Start the Realtime ML Manager"""
        if self.running:
            logger.warning("Realtime ML Manager already running")
            return False
        
        self.running = True
        
        # Start WebSocket integration
        self.ws_integration.start()
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_thread_function)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Started Realtime ML Manager")
        return True
    
    def stop(self):
        """Stop the Realtime ML Manager"""
        if not self.running:
            logger.warning("Realtime ML Manager not running")
            return False
        
        self.running = False
        
        # Stop WebSocket integration
        self.ws_integration.stop()
        
        logger.info("Stopped Realtime ML Manager")
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the Realtime ML Manager
        
        Returns:
            Status dictionary
        """
        self._load_data()
        
        return {
            'running': self.running,
            'trading_pairs': self.trading_pairs,
            'open_positions': len(self.positions),
            'closed_trades': len([t for t in self.trades if t.get('status') == 'closed']),
            'portfolio': {
                'initial_capital': self.initial_capital,
                'available_capital': self.portfolio.get('available_capital', 0),
                'total_value': self.portfolio.get('total_value', 0),
                'unrealized_pnl': self.portfolio.get('unrealized_pnl_usd', 0),
                'unrealized_pnl_pct': self.portfolio.get('unrealized_pnl_pct', 0)
            },
            'model_performance': self.model_performance
        }


# Example usage function
def example_usage():
    """Example usage of the Realtime ML Manager"""
    # Define trading pairs
    trading_pairs = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", 
                     "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"]
    
    # Create Realtime ML Manager
    manager = RealtimeMLManager(
        trading_pairs=trading_pairs,
        initial_capital=20000.0,
        max_open_positions=7,
        sandbox=True
    )
    
    # Start manager
    manager.start()
    
    try:
        # Keep the script running
        while True:
            # Get status every 5 minutes
            status = manager.get_status()
            logger.info(f"Status: {json.dumps(status, indent=2)}")
            
            time.sleep(300)
    
    except KeyboardInterrupt:
        logger.info("Realtime ML Manager stopped by user")
    
    finally:
        manager.stop()


if __name__ == "__main__":
    example_usage()