#!/usr/bin/env python3
"""
ML WebSocket Integration for Kraken Trading Bot

This module provides real-time integration between Kraken's WebSocket API
and ML models to optimize trading decisions in real-time.
"""
import os
import json
import time
import logging
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime

# Import our modules
from kraken_websocket_client import KrakenWebSocketClient
from ml_risk_manager import MLRiskManager
import kraken_api_client as kraken

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

class MLWebSocketIntegration:
    """
    Real-time integration between Kraken WebSocket API and ML models.
    
    This class manages the connection between real-time market data from
    Kraken's WebSocket API and the ML models used for trading decisions.
    """
    
    def __init__(
        self, 
        trading_pairs: List[str],
        ml_risk_manager: Optional[MLRiskManager] = None,
        update_interval: int = 5,
        sandbox: bool = True
    ):
        """
        Initialize the ML WebSocket integration
        
        Args:
            trading_pairs: List of trading pairs to monitor
            ml_risk_manager: ML risk manager instance (creates one if not provided)
            update_interval: Interval in seconds for updating positions
            sandbox: Whether to run in sandbox mode
        """
        self.trading_pairs = trading_pairs
        self.risk_manager = ml_risk_manager or MLRiskManager()
        self.update_interval = update_interval
        self.sandbox = sandbox
        
        # Internal state
        self.ws_client = None
        self.current_prices = {}
        self.last_predictions = {}
        self.running = False
        self.positions = []
        self.portfolio = {}
        self.ml_callbacks = []
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        logger.info(f"Initialized ML WebSocket integration for {len(trading_pairs)} pairs")
    
    def register_callback(self, callback: Callable[[str, Dict[str, Any], Dict[str, Any]], None]):
        """
        Register a callback for ML predictions
        
        Args:
            callback: Function to call with (pair, prediction, market_data)
        """
        self.ml_callbacks.append(callback)
    
    def _load_data(self):
        """Load portfolio and positions data"""
        try:
            if os.path.exists(PORTFOLIO_FILE):
                with open(PORTFOLIO_FILE, 'r') as f:
                    self.portfolio = json.load(f)
            
            if os.path.exists(POSITIONS_FILE):
                with open(POSITIONS_FILE, 'r') as f:
                    self.positions = json.load(f)
        except Exception as e:
            logger.error(f"Error loading data: {e}")
    
    def _save_positions(self):
        """Save positions to file"""
        try:
            with open(POSITIONS_FILE, 'w') as f:
                json.dump(self.positions, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving positions: {e}")
    
    def _ticker_callback(self, pair: str, data: Dict[str, Any]):
        """
        Process ticker data from WebSocket
        
        Args:
            pair: Trading pair
            data: Ticker data
        """
        # Extract the latest price
        if 'c' in data and data['c'] and len(data['c']) > 0:
            try:
                price = float(data['c'][0])
                
                # Update current price
                self.current_prices[pair] = price
                
                # Update position if we have one for this pair
                for position in self.positions:
                    if position.get('pair') == pair:
                        position['current_price'] = price
                        
                        # Calculate PnL
                        entry_price = position.get('entry_price', price)
                        size = position.get('position_size', 0)
                        leverage = position.get('leverage', 1)
                        direction = position.get('direction', 'long')
                        
                        if direction.lower() == 'long':
                            pnl_percentage = (price - entry_price) / entry_price * leverage
                            pnl_amount = size * pnl_percentage
                        else:  # short
                            pnl_percentage = (entry_price - price) / entry_price * leverage
                            pnl_amount = size * pnl_percentage
                        
                        position['unrealized_pnl_pct'] = pnl_percentage * 100
                        position['unrealized_pnl_amount'] = pnl_amount
                
                # Save updated positions
                self._save_positions()
                
                # Check for ML model updates if needed
                self._check_ml_predictions(pair, price, data)
            
            except (ValueError, TypeError) as e:
                logger.error(f"Error processing ticker data for {pair}: {e}")
    
    def _check_ml_predictions(self, pair: str, price: float, market_data: Dict[str, Any]):
        """
        Check if we need to update ML predictions
        
        Args:
            pair: Trading pair
            price: Current price
            market_data: Market data from WebSocket
        """
        # Only run ML predictions occasionally to avoid excessive CPU usage
        last_prediction_time = self.last_predictions.get(pair, {}).get('timestamp', 0)
        now = time.time()
        
        # Run prediction every 5 minutes (300 seconds)
        if now - last_prediction_time >= 300:
            # Generate market features from WebSocket data
            features = self._extract_market_features(pair, market_data)
            
            # Get ML prediction using these features
            prediction = self.risk_manager.get_real_time_prediction(pair, features)
            
            if prediction:
                prediction['timestamp'] = now
                self.last_predictions[pair] = prediction
                
                # Notify callbacks
                for callback in self.ml_callbacks:
                    try:
                        callback(pair, prediction, market_data)
                    except Exception as e:
                        logger.error(f"Error in ML callback: {e}")
                
                # Check if we need to adjust any existing positions
                self._check_position_adjustments(pair, prediction, price)
    
    def _extract_market_features(self, pair: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract market features from WebSocket data
        
        Args:
            pair: Trading pair
            market_data: Market data from WebSocket
            
        Returns:
            Dictionary of features for ML models
        """
        features = {
            'pair': pair,
            'price': self.current_prices.get(pair, 0),
            'timestamp': datetime.now().timestamp()
        }
        
        # Extract ask/bid prices
        if 'a' in market_data and market_data['a'] and len(market_data['a']) > 0:
            try:
                features['ask'] = float(market_data['a'][0])
            except (ValueError, TypeError):
                pass
        
        if 'b' in market_data and market_data['b'] and len(market_data['b']) > 0:
            try:
                features['bid'] = float(market_data['b'][0])
            except (ValueError, TypeError):
                pass
        
        # Extract volume
        if 'v' in market_data and market_data['v'] and len(market_data['v']) > 1:
            try:
                features['volume_24h'] = float(market_data['v'][1])
            except (ValueError, TypeError):
                pass
        
        # Calculate spread
        if 'ask' in features and 'bid' in features:
            features['spread'] = features['ask'] - features['bid']
            features['spread_pct'] = (features['ask'] / features['bid']) - 1 if features['bid'] > 0 else 0
        
        return features
    
    def _check_position_adjustments(self, pair: str, prediction: Dict[str, Any], price: float):
        """
        Check if we need to adjust positions based on new ML predictions
        
        Args:
            pair: Trading pair
            prediction: ML prediction
            price: Current price
        """
        # Find any existing position for this pair
        for position in self.positions:
            if position.get('pair') == pair:
                # Check if we need to exit the position
                if self._should_exit_position(position, prediction, price):
                    self._exit_position(position, price, prediction)
                # Check if we need to adjust stop loss or take profit
                elif self._should_adjust_risk_parameters(position, prediction):
                    self._adjust_risk_parameters(position, prediction)
    
    def _should_exit_position(self, 
                             position: Dict[str, Any], 
                             prediction: Dict[str, Any], 
                             price: float) -> bool:
        """
        Determine if we should exit a position based on new ML prediction
        
        Args:
            position: Position data
            prediction: ML prediction
            price: Current price
            
        Returns:
            True if position should be exited, False otherwise
        """
        # Get position details
        direction = position.get('direction', '').lower()
        entry_price = position.get('entry_price', price)
        leverage = position.get('leverage', 1)
        confidence = position.get('confidence', 0.5)
        stop_loss_pct = position.get('stop_loss_pct', 4.0)
        take_profit_pct = position.get('take_profit_pct', 8.0)
        
        # Calculate current PnL
        if direction == 'long':
            pnl_pct = ((price / entry_price) - 1) * 100 * leverage
        else:  # short
            pnl_pct = ((entry_price / price) - 1) * 100 * leverage
        
        # Check for stop loss or take profit
        if pnl_pct <= -stop_loss_pct:
            logger.info(f"Exit signal: Stop loss triggered for {position.get('pair')}")
            return True
        
        if pnl_pct >= take_profit_pct:
            logger.info(f"Exit signal: Take profit triggered for {position.get('pair')}")
            return True
        
        # Check for model signal reversal
        predicted_direction = prediction.get('direction', '').lower()
        predicted_confidence = prediction.get('confidence', 0)
        
        if predicted_direction and predicted_direction != direction and predicted_confidence >= 0.75:
            logger.info(f"Exit signal: Model signal reversed for {position.get('pair')} with {predicted_confidence:.2f} confidence")
            return True
        
        # Check for significantly reduced confidence in current direction
        if predicted_direction == direction and predicted_confidence < (confidence * 0.7):
            logger.info(f"Exit signal: Model confidence decreased significantly for {position.get('pair')}: {confidence:.2f} -> {predicted_confidence:.2f}")
            return True
        
        return False
    
    def _should_adjust_risk_parameters(self, 
                                      position: Dict[str, Any], 
                                      prediction: Dict[str, Any]) -> bool:
        """
        Determine if we should adjust risk parameters for a position
        
        Args:
            position: Position data
            prediction: ML prediction
            
        Returns:
            True if risk parameters should be adjusted, False otherwise
        """
        direction = position.get('direction', '').lower()
        predicted_direction = prediction.get('direction', '').lower()
        current_confidence = position.get('confidence', 0.5)
        predicted_confidence = prediction.get('confidence', 0)
        
        # Only adjust if prediction matches position direction
        if direction != predicted_direction:
            return False
        
        # Adjust if confidence changed significantly
        confidence_change = abs(predicted_confidence - current_confidence)
        return confidence_change >= 0.1
    
    def _adjust_risk_parameters(self, position: Dict[str, Any], prediction: Dict[str, Any]):
        """
        Adjust risk parameters for a position based on new ML prediction
        
        Args:
            position: Position data
            prediction: ML prediction
        """
        direction = position.get('direction', '').lower()
        pair = position.get('pair', '')
        current_confidence = position.get('confidence', 0.5)
        predicted_confidence = prediction.get('confidence', current_confidence)
        
        # Update confidence
        position['confidence'] = predicted_confidence
        
        # Get new risk parameters based on updated confidence
        entry_parameters = self.risk_manager.get_entry_parameters(
            pair=pair,
            confidence=predicted_confidence,
            account_balance=20000.0,  # Default, will be scaled appropriately
            current_price=position.get('current_price', position.get('entry_price', 0))
        )
        
        # Update stop loss and take profit
        position['stop_loss_pct'] = entry_parameters.get('stop_loss_pct', position.get('stop_loss_pct', 4.0))
        position['take_profit_pct'] = entry_parameters.get('take_profit_pct', position.get('take_profit_pct', 8.0))
        
        logger.info(f"Adjusted risk parameters for {pair} based on new confidence: {current_confidence:.2f} -> {predicted_confidence:.2f}")
        
        # Save updated positions
        self._save_positions()
    
    def _exit_position(self, position: Dict[str, Any], price: float, prediction: Dict[str, Any]):
        """
        Exit a position and record the trade
        
        Args:
            position: Position data
            price: Current price
            prediction: ML prediction that triggered the exit
        """
        pair = position.get('pair', '')
        direction = position.get('direction', '').lower()
        entry_price = position.get('entry_price', price)
        size = position.get('position_size', 0)
        leverage = position.get('leverage', 1)
        
        # Calculate PnL
        if direction == 'long':
            pnl_pct = ((price / entry_price) - 1) * 100 * leverage
            pnl_amount = size * ((price / entry_price) - 1) * leverage
        else:  # short
            pnl_pct = ((entry_price / price) - 1) * 100 * leverage
            pnl_amount = size * ((entry_price / price) - 1) * leverage
        
        # Record exit details
        position['exit_price'] = price
        position['exit_time'] = datetime.now().isoformat()
        position['pnl_pct'] = pnl_pct
        position['pnl_amount'] = pnl_amount
        position['exit_reason'] = f"ML_SIGNAL_{prediction.get('direction', '').upper()}"
        
        # Add to trade history
        try:
            trades = []
            if os.path.exists(TRADES_FILE):
                with open(TRADES_FILE, 'r') as f:
                    trades = json.load(f)
            
            # Add this trade
            trades.append({
                'id': f"trade_{len(trades) + 1}",
                'pair': pair,
                'direction': direction,
                'entry_price': entry_price,
                'position_size': size,
                'leverage': leverage,
                'entry_time': position.get('entry_time', ''),
                'exit_time': position.get('exit_time', ''),
                'exit_price': price,
                'pnl': pnl_amount,
                'pnl_pct': pnl_pct,
                'status': 'closed',
                'exit_reason': position.get('exit_reason', '')
            })
            
            with open(TRADES_FILE, 'w') as f:
                json.dump(trades, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
        
        # Remove from active positions
        self.positions = [p for p in self.positions if p.get('pair') != pair]
        self._save_positions()
        
        logger.info(f"Exited position: {pair} {direction} with PnL: {pnl_pct:.2f}% (${pnl_amount:.2f})")
    
    async def _start_websocket(self):
        """Start WebSocket connection for real-time data"""
        try:
            # Create WebSocket client
            self.ws_client = KrakenWebSocketClient()
            
            # Connect to WebSocket
            await self.ws_client.connect()
            
            # Subscribe to ticker data for all pairs
            for pair in self.trading_pairs:
                self.ws_client.register_ticker_callback(pair, self._ticker_callback)
            
            await self.ws_client.subscribe_ticker(self.trading_pairs)
            
            logger.info(f"Connected to Kraken WebSocket API for {len(self.trading_pairs)} pairs")
            
            # Keep connection alive
            while self.running:
                # Check for disconnects and reconnect if needed
                if not self.ws_client.running:
                    logger.warning("WebSocket connection lost, reconnecting...")
                    await self.ws_client.connect()
                    await self.ws_client.subscribe_ticker(self.trading_pairs)
                
                # Periodically update positions with latest prices
                self._load_data()
                
                # Sleep to avoid excessive CPU usage
                await asyncio.sleep(self.update_interval)
        
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
        
        finally:
            if self.ws_client:
                await self.ws_client.disconnect()
    
    def start(self):
        """Start the ML WebSocket integration"""
        if self.running:
            logger.warning("ML WebSocket integration already running")
            return False
        
        self.running = True
        
        # Start WebSocket connection in a separate thread
        def run_websocket():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._start_websocket())
        
        websocket_thread = threading.Thread(target=run_websocket)
        websocket_thread.daemon = True
        websocket_thread.start()
        
        logger.info("Started ML WebSocket integration")
        return True
    
    def stop(self):
        """Stop the ML WebSocket integration"""
        if not self.running:
            logger.warning("ML WebSocket integration not running")
            return False
        
        self.running = False
        logger.info("Stopping ML WebSocket integration")
        return True
    
    def get_position_updates(self) -> List[Dict[str, Any]]:
        """
        Get updated position data with real-time prices and PnL
        
        Returns:
            List of updated positions
        """
        self._load_data()
        return self.positions.copy()


# Example usage function
def example_usage():
    """Example usage of the ML WebSocket integration"""
    # Define trading pairs
    trading_pairs = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD"]
    
    # Create ML risk manager
    risk_manager = MLRiskManager()
    
    # Create ML WebSocket integration
    integration = MLWebSocketIntegration(
        trading_pairs=trading_pairs,
        ml_risk_manager=risk_manager,
        update_interval=5,
        sandbox=True
    )
    
    # Define callback for ML predictions
    def ml_prediction_callback(pair, prediction, market_data):
        logger.info(f"ML prediction for {pair}: {prediction.get('direction', '').upper()} "
                   f"with {prediction.get('confidence', 0):.2f} confidence")
    
    # Register callback
    integration.register_callback(ml_prediction_callback)
    
    # Start integration
    integration.start()
    
    try:
        # Keep the script running
        while True:
            # Get updated positions every 30 seconds
            positions = integration.get_position_updates()
            logger.info(f"Current positions: {len(positions)}")
            
            time.sleep(30)
    
    except KeyboardInterrupt:
        logger.info("ML WebSocket integration stopped by user")
    
    finally:
        integration.stop()


if __name__ == "__main__":
    example_usage()