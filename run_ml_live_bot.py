#!/usr/bin/env python3
"""
Run ML Live Bot

This module provides the main entry point for running the ML-enhanced trading bot.
"""

import os
import sys
import json
import logging
import argparse
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import ML components
from ml_live_trading_integration import MLLiveTradingIntegration
from model_collaboration_integrator import ModelCollaborationIntegrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_bot.log')
    ]
)
logger = logging.getLogger(__name__)

class MLLiveBot:
    """
    ML Live Trading Bot
    
    This class integrates ML predictions with the trading system
    to execute ML-enhanced trading strategies.
    """
    
    def __init__(
        self,
        pairs: List[str] = ["SOL/USD", "ETH/USD", "BTC/USD"],
        sandbox: bool = True,
        extreme_leverage: bool = False,
        ml_position_sizing: bool = False,
        initial_capital: float = 20000.0,
        capital_allocation: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the ML live bot
        
        Args:
            pairs: List of trading pairs to trade
            sandbox: Whether to use sandbox mode
            extreme_leverage: Whether to use extreme leverage settings
            ml_position_sizing: Whether to use ML-enhanced position sizing
            initial_capital: Initial capital
            capital_allocation: Capital allocation by asset
        """
        self.pairs = pairs
        self.sandbox = sandbox
        self.extreme_leverage = extreme_leverage
        self.ml_position_sizing = ml_position_sizing
        self.initial_capital = initial_capital
        
        # Set default capital allocation if not provided
        if capital_allocation is None:
            self.capital_allocation = {
                "SOL/USD": 0.40,  # 40% allocation to SOL
                "ETH/USD": 0.35,  # 35% allocation to ETH
                "BTC/USD": 0.25   # 25% allocation to BTC
            }
        else:
            self.capital_allocation = capital_allocation
        
        # Initialize ML components
        self.ml_integration = MLLiveTradingIntegration(
            assets=pairs,
            use_extreme_leverage=extreme_leverage
        )
        
        self.collaboration_integrator = ModelCollaborationIntegrator(
            assets=pairs,
            use_ml_position_sizing=ml_position_sizing
        )
        
        # Initialize trading state
        self.positions = {}
        self.order_history = []
        self.market_data = {}
        
        logger.info(f"ML Live Bot initialized with pairs: {pairs}")
        logger.info(f"Sandbox mode: {sandbox}")
        logger.info(f"Extreme leverage: {extreme_leverage}")
        logger.info(f"ML position sizing: {ml_position_sizing}")
        logger.info(f"Initial capital: ${initial_capital:.2f}")
        logger.info(f"Capital allocation: {self.capital_allocation}")
    
    def fetch_market_data(self, pair: str) -> Dict[str, Any]:
        """
        Fetch market data for a trading pair
        
        Args:
            pair: Trading pair
            
        Returns:
            Dict: Market data
        """
        try:
            # In a real implementation, this would use the Kraken API
            # to fetch real-time market data
            
            # For now, just return dummy data
            current_time = datetime.now().isoformat()
            
            # Generate random price movement
            import random
            base_price = self.market_data.get(pair, {}).get("ticker", {}).get("c", ["100.0"])[0]
            if not isinstance(base_price, float):
                base_price = float(base_price)
            
            price_movement = random.uniform(-0.001, 0.001) * base_price
            current_price = base_price + price_movement
            
            data = {
                "pair": pair,
                "timestamp": current_time,
                "ticker": {
                    "c": [str(current_price)],
                    "v": ["1000.0"],
                    "p": [str(current_price * 0.999), str(current_price * 1.001)],
                    "t": ["100", "1000"],
                    "l": [str(current_price * 0.998)],
                    "h": [str(current_price * 1.002)],
                    "o": [str(base_price)]
                }
            }
            
            # Store data
            self.market_data[pair] = data
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {pair}: {e}")
            return {}
    
    def get_available_capital(self, pair: str) -> float:
        """
        Get available capital for a trading pair
        
        Args:
            pair: Trading pair
            
        Returns:
            float: Available capital
        """
        allocation = self.capital_allocation.get(pair, 0.0)
        available_capital = self.initial_capital * allocation
        
        return available_capital
    
    def generate_ml_predictions(self, pair: str) -> Optional[Dict[str, Any]]:
        """
        Generate ML predictions for a trading pair
        
        Args:
            pair: Trading pair
            
        Returns:
            Dict: ML predictions
        """
        # Get market data
        market_data = self.fetch_market_data(pair)
        
        if not market_data:
            logger.warning(f"No market data available for {pair}")
            return None
        
        # Generate ML prediction
        prediction = self.ml_integration.predict(pair, market_data)
        
        return prediction
    
    def execute_trades(self, pair: str, prediction: Dict[str, Any]) -> bool:
        """
        Execute trades based on ML predictions
        
        Args:
            pair: Trading pair
            prediction: ML prediction
            
        Returns:
            bool: Whether a trade was executed
        """
        try:
            # Generate trading signal from prediction
            signal = self.ml_integration.generate_trading_signal(prediction)
            
            if signal["signal_type"] == "NEUTRAL":
                logger.info(f"NEUTRAL signal for {pair}, no trade executed")
                return False
            
            # Get current price
            current_price = float(self.market_data[pair]["ticker"]["c"][0])
            
            # Get available capital
            available_capital = self.get_available_capital(pair)
            
            # Calculate position parameters
            position_params = self.ml_integration.calculate_position_parameters(
                signal=signal,
                available_capital=available_capital,
                current_price=current_price
            )
            
            # Log trade details
            logger.info(f"Executing {signal['signal_type']} trade for {pair}")
            logger.info(f"  - Price: ${current_price:.2f}")
            logger.info(f"  - Leverage: {position_params.get('leverage', 1.0):.1f}x")
            logger.info(f"  - Position size: ${position_params.get('position_size', 0.0):.2f}")
            logger.info(f"  - Stop price: ${position_params.get('stop_price', 0.0):.2f}")
            logger.info(f"  - Target price: ${position_params.get('target_price', 0.0):.2f}")
            
            # In a real implementation, this would use the Kraken API
            # to execute the trade
            
            # Record the trade
            trade = {
                "pair": pair,
                "timestamp": datetime.now().isoformat(),
                "signal_type": signal["signal_type"],
                "confidence": signal["confidence"],
                "price": current_price,
                **position_params
            }
            
            self.order_history.append(trade)
            
            # Update position state
            self.positions[pair] = {
                "signal_type": signal["signal_type"],
                "entry_price": current_price,
                "entry_time": datetime.now().isoformat(),
                "confidence": signal["confidence"],
                **position_params
            }
            
            logger.info(f"Trade executed successfully for {pair}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade for {pair}: {e}")
            return False
    
    def run_trading_iteration(self) -> Dict[str, Any]:
        """
        Run one iteration of the trading loop
        
        Returns:
            Dict: Trading iteration results
        """
        results = {}
        
        for pair in self.pairs:
            logger.info(f"Processing {pair}")
            
            # Generate ML predictions
            prediction = self.generate_ml_predictions(pair)
            
            if prediction:
                # Execute trades
                trade_executed = self.execute_trades(pair, prediction)
                
                # Store results
                results[pair] = {
                    "prediction": prediction,
                    "trade_executed": trade_executed
                }
            else:
                results[pair] = {
                    "prediction": None,
                    "trade_executed": False
                }
        
        return results
    
    def display_status(self):
        """Display current bot status"""
        logger.info("=" * 80)
        logger.info(f"ML LIVE BOT STATUS [{datetime.now().isoformat()}]")
        logger.info("=" * 80)
        
        # Display portfolio status
        logger.info("PORTFOLIO STATUS:")
        logger.info(f"  Initial capital: ${self.initial_capital:.2f}")
        
        # Display current market prices
        logger.info("CURRENT PRICES:")
        for pair in self.pairs:
            if pair in self.market_data:
                current_price = float(self.market_data[pair]["ticker"]["c"][0])
                logger.info(f"  {pair}: ${current_price:.2f}")
        
        # Display open positions
        logger.info("OPEN POSITIONS:")
        for pair, position in self.positions.items():
            signal_type = position.get("signal_type", "UNKNOWN")
            entry_price = position.get("entry_price", 0.0)
            leverage = position.get("leverage", 1.0)
            
            # Calculate current P&L if we have market data
            if pair in self.market_data:
                current_price = float(self.market_data[pair]["ticker"]["c"][0])
                if signal_type == "BUY":
                    pnl_pct = (current_price - entry_price) / entry_price * 100 * leverage
                else:  # SELL
                    pnl_pct = (entry_price - current_price) / entry_price * 100 * leverage
                
                logger.info(f"  {pair}: {signal_type} @ ${entry_price:.2f}, {leverage:.1f}x, P&L: {pnl_pct:.2f}%")
            else:
                logger.info(f"  {pair}: {signal_type} @ ${entry_price:.2f}, {leverage:.1f}x")
        
        # Display recent trades
        if self.order_history:
            logger.info("RECENT TRADES:")
            for trade in self.order_history[-5:]:
                pair = trade.get("pair", "UNKNOWN")
                signal_type = trade.get("signal_type", "UNKNOWN")
                price = trade.get("price", 0.0)
                timestamp = trade.get("timestamp", "UNKNOWN")
                
                logger.info(f"  {pair}: {signal_type} @ ${price:.2f} [{timestamp}]")
        
        logger.info("=" * 80)
    
    def run_trading_loop(
        self,
        interval: int = 60,
        max_iterations: Optional[int] = None
    ):
        """
        Run the trading loop
        
        Args:
            interval: Seconds between iterations
            max_iterations: Maximum number of iterations (None for infinite)
        """
        logger.info(f"Starting ML Live Bot trading loop")
        logger.info(f"  - Interval: {interval} seconds")
        logger.info(f"  - Max iterations: {max_iterations or 'Infinite'}")
        
        iteration = 0
        
        try:
            while max_iterations is None or iteration < max_iterations:
                logger.info(f"Trading iteration {iteration + 1}")
                
                # Run trading iteration
                self.run_trading_iteration()
                
                # Display status
                self.display_status()
                
                # Increment iteration counter
                iteration += 1
                
                # Sleep for the interval
                if max_iterations is None or iteration < max_iterations:
                    logger.info(f"Sleeping for {interval} seconds...")
                    time.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("Trading loop interrupted")
        
        logger.info(f"Trading loop completed after {iteration} iterations")

def main():
    """Run the ML live bot"""
    parser = argparse.ArgumentParser(description='Run ML-enhanced trading bot')
    
    parser.add_argument('--pairs', nargs='+', default=["SOL/USD", "ETH/USD", "BTC/USD"],
                      help='Trading pairs to trade')
    
    parser.add_argument('--sandbox', action='store_true', default=True,
                      help='Use sandbox mode (no real trades)')
    
    parser.add_argument('--live', action='store_true',
                      help='Run in live trading mode (not sandbox)')
    
    parser.add_argument('--extreme-leverage', action='store_true',
                      help='Use extreme leverage settings (20-125x)')
    
    parser.add_argument('--ml-position-sizing', action='store_true',
                      help='Use ML-enhanced position sizing')
    
    parser.add_argument('--interval', type=int, default=60,
                      help='Seconds between trading iterations')
    
    parser.add_argument('--max-iterations', type=int, default=None,
                      help='Maximum number of iterations (None for infinite)')
    
    parser.add_argument('--initial-capital', type=float, default=20000.0,
                      help='Initial capital')
    
    args = parser.parse_args()
    
    # Override sandbox if live is specified
    if args.live:
        args.sandbox = False
    
    # Create bot
    bot = MLLiveBot(
        pairs=args.pairs,
        sandbox=args.sandbox,
        extreme_leverage=args.extreme_leverage,
        ml_position_sizing=args.ml_position_sizing,
        initial_capital=args.initial_capital
    )
    
    # Run trading loop
    bot.run_trading_loop(
        interval=args.interval,
        max_iterations=args.max_iterations
    )

if __name__ == "__main__":
    main()