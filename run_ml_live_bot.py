#!/usr/bin/env python3
"""
Run ML Live Bot

This module provides the main entry point for running the ML-enhanced trading bot
with real-time ML predictions integrated into the decision-making process.
"""

import os
import sys
import logging
import argparse
import time
from typing import List, Dict, Any, Optional

# Import trading components
from ml_live_trading_integration import MLLiveTradingIntegration
from model_collaboration_integrator import ModelCollaborationIntegrator
from bot_manager import BotManager
from kraken_api import KrakenAPI
from market_context import MarketContextAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_live_bot.log')
    ]
)
logger = logging.getLogger(__name__)

class MLTradingBot:
    """
    ML-enhanced trading bot that integrates ML predictions with trading strategies
    """
    
    def __init__(
        self,
        trading_pairs: List[str] = ["SOL/USD", "ETH/USD", "BTC/USD"],
        use_sandbox: bool = True,
        use_extreme_leverage: bool = False,
        use_ml_position_sizing: bool = False,
        capital_allocation: Optional[Dict[str, float]] = None,
        log_level: str = "INFO"
    ):
        """
        Initialize the ML trading bot
        
        Args:
            trading_pairs: List of trading pairs to trade
            use_sandbox: Whether to use sandbox mode
            use_extreme_leverage: Whether to use extreme leverage settings
            use_ml_position_sizing: Whether to use ML-enhanced position sizing
            capital_allocation: Capital allocation by asset (None for default)
            log_level: Logging level
        """
        self.trading_pairs = trading_pairs
        self.use_sandbox = use_sandbox
        self.use_extreme_leverage = use_extreme_leverage
        self.use_ml_position_sizing = use_ml_position_sizing
        
        # Set default capital allocation if not provided
        if capital_allocation is None:
            self.capital_allocation = {
                "SOL/USD": 0.40,  # 40% for SOL
                "ETH/USD": 0.35,  # 35% for ETH
                "BTC/USD": 0.25   # 25% for BTC
            }
        else:
            self.capital_allocation = capital_allocation
        
        # Set logging level
        logger.setLevel(getattr(logging, log_level))
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"ML Trading Bot initialized with {len(trading_pairs)} trading pairs")
    
    def _initialize_components(self):
        """Initialize trading bot components"""
        # Initialize Kraken API
        self.api = KrakenAPI(sandbox=self.use_sandbox)
        
        # Initialize ML trading integration
        self.ml_integration = MLLiveTradingIntegration(
            assets=self.trading_pairs,
            use_extreme_leverage=self.use_extreme_leverage
        )
        
        # Initialize model collaboration integrator
        self.model_integrator = ModelCollaborationIntegrator(
            enable_adaptive_weights=True
        )
        
        # Initialize market context analyzer
        self.market_analyzer = MarketContextAnalyzer(
            pairs=self.trading_pairs
        )
        
        # Initialize bot manager
        if self.use_sandbox:
            capital = 25000.0
        else:
            # Get actual capital in live mode
            account_info = self.api.get_account_balance()
            capital = float(account_info.get("total", 10000.0))
        
        self.bot_manager = BotManager(
            api=self.api,
            starting_capital=capital,
            trading_enabled=not self.use_sandbox
        )
        
        # Add strategies to bot manager
        self._add_trading_strategies()
    
    def _add_trading_strategies(self):
        """Add trading strategies to bot manager"""
        logger.info("Adding trading strategies to bot manager")
        
        # Define base leverage and margin for each asset
        leverage_settings = {
            "SOL/USD": {
                "standard": 5,
                "extreme": {
                    "min": 20,
                    "max": 125,
                    "default": 35
                }
            },
            "ETH/USD": {
                "standard": 3,
                "extreme": {
                    "min": 15,
                    "max": 100,
                    "default": 30
                }
            },
            "BTC/USD": {
                "standard": 2,
                "extreme": {
                    "min": 12,
                    "max": 85,
                    "default": 25
                }
            }
        }
        
        # Add bots for each trading pair
        for pair in self.trading_pairs:
            # Get leverage for this pair
            if self.use_extreme_leverage:
                leverage = leverage_settings.get(pair, {}).get("extreme", {}).get("default", 20)
            else:
                leverage = leverage_settings.get(pair, {}).get("standard", 3)
            
            # Calculate position size based on capital allocation
            allocation = self.capital_allocation.get(pair, 1.0 / len(self.trading_pairs))
            position_size = allocation * self.bot_manager.starting_capital
            
            # Add ARIMA strategy
            self.bot_manager.add_bot(
                pair=pair,
                strategy_name="ARIMAStrategy",
                position_size=position_size * 0.5,  # 50% of allocation
                leverage=leverage
            )
            
            # Add Adaptive strategy
            self.bot_manager.add_bot(
                pair=pair,
                strategy_name="AdaptiveStrategy",
                position_size=position_size * 0.5,  # 50% of allocation
                leverage=leverage
            )
            
            logger.info(f"Added strategies for {pair} with leverage {leverage}x")
    
    def run_trading_loop(self, interval: int = 60, max_iterations: Optional[int] = None):
        """
        Run the ML-enhanced trading loop
        
        Args:
            interval: Seconds between iterations
            max_iterations: Maximum number of iterations (None for infinite)
        """
        logger.info(f"Starting ML trading loop with interval {interval}s")
        
        iteration = 0
        
        try:
            while max_iterations is None or iteration < max_iterations:
                # Get current market data
                market_data = self._get_current_market_data()
                
                # Detect market regime
                for pair in self.trading_pairs:
                    if pair in market_data:
                        regime = self.market_analyzer.detect_market_regime(market_data[pair])
                        self.model_integrator.update_market_regime(regime)
                        logger.info(f"Detected {regime} market regime for {pair}")
                
                # Generate ML predictions and integrate with trading signals
                self._process_markets(market_data)
                
                # Execute trades
                self.bot_manager.execute_pending_trades()
                
                # Display portfolio status
                self.bot_manager.display_portfolio_status()
                
                # Increment iteration counter
                iteration += 1
                
                # Sleep until next iteration
                logger.debug(f"Completed iteration {iteration}, sleeping for {interval}s")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("ML trading loop stopped by user")
        except Exception as e:
            logger.error(f"Error in ML trading loop: {e}")
    
    def _get_current_market_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current market data for all trading pairs
        
        Returns:
            Dict: Market data by trading pair
        """
        market_data = {}
        
        for pair in self.trading_pairs:
            try:
                # Get OHLC data
                ohlc = self.api.get_ohlc_data(pair, interval=5, count=100)
                
                # Get ticker data
                ticker = self.api.get_ticker(pair)
                
                # Get order book
                order_book = self.api.get_order_book(pair, count=10)
                
                # Combine data
                market_data[pair] = {
                    "ohlc": ohlc,
                    "ticker": ticker,
                    "order_book": order_book,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                logger.error(f"Error getting market data for {pair}: {e}")
        
        return market_data
    
    def _process_markets(self, market_data: Dict[str, Dict[str, Any]]):
        """
        Process markets data and generate trading signals
        
        Args:
            market_data: Market data by trading pair
        """
        for pair in self.trading_pairs:
            if pair not in market_data:
                logger.warning(f"No market data available for {pair}")
                continue
            
            try:
                # Generate ML prediction
                ml_prediction = self.ml_integration.predict(pair, market_data[pair])
                
                # Generate ML trading signal
                if ml_prediction:
                    ml_signal = self.ml_integration.generate_trading_signal(ml_prediction)
                    
                    # Get current signals from bot manager
                    bot_signals = self.bot_manager.get_current_signals(pair)
                    
                    # Integrate ML signal with bot signals
                    integrated_signal = self.model_integrator.integrate_signals(
                        signals=bot_signals,
                        ml_prediction=ml_signal
                    )
                    
                    # Calculate position parameters if using ML position sizing
                    if self.use_ml_position_sizing and integrated_signal["signal_type"] != "NEUTRAL":
                        # Get current price
                        current_price = float(market_data[pair]["ticker"]["c"][0])
                        
                        # Calculate available capital
                        available_capital = self.bot_manager.get_available_capital()
                        
                        # Calculate position parameters
                        position_params = self.ml_integration.calculate_position_parameters(
                            signal=integrated_signal,
                            available_capital=available_capital,
                            current_price=current_price
                        )
                        
                        # Update signal with position parameters
                        integrated_signal["params"].update(position_params)
                    
                    # Apply integrated signal to bot manager
                    self._apply_integrated_signal(pair, integrated_signal)
                
            except Exception as e:
                logger.error(f"Error processing market data for {pair}: {e}")
    
    def _apply_integrated_signal(self, pair: str, signal: Dict[str, Any]):
        """
        Apply integrated signal to bot manager
        
        Args:
            pair: Trading pair
            signal: Integrated signal
        """
        # Extract signal components
        signal_type = signal.get("signal_type", "NEUTRAL")
        strength = signal.get("strength", 0.0)
        confidence = signal.get("confidence", 0.0)
        params = signal.get("params", {})
        
        # Only proceed if signal is strong enough
        if strength < 0.3:
            logger.info(f"Signal strength too low for {pair}: {strength:.2f}")
            return
        
        # Apply signal to bot manager
        if signal_type == "BUY":
            logger.info(f"Applying BUY signal to {pair} with strength {strength:.2f} and confidence {confidence:.2f}")
            
            # Set leverage if provided in params
            leverage = params.get("leverage", None)
            if leverage is not None:
                self.bot_manager.set_leverage(pair, int(leverage))
            
            # Execute buy
            self.bot_manager.execute_buy(pair)
            
        elif signal_type == "SELL":
            logger.info(f"Applying SELL signal to {pair} with strength {strength:.2f} and confidence {confidence:.2f}")
            
            # Set leverage if provided in params
            leverage = params.get("leverage", None)
            if leverage is not None:
                self.bot_manager.set_leverage(pair, int(leverage))
            
            # Execute sell
            self.bot_manager.execute_sell(pair)
        
        else:  # NEUTRAL
            logger.debug(f"No action taken for NEUTRAL signal on {pair}")

def main():
    """Run ML-enhanced trading bot"""
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
    
    args = parser.parse_args()
    
    # Override sandbox if live is specified
    if args.live:
        args.sandbox = False
    
    # Initialize bot
    ml_bot = MLTradingBot(
        trading_pairs=args.pairs,
        use_sandbox=args.sandbox,
        use_extreme_leverage=args.extreme_leverage,
        use_ml_position_sizing=args.ml_position_sizing
    )
    
    # Run trading loop
    ml_bot.run_trading_loop(
        interval=args.interval,
        max_iterations=args.max_iterations
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())