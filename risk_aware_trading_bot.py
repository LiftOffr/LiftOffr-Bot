#!/usr/bin/env python3
"""
Risk-Aware Trading Bot

This script integrates the advanced risk management system with the trading bot, enabling
dynamic parameters based on confidence levels with enhanced risk controls to prevent
liquidations and large losses while maximizing profits.

When executed, it:
1. Loads optimized parameters for each pair
2. Initializes the integrated risk management system
3. Sets up real-time market data analysis and risk monitoring
4. Runs the trading bot with advanced risk-aware parameters

Usage:
    python risk_aware_trading_bot.py [--pair PAIR] [--sandbox]
"""

import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Import our components
from integrated_risk_manager import integrated_risk_manager
from utils.risk_manager import risk_manager
from utils.market_analyzer import MarketAnalyzer
from utils.data_loader import HistoricalDataLoader
from dynamic_parameter_optimizer import DynamicParameterOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("risk_aware_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = "config/risk_config.json"
INT_RISK_CONFIG_PATH = "config/integrated_risk_config.json"
DATA_DIR = "historical_data"

class RiskAwareTradingBot:
    """
    Trading bot with integrated risk management to prevent liquidations and
    large losses while maximizing profits.
    """
    
    def __init__(self, pair: str = "SOL/USD", sandbox: bool = True):
        """
        Initialize the risk-aware trading bot.
        
        Args:
            pair: Trading pair to trade
            sandbox: Whether to run in sandbox mode
        """
        self.pair = pair
        self.sandbox = sandbox
        
        # Load configurations
        self.load_configs()
        
        # Initialize components
        self.integrated_risk_manager = integrated_risk_manager  # Use singleton
        self.risk_manager = risk_manager  # Use singleton
        self.market_analyzer = MarketAnalyzer()
        self.data_loader = HistoricalDataLoader()
        self.parameter_optimizer = DynamicParameterOptimizer()
        
        # Initialize state variables
        self.portfolio_value = 0.0
        self.current_positions = {}
        self.historical_data = None
        self.market_state = None
        self.current_price = 0.0
        self.last_risk_check_time = time.time()
        self.risk_check_interval = 60  # seconds
        
        # Load historical data for analysis
        self._load_historical_data()
        
        logger.info(f"Risk-aware trading bot initialized for {pair} (Sandbox: {sandbox})")
    
    def load_configs(self):
        """Load risk management and trading configurations"""
        try:
            # Load risk config
            with open(CONFIG_PATH, 'r') as f:
                self.risk_config = json.load(f)
            logger.info(f"Loaded risk configuration from {CONFIG_PATH}")
            
            # Load integrated risk config
            with open(INT_RISK_CONFIG_PATH, 'r') as f:
                self.int_risk_config = json.load(f)
            logger.info(f"Loaded integrated risk configuration from {INT_RISK_CONFIG_PATH}")
            
            # Load optimized parameters
            opt_params_path = "config/optimized_params.json"
            if os.path.exists(opt_params_path):
                with open(opt_params_path, 'r') as f:
                    self.optimized_params = json.load(f)
                logger.info(f"Loaded optimized parameters from {opt_params_path}")
            else:
                self.optimized_params = {}
                logger.warning(f"No optimized parameters found at {opt_params_path}")
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            # Use default configurations
            self.risk_config = {}
            self.int_risk_config = {"enable_integrated_risk": True}
            self.optimized_params = {}
    
    def _load_historical_data(self):
        """Load historical data for the trading pair"""
        try:
            # Load sufficient historical data for analysis (30 days)
            self.historical_data = self.data_loader.fetch_historical_data(
                pair=self.pair,
                timeframe="1h",
                days=30
            )
            
            # Add technical indicators
            self.historical_data = self.data_loader.add_technical_indicators(self.historical_data)
            
            logger.info(f"Loaded {len(self.historical_data)} data points for {self.pair}")
            
            # Initial market analysis
            self._analyze_market()
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self.historical_data = None
    
    def _analyze_market(self):
        """Analyze current market conditions"""
        if self.historical_data is None or len(self.historical_data) == 0:
            logger.warning("No historical data available for market analysis")
            return
        
        try:
            # Analyze market regimes
            self.market_state = self.market_analyzer.analyze_market_regimes(self.historical_data)
            
            # Assess volatility
            volatility_metrics = self.risk_manager.assess_volatility(self.pair, self.historical_data)
            
            # Update integrated risk manager
            self.integrated_risk_manager.update_market_data(self.pair, self.historical_data)
            
            logger.info(f"Market analysis for {self.pair}: "
                      f"Regime={self.market_state.get('current_regime')}, "
                      f"Volatility={volatility_metrics.get('volatility_category')}")
        except Exception as e:
            logger.error(f"Error analyzing market: {e}")
    
    def update_market_data(self, new_candle):
        """
        Update market data with a new candle.
        
        Args:
            new_candle: New price candle data
        """
        if self.historical_data is None:
            self._load_historical_data()
            return
        
        try:
            # Append new candle to historical data
            # In a real implementation, this would handle DataFrame operations properly
            self.historical_data = self.historical_data.append(new_candle)
            
            # Keep only the most recent data (e.g., 1000 candles)
            if len(self.historical_data) > 1000:
                self.historical_data = self.historical_data.iloc[-1000:]
            
            # Update current price
            if "close" in new_candle:
                self.current_price = new_candle["close"]
            
            # Check if it's time to re-analyze the market
            current_time = time.time()
            if current_time - self.last_risk_check_time >= self.risk_check_interval:
                self._analyze_market()
                self.last_risk_check_time = current_time
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def update_portfolio(self, portfolio_value, positions):
        """
        Update portfolio status and positions.
        
        Args:
            portfolio_value: Current portfolio value
            positions: Dictionary of current positions
        """
        self.portfolio_value = portfolio_value
        self.current_positions = positions
        
        # Update integrated risk manager
        self.integrated_risk_manager.update_portfolio(portfolio_value, positions)
    
    def get_trade_parameters(self, strategy, confidence, signal_strength):
        """
        Get risk-adjusted parameters for a trade.
        
        Args:
            strategy: Strategy name
            confidence: Signal confidence (0-1)
            signal_strength: Signal strength (0-1)
            
        Returns:
            Dictionary with risk-adjusted trade parameters
        """
        # Get parameters from integrated risk manager
        return self.integrated_risk_manager.get_trade_parameters(
            pair=self.pair,
            strategy=strategy,
            confidence=confidence,
            signal_strength=signal_strength
        )
    
    def update_active_trade(self, trade_id, current_price, position_data):
        """
        Update an active trade with risk management (e.g., trailing stops).
        
        Args:
            trade_id: Unique trade identifier
            current_price: Current market price
            position_data: Position data
            
        Returns:
            Updated position data
        """
        return self.integrated_risk_manager.update_active_trade(
            trade_id=trade_id,
            current_price=current_price,
            position_data=position_data
        )
    
    def register_trade_result(self, trade_result):
        """
        Register a completed trade result for performance tracking.
        
        Args:
            trade_result: Dictionary with trade result details
        """
        self.integrated_risk_manager.register_trade_result(trade_result)
    
    def run(self):
        """
        Run the risk-aware trading bot.
        
        This would normally integrate with the actual trading bot execution,
        but here we'll just demonstrate the risk management integration.
        """
        logger.info(f"Starting risk-aware trading bot for {self.pair} (Sandbox: {self.sandbox})")
        
        # In a real implementation, this would connect to the exchange, fetch real-time data,
        # and execute trades with risk management. For demonstration purposes, we'll just
        # showcase how the risk management would be integrated.
        
        # Example trade parameters for ARIMA strategy
        arima_params = self.get_trade_parameters(
            strategy="arima",
            confidence=0.85,
            signal_strength=0.78
        )
        
        logger.info(f"ARIMA strategy trade parameters: {arima_params}")
        
        # Example trade parameters for Adaptive strategy
        adaptive_params = self.get_trade_parameters(
            strategy="adaptive",
            confidence=0.72,
            signal_strength=0.65
        )
        
        logger.info(f"Adaptive strategy trade parameters: {adaptive_params}")
        
        # Example of updating an active trade
        example_position = {
            "entry_price": 130.0,
            "direction": "long",
            "size": 1.0,
            "leverage": 20.0,
            "entry_time": datetime.now().isoformat(),
            "initial_stop_loss": 125.0,
            "stop_loss": 125.0
        }
        
        updated_position = self.update_active_trade(
            trade_id="example_trade_1",
            current_price=135.0,  # Price moved in favor
            position_data=example_position
        )
        
        logger.info(f"Updated position: Entry={updated_position['entry_price']}, "
                  f"Current stop={updated_position['stop_loss']}, "
                  f"Profit %={updated_position.get('profit_percentage', 0):.2%}")
        
        # Example of registering a trade result
        example_trade_result = {
            "pair": self.pair,
            "strategy": "arima",
            "type": "long",
            "entry_price": 130.0,
            "exit_price": 135.0,
            "size": 1.0,
            "leverage": 20.0,
            "profit": 100.0,
            "profit_percentage": 0.0385,  # 3.85%
            "entry_timestamp": (datetime.now() - timedelta(hours=5)).isoformat(),
            "exit_timestamp": datetime.now().isoformat(),
            "exit_reason": "take_profit"
        }
        
        self.register_trade_result(example_trade_result)
        
        logger.info("Trade result registered")
        
        # In a real implementation, this would be an infinite loop that:
        # 1. Fetches real-time market data
        # 2. Updates market analysis
        # 3. Gets trading signals from strategies
        # 4. Applies risk management to trade parameters
        # 5. Executes trades and monitors positions
        # 6. Updates trailing stops and risk metrics
        
        logger.info("Risk-aware trading bot demonstration completed")
        
        return {
            "status": "success",
            "message": "Risk-aware trading bot integration demonstrated",
            "arima_params": arima_params,
            "adaptive_params": adaptive_params,
            "updated_position": updated_position
        }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Risk-aware trading bot")
    parser.add_argument("--pair", type=str, default="SOL/USD", help="Trading pair")
    parser.add_argument("--sandbox", action="store_true", help="Run in sandbox mode")
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Create and run the bot
    bot = RiskAwareTradingBot(pair=args.pair, sandbox=args.sandbox)
    result = bot.run()
    
    print("\n" + "=" * 80)
    print("RISK-AWARE TRADING BOT")
    print("=" * 80)
    print(f"Pair: {args.pair}")
    print(f"Sandbox Mode: {args.sandbox}")
    print("\nParameters for ARIMA strategy:")
    for key, value in result["arima_params"].items():
        print(f"  {key}: {value}")
    print("\nParameters for Adaptive strategy:")
    for key, value in result["adaptive_params"].items():
        print(f"  {key}: {value}")
    print("\nUpdated Position:")
    for key, value in result["updated_position"].items():
        if key == "profit_percentage":
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())