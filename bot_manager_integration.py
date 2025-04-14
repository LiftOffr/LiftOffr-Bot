#!/usr/bin/env python3
"""
Bot Manager Integration with ML Components

This module provides integration between the trading bot manager and the ML components,
including ML signal processing and model collaboration.
"""

import os
import sys
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot_manager_integration.log')
    ]
)
logger = logging.getLogger(__name__)

class MLTradingBotManager:
    """
    ML Trading Bot Manager
    
    This class extends the basic trading bot manager with ML capabilities,
    integrating model predictions and collaboration into the trading decision process.
    
    It handles:
    1. Processing ML signals alongside traditional strategy signals
    2. Applying ML-based position sizing
    3. Integrating market regime awareness
    4. Tracking ML model performance
    5. Dynamic strategy weighting
    """
    
    def __init__(
        self,
        trading_pairs: List[str] = ["SOL/USD"],
        initial_capital: float = 20000.0,
        ml_integration = None,
        model_collaboration = None,
        sandbox_mode: bool = True,
        use_ml_position_sizing: bool = True,
        config_path: str = "config",
        enable_adaptive_weights: bool = True
    ):
        """
        Initialize the ML trading bot manager
        
        Args:
            trading_pairs: List of trading pairs to trade
            initial_capital: Initial trading capital
            ml_integration: ML trading integration
            model_collaboration: Model collaboration integrator
            sandbox_mode: Whether to run in sandbox mode
            use_ml_position_sizing: Whether to use ML for position sizing
            config_path: Path to configuration files
            enable_adaptive_weights: Whether to adapt weights based on performance
        """
        # Import here to avoid circular imports
        from bot_manager import BotManager
        
        self.trading_pairs = trading_pairs
        self.initial_capital = initial_capital
        self.ml_integration = ml_integration
        self.model_collaboration = model_collaboration
        self.sandbox_mode = sandbox_mode
        self.use_ml_position_sizing = use_ml_position_sizing
        self.config_path = config_path
        self.enable_adaptive_weights = enable_adaptive_weights
        
        # Create the underlying bot manager
        self.bot_manager = BotManager(
            initial_capital=initial_capital,
            sandbox_mode=sandbox_mode
        )
        
        # Initialize trading strategies
        self._initialize_trading_strategies()
        
        # Signal tracking
        self.ml_signals = {pair: {} for pair in trading_pairs}
        self.traditional_signals = {pair: {} for pair in trading_pairs}
        self.combined_signals = {pair: {} for pair in trading_pairs}
        
        # Market data tracking
        self.market_data = {pair: {} for pair in trading_pairs}
        
        logger.info(f"ML Trading Bot Manager initialized for {len(trading_pairs)} pairs")
        logger.info(f"ML position sizing: {use_ml_position_sizing}")
        logger.info(f"Sandbox mode: {sandbox_mode}")
    
    def _initialize_trading_strategies(self) -> None:
        """Initialize trading strategies for each trading pair"""
        try:
            for pair in self.trading_pairs:
                # Configure SOL trading with ARIMA and Adaptive strategies
                if "SOL" in pair:
                    self._add_traditional_strategies(pair)
                
                # Add more asset-specific strategies for ETH, BTC etc.
                elif "ETH" in pair:
                    self._add_traditional_strategies(pair)
                
                elif "BTC" in pair:
                    self._add_traditional_strategies(pair)
            
            logger.info("Trading strategies initialized")
            
        except Exception as e:
            logger.error(f"Error initializing trading strategies: {e}")
    
    def _add_traditional_strategies(self, pair: str) -> None:
        """
        Add traditional strategies for a trading pair
        
        Args:
            pair: Trading pair
        """
        try:
            from arima_strategy import ARIMAStrategy
            from fixed_strategy import AdaptiveStrategy
            
            # Default configuration with small quantity for testing
            quantity = 0.001  # Small quantity for testing
            
            # Add ARIMA strategy
            arima_bot = self.bot_manager.add_bot(
                symbol=pair,
                strategy="ARIMAStrategy",
                quantity=quantity,
                leverage=5,
                stop_percent=None,  # Use ATR for stops
                take_profit_percent=None,  # Use ATR for take profit
                enable_dual_limit_orders=True
            )
            
            # Add Adaptive strategy
            adaptive_bot = self.bot_manager.add_bot(
                symbol=pair,
                strategy="AdaptiveStrategy",
                quantity=quantity,
                leverage=5,
                stop_percent=None,  # Use ATR for stops
                take_profit_percent=None,  # Use ATR for take profit
                enable_dual_limit_orders=True
            )
            
            logger.info(f"Added traditional strategies for {pair}")
            
        except Exception as e:
            logger.error(f"Error adding traditional strategies for {pair}: {e}")
    
    def update_market_data(self) -> None:
        """Update market data for all trading pairs"""
        try:
            # Use bot manager to get latest market data
            self.bot_manager.update_market_data()
            
            # Extract market data from bot manager
            for pair in self.trading_pairs:
                # Get market data from bot manager
                pair_data = self.bot_manager.get_market_data(pair)
                
                if pair_data:
                    self.market_data[pair] = pair_data
            
            # If we have model collaboration, update market regime
            if self.model_collaboration:
                self.model_collaboration.update_market_regime(self.market_data)
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def get_market_data(self) -> Dict[str, Any]:
        """
        Get current market data
        
        Returns:
            Dict: Market data by trading pair
        """
        return self.market_data
    
    def process_ml_signal(self, pair: str, signal: Dict[str, Any]) -> None:
        """
        Process a trading signal from ML models
        
        Args:
            pair: Trading pair
            signal: Signal data
        """
        try:
            if pair not in self.trading_pairs:
                logger.warning(f"Ignoring ML signal for unknown pair: {pair}")
                return
            
            # Extract signal components
            direction = signal.get("signal", 0)  # 1 for BUY, -1 for SELL, 0 for HOLD
            confidence = signal.get("confidence", 0.0)
            position_sizing = signal.get("position_sizing", {})
            details = signal.get("details", {})
            
            # Store the ML signal
            self.ml_signals[pair] = {
                "direction": direction,
                "confidence": confidence,
                "position_sizing": position_sizing,
                "details": details,
                "timestamp": datetime.now().isoformat()
            }
            
            # Register with model collaboration if available
            if self.model_collaboration:
                self.model_collaboration.register_signal(
                    pair, "ml_ensemble", direction, confidence, details
                )
            
            # Process the signal for traditional strategies
            if direction != 0 and confidence >= 0.65:  # Only process BUY/SELL with sufficient confidence
                self._apply_ml_signal_to_strategies(pair, direction, confidence, position_sizing)
                
                # Log the processed signal
                logger.info(f"Processed ML signal for {pair}: "
                          f"{'BUY' if direction == 1 else 'SELL' if direction == -1 else 'HOLD'}, "
                          f"Confidence={confidence:.2f}")
            else:
                logger.info(f"Ignoring low confidence ML signal for {pair}: "
                          f"{'BUY' if direction == 1 else 'SELL' if direction == -1 else 'HOLD'}, "
                          f"Confidence={confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Error processing ML signal for {pair}: {e}")
    
    def _apply_ml_signal_to_strategies(
        self, 
        pair: str, 
        direction: int, 
        confidence: float,
        position_sizing: Dict[str, Any]
    ) -> None:
        """
        Apply ML signal to traditional strategies
        
        Args:
            pair: Trading pair
            direction: Signal direction (1 for BUY, -1 for SELL, 0 for HOLD)
            confidence: Signal confidence (0.0-1.0)
            position_sizing: Position sizing information
        """
        try:
            # Get all bots for this trading pair
            bots = self.bot_manager.get_bots_for_symbol(pair)
            
            if not bots:
                logger.warning(f"No bots found for {pair}, cannot apply ML signal")
                return
            
            # Apply position sizing if enabled
            if self.use_ml_position_sizing:
                leverage = position_sizing.get("leverage", 5.0)
                size = position_sizing.get("size", 0.5)
                
                # Apply position sizing to all bots
                for bot in bots:
                    # Update leverage
                    bot.leverage = int(leverage)  # Convert to int as required by bot_manager
                    
                    # Adjust quantity based on size
                    base_quantity = 0.001  # Base quantity
                    bot.quantity = base_quantity * size
                    
                    logger.info(f"Applied ML position sizing to {bot.strategy} for {pair}: "
                              f"Leverage={leverage:.1f}x, Size={size:.2f}")
            
            # Apply ML direction to strategy decision making
            # In a real implementation, we'd modify the strategy's decision process
            # For now, we just log that we would apply the signal
            logger.info(f"Would apply ML signal direction ({direction}) to strategies for {pair}")
            
        except Exception as e:
            logger.error(f"Error applying ML signal to strategies for {pair}: {e}")
    
    def evaluate_strategies(self) -> None:
        """Evaluate all trading strategies"""
        try:
            # Run the bot manager's strategy evaluation
            self.bot_manager.evaluate_all_strategies()
            
            # Collect traditional strategy signals
            for pair in self.trading_pairs:
                self._collect_traditional_signals(pair)
            
            # If we have model collaboration, get weighted signals
            if self.model_collaboration:
                self._apply_model_collaboration()
            
            # Update the bot manager with final decisions
            self._update_bot_manager_with_decisions()
            
        except Exception as e:
            logger.error(f"Error evaluating strategies: {e}")
    
    def _collect_traditional_signals(self, pair: str) -> None:
        """
        Collect signals from traditional strategies
        
        Args:
            pair: Trading pair
        """
        try:
            # Get all bots for this pair
            bots = self.bot_manager.get_bots_for_symbol(pair)
            
            if not bots:
                return
            
            # Get signals from the bot manager's signal registry
            for bot in bots:
                strategy = bot.strategy
                signal = self.bot_manager.get_latest_signal(pair, strategy)
                
                if signal:
                    # Convert to standardized format
                    direction = 0
                    if signal["action"] == "BUY":
                        direction = 1
                    elif signal["action"] == "SELL":
                        direction = -1
                    
                    # Estimate confidence from strength
                    confidence = min(1.0, max(0.1, signal["strength"]))
                    
                    # Store the signal
                    self.traditional_signals[pair][strategy] = {
                        "direction": direction,
                        "confidence": confidence,
                        "details": signal,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Register with model collaboration if available
                    if self.model_collaboration:
                        self.model_collaboration.register_signal(
                            pair, strategy, direction, confidence, signal
                        )
            
        except Exception as e:
            logger.error(f"Error collecting traditional signals for {pair}: {e}")
    
    def _apply_model_collaboration(self) -> None:
        """Apply model collaboration to generate consensus signals"""
        try:
            for pair in self.trading_pairs:
                # Get weighted signal from collaboration
                direction, confidence, details = self.model_collaboration.get_weighted_signal(pair)
                
                # Store the combined signal
                self.combined_signals[pair] = {
                    "direction": direction,
                    "confidence": confidence,
                    "details": details,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"Model collaboration signal for {pair}: "
                          f"{'BUY' if direction == 1 else 'SELL' if direction == -1 else 'HOLD'}, "
                          f"Confidence={confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Error applying model collaboration: {e}")
    
    def _update_bot_manager_with_decisions(self) -> None:
        """Update bot manager with final trading decisions"""
        try:
            # In a real implementation, we would adjust the bot manager's
            # decision-making based on combined signals
            
            # For now, just log the decisions
            for pair in self.trading_pairs:
                if pair in self.combined_signals:
                    signal = self.combined_signals[pair]
                    logger.info(f"Final decision for {pair}: "
                              f"{'BUY' if signal['direction'] == 1 else 'SELL' if signal['direction'] == -1 else 'HOLD'}, "
                              f"Confidence={signal['confidence']:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating bot manager with decisions: {e}")
    
    def execute_trades(self) -> None:
        """Execute trades based on current signals"""
        try:
            # Use the bot manager to execute trades
            self.bot_manager.execute_all_strategies()
            
            # Process trade outcomes for ML performance tracking
            self._process_trade_outcomes()
            
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
    
    def _process_trade_outcomes(self) -> None:
        """Process trade outcomes for ML performance tracking"""
        try:
            # Get recent trades from bot manager
            recent_trades = self.bot_manager.get_recent_trades()
            
            # Register performance with ML components
            for trade in recent_trades:
                pair = trade.get("symbol")
                if pair not in self.trading_pairs:
                    continue
                
                # Get relevant ML signals (if any)
                ml_signal = self.ml_signals.get(pair, {})
                ml_direction = ml_signal.get("direction", 0)
                
                # Determine outcome
                trade_direction = 1 if trade.get("side") == "buy" else -1
                profit_loss = trade.get("profit_loss", 0.0)
                
                # Consider correct if:
                # - ML predicted buy and trade made money, or
                # - ML predicted sell and trade lost money
                outcome = 0  # Neutral by default
                if ml_direction != 0:  # If ML had an opinion
                    if (ml_direction == trade_direction and profit_loss > 0) or \
                       (ml_direction != trade_direction and profit_loss < 0):
                        outcome = 1  # Correct
                    else:
                        outcome = -1  # Incorrect
                
                # Register with ML integration if available
                if self.ml_integration:
                    self.ml_integration.update_performance(pair, {
                        "entry_price": trade.get("entry_price"),
                        "exit_price": trade.get("exit_price"),
                        "profit_loss": profit_loss,
                        "direction": trade_direction,
                        "leverage": trade.get("leverage", 1),
                        "timestamp": trade.get("timestamp", datetime.now().isoformat())
                    })
                
                # Register with model collaboration if available
                if self.model_collaboration:
                    self.model_collaboration.register_performance(
                        pair, "ml_ensemble", ml_direction, outcome, trade
                    )
                
                # Log the outcome
                logger.info(f"Processed trade outcome for {pair}: "
                          f"P&L={profit_loss:.2f}%, "
                          f"ML Outcome={'Correct' if outcome == 1 else 'Incorrect' if outcome == -1 else 'Neutral'}")
            
        except Exception as e:
            logger.error(f"Error processing trade outcomes: {e}")
    
    def display_status(self) -> None:
        """Display current status of the trading bot"""
        try:
            # Display bot manager status
            self.bot_manager.display_status()
            
            # Display additional ML-specific status
            self._display_ml_status()
            
        except Exception as e:
            logger.error(f"Error displaying status: {e}")
    
    def _display_ml_status(self) -> None:
        """Display ML-specific status"""
        try:
            # Only display if ML components are available
            if not self.ml_integration and not self.model_collaboration:
                return
            
            # Display ML performance metrics
            if self.ml_integration:
                print("\n📊 ML PERFORMANCE METRICS:")
                for pair in self.trading_pairs:
                    metrics = self.ml_integration.get_performance_metrics(pair)
                    if metrics.get("total_trades", 0) > 0:
                        print(f"  {pair}: Win Rate={metrics.get('win_rate', 0.0):.2f}, "
                             f"P&L={metrics.get('total_pnl', 0.0):.2f}%, "
                             f"Trades={metrics.get('total_trades', 0)}")
            
            # Display model collaboration metrics
            if self.model_collaboration:
                print("\n🧠 MODEL COLLABORATION:")
                for pair in self.trading_pairs:
                    regime = self.model_collaboration.market_regimes.get(pair, "unknown")
                    weights = self.model_collaboration.get_strategy_weights(pair)
                    
                    print(f"  {pair}: Regime={regime}")
                    for strategy, weight in weights.items():
                        print(f"    {strategy}: {weight:.2f}")
                    
                    metrics = self.model_collaboration.get_performance_metrics(pair)
                    if metrics:
                        for strategy, perf in metrics.items():
                            if perf.get("total_predictions", 0) > 0:
                                print(f"    {strategy}: Accuracy={perf.get('accuracy', 0.0):.2f}, "
                                     f"Predictions={perf.get('total_predictions', 0)}")
            
        except Exception as e:
            logger.error(f"Error displaying ML status: {e}")

def main():
    """Test the ML trading bot manager"""
    try:
        # Import ML components
        from ml_live_trading_integration import MLLiveTradingIntegration
        from model_collaboration_integrator import ModelCollaborationIntegrator
        
        # Initialize ML components
        ml_integration = MLLiveTradingIntegration(
            trading_pairs=["SOL/USD", "ETH/USD"],
            use_extreme_leverage=True
        )
        
        model_collaboration = ModelCollaborationIntegrator(
            trading_pairs=["SOL/USD", "ETH/USD"],
            enable_adaptive_weights=True
        )
        
        # Initialize ML trading bot manager
        bot_manager = MLTradingBotManager(
            trading_pairs=["SOL/USD", "ETH/USD"],
            initial_capital=20000.0,
            ml_integration=ml_integration,
            model_collaboration=model_collaboration,
            sandbox_mode=True,
            use_ml_position_sizing=True
        )
        
        # Update market data
        bot_manager.update_market_data()
        
        # Simulate some ML signals
        signal = {
            "signal": 1,  # BUY
            "confidence": 0.85,
            "position_sizing": {
                "size": 0.8,
                "leverage": 25.0,
                "confidence": 0.85,
                "risk_level": "high"
            },
            "details": {
                "timestamp": datetime.now().isoformat(),
                "models_used": ["transformer", "tcn", "lstm"]
            }
        }
        
        bot_manager.process_ml_signal("SOL/USD", signal)
        
        # Evaluate strategies
        bot_manager.evaluate_strategies()
        
        # Display status
        bot_manager.display_status()
        
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == "__main__":
    main()