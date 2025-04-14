#!/usr/bin/env python3
"""
Bot Manager Integration with ML Collaboration System

This module integrates the ML-based collaborative model system with the existing
bot manager for portfolio management and signal handling. It adds advanced
collaboration capabilities for strategy coordination and ML model integration.
"""

import os
import sys
import json
import logging
import datetime
from typing import Dict, List, Any, Tuple, Optional

# Import collaboration components
from model_collaboration_integrator import ModelCollaborationIntegrator

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

class CollaborativeBotManager:
    """
    Enhanced bot manager that integrates ML model collaboration
    """
    
    def __init__(
        self,
        config_path: str = "config/trading_config.json",
        collaboration_config_path: str = "models/ensemble/strategy_ensemble_weights.json",
        enable_collaboration: bool = True,
        enable_adaptive_weights: bool = True,
        minimum_signal_strength: float = 0.3,
        max_open_positions_per_asset: int = 1,
        log_level: str = "INFO"
    ):
        """
        Initialize the collaborative bot manager
        
        Args:
            config_path: Path to trading configuration
            collaboration_config_path: Path to collaboration configuration
            enable_collaboration: Whether to enable collaboration
            enable_adaptive_weights: Whether to allow adaptive weights
            minimum_signal_strength: Minimum signal strength to consider
            max_open_positions_per_asset: Maximum open positions per asset
            log_level: Logging level
        """
        self.config_path = config_path
        self.collaboration_config_path = collaboration_config_path
        self.enable_collaboration = enable_collaboration
        self.enable_adaptive_weights = enable_adaptive_weights
        self.minimum_signal_strength = minimum_signal_strength
        self.max_open_positions_per_asset = max_open_positions_per_asset
        
        # Set logging level
        logger.setLevel(getattr(logging, log_level))
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize collaboration integrator
        if self.enable_collaboration:
            self.integrator = ModelCollaborationIntegrator(
                config_path=collaboration_config_path,
                enable_adaptive_weights=enable_adaptive_weights
            )
        else:
            self.integrator = None
        
        # Initialize state
        self.pending_signals = {}
        self.active_positions = {}
        self.position_history = []
        self.current_market_regimes = {}
        
        logger.info(f"Collaborative Bot Manager initialized with collaboration {'enabled' if enable_collaboration else 'disabled'}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load trading configuration
        
        Returns:
            Dict: Trading configuration
        """
        default_config = {
            "assets": ["SOL/USD", "ETH/USD", "BTC/USD"],
            "sandbox_mode": True,
            "use_extreme_leverage": False,
            "use_ml_position_sizing": False,
            "trading_interval": 60,
            "max_leverage": {
                "SOL/USD": 125.0,
                "ETH/USD": 100.0,
                "BTC/USD": 85.0
            },
            "min_leverage": {
                "SOL/USD": 20.0,
                "ETH/USD": 15.0,
                "BTC/USD": 12.0
            },
            "capital_allocation": {
                "SOL/USD": 0.4,
                "ETH/USD": 0.35,
                "BTC/USD": 0.25
            }
        }
        
        try:
            # Create directory for config if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Load from file if it exists
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded trading configuration from {self.config_path}")
            else:
                # Create default config file
                config = default_config
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                logger.info(f"Created default trading configuration at {self.config_path}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading trading configuration: {e}")
            return default_config
    
    def register_signal(
        self,
        strategy_name: str,
        signal_type: str,
        strength: float,
        pair: str,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Register a trading signal from a strategy
        
        Args:
            strategy_name: Name of the strategy generating the signal
            signal_type: Type of signal (BUY, SELL, NEUTRAL)
            strength: Signal strength (0-1)
            pair: Trading pair
            price: Current price
            params: Additional signal parameters
        """
        if strength < self.minimum_signal_strength and signal_type != "NEUTRAL":
            logger.debug(f"Signal from {strategy_name} for {pair} ignored (strength {strength:.2f} < minimum {self.minimum_signal_strength:.2f})")
            return
        
        # Create signal record
        signal = {
            "strategy": strategy_name,
            "signal_type": signal_type,
            "strength": strength,
            "pair": pair,
            "price": price,
            "timestamp": datetime.datetime.now().isoformat(),
            "params": params or {}
        }
        
        # Add to pending signals
        signal_key = f"{strategy_name}-{pair}"
        self.pending_signals[signal_key] = signal
        
        logger.info(f"[BotManager] Registered {signal_type} signal from {strategy_name}-{pair} with strength {strength:.2f}")
    
    def update_market_regime(self, pair: str, regime: str):
        """
        Update market regime for a trading pair
        
        Args:
            pair: Trading pair
            regime: Market regime
        """
        self.current_market_regimes[pair] = regime
        
        # Update in collaboration integrator
        if self.enable_collaboration and self.integrator:
            self.integrator.update_market_regime(regime)
        
        logger.info(f"[BotManager] Updated market regime for {pair} to {regime}")
    
    def process_pending_signals(self, market_data: Optional[Dict[str, Any]] = None):
        """
        Process all pending signals
        
        Args:
            market_data: Current market data (optional)
        """
        if not self.pending_signals:
            return
        
        logger.info(f"[BotManager] Processing {len(self.pending_signals)} pending signals")
        
        # Group signals by trading pair
        signals_by_pair = {}
        
        for signal_key, signal in self.pending_signals.items():
            pair = signal["pair"]
            if pair not in signals_by_pair:
                signals_by_pair[pair] = {}
            
            signals_by_pair[pair][signal["strategy"]] = signal
        
        # Process each pair
        for pair, signals in signals_by_pair.items():
            # Get market regime for this pair
            regime = self.current_market_regimes.get(pair, "neutral")
            
            # Collaborative decision making
            if self.enable_collaboration and self.integrator:
                # Use model collaboration integrator to arbitrate signals
                arbitrated_signal = self.integrator.arbitrate_signals(signals, regime)
                
                # Execute the arbitrated signal
                self._execute_signal(arbitrated_signal, pair, market_data)
            else:
                # Simple decision making (take strongest signal)
                strongest_signal = None
                max_strength = 0.0
                
                for signal in signals.values():
                    if signal["signal_type"] != "NEUTRAL" and signal["strength"] > max_strength:
                        strongest_signal = signal
                        max_strength = signal["strength"]
                
                if strongest_signal:
                    self._execute_signal(strongest_signal, pair, market_data)
        
        # Clear pending signals
        self.pending_signals = {}
    
    def _execute_signal(
        self,
        signal: Dict[str, Any],
        pair: str,
        market_data: Optional[Dict[str, Any]] = None
    ):
        """
        Execute a trading signal
        
        Args:
            signal: Trading signal
            pair: Trading pair
            market_data: Current market data
        """
        signal_type = signal.get("signal_type", "NEUTRAL")
        strategy = signal.get("strategy", "Unknown")
        strength = signal.get("strength", 0.0)
        price = signal.get("price")
        params = signal.get("params", {})
        
        # Skip neutral signals
        if signal_type == "NEUTRAL":
            logger.info(f"[BotManager] Skipping NEUTRAL signal for {pair}")
            return
        
        # Check if we already have a position for this pair
        has_position = pair in self.active_positions
        
        # Skip if we already have max positions for this pair
        if has_position and len(self.active_positions[pair]) >= self.max_open_positions_per_asset:
            logger.info(f"[BotManager] Maximum positions reached for {pair}, skipping signal")
            return
        
        # Calculate position size and leverage
        position_size, leverage = self._calculate_position_parameters(signal, pair)
        
        # Record position
        position = {
            "pair": pair,
            "signal_type": signal_type,
            "strategy": strategy,
            "entry_price": price,
            "entry_time": datetime.datetime.now().isoformat(),
            "size": position_size,
            "leverage": leverage,
            "params": params,
            "status": "open"
        }
        
        # Add to active positions
        if pair not in self.active_positions:
            self.active_positions[pair] = []
        
        self.active_positions[pair].append(position)
        
        # Add to history
        self.position_history.append(position)
        
        logger.info(f"[BotManager] Executed {signal_type} signal for {pair} with size {position_size:.4f} and leverage {leverage:.1f}x")
    
    def _calculate_position_parameters(
        self,
        signal: Dict[str, Any],
        pair: str
    ) -> Tuple[float, float]:
        """
        Calculate position size and leverage
        
        Args:
            signal: Trading signal
            pair: Trading pair
            
        Returns:
            Tuple: (position_size, leverage)
        """
        # Get params from signal
        params = signal.get("params", {})
        
        # Use parameters from signal if available
        if "leverage" in params:
            leverage = params["leverage"]
        else:
            # Use default leverage from config
            leverage_range = self.config.get("max_leverage", {})
            leverage = leverage_range.get(pair, 5.0)
        
        # Cap leverage to configured max
        max_leverage = self.config.get("max_leverage", {}).get(pair, 125.0)
        min_leverage = self.config.get("min_leverage", {}).get(pair, 1.0)
        leverage = max(min_leverage, min(leverage, max_leverage))
        
        # Calculate position size
        if "margin_pct" in params:
            position_size = params["margin_pct"]
        elif "position_size" in params:
            position_size = params["position_size"]
        else:
            # Use default position size (1% of capital)
            position_size = 0.01
        
        # Cap position size to reasonable value
        position_size = max(0.001, min(position_size, 0.25))
        
        return position_size, leverage
    
    def close_position(
        self,
        pair: str,
        strategy: Optional[str] = None,
        reason: str = "manual",
        price: Optional[float] = None
    ):
        """
        Close a trading position
        
        Args:
            pair: Trading pair
            strategy: Strategy name (None for all)
            reason: Reason for closing
            price: Exit price
        """
        if pair not in self.active_positions:
            logger.warning(f"[BotManager] No active positions for {pair}")
            return
        
        # Find positions to close
        positions_to_close = []
        
        for position in self.active_positions[pair]:
            if strategy is None or position["strategy"] == strategy:
                positions_to_close.append(position)
        
        # Close positions
        for position in positions_to_close:
            # Update position
            position["exit_price"] = price
            position["exit_time"] = datetime.datetime.now().isoformat()
            position["status"] = "closed"
            position["close_reason"] = reason
            
            # Calculate profit/loss if prices are available
            if price is not None and position["entry_price"] is not None:
                if position["signal_type"] == "BUY":
                    pnl_pct = (price / position["entry_price"]) - 1.0
                else:  # SELL
                    pnl_pct = 1.0 - (price / position["entry_price"])
                
                # Apply leverage
                pnl_pct *= position["leverage"]
                
                position["pnl_pct"] = pnl_pct
                logger.info(f"[BotManager] Position P&L: {pnl_pct:.2%}")
            
            # Remove from active positions
            self.active_positions[pair].remove(position)
            
            # Add to history (update the existing entry)
            for i, hist_position in enumerate(self.position_history):
                if (hist_position["pair"] == position["pair"] and
                    hist_position["entry_time"] == position["entry_time"] and
                    hist_position["strategy"] == position["strategy"]):
                    self.position_history[i] = position
                    break
            
            logger.info(f"[BotManager] Closed {position['signal_type']} position for {pair} ({reason})")
            
            # Record performance in collaboration integrator
            if self.enable_collaboration and self.integrator:
                # Determine outcome (-1 to 1)
                if position.get("pnl_pct") is not None:
                    pnl = position["pnl_pct"]
                    # Normalize to -1 to 1 range
                    outcome = min(1.0, max(-1.0, pnl))
                else:
                    # No P&L information, use neutral outcome
                    outcome = 0.0
                
                # Get regime
                regime = self.current_market_regimes.get(pair, "neutral")
                
                # Register performance
                self.integrator.register_performance(
                    strategy=position["strategy"],
                    outcome=outcome,
                    signal_type=position["signal_type"],
                    regime=regime,
                    details={
                        "pair": pair,
                        "entry_price": position["entry_price"],
                        "exit_price": position["exit_price"],
                        "pnl_pct": position.get("pnl_pct"),
                        "close_reason": reason
                    }
                )
        
        # If no positions were closed, log warning
        if not positions_to_close:
            logger.warning(f"[BotManager] No matching positions found to close for {pair}" + 
                          (f" with strategy {strategy}" if strategy else ""))
    
    def get_active_positions(self, pair: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get active positions
        
        Args:
            pair: Trading pair (None for all)
            
        Returns:
            Dict: Active positions by pair
        """
        if pair is not None:
            return {pair: self.active_positions.get(pair, [])}
        else:
            return self.active_positions
    
    def get_position_history(self) -> List[Dict[str, Any]]:
        """
        Get position history
        
        Returns:
            List: Position history
        """
        return self.position_history
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """
        Get portfolio status
        
        Returns:
            Dict: Portfolio status
        """
        # Calculate total positions value
        total_position_value = 0.0
        position_details = []
        
        for pair, positions in self.active_positions.items():
            for position in positions:
                # Calculate position value (placeholder)
                position_value = position.get("size", 0.0) * 1000.0  # Simplified
                total_position_value += position_value
                
                position_details.append({
                    "pair": pair,
                    "strategy": position["strategy"],
                    "type": position["signal_type"],
                    "size": position["size"],
                    "leverage": position["leverage"],
                    "entry_price": position["entry_price"],
                    "entry_time": position["entry_time"],
                    "value": position_value
                })
        
        # Get collaboration metrics if available
        collaboration_metrics = {}
        
        if self.enable_collaboration and self.integrator:
            collaboration_metrics = {
                "performance_metrics": self.integrator.get_performance_metrics(),
                "weights": {
                    regime: self.integrator.get_strategy_weights(regime)
                    for regime in ["trending_bullish", "trending_bearish", "volatile", "neutral", "ranging"]
                }
            }
        
        return {
            "total_position_value": total_position_value,
            "positions": position_details,
            "market_regimes": self.current_market_regimes,
            "collaboration_metrics": collaboration_metrics,
            "timestamp": datetime.datetime.now().isoformat()
        }

def main():
    """Test the collaborative bot manager"""
    logging.basicConfig(level=logging.INFO)
    
    manager = CollaborativeBotManager(
        enable_collaboration=True,
        enable_adaptive_weights=True
    )
    
    # Register a test signal
    manager.register_signal(
        strategy_name="TestStrategy",
        signal_type="BUY",
        strength=0.8,
        pair="SOL/USD",
        price=125.0
    )
    
    # Process signals
    manager.process_pending_signals()
    
    # Print status
    print(json.dumps(manager.get_portfolio_status(), indent=2))

if __name__ == "__main__":
    main()