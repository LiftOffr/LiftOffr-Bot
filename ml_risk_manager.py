#!/usr/bin/env python3
"""
ML-Based Risk Manager

This module calculates optimal risk parameters and leverage based on
ML model prediction confidence and market conditions.
"""
import os
import json
import math
import logging
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
MIN_LEVERAGE = 5.0     # Minimum leverage (user specified)
MAX_LEVERAGE = 125.0   # Maximum leverage (user specified)
BASE_LEVERAGE = 10.0   # Default leverage when confidence is medium
MIN_RISK_PCT = 0.01    # Minimum risk percentage (1% of portfolio)
MAX_RISK_PCT = 0.05    # Maximum risk percentage (5% of portfolio)
BASE_RISK_PCT = 0.02   # Default risk percentage (2% of portfolio)

# Market regime types
MARKET_REGIMES = ["trending", "ranging", "volatile", "stable"]

class MLRiskManager:
    """
    ML-based risk manager that calculates optimal leverage and
    position sizing based on prediction confidence and market conditions.
    """
    
    def __init__(self, model_file: str = None):
        """
        Initialize the risk manager
        
        Args:
            model_file: Path to custom risk model parameters (if any)
        """
        self.asset_metrics = {}
        self.market_regimes = {}
        self.pair_performance = {}
        
        # Load model parameters if provided
        if model_file and os.path.exists(model_file):
            try:
                with open(model_file, 'r') as f:
                    self.parameters = json.load(f)
                logger.info(f"Loaded risk model parameters from {model_file}")
            except Exception as e:
                logger.error(f"Error loading risk model: {e}")
                self.parameters = self._default_parameters()
        else:
            self.parameters = self._default_parameters()
    
    def _default_parameters(self) -> Dict[str, Any]:
        """
        Create default risk model parameters
        
        Returns:
            Default parameters dictionary
        """
        return {
            # Base weight multipliers for different factors
            "confidence_weight": 0.5,
            "market_regime_weight": 0.2,
            "asset_volatility_weight": 0.15,
            "historical_performance_weight": 0.15,
            
            # Leverage scaling factors for different market regimes
            "regime_leverage_factors": {
                "trending": 1.2,
                "ranging": 0.8,
                "volatile": 0.6,
                "stable": 1.0
            },
            
            # Risk percentage scaling factors for different market regimes
            "regime_risk_factors": {
                "trending": 1.1,
                "ranging": 0.9,
                "volatile": 0.7,
                "stable": 1.0
            },
            
            # Asset-specific adjustment factors (can be trained/updated)
            "asset_factors": {
                "BTC/USD": {"leverage": 1.0, "risk": 1.0},
                "ETH/USD": {"leverage": 1.0, "risk": 1.0},
                "SOL/USD": {"leverage": 0.9, "risk": 0.9},
                "ADA/USD": {"leverage": 0.85, "risk": 0.85},
                "DOT/USD": {"leverage": 0.85, "risk": 0.85},
                "LINK/USD": {"leverage": 0.9, "risk": 0.9},
                "AVAX/USD": {"leverage": 0.85, "risk": 0.85},
                "MATIC/USD": {"leverage": 0.85, "risk": 0.85},
                "UNI/USD": {"leverage": 0.8, "risk": 0.8},
                "ATOM/USD": {"leverage": 0.85, "risk": 0.85}
            }
        }
    
    def update_market_regime(self, pair: str, regime: str):
        """
        Update current market regime for a specific pair
        
        Args:
            pair: Trading pair
            regime: Market regime (trending, ranging, volatile, stable)
        """
        if regime not in MARKET_REGIMES:
            logger.warning(f"Unknown market regime: {regime}, using 'stable'")
            regime = "stable"
        
        self.market_regimes[pair] = regime
        logger.debug(f"Updated market regime for {pair}: {regime}")
    
    def update_asset_metrics(self, pair: str, volatility: float, liquidity: float, 
                            strength: float):
        """
        Update asset metrics for a specific pair
        
        Args:
            pair: Trading pair
            volatility: Asset volatility score (0-1)
            liquidity: Asset liquidity score (0-1)
            strength: Trend strength score (0-1)
        """
        self.asset_metrics[pair] = {
            "volatility": volatility,
            "liquidity": liquidity,
            "strength": strength,
            "updated_at": datetime.now().isoformat()
        }
        logger.debug(f"Updated metrics for {pair}: vol={volatility:.2f}, liq={liquidity:.2f}, str={strength:.2f}")
    
    def update_pair_performance(self, pair: str, win_rate: float, avg_profit: float, 
                               avg_loss: float, trades_count: int):
        """
        Update trading performance metrics for a specific pair
        
        Args:
            pair: Trading pair
            win_rate: Win rate (0-1)
            avg_profit: Average profit per winning trade
            avg_loss: Average loss per losing trade
            trades_count: Number of trades
        """
        self.pair_performance[pair] = {
            "win_rate": win_rate,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "trades_count": trades_count,
            "profit_factor": abs(avg_profit * win_rate / (avg_loss * (1 - win_rate))) if avg_loss and (1 - win_rate) else 1.0,
            "updated_at": datetime.now().isoformat()
        }
        logger.debug(f"Updated performance for {pair}: win_rate={win_rate:.2f}, profit_factor={self.pair_performance[pair]['profit_factor']:.2f}")
    
    def get_optimal_leverage(self, pair: str, confidence: float) -> float:
        """
        Calculate optimal leverage based on prediction confidence and other factors
        
        Args:
            pair: Trading pair
            confidence: ML model prediction confidence (0-1)
            
        Returns:
            Optimal leverage value
        """
        # Validate confidence
        confidence = max(0.0, min(1.0, confidence))
        
        # Get current market regime or default to 'stable'
        regime = self.market_regimes.get(pair, "stable")
        
        # Get asset-specific factors
        asset_factors = self.parameters["asset_factors"].get(pair, {"leverage": 1.0, "risk": 1.0})
        
        # Calculate base dynamic leverage from confidence
        # Scale from MIN_LEVERAGE to MAX_LEVERAGE based on confidence
        # Higher confidence = higher leverage
        confidence_adjusted = (confidence - 0.5) * 2  # Scale from -1 to 1
        if confidence_adjusted < 0:
            # Lower confidence results in minimum leverage
            base_leverage = MIN_LEVERAGE
        else:
            # Higher confidence scales between BASE_LEVERAGE and MAX_LEVERAGE
            base_leverage = BASE_LEVERAGE + (MAX_LEVERAGE - BASE_LEVERAGE) * (confidence_adjusted ** 2)
        
        # Apply market regime adjustment
        regime_factor = self.parameters["regime_leverage_factors"].get(regime, 1.0)
        
        # Apply asset-specific adjustment
        asset_factor = asset_factors["leverage"]
        
        # Apply asset volatility adjustment if available
        volatility_factor = 1.0
        if pair in self.asset_metrics:
            # Higher volatility = lower leverage
            volatility = self.asset_metrics[pair]["volatility"]
            volatility_factor = 1.0 - (volatility * 0.5)  # Scale between 0.5 and 1.0
        
        # Apply historical performance adjustment if available
        performance_factor = 1.0
        if pair in self.pair_performance:
            # Higher win rate = higher leverage
            win_rate = self.pair_performance[pair]["win_rate"]
            profit_factor = self.pair_performance[pair]["profit_factor"]
            
            # Combine win rate and profit factor for overall performance score
            performance_score = (win_rate * 0.7) + (min(profit_factor / 3, 1.0) * 0.3)
            performance_factor = 0.8 + (performance_score * 0.4)  # Scale between 0.8 and 1.2
        
        # Calculate final leverage with all adjustments
        leverage = base_leverage * regime_factor * asset_factor * volatility_factor * performance_factor
        
        # Ensure leverage is within allowed range
        leverage = max(MIN_LEVERAGE, min(MAX_LEVERAGE, leverage))
        
        logger.debug(f"Calculated leverage for {pair} (confidence={confidence:.2f}): {leverage:.2f}x")
        logger.debug(f"  Base: {base_leverage:.2f}x, Regime: {regime_factor:.2f}, Asset: {asset_factor:.2f}, "
                   f"Volatility: {volatility_factor:.2f}, Performance: {performance_factor:.2f}")
        
        return leverage
    
    def get_position_size(self, pair: str, confidence: float, account_balance: float, 
                         current_price: float, leverage: float = None) -> Tuple[float, float]:
        """
        Calculate optimal position size based on confidence and risk parameters
        
        Args:
            pair: Trading pair
            confidence: ML model prediction confidence (0-1)
            account_balance: Current account balance
            current_price: Current price of the asset
            leverage: Leverage to use (if None, will be calculated)
            
        Returns:
            Tuple of (position size, risk percentage)
        """
        # Validate confidence
        confidence = max(0.0, min(1.0, confidence))
        
        # Calculate leverage if not provided
        if leverage is None:
            leverage = self.get_optimal_leverage(pair, confidence)
        
        # Get current market regime or default to 'stable'
        regime = self.market_regimes.get(pair, "stable")
        
        # Get asset-specific factors
        asset_factors = self.parameters["asset_factors"].get(pair, {"leverage": 1.0, "risk": 1.0})
        
        # Calculate base risk percentage from confidence
        # Scale from MIN_RISK_PCT to MAX_RISK_PCT based on confidence
        confidence_adjusted = (confidence - 0.5) * 2  # Scale from -1 to 1
        if confidence_adjusted < 0:
            # Lower confidence results in minimum risk
            base_risk = MIN_RISK_PCT
        else:
            # Higher confidence scales between BASE_RISK_PCT and MAX_RISK_PCT
            base_risk = BASE_RISK_PCT + (MAX_RISK_PCT - BASE_RISK_PCT) * (confidence_adjusted ** 2)
        
        # Apply market regime adjustment
        regime_factor = self.parameters["regime_risk_factors"].get(regime, 1.0)
        
        # Apply asset-specific adjustment
        asset_factor = asset_factors["risk"]
        
        # Calculate final risk percentage with all adjustments
        risk_percentage = base_risk * regime_factor * asset_factor
        
        # Calculate position value based on risk
        position_value = account_balance * risk_percentage
        
        # Calculate position size in base units
        position_size = position_value / current_price
        
        logger.debug(f"Calculated position for {pair} (confidence={confidence:.2f}): "
                   f"size={position_size:.6f}, risk={risk_percentage:.2%}")
        
        return position_size, risk_percentage
    
    def get_entry_parameters(self, pair: str, confidence: float, account_balance: float, 
                           current_price: float) -> Dict[str, Any]:
        """
        Get complete set of entry parameters for a trade
        
        Args:
            pair: Trading pair
            confidence: ML model prediction confidence (0-1)
            account_balance: Current account balance
            current_price: Current price of the asset
            
        Returns:
            Dictionary of entry parameters
        """
        # Calculate optimal leverage
        leverage = self.get_optimal_leverage(pair, confidence)
        
        # Calculate position size and risk
        position_size, risk_percentage = self.get_position_size(
            pair, confidence, account_balance, current_price, leverage
        )
        
        # Calculate required margin
        margin = (position_size * current_price) / leverage
        
        # Calculate liquidation price (approximate)
        # For longs: entry_price * (1 - (1 - margin_requirement) / leverage)
        # For shorts: entry_price * (1 + (1 - margin_requirement) / leverage)
        # Using a standard 10% margin requirement for calculation
        margin_requirement = 0.1 if leverage <= 10 else 0.15
        liquidation_price_long = current_price * (1 - (1 - margin_requirement) / leverage)
        liquidation_price_short = current_price * (1 + (1 - margin_requirement) / leverage)
        
        return {
            "pair": pair,
            "confidence": confidence,
            "leverage": leverage,
            "position_size": position_size,
            "risk_percentage": risk_percentage,
            "margin": margin,
            "liquidation_price_long": liquidation_price_long,
            "liquidation_price_short": liquidation_price_short,
            "current_price": current_price,
            "calculated_at": datetime.now().isoformat()
        }
    
    def simulate_ml_prediction(self, pair: str) -> Dict[str, Any]:
        """
        Simulate ML model prediction for testing
        
        Args:
            pair: Trading pair
            
        Returns:
            Simulated prediction dictionary
        """
        # Generate random confidence between 0.65 and 0.95
        confidence = random.uniform(0.65, 0.95)
        
        # Generate random direction (long/short)
        direction = "long" if random.random() > 0.5 else "short"
        
        # Generate random market regime
        regime = random.choice(MARKET_REGIMES)
        
        # Generate random metrics
        volatility = random.uniform(0.2, 0.8)
        liquidity = random.uniform(0.5, 0.9)
        strength = random.uniform(0.3, 0.9)
        
        # Update state
        self.update_market_regime(pair, regime)
        self.update_asset_metrics(pair, volatility, liquidity, strength)
        
        return {
            "pair": pair,
            "direction": direction,
            "confidence": confidence,
            "regime": regime,
            "metrics": {
                "volatility": volatility,
                "liquidity": liquidity,
                "strength": strength
            }
        }

def test_risk_manager():
    """Test the risk manager functionality"""
    # Create risk manager
    risk_manager = MLRiskManager()
    
    # Test with different pairs and confidence levels
    pairs = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "LINK/USD"]
    
    # Add some mock performance data
    for pair in pairs:
        risk_manager.update_pair_performance(
            pair=pair,
            win_rate=random.uniform(0.65, 0.95),
            avg_profit=random.uniform(0.05, 0.15),
            avg_loss=random.uniform(0.03, 0.08),
            trades_count=random.randint(20, 100)
        )
    
    print("\nTesting ML Risk Manager")
    print("=" * 50)
    
    for pair in pairs:
        # Generate simulated prediction
        prediction = risk_manager.simulate_ml_prediction(pair)
        
        print(f"\nPair: {pair}")
        print(f"Direction: {prediction['direction']}")
        print(f"Confidence: {prediction['confidence']:.2f}")
        print(f"Market Regime: {prediction['regime']}")
        
        # Calculate entry parameters
        account_balance = 20000.0
        current_price = random.uniform(10, 1000)  # Simulate different price levels
        
        entry_params = risk_manager.get_entry_parameters(
            pair=pair,
            confidence=prediction['confidence'],
            account_balance=account_balance,
            current_price=current_price
        )
        
        print(f"Current Price: ${current_price:.2f}")
        print(f"Leverage: {entry_params['leverage']:.2f}x")
        print(f"Risk Percentage: {entry_params['risk_percentage']:.2%}")
        print(f"Position Size: {entry_params['position_size']:.6f}")
        print(f"Margin: ${entry_params['margin']:.2f}")
        print(f"Liquidation Price (Long): ${entry_params['liquidation_price_long']:.2f}")
        print(f"Liquidation Price (Short): ${entry_params['liquidation_price_short']:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    test_risk_manager()