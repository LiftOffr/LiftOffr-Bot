#!/usr/bin/env python3
"""
ML Live Trading Integration

This module provides the integration layer between ML models and the live trading system.
"""

import os
import sys
import json
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ml_live_integration.log')
    ]
)
logger = logging.getLogger(__name__)

class MLLiveTradingIntegration:
    """
    Integration layer between ML models and live trading system
    """
    
    def __init__(
        self,
        assets: List[str] = ["SOL/USD", "ETH/USD", "BTC/USD"],
        use_extreme_leverage: bool = False,
        log_level: str = "INFO"
    ):
        """
        Initialize the ML trading integration
        
        Args:
            assets: List of assets to predict
            use_extreme_leverage: Whether to use extreme leverage settings
            log_level: Logging level
        """
        self.assets = assets
        self.use_extreme_leverage = use_extreme_leverage
        
        # Set logging level
        logger.setLevel(getattr(logging, log_level))
        
        # Load ML models
        self.models = self._load_models()
        
        logger.info(f"ML Live Trading Integration initialized with {len(assets)} assets")
    
    def _load_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Load ML models for all assets
        
        Returns:
            Dict: ML models by asset
        """
        models = {}
        
        # Load models for each asset
        for asset in self.assets:
            asset_filename = asset.replace("/", "")
            
            # Check for ensemble model
            ensemble_path = f"models/ensemble/{asset_filename}_ensemble.json"
            position_sizing_path = f"models/ensemble/{asset_filename}_position_sizing.json"
            
            if os.path.exists(ensemble_path) and os.path.exists(position_sizing_path):
                # Load ensemble model
                try:
                    with open(ensemble_path, 'r') as f:
                        ensemble_config = json.load(f)
                    
                    # Load position sizing model
                    with open(position_sizing_path, 'r') as f:
                        position_sizing_config = json.load(f)
                    
                    # Store models
                    models[asset] = {
                        "ensemble": ensemble_config,
                        "position_sizing": position_sizing_config,
                        "transformer": self._load_transformer_model(asset),
                        "tcn": self._load_tcn_model(asset),
                        "lstm": self._load_lstm_model(asset)
                    }
                    
                    logger.info(f"Loaded ML models for {asset}")
                    
                except Exception as e:
                    logger.error(f"Error loading ML models for {asset}: {e}")
            else:
                logger.warning(f"ML models for {asset} not found")
        
        return models
    
    def _load_transformer_model(self, asset: str) -> Optional[Dict[str, Any]]:
        """
        Load transformer model for an asset
        
        Args:
            asset: Trading pair
            
        Returns:
            Dict: Transformer model (None if not found)
        """
        asset_filename = asset.replace("/", "")
        model_path = f"models/transformer/{asset_filename}_transformer.h5"
        
        if os.path.exists(model_path):
            # In a real implementation, this would load a proper model
            # For now, just return a placeholder
            return {"type": "transformer", "path": model_path}
        
        return None
    
    def _load_tcn_model(self, asset: str) -> Optional[Dict[str, Any]]:
        """
        Load TCN model for an asset
        
        Args:
            asset: Trading pair
            
        Returns:
            Dict: TCN model (None if not found)
        """
        asset_filename = asset.replace("/", "")
        model_path = f"models/tcn/{asset_filename}_tcn.h5"
        
        if os.path.exists(model_path):
            # In a real implementation, this would load a proper model
            # For now, just return a placeholder
            return {"type": "tcn", "path": model_path}
        
        return None
    
    def _load_lstm_model(self, asset: str) -> Optional[Dict[str, Any]]:
        """
        Load LSTM model for an asset
        
        Args:
            asset: Trading pair
            
        Returns:
            Dict: LSTM model (None if not found)
        """
        asset_filename = asset.replace("/", "")
        model_path = f"models/lstm/{asset_filename}_lstm.h5"
        
        if os.path.exists(model_path):
            # In a real implementation, this would load a proper model
            # For now, just return a placeholder
            return {"type": "lstm", "path": model_path}
        
        return None
    
    def predict(
        self,
        asset: str,
        market_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate ML prediction for an asset
        
        Args:
            asset: Trading pair
            market_data: Market data for the asset
            
        Returns:
            Dict: ML prediction
        """
        if asset not in self.models:
            logger.warning(f"No ML models available for {asset}")
            return None
        
        logger.info(f"Generating ML prediction for {asset}")
        
        try:
            # In a real implementation, this would use the models to generate
            # a proper prediction based on the market data
            
            # For now, just return a placeholder prediction
            current_price = float(market_data.get("ticker", {}).get("c", [0])[0])
            
            # Simulate random prediction
            import random
            direction = random.choice(["BUY", "SELL", "NEUTRAL"])
            confidence = random.uniform(0.6, 0.95)
            
            # Generate prediction
            prediction = {
                "asset": asset,
                "timestamp": datetime.now().isoformat(),
                "current_price": current_price,
                "direction": direction,
                "confidence": confidence,
                "predicted_price_change": random.uniform(-0.02, 0.02) * current_price,
                "models": {
                    "transformer": {"weight": 0.4, "direction": direction},
                    "tcn": {"weight": 0.3, "direction": direction},
                    "lstm": {"weight": 0.3, "direction": direction}
                }
            }
            
            logger.info(f"Generated {direction} prediction with {confidence:.2f} confidence for {asset}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error generating prediction for {asset}: {e}")
            return None
    
    def generate_trading_signal(
        self,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate trading signal from ML prediction
        
        Args:
            prediction: ML prediction
            
        Returns:
            Dict: Trading signal
        """
        try:
            # Extract prediction components
            asset = prediction.get("asset", "")
            direction = prediction.get("direction", "NEUTRAL")
            confidence = prediction.get("confidence", 0.0)
            
            # Map direction to signal type
            signal_type = "NEUTRAL"
            if direction == "BUY":
                signal_type = "BUY"
            elif direction == "SELL":
                signal_type = "SELL"
            
            # Calculate signal strength based on confidence
            strength = confidence
            
            # Determine leverage based on confidence
            leverage = self._determine_leverage(asset, confidence)
            
            # Build trading signal
            signal = {
                "signal_type": signal_type,
                "strength": strength,
                "confidence": confidence,
                "strategy": "MLStrategy",
                "pair": asset,
                "params": {
                    "leverage": leverage,
                    "confidence": confidence
                }
            }
            
            logger.info(f"Generated {signal_type} signal with strength {strength:.2f} for {asset}")
            return signal
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return {
                "signal_type": "NEUTRAL",
                "strength": 0.0,
                "confidence": 0.0,
                "strategy": "MLStrategy",
                "pair": prediction.get("asset", ""),
                "params": {}
            }
    
    def _determine_leverage(
        self,
        asset: str,
        confidence: float
    ) -> float:
        """
        Determine leverage based on confidence
        
        Args:
            asset: Trading pair
            confidence: Prediction confidence
            
        Returns:
            float: Leverage
        """
        # Get position sizing config for the asset
        position_sizing = self.models.get(asset, {}).get("position_sizing", {})
        
        # Get leverage range
        leverage_range = position_sizing.get("leverage_range", {})
        
        if self.use_extreme_leverage:
            # Use extreme leverage settings
            min_leverage = leverage_range.get("min", 5.0)
            max_leverage = leverage_range.get("max", 20.0)
        else:
            # Use normal leverage settings
            min_leverage = 1.0
            max_leverage = leverage_range.get("default", 5.0)
        
        # Calculate leverage based on confidence
        confidence_adjusted = (confidence - 0.5) * 2.0  # Scale confidence from 0.5-1.0 to 0.0-1.0
        confidence_adjusted = max(0.0, min(1.0, confidence_adjusted))
        
        leverage = min_leverage + confidence_adjusted * (max_leverage - min_leverage)
        
        # Round to nearest 0.5
        leverage = round(leverage * 2) / 2.0
        
        logger.info(f"Determined leverage {leverage:.1f}x for {asset} with confidence {confidence:.2f}")
        return leverage
    
    def calculate_position_parameters(
        self,
        signal: Dict[str, Any],
        available_capital: float,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Calculate position parameters for a signal
        
        Args:
            signal: Trading signal
            available_capital: Available capital
            current_price: Current price
            
        Returns:
            Dict: Position parameters
        """
        try:
            # Extract signal components
            asset = signal.get("pair", "")
            confidence = signal.get("confidence", 0.0)
            signal_type = signal.get("signal_type", "NEUTRAL")
            
            # Skip if neutral signal
            if signal_type == "NEUTRAL":
                return {}
            
            # Get position sizing config for the asset
            position_sizing = self.models.get(asset, {}).get("position_sizing", {})
            
            # Extract position sizing parameters
            base_size = position_sizing.get("position_sizing", {}).get("base_size", 0.05)
            min_size = position_sizing.get("position_sizing", {}).get("min_size", 0.01)
            max_size = position_sizing.get("position_sizing", {}).get("max_size", 0.20)
            
            # Calculate position size based on confidence
            if position_sizing.get("position_sizing", {}).get("confidence_scaling", True):
                # Scale position size based on confidence
                confidence_adjusted = (confidence - 0.5) * 2.0  # Scale confidence from 0.5-1.0 to 0.0-1.0
                confidence_adjusted = max(0.0, min(1.0, confidence_adjusted))
                
                position_size = min_size + confidence_adjusted * (max_size - min_size)
            else:
                # Use base position size
                position_size = base_size
            
            # Calculate capital allocation
            capital_allocation = position_size * available_capital
            
            # Calculate position parameters
            leverage = signal.get("params", {}).get("leverage", 1.0)
            
            # Extract risk parameters
            stop_loss = position_sizing.get("risk_parameters", {}).get("stop_loss", 0.04)
            profit_target = position_sizing.get("risk_parameters", {}).get("profit_target", 0.30)
            trailing_stop = position_sizing.get("risk_parameters", {}).get("trailing_stop", True)
            
            # Calculate stop prices
            if signal_type == "BUY":
                stop_price = current_price * (1.0 - stop_loss)
                target_price = current_price * (1.0 + profit_target)
            else:  # SELL
                stop_price = current_price * (1.0 + stop_loss)
                target_price = current_price * (1.0 - profit_target)
            
            # Build position parameters
            params = {
                "leverage": leverage,
                "position_size": capital_allocation,
                "stop_price": stop_price,
                "target_price": target_price,
                "trailing_stop": trailing_stop
            }
            
            logger.info(f"Calculated position parameters for {asset}: {params}")
            return params
            
        except Exception as e:
            logger.error(f"Error calculating position parameters: {e}")
            return {}

def main():
    """Test the ML live trading integration"""
    logging.basicConfig(level=logging.INFO)
    
    # Create integration
    integration = MLLiveTradingIntegration()
    
    # Create test market data
    market_data = {
        "ticker": {
            "c": ["100.0"]
        }
    }
    
    # Generate prediction
    prediction = integration.predict("SOL/USD", market_data)
    
    if prediction:
        # Generate trading signal
        signal = integration.generate_trading_signal(prediction)
        
        # Calculate position parameters
        params = integration.calculate_position_parameters(
            signal=signal,
            available_capital=10000.0,
            current_price=100.0
        )
        
        # Print results
        print(json.dumps(prediction, indent=2))
        print(json.dumps(signal, indent=2))
        print(json.dumps(params, indent=2))

if __name__ == "__main__":
    main()