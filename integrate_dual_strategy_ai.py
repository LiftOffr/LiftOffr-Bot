#!/usr/bin/env python3
"""
Dual Strategy AI Integration Script

This script integrates the Dual Strategy AI system with the trading bot by:
1. Loading the Dual Strategy AI for each trading pair
2. Registering it with the BotManager
3. Configuring the bot to use AI-enhanced signals
4. Starting the trading bot with AI augmentation

Usage:
    python integrate_dual_strategy_ai.py --pairs SOL/USD ETH/USD BTC/USD --sandbox
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_integration.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TRADING_PAIRS = ["SOL/USD", "ETH/USD", "BTC/USD", "DOT/USD", "LINK/USD"]
CONFIG_PATH = "ml_enhanced_config.json"

def load_config(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """
    Load ML configuration from file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        else:
            logger.warning(f"Configuration file {config_path} not found, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def integrate_ai_with_trading_bot(
    trading_pairs: List[str],
    use_sandbox: bool = True,
    max_leverage: int = 125,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Integrate Dual Strategy AI with the trading bot
    
    Args:
        trading_pairs: List of trading pairs to trade
        use_sandbox: Whether to use sandbox mode
        max_leverage: Maximum allowed leverage
        confidence_threshold: Minimum confidence threshold for trades
        
    Returns:
        Dictionary with integration results
    """
    try:
        # Import required modules
        from dual_strategy_ai_integration import DualStrategyAI

        # Make sure we can import the trading bot
        import_bot = importlib.util.find_spec("main")
        if import_bot is None:
            logger.error("Cannot find main trading bot module")
            return {"success": False, "error": "Trading bot module not found"}
        
        # Load the configuration
        config = load_config()
        
        # Create AI instances for each trading pair
        ai_instances = {}
        for pair in trading_pairs:
            try:
                ai = DualStrategyAI(
                    trading_pair=pair,
                    timeframe="1h",
                    config_path=CONFIG_PATH,
                    max_leverage=max_leverage,
                    confidence_threshold=confidence_threshold
                )
                ai_instances[pair] = ai
                logger.info(f"Created Dual Strategy AI for {pair}")
            except Exception as e:
                logger.error(f"Error creating AI for {pair}: {e}")
        
        # Create modified command-line arguments for the trading bot
        import sys
        original_args = sys.argv.copy()
        
        # Build new command line arguments
        bot_args = ["main.py"]
        for pair in trading_pairs:
            bot_args.extend(["--pair", pair])
            
        # Add standard arguments
        if use_sandbox:
            bot_args.append("--sandbox")
        bot_args.extend(["--strategy", "integrated"])
        bot_args.extend(["--multi-strategy", "true"])
        
        # Replace system arguments temporarily
        sys.argv = bot_args
        
        # Store AI instances globally for the bot to access
        import builtins
        setattr(builtins, "dual_strategy_ai_instances", ai_instances)
        
        # Import and patch the bot manager to use our AI
        try:
            from bot_manager import BotManager
            
            # Store original method
            original_process_signals = BotManager.process_signals
            
            # Define our patch to enhance signal processing with AI
            def enhanced_process_signals(self, *args, **kwargs):
                """
                Enhanced signal processing method that incorporates AI recommendations
                """
                # First, call the original method
                signals, strengths = args
                if not signals:
                    return original_process_signals(self, *args, **kwargs)
                
                # Check what pairs we have signals for
                pairs_with_signals = set()
                strategy_signals = {}
                
                for strategy_id, signal in signals.items():
                    if '/' in strategy_id:
                        # Extract pair from strategy ID
                        pair = strategy_id.split('-')[-1]
                        pairs_with_signals.add(pair)
                        
                        # Group signals by pair and strategy
                        if pair not in strategy_signals:
                            strategy_signals[pair] = {}
                        
                        strategy_type = "arima" if "ARIMA" in strategy_id else "adaptive"
                        strategy_signals[pair][strategy_type] = {
                            "signal": signal,
                            "strength": strengths.get(strategy_id, 0.5)
                        }
                
                # Apply AI enhancement for pairs with signals
                for pair in pairs_with_signals:
                    if pair in ai_instances:
                        ai = ai_instances[pair]
                        
                        # Get ARIMA and Adaptive signals
                        arima_signal = strategy_signals[pair].get("arima", {"signal": "neutral", "strength": 0})
                        adaptive_signal = strategy_signals[pair].get("adaptive", {"signal": "neutral", "strength": 0})
                        
                        # Get market data
                        # For now, we'll use simple price data since we don't have full history here
                        import pandas as pd
                        from datetime import datetime, timedelta
                        
                        # Create minimal market data
                        current_price = self.get_current_price(pair.replace("/", ""))
                        minimal_data = pd.DataFrame({
                            "timestamp": [datetime.now() - timedelta(hours=i) for i in range(10, 0, -1)],
                            "open": [current_price * 0.99] * 10,
                            "high": [current_price * 1.01] * 10,
                            "low": [current_price * 0.99] * 10,
                            "close": [current_price] * 10,
                            "volume": [1000] * 10,
                            "atr": [current_price * 0.01] * 10,  # 1% ATR as placeholder
                            "volatility": [0.002] * 10,  # 0.2% volatility as placeholder
                        })
                        
                        # Get AI recommendation
                        recommendation = ai.get_recommendation(
                            current_price=current_price,
                            market_data=minimal_data,
                            arima_signal=arima_signal,
                            adaptive_signal=adaptive_signal
                        )
                        
                        # Create an integrated strategy signal
                        integrated_strategy_id = f"IntegratedStrategy-{pair}"
                        signals[integrated_strategy_id] = recommendation["signal"]
                        strengths[integrated_strategy_id] = recommendation["confidence"]
                        
                        # Log the recommendation
                        logger.info(f"AI Recommendation for {pair}: {recommendation['signal'].upper()} (confidence: {recommendation['confidence']:.2f})")
                        logger.info(f"Regime: {recommendation['regime']}, Leverage: {recommendation['leverage']}x")
                        
                        if recommendation["signal"] != "neutral":
                            # Log stop levels if we have a signal
                            stop_info = f"Stop Loss: ${recommendation['stop_loss']:.2f}, Take Profit: ${recommendation['take_profit']:.2f}"
                            logger.info(f"AI-Enhanced Position Management: {stop_info}")
                
                # Now call the original method with our enhanced signals
                return original_process_signals(self, signals, strengths, **kwargs)
            
            # Apply our patch
            BotManager.process_signals = enhanced_process_signals
            logger.info("Enhanced BotManager with AI signal processing")
            
        except Exception as e:
            logger.error(f"Error patching BotManager: {e}")
            
        # Import and run the trading bot
        try:
            import main
            logger.info("Starting trading bot with AI integration...")
            main.main()
        except Exception as e:
            logger.error(f"Error running trading bot: {e}")
            
        # Restore original arguments
        sys.argv = original_args
        
        return {
            "success": True,
            "ai_instances": len(ai_instances),
            "trading_pairs": trading_pairs
        }
        
    except Exception as e:
        logger.error(f"Error integrating AI with trading bot: {e}")
        return {"success": False, "error": str(e)}

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Integrate Dual Strategy AI with Trading Bot")
    parser.add_argument("--pairs", nargs="+", default=DEFAULT_TRADING_PAIRS,
                       help="Trading pairs to use")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox mode")
    parser.add_argument("--max-leverage", type=int, default=125,
                       help="Maximum leverage to use")
    parser.add_argument("--confidence", type=float, default=0.7,
                       help="Minimum confidence threshold for trades")
    args = parser.parse_args()
    
    # Run the integration
    result = integrate_ai_with_trading_bot(
        trading_pairs=args.pairs,
        use_sandbox=args.sandbox,
        max_leverage=args.max_leverage,
        confidence_threshold=args.confidence
    )
    
    # Log the results
    if result.get("success", False):
        logger.info("Successfully integrated AI with trading bot")
    else:
        logger.error(f"Failed to integrate AI: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()