"""
Dynamic Position Sizing Update Script

This script updates the ultra-aggressive SOL/USD strategy with the dynamic position sizing
functionality. It integrates the ML-driven position sizing with the backtester
and trading bot for production use.
"""

import os
import sys
import logging
import json
from datetime import datetime
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s [%(levelname)s] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Import our dynamic position sizing modules
from dynamic_position_sizing import DynamicPositionSizer
from dynamic_position_sizing_implementation import MLDrivenPositionManager

# We'll need these for updating the backtester and bot manager
sys.path.append('.')
try:
    from aggressive_sol_backtest import MLEnsembleBacktester
    from advanced_ensemble_model import DynamicWeightedEnsemble
    from bot_manager import BotManager
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Make sure you're running this script from the project root directory.")
    sys.exit(1)

def update_backtester_for_dynamic_sizing():
    """
    Update the MLEnsembleBacktester class to use dynamic position sizing
    
    This function modifies the MLEnsembleBacktester to use the dynamic position
    sizing functionality instead of fixed position sizes.
    
    Returns:
        bool: Success status
    """
    try:
        logger.info("Updating backtest engine for dynamic position sizing...")
        
        # Create the position manager
        position_manager = MLDrivenPositionManager("SOL/USD", "1h")
        
        # Update the backtester's calculate_position_size method with a new version
        # This is done by monkey patching the class
        original_calc_position = MLEnsembleBacktester.calculate_position_size
        
        def dynamic_position_size(self, signal_direction, available_capital, market_data=None, signal_strength=None):
            """
            Calculate position size dynamically based on ML confidence and market conditions
            
            Args:
                signal_direction: Direction of the trade (1 for long, -1 for short)
                available_capital: Available capital for the trade
                market_data: Market data for ML analysis
                signal_strength: Signal strength from strategy
                
            Returns:
                float: Position size in base currency
            """
            # If we don't have the necessary data, fall back to the original method
            if market_data is None or signal_strength is None:
                logger.warning("Falling back to original position sizing due to missing data")
                return original_calc_position(self, signal_direction, available_capital)
            
            # Get current exposure for portfolio allocation
            current_exposure = sum(position.size_pct for position in self.open_positions) / 100.0
            
            # Get ATR value from market data if available
            atr_value = market_data['atr'].iloc[-1] if 'atr' in market_data.columns else None
            
            # Calculate dynamic position size
            position_size_pct, position_value, _ = position_manager.calculate_position_size(
                available_capital=available_capital,
                market_data=market_data,
                signal_strength=signal_strength,
                current_exposure=current_exposure,
                atr_value=atr_value,
                trailing_stop_distance=self.trailing_stop_distance if hasattr(self, 'trailing_stop_distance') else None
            )
            
            # Convert to quote currency amount
            price = market_data['close'].iloc[-1]
            base_currency_amount = position_value / price
            
            logger.info(f"Dynamic position size: {position_size_pct:.2%} ({base_currency_amount:.4f} SOL at ${price:.2f})")
            
            # Return the position size in base currency
            return base_currency_amount
        
        # Replace the method
        MLEnsembleBacktester.calculate_position_size = dynamic_position_size
        
        logger.info("Successfully updated backtest engine with dynamic position sizing")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update backtest engine: {e}")
        return False

def update_bot_manager_for_dynamic_sizing():
    """
    Update the BotManager to use dynamic position sizing
    
    This function modifies the BotManager to use the dynamic position
    sizing functionality instead of fixed position sizes.
    
    Returns:
        bool: Success status
    """
    try:
        logger.info("Updating bot manager for dynamic position sizing...")
        
        # Create position managers for each trading pair
        position_managers = {
            "SOL/USD": MLDrivenPositionManager("SOL/USD", "1h"),
            "BTC/USD": MLDrivenPositionManager("BTC/USD", "1h"),
            "ETH/USD": MLDrivenPositionManager("ETH/USD", "1h")
        }
        
        # Store the original method
        original_calc_position = BotManager.calculate_order_size
        
        def dynamic_order_size(self, strategy_name, trading_pair, signal_direction, signal_strength):
            """
            Calculate order size dynamically based on ML confidence and market conditions
            
            Args:
                strategy_name: Name of the strategy
                trading_pair: Trading pair
                signal_direction: Direction of the trade (1 for long, -1 for short)
                signal_strength: Signal strength
                
            Returns:
                float: Order size in base currency
            """
            # If we don't have a position manager for this pair, fall back to original method
            if trading_pair not in position_managers:
                logger.warning(f"No position manager for {trading_pair}, falling back to original sizing")
                return original_calc_position(self, strategy_name, trading_pair, signal_direction)
            
            # Get available capital
            available_capital = self.get_available_capital()
            
            # Get market data from the data manager
            market_data = self.get_market_data(trading_pair, limit=50)
            if market_data is None or len(market_data) < 20:
                logger.warning(f"Insufficient market data for {trading_pair}, falling back to original sizing")
                return original_calc_position(self, strategy_name, trading_pair, signal_direction)
            
            # Get current portfolio exposure
            current_exposure = self.get_portfolio_exposure()
            
            # Get ATR value
            atr_value = None
            for strategy in self.strategies.values():
                if strategy.trading_pair == trading_pair:
                    atr_value = strategy.get_atr()
                    break
            
            # Get trailing stop settings
            trailing_stop_distance = None
            for strategy in self.strategies.values():
                if strategy.trading_pair == trading_pair:
                    trailing_stop_distance = strategy.trailing_stop_distance if hasattr(strategy, 'trailing_stop_distance') else None
                    break
            
            # Calculate dynamic position size
            position_manager = position_managers[trading_pair]
            position_size_pct, position_value, metadata = position_manager.calculate_position_size(
                available_capital=available_capital,
                market_data=market_data,
                signal_strength=signal_strength,
                current_exposure=current_exposure,
                atr_value=atr_value,
                trailing_stop_distance=trailing_stop_distance
            )
            
            # Convert to base currency amount
            price = market_data['close'].iloc[-1] if 'close' in market_data.columns else self.get_current_price(trading_pair)
            base_currency_amount = position_value / price
            
            logger.info(f"Dynamic order size for {trading_pair}: {position_size_pct:.2%} ({base_currency_amount:.4f} at ${price:.2f})")
            logger.info(f"ML confidence: {metadata.get('confidence', 0):.2%} in {metadata.get('market_regime', 'unknown')} regime")
            
            # Apply maximum and minimum order sizes
            min_order = self.get_min_order_size(trading_pair)
            max_order = self.get_max_order_size(trading_pair)
            
            base_currency_amount = max(min_order, min(max_order, base_currency_amount))
            
            return base_currency_amount
        
        # Replace the method
        BotManager.calculate_order_size = dynamic_order_size
        
        logger.info("Successfully updated bot manager with dynamic position sizing")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update bot manager: {e}")
        return False

def create_configuration_files():
    """
    Create default configuration files for the dynamic position sizer
    
    These files can be edited manually to tune the position sizing parameters.
    """
    try:
        logger.info("Creating configuration files...")
        
        # Default configuration
        default_config = {
            'base_position_size': 0.35,  # Base position size (35% of available capital)
            'min_position_size': 0.15,   # Minimum position size (15% of available capital)
            'max_position_size': 0.50,   # Maximum position size (50% of available capital)
            'confidence_weight': 0.40,   # Weight given to ML confidence
            'signal_weight': 0.30,       # Weight given to signal strength
            'volatility_weight': 0.20,   # Weight given to volatility
            'trend_weight': 0.10,        # Weight given to trend strength
            'max_portfolio_exposure': 0.80,  # Maximum portfolio exposure (80%)
            'volatility_scaling_factor': 1.5, # Volatility scaling factor
            'regime_allocation': {       # Allocation adjustments for different market regimes
                'volatile_trending_up': 1.2,
                'volatile_trending_down': 0.8,
                'normal_trending_up': 1.1,
                'normal_trending_down': 0.9,
                'neutral': 1.0
            }
        }
        
        # Create ultra-aggressive configuration
        ultra_aggressive_config = default_config.copy()
        ultra_aggressive_config.update({
            'base_position_size': 0.40,  # Higher base position
            'min_position_size': 0.20,   # Higher minimum
            'max_position_size': 0.60,   # Higher maximum
            'confidence_weight': 0.45,   # More weight on ML confidence
            'max_portfolio_exposure': 0.90,  # Higher max exposure
        })
        
        # Create configurations directory
        os.makedirs('config', exist_ok=True)
        
        # Save default configuration
        with open('config/position_sizing_config.json', 'w') as f:
            json.dump(default_config, f, indent=4)
        
        # Save ultra-aggressive configuration
        with open('config/position_sizing_config_ultra_aggressive.json', 'w') as f:
            json.dump(ultra_aggressive_config, f, indent=4)
        
        logger.info("Configuration files created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create configuration files: {e}")
        return False

def run_test_with_dynamic_sizing():
    """
    Run a test with dynamic position sizing
    
    This function runs a simple test with the dynamic position sizing
    to verify that it's working correctly.
    """
    try:
        logger.info("Running dynamic position sizing test...")
        
        # Create a position manager
        position_manager = MLDrivenPositionManager("SOL/USD", "1h")
        
        # Try to load historical data
        data_file = "historical_data/SOLUSD_1h.csv"
        if not os.path.exists(data_file):
            logger.warning(f"Historical data file {data_file} not found. Skipping test.")
            return False
        
        import pandas as pd
        market_data = pd.read_csv(data_file)
        
        # Define test scenarios
        test_scenarios = [
            {
                "name": "High confidence bullish",
                "available_capital": 10000,
                "signal_strength": 0.85,
                "current_exposure": 0.2,
                "atr_value": 0.22,
                "trailing_stop_distance": 0.01
            },
            {
                "name": "Moderate confidence neutral",
                "available_capital": 10000,
                "signal_strength": 0.60,
                "current_exposure": 0.45,
                "atr_value": 0.18,
                "trailing_stop_distance": 0.015
            },
            {
                "name": "Low confidence bearish",
                "available_capital": 10000,
                "signal_strength": 0.40,
                "current_exposure": 0.65,
                "atr_value": 0.25,
                "trailing_stop_distance": 0.008
            }
        ]
        
        # Run each scenario
        for scenario in test_scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")
            
            position_size_pct, position_value, metadata = position_manager.calculate_position_size(
                available_capital=scenario["available_capital"],
                market_data=market_data.tail(50),
                signal_strength=scenario["signal_strength"],
                current_exposure=scenario["current_exposure"],
                atr_value=scenario["atr_value"],
                trailing_stop_distance=scenario["trailing_stop_distance"]
            )
            
            logger.info(f"Position size: {position_size_pct:.2%} (${position_value:.2f})")
            logger.info(f"ML prediction: {metadata.get('prediction', 'unknown')} with {metadata.get('confidence', 0):.2%} confidence")
            logger.info(f"Market regime: {metadata.get('market_regime', 'unknown')}")
            logger.info(f"Adjustments: Signal {metadata.get('position_details', {}).get('signal_strength_adjustment', 0):.2f}, "
                       f"Confidence {metadata.get('position_details', {}).get('ml_confidence_adjustment', 0):.2f}, "
                       f"Volatility {metadata.get('position_details', {}).get('volatility_adjustment', 0):.2f}")
            logger.info("---")
        
        logger.info("Dynamic position sizing test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def main():
    """Main function to update the system with dynamic position sizing"""
    parser = argparse.ArgumentParser(description="Update trading system with dynamic position sizing")
    parser.add_argument("--config-only", action="store_true", help="Only create configuration files without updating system")
    parser.add_argument("--test-only", action="store_true", help="Run a test without updating system")
    parser.add_argument("--force", action="store_true", help="Force update even if errors occur")
    args = parser.parse_args()
    
    logger.info("Dynamic Position Sizing Update Script")
    logger.info("====================================")
    
    success = True
    
    # Create configuration files
    config_success = create_configuration_files()
    success = success and config_success
    
    if args.config_only:
        logger.info("Configuration files created. Exiting as requested.")
        sys.exit(0 if success else 1)
    
    # Run test
    if args.test_only:
        test_success = run_test_with_dynamic_sizing()
        logger.info("Test completed. Exiting as requested.")
        sys.exit(0 if test_success else 1)
    
    # Update backtest engine
    backtest_success = update_backtester_for_dynamic_sizing()
    success = success and backtest_success
    
    # Update bot manager
    bot_success = update_bot_manager_for_dynamic_sizing()
    success = success and bot_success
    
    # Final status
    if success:
        logger.info("System successfully updated with dynamic position sizing!")
        logger.info("You can now run the updated backtest or trading bot with dynamic position sizing.")
    else:
        logger.error("Some updates failed. Check the log for details.")
        if not args.force:
            logger.error("Use --force to proceed with partial updates.")
            sys.exit(1)
        else:
            logger.warning("Proceeding with partial updates due to --force flag.")
    
    # Run a test
    logger.info("Running test to verify dynamic position sizing...")
    test_success = run_test_with_dynamic_sizing()
    
    logger.info("Update completed. System is ready for dynamic position sizing.")
    sys.exit(0 if (success and test_success) else 1)

if __name__ == "__main__":
    main()