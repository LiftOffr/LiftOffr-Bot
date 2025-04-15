#!/usr/bin/env python3
"""
Run Trading Bot with Real-time Data

This script starts the enhanced trading bot with real-time market data integration
for 10 cryptocurrency pairs in sandbox mode.
"""
import os
import sys
import logging
import time
from integration_controller import IntegrationController

def main():
    """Main function"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("trading_bot.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Parse trading pairs
    pairs = [
        "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", 
        "DOT/USD", "LINK/USD", "AVAX/USD", "MATIC/USD", 
        "UNI/USD", "ATOM/USD"
    ]
    
    try:
        # Initialize integration controller for real-time market data
        logger.info("Initializing integration controller for real-time market data")
        controller = IntegrationController(pairs=pairs)
        
        # Start the integration controller
        controller.start()
        logger.info("Started real-time market data integration")
        
        # Log all trading pairs
        logger.info(f"Monitoring prices for: {', '.join(pairs)}")
        
        # Initial price fetch
        logger.info("Fetching initial prices...")
        controller.update_prices()
        
        # Display current prices
        prices = controller.latest_prices
        if prices:
            logger.info("Current prices:")
            for pair, price in sorted(prices.items()):
                logger.info(f"  {pair}: ${price:.2f}")
        else:
            logger.warning("No price data available yet")
        
        # Calculate liquidation prices for sample positions
        logger.info("\nLiquidation price examples:")
        example_entry_price = 70000.0  # BTC/USD
        for leverage in [10, 25, 50, 75, 100]:
            # Long position
            long_liq_price = controller.calculate_liquidation_price(
                example_entry_price, leverage, "Long"
            )
            long_buffer_pct = (example_entry_price - long_liq_price) / example_entry_price * 100
            
            # Short position
            short_liq_price = controller.calculate_liquidation_price(
                example_entry_price, leverage, "Short"
            )
            short_buffer_pct = (short_liq_price - example_entry_price) / example_entry_price * 100
            
            logger.info(
                f"  {leverage}x leverage: "
                f"Long liq @ ${long_liq_price:.2f} ({long_buffer_pct:.2f}% buffer), "
                f"Short liq @ ${short_liq_price:.2f} ({short_buffer_pct:.2f}% buffer)"
            )
        
        # Load and update current positions
        logger.info("\nUpdating current positions with real-time prices")
        controller.update_position_prices()
        
        # Check for liquidation risks
        logger.info("Checking for liquidation risks in current positions")
        controller.check_liquidations()
        
        # Update portfolio with current unrealized P&L
        logger.info("Updating portfolio with current market data")
        controller.update_portfolio()
        
        # Keep running until interrupted
        logger.info("\nReal-time market data integration running. Press Ctrl+C to stop.")
        while True:
            # Periodically update prices and check for liquidations
            time.sleep(30)
            controller.update_prices()
            controller.update_position_prices()
            controller.check_liquidations()
            controller.update_portfolio()
            
            # Log current prices periodically
            current_time = time.strftime("%H:%M:%S", time.localtime())
            prices = controller.latest_prices
            if prices:
                btc_price = prices.get("BTC/USD", 0)
                eth_price = prices.get("ETH/USD", 0)
                sol_price = prices.get("SOL/USD", 0)
                logger.info(f"[{current_time}] BTC: ${btc_price:.2f}, ETH: ${eth_price:.2f}, SOL: ${sol_price:.2f}")
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Ensure controller is stopped
        if 'controller' in locals():
            controller.stop()
            logger.info("Stopped real-time market data integration")

if __name__ == "__main__":
    main()