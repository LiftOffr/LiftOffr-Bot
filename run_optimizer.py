#!/usr/bin/env python3
"""
Run Trade Optimizer Integration

This script runs the trade optimizer integration with the ML trading system.
It optimizes entry and exit timing for maximum profits and minimum losses.
"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to run the optimizer integration"""
    # Check if API keys are set
    if not os.environ.get("KRAKEN_API_KEY") or not os.environ.get("KRAKEN_API_SECRET"):
        logger.error("KRAKEN_API_KEY and KRAKEN_API_SECRET must be set")
        sys.exit(1)
    
    # Import after environment check to avoid import errors
    from trade_optimizer import TradeOptimizer
    import kraken_api_client as kraken
    
    # Default trading pairs
    trading_pairs = [
        "SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD",
        "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
    ]
    
    logger.info(f"Starting trade optimizer for {len(trading_pairs)} pairs")
    
    # Get current prices for all pairs
    current_prices = {}
    kraken_client = kraken.KrakenAPIClient()
    for pair in trading_pairs:
        try:
            # Get ticker data from Kraken API
            ticker = kraken_client.get_ticker([pair])
            
            # Extract standardized pair name from Kraken's response
            kraken_pair = None
            for key in ticker.keys():
                if key.upper().replace('X', '').replace('Z', '').startswith(pair.split('/')[0].upper()):
                    kraken_pair = key
                    break
            
            if kraken_pair and 'c' in ticker[kraken_pair] and ticker[kraken_pair]['c']:
                current_prices[pair] = float(ticker[kraken_pair]['c'][0])
                logger.info(f"Current price for {pair}: ${current_prices[pair]}")
        except Exception as e:
            logger.error(f"Error getting price for {pair}: {e}")
            # Use base prices from sandbox mode for testing
            if pair == "SOL/USD":
                current_prices[pair] = 142.50
            elif pair == "BTC/USD":
                current_prices[pair] = 62350.0
            elif pair == "ETH/USD":
                current_prices[pair] = 3050.0
            elif pair == "ADA/USD":
                current_prices[pair] = 0.45
            elif pair == "DOT/USD":
                current_prices[pair] = 6.75
            elif pair == "LINK/USD":
                current_prices[pair] = 15.30
            elif pair == "AVAX/USD":
                current_prices[pair] = 35.25
            elif pair == "MATIC/USD":
                current_prices[pair] = 0.65
            elif pair == "UNI/USD":
                current_prices[pair] = 9.80
            elif pair == "ATOM/USD":
                current_prices[pair] = 8.45
            else:
                current_prices[pair] = 100.0
            logger.info(f"Using default price for {pair}: ${current_prices[pair]}")
    
    # Create and run the optimizer
    optimizer = TradeOptimizer(trading_pairs)
    
    # Run optimization
    results = optimizer.run_optimization(current_prices)
    
    # Print optimization results
    logger.info("Optimization complete. Results:")
    logger.info(f"Market states: {results.get('market_states', {})}")
    
    # Check for volatility data
    volatility_data = results.get('volatility_data', {})
    logger.info(f"Volatility regimes: {volatility_data}")
    
    # Check for optimal trading hours
    optimal_hours = results.get('optimal_hours', {})
    for pair, hours in optimal_hours.items():
        logger.info(f"Optimal hours for {pair}: Entry {hours.get('entry')}, Exit {hours.get('exit')}")
    
    # Check for position optimizations
    position_opts = results.get('position_optimizations', {})
    if position_opts:
        logger.info(f"Position adjustments: {len(position_opts.get('adjustments', []))}")
        logger.info(f"Recommended exits: {len(position_opts.get('exits', []))}")
        
        # Show detailed position adjustment recommendations
        adjustments = position_opts.get('adjustments', [])
        if adjustments:
            logger.info("Detailed position adjustments:")
            for adjustment in adjustments:
                pair = adjustment.get('pair')
                current_price = adjustment.get('current_price')
                pnl = adjustment.get('current_pnl_pct')
                signals = adjustment.get('signals', [])
                
                logger.info(f"  {pair} (Price: ${current_price}, PnL: {pnl:.2f}%):")
                for signal_type, value in signals:
                    if isinstance(value, dict):
                        logger.info(f"    - {signal_type}: {value}")
                    else:
                        logger.info(f"    - {signal_type}: {value:.2f}")
    
    # Test optimal entry price calculations
    logger.info("Sample optimal entry price calculations:")
    for pair in ["BTC/USD", "ETH/USD", "SOL/USD"]:
        if pair in current_prices:
            current_price = current_prices[pair]
            
            # Test with different confidence levels
            for direction in ["long", "short"]:
                for confidence in [0.6, 0.75, 0.9]:
                    optimal_entry = optimizer.calculate_optimal_entry_price(
                        pair, current_price, direction, confidence
                    )
                    diff_pct = ((optimal_entry / current_price) - 1) * 100
                    logger.info(
                        f"  {pair} {direction} (conf: {confidence:.2f}): "
                        f"${optimal_entry:.2f} ({diff_pct:+.2f}% from current ${current_price:.2f})"
                    )
    
    # Display portfolio allocation recommendations
    allocations = results.get('allocations', {})
    logger.info("Recommended portfolio allocations:")
    for pair, allocation in allocations.items():
        logger.info(f"  {pair}: {allocation*100:.2f}%")
    
    # Connect optimizer to real-time system
    logger.info("To activate the optimizer in the trading system, run:")
    logger.info("  python integrate_trade_optimizer.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())