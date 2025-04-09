import logging
import argparse
import os
from kraken_trading_bot import KrakenTradingBot
from app import app
from config import (
    TRADING_PAIR, TRADE_QUANTITY, STRATEGY_TYPE, USE_SANDBOX
)

def main():
    """
    Main entry point for the Kraken Trading Bot
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Kraken Trading Bot')
    parser.add_argument('--pair', type=str, help='Trading pair (e.g. XBTUSD)')
    parser.add_argument('--quantity', type=float, help='Trade quantity')
    parser.add_argument('--strategy', type=str, help='Trading strategy (simple_moving_average, rsi, combined)')
    parser.add_argument('--sandbox', action='store_true', help='Run in sandbox/test mode')
    
    args = parser.parse_args()
    
    # Get parameters from arguments or environment variables
    trading_pair = args.pair or TRADING_PAIR
    trade_quantity = args.quantity or TRADE_QUANTITY
    strategy_type = args.strategy or STRATEGY_TYPE
    sandbox_mode = args.sandbox or USE_SANDBOX
    
    # Override environment variables for later use
    if args.pair:
        os.environ['TRADING_PAIR'] = args.pair
    if args.quantity:
        os.environ['TRADE_QUANTITY'] = str(args.quantity)
    if args.strategy:
        os.environ['STRATEGY_TYPE'] = args.strategy
    if args.sandbox:
        os.environ['USE_SANDBOX'] = 'True'
    
    # Initialize and run the trading bot
    bot = KrakenTradingBot(
        trading_pair=trading_pair,
        trade_quantity=trade_quantity,
        strategy_type=strategy_type
    )
    bot.run()

if __name__ == '__main__':
    main()
