"""
Web server for the Kraken trading bot.
This file is used to start the web server without the argparse arguments from main.py.
"""
import os
import logging
from bot_manager import BotManager
from config import TRADING_PAIR, TRADE_QUANTITY, STRATEGY_TYPE
from app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to start the web server with a bot manager.
    """
    logger.info("Starting web server for Kraken trading bot")
    
    # Create bot manager
    bot_manager = BotManager()
    
    # Add adaptive strategy
    bot_manager.add_bot(
        strategy_type="adaptive",
        trading_pair=TRADING_PAIR,
        trade_quantity=TRADE_QUANTITY
    )
    
    # Add ARIMA strategy
    bot_manager.add_bot(
        strategy_type="arima",
        trading_pair=TRADING_PAIR,
        trade_quantity=TRADE_QUANTITY
    )
    
    # Start bot manager
    bot_manager.start_all()
    
    # Make bot manager available to Flask
    app.config['BOT_MANAGER'] = bot_manager
    
    # Run Flask app on port 5001
    app.run(host='0.0.0.0', port=5001)

if __name__ == "__main__":
    main()