#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from notifications import send_trade_entry_notification, send_trade_exit_notification

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def show_notification_status():
    """
    Display the current notification configuration status
    """
    from notifications import email_available, NOTIFICATION_EMAIL, NOTIFICATIONS_LOG_FILE
    
    logger.info("Notification System Status:")
    logger.info(f"Email notifications: {'ENABLED' if email_available else 'DISABLED'}")
    logger.info(f"Notification email: {NOTIFICATION_EMAIL}")
    logger.info(f"Log file fallback: {NOTIFICATIONS_LOG_FILE}")
    
    # Check if the log file exists
    if os.path.exists(NOTIFICATIONS_LOG_FILE):
        file_size = os.path.getsize(NOTIFICATIONS_LOG_FILE)
        logger.info(f"Log file exists: Yes ({file_size} bytes)")
    else:
        logger.info("Log file exists: No (will be created when needed)")

def test_entry_notification():
    """
    Test trade entry notification
    """
    trading_pair = "SOL/USD"
    entry_price = 113.85
    quantity = 43.92
    atr = 0.1629
    volatility_stop = 113.53
    
    logger.info("Testing trade entry notification...")
    entry_result = send_trade_entry_notification(
        trading_pair=trading_pair,
        entry_price=entry_price,
        quantity=quantity,
        atr=atr,
        volatility_stop=volatility_stop
    )
    
    if entry_result:
        logger.info("Trade entry notification sent successfully!")
    else:
        logger.error("Failed to send trade entry notification.")
    
    return entry_result

def test_exit_notification():
    """
    Test trade exit notification
    """
    trading_pair = "SOL/USD"
    entry_price = 113.85
    exit_price = 115.20
    quantity = 43.92
    trade_profit_usd = (exit_price - entry_price) * quantity
    trade_profit_percent = 1.18  # (trade_profit_usd / margin_amount) * 100
    portfolio_value = 20153.22
    portfolio_profit_usd = 153.22
    portfolio_profit_percent = 0.77
    total_trades = 1
    
    logger.info("Testing trade exit notification...")
    exit_result = send_trade_exit_notification(
        trading_pair=trading_pair,
        exit_price=exit_price,
        entry_price=entry_price,
        quantity=quantity,
        trade_profit_usd=trade_profit_usd,
        trade_profit_percent=trade_profit_percent,
        portfolio_value=portfolio_value,
        portfolio_profit_usd=portfolio_profit_usd,
        portfolio_profit_percent=portfolio_profit_percent,
        total_trades=total_trades
    )
    
    if exit_result:
        logger.info("Trade exit notification sent successfully!")
    else:
        logger.error("Failed to send trade exit notification.")
    
    return exit_result

def show_log_preview():
    """
    Display a preview of the notification log file
    """
    from notifications import NOTIFICATIONS_LOG_FILE
    
    if not os.path.exists(NOTIFICATIONS_LOG_FILE):
        logger.error(f"Notification log file not found: {NOTIFICATIONS_LOG_FILE}")
        return
    
    logger.info(f"Showing preview of notification log file: {NOTIFICATIONS_LOG_FILE}")
    try:
        with open(NOTIFICATIONS_LOG_FILE, 'r') as f:
            content = f.read()
            if len(content) > 1000:
                logger.info(f"File is large ({len(content)} bytes), showing last 1000 bytes:")
                print(f"\n{'-'*80}\n{content[-1000:]}\n{'-'*80}")
            else:
                logger.info("File content:")
                print(f"\n{'-'*80}\n{content}\n{'-'*80}")
    except Exception as e:
        logger.error(f"Error reading log file: {e}")

def main():
    """
    Test the notification functionality
    """
    parser = argparse.ArgumentParser(description="Test trade notifications")
    parser.add_argument("--status", action="store_true", help="Show notification system status")
    parser.add_argument("--entry", action="store_true", help="Test trade entry notification")
    parser.add_argument("--exit", action="store_true", help="Test trade exit notification")
    parser.add_argument("--log", action="store_true", help="Show notification log preview")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # If no args specified, show help and status
    if not any(vars(args).values()):
        parser.print_help()
        print("\n")
        show_notification_status()
        return
    
    # Handle specific commands
    if args.status or args.all:
        show_notification_status()
        
    if args.entry or args.all:
        test_entry_notification()
        
    if args.exit or args.all:
        test_exit_notification()
        
    if args.log or args.all:
        show_log_preview()

if __name__ == "__main__":
    main()