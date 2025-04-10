import os
import sys
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Notification settings
NOTIFICATION_EMAIL = "cchapman.liftoffr@gmail.com"  # Email for notifications
NOTIFICATIONS_LOG_FILE = "trade_notifications.log"

# Try to import SendGrid if available
email_available = False
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content
    
    # Get SendGrid API key from environment
    SENDGRID_API_KEY = os.environ.get('SENDGRID_API_KEY')
    # FROM_EMAIL must be a verified sender in your SendGrid account
    FROM_EMAIL = "cchapman.liftoffr@gmail.com"  # Using recipient email as sender
    
    if SENDGRID_API_KEY:
        email_available = True
        logger.info("Email notifications enabled (using SendGrid)")
    else:
        logger.warning("SendGrid API key not set. Email notifications disabled.")
except ImportError:
    logger.warning("SendGrid package not available. Email notifications disabled.")
    SENDGRID_API_KEY = None


def log_notification_to_file(subject, content):
    """
    Log notification to a file as fallback when email fails
    
    Args:
        subject (str): Email subject
        content (str): Email content
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        with open(NOTIFICATIONS_LOG_FILE, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"NOTIFICATION: {subject}\n")
            f.write(f"TIME: {timestamp}\n")
            f.write(f"{'='*60}\n")
            f.write(f"{content}\n")
            f.write(f"{'='*60}\n")
        
        logger.info(f"Notification logged to file: {NOTIFICATIONS_LOG_FILE}")
        return True
    except Exception as e:
        logger.error(f"Failed to log notification to file: {e}")
        return False


def send_trade_entry_notification(trading_pair, entry_price, quantity, atr, volatility_stop):
    """
    Send notification when a trade is entered
    
    Args:
        trading_pair (str): Trading pair (e.g., SOL/USD)
        entry_price (float): Entry price
        quantity (float): Quantity traded
        atr (float): Average True Range
        volatility_stop (float): Stop loss price based on volatility
    """
    subject = f"🟢 TRADE ENTERED: {trading_pair}"
    
    # Format the message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    position_value = entry_price * quantity
    
    text_content = (
        f"TRADE ENTERED: {trading_pair}\n"
        f"Time: {timestamp}\n"
        f"Price: ${entry_price:.2f}\n"
        f"Quantity: {quantity}\n"
        f"Position Value: ${position_value:.2f}\n"
        f"ATR: ${atr:.4f}\n"
        f"Volatility Stop: ${volatility_stop:.2f}"
    )
    
    # Try to send email if available
    if email_available:
        try:
            # Create the email message
            message = Mail(
                from_email=Email(FROM_EMAIL),
                to_emails=To(NOTIFICATION_EMAIL),
                subject=subject,
                plain_text_content=Content("text/plain", text_content)
            )
            
            sg = SendGridAPIClient(SENDGRID_API_KEY)
            response = sg.send(message)
            logger.info(f"Trade entry notification sent. Status code: {response.status_code}")
            return True
        except Exception as e:
            logger.error(f"Failed to send trade entry email: {e}")
            logger.info("Falling back to file-based notification...")
    else:
        logger.info("Email notifications disabled, using file-based notification...")
    
    # Fallback to file logging
    return log_notification_to_file(subject, text_content)


def send_trade_exit_notification(
    trading_pair, exit_price, entry_price, quantity, 
    trade_profit_usd, trade_profit_percent, 
    portfolio_value, portfolio_profit_usd, portfolio_profit_percent, 
    total_trades):
    """
    Send notification when a trade is exited
    
    Args:
        trading_pair (str): Trading pair (e.g., SOL/USD)
        exit_price (float): Exit price
        entry_price (float): Original entry price
        quantity (float): Quantity traded
        trade_profit_usd (float): Profit/loss in USD for this trade
        trade_profit_percent (float): Profit/loss as percentage for this trade
        portfolio_value (float): Current portfolio value
        portfolio_profit_usd (float): Total portfolio profit/loss in USD
        portfolio_profit_percent (float): Total portfolio profit/loss as percentage
        total_trades (int): Total number of trades executed
    """
    # Set subject with emoji based on profit/loss
    emoji = "🔴" if trade_profit_usd < 0 else "💰"
    subject = f"{emoji} TRADE EXITED: {trading_pair}"
    
    # Format the message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    position_value = exit_price * quantity
    
    text_content = (
        f"TRADE EXITED: {trading_pair}\n"
        f"Time: {timestamp}\n"
        f"Exit Price: ${exit_price:.2f}\n"
        f"Entry Price: ${entry_price:.2f}\n"
        f"Quantity: {quantity}\n\n"
        f"TRADE PROFIT/LOSS:\n"
        f"${trade_profit_usd:.2f} ({trade_profit_percent:.2f}%)\n\n"
        f"PORTFOLIO STATUS:\n"
        f"Value: ${portfolio_value:.2f}\n"
        f"Total P/L: ${portfolio_profit_usd:.2f} ({portfolio_profit_percent:.2f}%)\n"
        f"Total Trades: {total_trades}"
    )
    
    # Try to send email if available
    if email_available:
        try:
            # Create the email message
            message = Mail(
                from_email=Email(FROM_EMAIL),
                to_emails=To(NOTIFICATION_EMAIL),
                subject=subject,
                plain_text_content=Content("text/plain", text_content)
            )
            
            sg = SendGridAPIClient(SENDGRID_API_KEY)
            response = sg.send(message)
            logger.info(f"Trade exit notification sent. Status code: {response.status_code}")
            return True
        except Exception as e:
            logger.error(f"Failed to send trade exit email: {e}")
            logger.info("Falling back to file-based notification...")
    else:
        logger.info("Email notifications disabled, using file-based notification...")
    
    # Fallback to file logging
    return log_notification_to_file(subject, text_content)