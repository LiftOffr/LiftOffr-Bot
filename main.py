import logging
import argparse
import os
import time
import sys
from flask import Flask
from threading import Thread

# Configure logging with a cleaner format
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to reduce noise in logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Filter out excessive websocket logs and specific error messages
class LogFilter(logging.Filter):
    def filter(self, record):
        # Filter out verbose websocket connection messages
        if "WebSocket connection established" in record.getMessage():
            return False
            
        # Filter out repeated subscription errors
        if "Unsupported event for unknown" in record.getMessage():
            return False
            
        # Allow important messages to pass through regardless of level
        if any(tag in record.getMessage() for tag in ["【TICKER】", "【MARKET】", "【ACCOUNT】", "【SIGNALS】", 
                                                    "【ANALYSIS】", "【CONDITIONS】", "【SIGNAL】"]):
            return True
            
        # Filter out detailed WebSocket data dumps
        if record.levelno <= logging.DEBUG and "Received WebSocket data:" in record.getMessage():
            return False
            
        # Filter out ticker processing debug information
        if record.levelno <= logging.DEBUG and ("Processing ticker data for" in record.getMessage() or 
                                               "Ticker data:" in record.getMessage() or
                                               "Received ticker data for" in record.getMessage()):
            return False
            
        # Filter out OHLC and trade processing debug info
        if record.levelno <= logging.DEBUG and ("Processing ohlc" in record.getMessage() or 
                                               "Processing trade data for" in record.getMessage()):
            return False
            
        return True

# Add filter to root logger
logging.getLogger().addFilter(LogFilter())

# Create logger
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Kraken Trading Bot')
parser.add_argument('--pair', type=str, help='Trading pair (e.g. XBTUSD)')
parser.add_argument('--quantity', type=float, help='Trade quantity')
parser.add_argument('--strategy', type=str, help='Trading strategy (simple_moving_average, rsi, adaptive)')
parser.add_argument('--sandbox', action='store_true', help='Run in sandbox/test mode')
parser.add_argument('--capital', type=float, help='Initial capital')
parser.add_argument('--leverage', type=int, help='Leverage')
parser.add_argument('--margin', type=float, help='Margin percent')
parser.add_argument('--web', action='store_true', help='Start web interface')
parser.add_argument('--live', action='store_true', help='Run in live trading mode (disables sandbox)')

args = parser.parse_args()
print(f"Command line arguments: sandbox={args.sandbox}, live={args.live}")

# Set environment variables based on command line before importing config
if args.live:
    os.environ['USE_SANDBOX'] = 'False'
    os.environ['FLASK_DEBUG'] = 'False'  
    logger.info("Running in live mode (--live flag)")
elif args.sandbox:
    os.environ['USE_SANDBOX'] = 'True'
    logger.info("Running in sandbox mode (--sandbox flag)")

# Now import configuration and other modules
from kraken_trading_bot import KrakenTradingBot
from config import (
    TRADING_PAIR, TRADE_QUANTITY, STRATEGY_TYPE, USE_SANDBOX,
    INITIAL_CAPITAL, LEVERAGE, MARGIN_PERCENT
)

# Create Flask app for web interface
app = Flask(__name__)

# Global bot instance that can be accessed from routes
bot_instance = None

@app.route('/')
def index():
    """
    Main page for trading bot web interface
    """
    return """
    <!DOCTYPE html>
    <html data-bs-theme="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Kraken Trading Bot</title>
        <link rel="stylesheet" href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            .container { max-width: 1200px; }
            .stats-card { min-height: 160px; }
            .chart-container { height: 400px; }
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <div class="row mb-4">
                <div class="col">
                    <h1>Kraken Adaptive Trading Bot</h1>
                    <p class="lead">Advanced algorithmic trading system for cryptocurrency markets</p>
                </div>
            </div>
            
            <div class="row mb-4">
                <!-- Market Stats Card -->
                <div class="col-md-4 mb-3">
                    <div class="card stats-card">
                        <div class="card-header">Market Data</div>
                        <div class="card-body">
                            <h5 class="card-title" id="current-pair">Trading Pair: <span id="trading-pair">--</span></h5>
                            <p class="card-text">Current Price: <span id="current-price">--</span></p>
                            <p class="card-text">24h Change: <span id="price-change">--</span></p>
                            <p class="card-text">24h Volume: <span id="volume">--</span></p>
                        </div>
                    </div>
                </div>
                
                <!-- Bot Stats Card -->
                <div class="col-md-4 mb-3">
                    <div class="card stats-card">
                        <div class="card-header">Bot Status</div>
                        <div class="card-body">
                            <h5 class="card-title">Strategy: <span id="strategy-type">--</span></h5>
                            <p class="card-text">Position: <span id="position-status">--</span></p>
                            <p class="card-text">Entry Price: <span id="entry-price">--</span></p>
                            <p class="card-text">Running Mode: <span id="running-mode">--</span></p>
                        </div>
                    </div>
                </div>
                
                <!-- Performance Card -->
                <div class="col-md-4 mb-3">
                    <div class="card stats-card">
                        <div class="card-header">Performance</div>
                        <div class="card-body">
                            <h5 class="card-title">Portfolio: <span id="portfolio-value">--</span></h5>
                            <p class="card-text">Total Profit: <span id="total-profit">--</span></p>
                            <p class="card-text">ROI: <span id="roi-percent">--</span></p>
                            <p class="card-text">Completed Trades: <span id="trade-count">--</span></p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Price Chart -->
            <div class="row mb-4">
                <div class="col">
                    <div class="card">
                        <div class="card-header">Price Chart</div>
                        <div class="card-body chart-container">
                            <canvas id="price-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Indicators -->
            <div class="row mb-4">
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header">Technical Indicators</div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-sm">
                                    <thead>
                                        <tr>
                                            <th>Indicator</th>
                                            <th>Value</th>
                                            <th>Signal</th>
                                        </tr>
                                    </thead>
                                    <tbody id="indicators-table">
                                        <tr><td>EMA (9/21)</td><td>--</td><td>--</td></tr>
                                        <tr><td>RSI (14)</td><td>--</td><td>--</td></tr>
                                        <tr><td>MACD</td><td>--</td><td>--</td></tr>
                                        <tr><td>ATR</td><td>--</td><td>--</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Recent Trades -->
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header">Recent Trades</div>
                        <div class="card-body">
                            <div class="table-responsive">
                                <table class="table table-sm">
                                    <thead>
                                        <tr>
                                            <th>Time</th>
                                            <th>Type</th>
                                            <th>Price</th>
                                            <th>Profit</th>
                                        </tr>
                                    </thead>
                                    <tbody id="trades-table">
                                        <tr><td colspan="4" class="text-center">No trades yet</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Dashboard JavaScript will go here in the future
            // For now, we'll keep it simple
            document.addEventListener('DOMContentLoaded', function() {
                // Set initial values
                document.getElementById('trading-pair').innerText = 'Loading...';
                document.getElementById('strategy-type').innerText = 'Loading...';
                document.getElementById('running-mode').innerText = 'Loading...';
                
                // Example price chart (would be updated via API)
                const ctx = document.getElementById('price-chart').getContext('2d');
                const priceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: ['Loading...'],
                        datasets: [{
                            label: 'Price',
                            data: [0],
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 2,
                            tension: 0.1,
                            fill: false
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: false
                            }
                        }
                    }
                });
                
                // In the future, we'll add WebSocket or polling to update data
            });
        </script>
    </body>
    </html>
    """

@app.route('/api/status')
def api_status():
    """
    API endpoint for getting bot status
    """
    if bot_instance:
        return {
            'running': bot_instance.running,
            'trading_pair': bot_instance.trading_pair,
            'position': bot_instance.position,
            'entry_price': bot_instance.entry_price,
            'current_price': bot_instance.current_price,
            'portfolio_value': bot_instance.portfolio_value,
            'total_profit': bot_instance.total_profit,
            'total_profit_percent': bot_instance.total_profit_percent,
            'trade_count': bot_instance.trade_count,
            'strategy_type': bot_instance.strategy.__class__.__name__
        }
    else:
        return {'error': 'Bot not initialized'}

def main():
    """
    Main entry point for the Kraken Trading Bot
    """
    global bot_instance
    
    # Command line arguments have already been parsed at the top level
    
    # Get parameters from arguments or environment variables
    trading_pair = args.pair or TRADING_PAIR
    trade_quantity = args.quantity or TRADE_QUANTITY
    strategy_type = args.strategy or STRATEGY_TYPE
    
    # Load API keys
    from config import API_KEY, API_SECRET
    api_key_present = API_KEY and len(API_KEY) > 10
    api_secret_present = API_SECRET and len(API_SECRET) > 10
    
    # Set the sandbox mode based on arguments and API keys
    if args.sandbox:
        sandbox_mode = True
        os.environ['USE_SANDBOX'] = 'True'
        logger.info("Running in sandbox mode (--sandbox flag)")
    elif args.live:
        sandbox_mode = False
        os.environ['USE_SANDBOX'] = 'False'
        logger.info("Running in live mode (--live flag)")
    elif api_key_present and api_secret_present:
        sandbox_mode = False
        os.environ['USE_SANDBOX'] = 'False'
        logger.info("API keys found, running in live mode with real trading")
    else:
        sandbox_mode = True
        os.environ['USE_SANDBOX'] = 'True'
        logger.info("No valid API keys or live flag, defaulting to sandbox mode")
    
    # Override environment variables for later use
    if args.pair:
        os.environ['TRADING_PAIR'] = args.pair
    if args.quantity:
        os.environ['TRADE_QUANTITY'] = str(args.quantity)
    if args.strategy:
        os.environ['STRATEGY_TYPE'] = args.strategy
    if args.sandbox:
        os.environ['USE_SANDBOX'] = 'True'
    if args.capital:
        os.environ['INITIAL_CAPITAL'] = str(args.capital)
    if args.leverage:
        os.environ['LEVERAGE'] = str(args.leverage)
    if args.margin:
        os.environ['MARGIN_PERCENT'] = str(args.margin)
    
    # Initialize the trading bot
    bot_instance = KrakenTradingBot(
        trading_pair=trading_pair,
        trade_quantity=trade_quantity,
        strategy_type=strategy_type
    )
    
    # Start web interface if requested
    if args.web:
        # Start the trading bot in a separate thread
        bot_thread = Thread(target=bot_instance.run)
        bot_thread.daemon = True
        bot_thread.start()
        
        # Start the Flask web server
        app.run(host='0.0.0.0', port=5000)
    else:
        # Just run the trading bot
        bot_instance.run()

if __name__ == '__main__':
    main()
