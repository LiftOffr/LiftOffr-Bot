"""
Simple web server script for the Kraken trading bot.
This file is used to start the web server without any additional dependencies.
"""
import os
import logging
from flask import Flask, render_template, jsonify, request, redirect, url_for
import os
import datetime
import pandas as pd
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "kraken_trading_bot_secret")

# Default portfolio value used in sandbox mode
DEFAULT_PORTFOLIO_VALUE = 20000.00
TRADES_CSV = 'trades.csv'

@app.route('/')
def index():
    """Main page for trading bot web interface"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """API endpoint for getting bot status"""
    try:
        # For demo purposes, use a default portfolio value in sandbox mode
        portfolio_value = DEFAULT_PORTFOLIO_VALUE
        
        # Demo metrics
        metrics = {
            "status": "running",
            "strategy": "ARIMA-based trading",
            "trading_pair": "SOL/USD",
            "portfolio_value": portfolio_value,
            "initial_value": 20000.00,
            "gain_loss_pct": ((portfolio_value / 20000.00) - 1) * 100,
            "position": "short",
            "entry_price": 112.69,
            "current_price": 110.00,
            "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error in API status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/trades')
def get_trades():
    """API endpoint for getting trade history"""
    try:
        # Check if trades CSV exists, if not return empty list
        if not os.path.exists(TRADES_CSV):
            return jsonify([])
            
        # Read trades from CSV
        trades_df = pd.read_csv(TRADES_CSV)
        trades = trades_df.to_dict('records')
        return jsonify(trades)
    except Exception as e:
        logger.error(f"Error getting trades: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolio/current')
def get_current_position():
    """API endpoint for getting current position"""
    try:
        # For demo purposes, use a fixed position
        position = {
            "symbol": "SOL/USD",
            "position_type": "short",
            "entry_price": 112.69,
            "current_price": 110.00,
            "quantity": 0.001,
            "pnl": 2.69,
            "pnl_percent": 2.39,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return jsonify(position)
    except Exception as e:
        logger.error(f"Error getting current position: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolio/history')
def get_portfolio_history():
    """API endpoint for getting historical portfolio values"""
    try:
        # For demo purposes, generate some sample historical data
        start_date = datetime.datetime.now() - datetime.timedelta(days=7)
        history = []
        
        # Generate sample data for the past 7 days
        start_value = 20000.00
        current_value = start_value
        for i in range(7):
            date = (start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            # Random daily change between -2% and +2%
            change = (110 - i/7) / 100  # Starts favorable, gets less favorable
            current_value = current_value * change
            history.append({
                "date": date,
                "value": current_value,
                "change_percent": (change - 1) * 100
            })
            
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error getting portfolio history: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting web server for Kraken trading bot")
    logger.info("Web interface available at http://localhost:5001")
    
    # Run Flask app on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)