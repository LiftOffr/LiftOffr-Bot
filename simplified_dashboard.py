#!/usr/bin/env python3
"""
Simplified Dashboard Application

This is a minimal Flask application that serves the dashboard
without any conflicts with other components.
"""
import os
import sys
import json
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request, redirect, url_for

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "trading_bot_secret_key")

def load_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return default
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default

def get_positions():
    """Get current positions with calculated metrics"""
    positions = load_file(POSITIONS_FILE, [])
    
    for position in positions:
        # Ensure all required fields exist
        if "unrealized_pnl_pct" not in position:
            position["unrealized_pnl_pct"] = 0.0
        if "unrealized_pnl_amount" not in position:
            position["unrealized_pnl_amount"] = 0.0
    
    return positions

def get_portfolio_data():
    """Get portfolio data with calculated metrics"""
    portfolio = load_file(PORTFOLIO_FILE, {"balance": 20000.0, "equity": 20000.0})
    positions = get_positions()
    portfolio_history = load_file(PORTFOLIO_HISTORY_FILE, [])
    
    # Calculate total unrealized PnL
    total_unrealized_pnl = 0
    for position in positions:
        if "unrealized_pnl_amount" in position:
            total_unrealized_pnl += position["unrealized_pnl_amount"]
    
    # Add to portfolio if not present
    if "unrealized_pnl_usd" not in portfolio:
        portfolio["unrealized_pnl_usd"] = total_unrealized_pnl
    if "unrealized_pnl_pct" not in portfolio:
        portfolio["unrealized_pnl_pct"] = (total_unrealized_pnl / 20000.0) * 100 if total_unrealized_pnl != 0 else 0.0
    
    # Set other required portfolio fields
    if "open_positions_count" not in portfolio:
        portfolio["open_positions_count"] = len(positions)
    
    return portfolio, positions, portfolio_history

def get_trades():
    """Get completed trades"""
    return load_file(TRADES_FILE, [])

@app.route('/')
def index():
    """Render dashboard index page"""
    try:
        logger.info("Dashboard request received")
        
        # Get portfolio data
        portfolio, positions, portfolio_history = get_portfolio_data()
        
        # Get trades
        trades = get_trades()
        
        # Placeholder for risk metrics
        risk_metrics = {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "current_risk_level": "Medium",
            "avg_leverage_used": 50.0  # Example value
        }
        
        # Placeholder for strategy performance
        strategy_performance = {
            "strategies": {
                "ARIMA": {"win_rate": 0.0, "profit_factor": 0.0, "trades": 0, "contribution": 0.0},
                "Adaptive": {"win_rate": 0.0, "profit_factor": 0.0, "trades": 0, "contribution": 0.0}
            },
            "categories": {
                "those dudes": {"win_rate": 0.0, "profit_factor": 0.0, "trades": 0, "contribution": 0.0},
                "him all along": {"win_rate": 0.0, "profit_factor": 0.0, "trades": 0, "contribution": 0.0}
            }
        }
        
        # Placeholder for ML metrics
        accuracy_data = {"BTC/USD": 0.95, "ETH/USD": 0.94, "SOL/USD": 0.96, "ADA/USD": 0.93}
        
        # If we have actual positions, update strategy performance
        if positions:
            models = {}
            categories = {}
            
            for position in positions:
                model = position.get("model", "Unknown")
                category = position.get("category", "Unknown")
                
                if model not in models:
                    models[model] = {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "contribution": 0.0}
                
                if category not in categories:
                    categories[category] = {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "contribution": 0.0}
                
                models[model]["trades"] += 1
                categories[category]["trades"] += 1
            
            # Calculate contributions
            total_models = len(positions)
            for model in models:
                models[model]["contribution"] = (models[model]["trades"] / total_models) * 100 if total_models > 0 else 0
                
            for category in categories:
                categories[category]["contribution"] = (categories[category]["trades"] / total_models) * 100 if total_models > 0 else 0
            
            strategy_performance["strategies"] = models
            strategy_performance["categories"] = categories
        
        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Render template with all data
        return render_template(
            'index.html',
            portfolio=portfolio,
            positions=positions,
            portfolio_history=portfolio_history,
            trades=trades,
            risk_metrics=risk_metrics,
            strategy_performance=strategy_performance,
            accuracy_data=accuracy_data,
            current_time=current_time
        )
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return f"<h1>Error rendering dashboard</h1><p>{str(e)}</p><pre>{str(sys.exc_info())}</pre>"

@app.route('/refresh')
def refresh_data():
    """Refresh dashboard data"""
    return redirect(url_for('index'))

@app.route('/api/portfolio')
def api_portfolio():
    """API endpoint for portfolio data"""
    portfolio, positions, portfolio_history = get_portfolio_data()
    return jsonify({
        "portfolio": portfolio,
        "positions": positions,
        "portfolio_history": portfolio_history
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting simplified dashboard on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)