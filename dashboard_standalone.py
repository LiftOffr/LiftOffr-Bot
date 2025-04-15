#!/usr/bin/env python3
"""
Standalone Dashboard Application

This file provides a completely standalone dashboard with no dependencies
on other components of the trading system.
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
app.secret_key = "trading_bot_dashboard_secret"

def load_file(filepath, default=None):
    """Load a JSON file or return default if not found"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        logger.warning(f"File not found: {filepath}, using default")
        return default
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return default

@app.route('/')
def index():
    """Render the dashboard index page"""
    try:
        # Load all data files
        portfolio = load_file(PORTFOLIO_FILE, {"balance": 20000.0, "equity": 20000.0})
        positions = load_file(POSITIONS_FILE, [])
        portfolio_history = load_file(PORTFOLIO_HISTORY_FILE, [])
        trades = load_file(TRADES_FILE, [])
        
        # Ensure positions have required fields
        for position in positions:
            position.setdefault("unrealized_pnl_pct", 0.0)
            position.setdefault("unrealized_pnl_amount", 0.0)
        
        # Generate placeholder risk metrics
        risk_metrics = {
            "win_rate": 0.75,
            "profit_factor": 2.5,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "max_drawdown": 0.15,
            "current_risk_level": "Medium",
            "avg_leverage_used": 50.0,
            "largest_win": 15.0,
            "largest_loss": 5.0,
        }
        
        # Create strategy performance data
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
        total_positions = len(positions)
        for model in models:
            models[model]["contribution"] = (models[model]["trades"] / total_positions) * 100 if total_positions > 0 else 0
            
        for category in categories:
            categories[category]["contribution"] = (categories[category]["trades"] / total_positions) * 100 if total_positions > 0 else 0
        
        strategy_performance = {
            "strategies": models,
            "categories": categories
        }
        
        # Generate placeholder ML metrics
        accuracy_data = {
            "BTC/USD": 0.95, 
            "ETH/USD": 0.94, 
            "SOL/USD": 0.96, 
            "ADA/USD": 0.93,
            "LINK/USD": 0.92,
            "DOT/USD": 0.91,
            "AVAX/USD": 0.93,
            "MATIC/USD": 0.92,
            "UNI/USD": 0.90,
            "ATOM/USD": 0.91
        }
        
        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
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
        return f"<h1>Error loading dashboard</h1><p>{str(e)}</p><pre>{str(sys.exc_info())}</pre>"

@app.route('/api/portfolio')
def api_portfolio():
    """API endpoint for portfolio data"""
    try:
        portfolio = load_file(PORTFOLIO_FILE, {"balance": 20000.0, "equity": 20000.0})
        positions = load_file(POSITIONS_FILE, [])
        portfolio_history = load_file(PORTFOLIO_HISTORY_FILE, [])
        
        return jsonify({
            "portfolio": portfolio,
            "positions": positions,
            "portfolio_history": portfolio_history
        })
    except Exception as e:
        logger.error(f"Error in portfolio API: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    print("Starting standalone dashboard on port 5000...")
    print("NOTE: Access the dashboard at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)