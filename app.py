#!/usr/bin/env python3

"""
Trading Bot Dashboard

This Flask application provides a web dashboard for the cryptocurrency
trading bot, displaying model performance, portfolio statistics, and
trading activities.
"""

import os
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
CONFIG_DIR = "config"
ML_CONFIG_FILE = f"{CONFIG_DIR}/ml_config.json"
DATA_DIR = "data"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
PORTFOLIO_HISTORY_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
DEFAULT_PAIRS = ["SOL/USD", "BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]

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

def get_ml_metrics():
    """Get ML metrics from the ML config file"""
    ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
    
    accuracy_data = {}
    for pair, config in ml_config.get("pairs", {}).items():
        accuracy_data[pair] = config.get("accuracy", 0.0)
    
    return accuracy_data

def get_portfolio_data():
    """Get portfolio data"""
    portfolio = load_file(PORTFOLIO_FILE, {"balance": 20000.0, "equity": 20000.0})
    positions = load_file(POSITIONS_FILE, [])
    portfolio_history = load_file(PORTFOLIO_HISTORY_FILE, [])
    
    # Calculate total unrealized PnL
    total_position_value = 0
    total_entry_value = 0
    unrealized_pnl_usd = 0
    
    for position in positions:
        position_size = position.get("position_size", 0.0)
        leverage = position.get("leverage", 1.0)
        entry_price = position.get("entry_price", 0.0)
        current_price = position.get("current_price", entry_price)
        direction = position.get("direction", "").lower()
        
        if direction == "long":
            position_pnl = (current_price - entry_price) / entry_price * leverage * position_size
        else:  # short
            position_pnl = (entry_price - current_price) / entry_price * leverage * position_size
            
        unrealized_pnl_usd += position_pnl
        
        if direction == "long":
            entry_value = position_size
            current_value = position_size * (current_price / entry_price)
        else:  # short
            entry_value = position_size
            current_value = position_size * (entry_price / current_price)
            
        total_entry_value += entry_value
        total_position_value += current_value
    
    # Calculate unrealized PnL percentage
    unrealized_pnl_pct = 0
    if total_entry_value > 0:
        unrealized_pnl_pct = (total_position_value / total_entry_value - 1) * 100
    
    # Add to portfolio
    portfolio["unrealized_pnl_usd"] = unrealized_pnl_usd
    portfolio["unrealized_pnl_pct"] = unrealized_pnl_pct
    portfolio["total_position_value"] = total_position_value
    
    return portfolio, positions, portfolio_history

def get_risk_metrics():
    """Calculate risk metrics from the trading data"""
    portfolio = load_file(PORTFOLIO_FILE, {"balance": 20000.0, "equity": 20000.0})
    positions = load_file(POSITIONS_FILE, [])
    history = load_file(PORTFOLIO_HISTORY_FILE, [])
    trades = portfolio.get("trades", [])
    
    # Default metrics
    risk_metrics = {
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_win_loss_ratio": 0.0,
        "sharpe_ratio": 0.0,
        "sortino_ratio": 0.0,
        "max_drawdown": 0.0,
        "value_at_risk": 0.0,
        "current_risk_level": "Medium",
        "optimal_position_size": "20-25% of balance",
        "optimal_risk_per_trade": "2-3% of balance"
    }
    
    # Calculate win rate
    if trades:
        winning_trades = sum(1 for t in trades if t.get("pnl_amount", 0.0) > 0)
        risk_metrics["win_rate"] = winning_trades / len(trades) if len(trades) > 0 else 0.0
    
    # Calculate profit factor
    total_profit = sum(t.get("pnl_amount", 0.0) for t in trades if t.get("pnl_amount", 0.0) > 0)
    total_loss = abs(sum(t.get("pnl_amount", 0.0) for t in trades if t.get("pnl_amount", 0.0) < 0))
    
    if total_loss > 0:
        risk_metrics["profit_factor"] = total_profit / total_loss
    else:
        risk_metrics["profit_factor"] = total_profit if total_profit > 0 else 1.0
    
    # Calculate average win/loss ratio
    if trades:
        winning_trades_count = sum(1 for t in trades if t.get("pnl_amount", 0.0) > 0)
        losing_trades_count = sum(1 for t in trades if t.get("pnl_amount", 0.0) < 0)
        
        if winning_trades_count > 0 and losing_trades_count > 0:
            avg_win = total_profit / winning_trades_count
            avg_loss = total_loss / losing_trades_count
            
            if avg_loss > 0:
                risk_metrics["avg_win_loss_ratio"] = avg_win / avg_loss
    
    # Set some reasonable defaults for other metrics based on model performance
    ml_config = load_file(ML_CONFIG_FILE, {"pairs": {}})
    avg_accuracy = 0.0
    avg_backtest_return = 0.0
    avg_leverage = 0.0
    count = 0
    
    for pair, config in ml_config.get("pairs", {}).items():
        avg_accuracy += config.get("accuracy", 0.0)
        avg_backtest_return += config.get("backtest_return", 0.0)
        avg_leverage += config.get("base_leverage", 20.0)
        count += 1
    
    if count > 0:
        avg_accuracy /= count
        avg_backtest_return /= count
        avg_leverage /= count
        
        # Estimate Sharpe and other metrics based on model performance
        risk_metrics["sharpe_ratio"] = max(0.5, min(5.0, avg_accuracy * 5.0))
        risk_metrics["max_drawdown"] = max(0.01, min(0.2, (1 - avg_accuracy) * 0.5))
        risk_metrics["value_at_risk"] = max(0.005, min(0.1, (1 - avg_accuracy) * 0.25))
        
        # Determine risk level based on model performance
        if avg_accuracy >= 0.97:
            risk_metrics["current_risk_level"] = "Low"
        elif avg_accuracy >= 0.94:
            risk_metrics["current_risk_level"] = "Medium"
        else:
            risk_metrics["current_risk_level"] = "High"
    
    return risk_metrics

def process_portfolio_history(history):
    """Process portfolio history for charting"""
    # Ensure we have at least two data points
    if len(history) < 2:
        now = datetime.now().isoformat()
        
        if len(history) == 0:
            # No history, create two points
            history = [
                {
                    "timestamp": (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).isoformat(),
                    "portfolio_value": 20000.0,
                    "balance": 20000.0,
                    "equity": 20000.0,
                    "num_positions": 0
                },
                {
                    "timestamp": now,
                    "portfolio_value": 20000.0,
                    "balance": 20000.0,
                    "equity": 20000.0,
                    "num_positions": 0
                }
            ]
        else:
            # Only one point, add another
            history.append({
                "timestamp": now,
                "portfolio_value": history[0].get("portfolio_value", 20000.0),
                "balance": history[0].get("balance", 20000.0),
                "equity": history[0].get("equity", 20000.0),
                "num_positions": history[0].get("num_positions", 0)
            })
    
    return history

def format_positions(positions):
    """Format position data for display"""
    for position in positions:
        # Calculate unrealized PnL if not set
        if "unrealized_pnl" not in position:
            entry_price = position.get("entry_price", 0.0)
            current_price = position.get("current_price", entry_price)
            leverage = position.get("leverage", 1.0)
            
            if position.get("direction", "").lower() == "long":
                price_change_pct = (current_price / entry_price) - 1 if entry_price > 0 else 0
            else:
                price_change_pct = (entry_price / current_price) - 1 if current_price > 0 else 0
            
            position["unrealized_pnl"] = price_change_pct * leverage
        
        # Calculate duration if not set
        if "duration" not in position:
            entry_time = position.get("entry_time", "")
            if entry_time:
                try:
                    entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                    now = datetime.now()
                    delta = now - entry_dt
                    hours, remainder = divmod(delta.seconds, 3600)
                    minutes, _ = divmod(remainder, 60)
                    position["duration"] = f"{delta.days}d {hours}h {minutes}m"
                except Exception:
                    position["duration"] = "Unknown"
            else:
                position["duration"] = "Unknown"
    
    return positions

def format_trades(trades):
    """Format trade data for display"""
    formatted_trades = []
    
    for trade in trades:
        try:
            # Calculate duration
            entry_time = trade.get("entry_time", "")
            exit_time = trade.get("exit_time", "")
            
            if entry_time and exit_time:
                entry_dt = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                exit_dt = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
                delta = exit_dt - entry_dt
                hours, remainder = divmod(delta.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                duration = f"{delta.days}d {hours}h {minutes}m"
            else:
                duration = "Unknown"
            
            # Create formatted trade
            formatted_trade = {
                "pair": trade.get("pair", "Unknown"),
                "direction": trade.get("direction", "Unknown"),
                "entry_price": trade.get("entry_price", 0.0),
                "exit_price": trade.get("exit_price", 0.0),
                "pnl_percentage": trade.get("pnl_percentage", 0.0) * 100,  # Convert to percentage
                "pnl_amount": trade.get("pnl_amount", 0.0),
                "exit_reason": trade.get("exit_reason", "Unknown"),
                "duration": duration,
                "leverage": trade.get("leverage", 1.0),
                "confidence": trade.get("confidence", 0.0)
            }
            
            formatted_trades.append(formatted_trade)
        except Exception as e:
            logger.error(f"Error formatting trade: {e}")
    
    return formatted_trades

@app.route('/')
def index():
    """Render the dashboard index page"""
    # Get ML metrics
    accuracy_data = get_ml_metrics()
    
    # Get portfolio data
    portfolio, positions, portfolio_history = get_portfolio_data()
    
    # Process portfolio history
    portfolio_history = process_portfolio_history(portfolio_history)
    
    # Format positions
    positions = format_positions(positions)
    
    # Get trades
    trades = portfolio.get("trades", [])
    formatted_trades = format_trades(trades)
    
    # Get risk metrics
    risk_metrics = get_risk_metrics()
    
    # Get current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return render_template(
        'index.html',
        accuracy_data=accuracy_data,
        portfolio=portfolio,
        positions=positions,
        portfolio_history=portfolio_history,
        trades=formatted_trades,
        risk_metrics=risk_metrics,
        current_time=current_time
    )

@app.route('/refresh')
def refresh_data():
    """Refresh dashboard data"""
    # We'll just redirect to the index page
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

@app.route('/api/models')
def api_models():
    """API endpoint for ML model metrics"""
    accuracy_data = get_ml_metrics()
    
    return jsonify(accuracy_data)

@app.route('/api/risk')
def api_risk():
    """API endpoint for risk metrics"""
    risk_metrics = get_risk_metrics()
    
    return jsonify(risk_metrics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)