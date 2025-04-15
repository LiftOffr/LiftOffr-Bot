#!/usr/bin/env python3

"""
Update Dashboard

This script updates the Flask web application with the latest model metrics
and trading performance data from the sandbox trader.

Usage:
    python update_dashboard.py
"""

import os
import sys
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
        "current_risk_level": "Medium"
    }
    
    # Calculate win rate
    if trades:
        winning_trades = sum(1 for t in trades if t.get("pnl_amount", 0.0) > 0)
        risk_metrics["win_rate"] = winning_trades / len(trades)
    
    # Calculate profit factor
    total_profit = sum(t.get("pnl_amount", 0.0) for t in trades if t.get("pnl_amount", 0.0) > 0)
    total_loss = abs(sum(t.get("pnl_amount", 0.0) for t in trades if t.get("pnl_amount", 0.0) < 0))
    
    if total_loss > 0:
        risk_metrics["profit_factor"] = total_profit / total_loss
    else:
        risk_metrics["profit_factor"] = total_profit if total_profit > 0 else 1.0
    
    # Calculate average win/loss ratio
    if trades:
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / (len(trades) - winning_trades) if len(trades) - winning_trades > 0 else 1
        
        if avg_loss > 0:
            risk_metrics["avg_win_loss_ratio"] = avg_win / avg_loss
    
    # Calculate Sharpe ratio from portfolio history
    if len(history) > 1:
        returns = []
        for i in range(1, len(history)):
            prev_value = history[i-1].get("portfolio_value", 20000.0)
            curr_value = history[i].get("portfolio_value", 20000.0)
            
            if prev_value > 0:
                returns.append(curr_value / prev_value - 1)
        
        if returns:
            import numpy as np
            avg_return = sum(returns) / len(returns)
            std_return = np.std(returns) if len(returns) > 1 else 0.01
            
            if std_return > 0:
                risk_metrics["sharpe_ratio"] = avg_return / std_return * (252 ** 0.5)  # Annualized
    
    # Calculate Sortino ratio (similar to Sharpe but only considers downside risk)
    if len(history) > 1:
        neg_returns = [r for r in returns if r < 0]
        
        if neg_returns:
            downside_std = np.std(neg_returns)
            
            if downside_std > 0:
                risk_metrics["sortino_ratio"] = avg_return / downside_std * (252 ** 0.5)
    
    # Calculate max drawdown
    if len(history) > 1:
        peak = history[0].get("portfolio_value", 20000.0)
        max_dd = 0.0
        
        for h in history[1:]:
            curr_value = h.get("portfolio_value", 20000.0)
            
            if curr_value > peak:
                peak = curr_value
            else:
                dd = (peak - curr_value) / peak
                max_dd = max(max_dd, dd)
        
        risk_metrics["max_drawdown"] = max_dd
    
    # Calculate Value at Risk (VaR)
    if len(returns) > 1:
        returns.sort()
        var_95 = returns[int(len(returns) * 0.05)]  # 95% VaR
        risk_metrics["value_at_risk"] = abs(var_95)
    
    # Determine risk level
    if risk_metrics["max_drawdown"] < 0.05 and risk_metrics["sharpe_ratio"] > 2.0:
        risk_metrics["current_risk_level"] = "Low"
    elif risk_metrics["max_drawdown"] > 0.15 or risk_metrics["sharpe_ratio"] < 0.5:
        risk_metrics["current_risk_level"] = "High"
    else:
        risk_metrics["current_risk_level"] = "Medium"
    
    return risk_metrics

def update_dashboard():
    """Update the dashboard with the latest data"""
    # Get ML metrics
    accuracy_data = get_ml_metrics()
    
    # Get portfolio data
    portfolio, positions, portfolio_history = get_portfolio_data()
    
    # Get risk metrics
    risk_metrics = get_risk_metrics()
    
    # Get the trades
    trades = portfolio.get("trades", [])
    
    logger.info(f"Updated dashboard data: {len(positions)} positions, {len(trades)} trades")
    
    return {
        "accuracy_data": accuracy_data,
        "portfolio": portfolio,
        "positions": positions,
        "portfolio_history": portfolio_history,
        "risk_metrics": risk_metrics,
        "trades": trades
    }

def main():
    """Main function"""
    logger.info("Updating dashboard data")
    
    # Update dashboard data
    data = update_dashboard()
    
    # Print summary
    logger.info(f"Portfolio balance: ${data['portfolio'].get('balance', 0.0):.2f}")
    logger.info(f"Portfolio equity: ${data['portfolio'].get('equity', 0.0):.2f}")
    logger.info(f"Open positions: {len(data['positions'])}")
    logger.info(f"Completed trades: {len(data['trades'])}")
    
    # Print accuracy data
    logger.info("\nML Model Accuracy:")
    for pair, accuracy in data["accuracy_data"].items():
        logger.info(f"  {pair}: {accuracy*100:.2f}%")
    
    # Print risk metrics
    logger.info("\nRisk Metrics:")
    for key, value in data["risk_metrics"].items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())