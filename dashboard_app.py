#!/usr/bin/env python3
"""
ML Trading Dashboard

This module implements a web dashboard for monitoring and controlling
the ML-powered trading bot, showing portfolio performance, trade history,
and current predictions.
"""

import json
import logging
import os
from datetime import datetime
from flask import Flask, render_template, jsonify, request, redirect, url_for

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename='logs/dashboard.log'
)

# Ensure directories exist
os.makedirs('logs', exist_ok=True)
os.makedirs('config', exist_ok=True)


@app.route('/')
def dashboard():
    """Render the main dashboard"""
    try:
        # Load portfolio data
        portfolio = load_portfolio()
        
        # Load ML configuration
        ml_config = load_ml_config()
        
        # Load recent trades
        recent_trades = load_recent_trades()
        
        # Load current predictions
        predictions = load_predictions()
        
        return render_template('dashboard.html', 
                              portfolio=portfolio,
                              ml_config=ml_config,
                              recent_trades=recent_trades,
                              predictions=predictions)
    except Exception as e:
        logging.error(f"Error loading dashboard: {e}")
        return render_template('dashboard.html')


@app.route('/api/portfolio')
def api_portfolio():
    """Return portfolio data as JSON"""
    try:
        portfolio = load_portfolio()
        return jsonify(portfolio)
    except Exception as e:
        logging.error(f"Error loading portfolio data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/trades')
def api_trades():
    """Return trade history as JSON"""
    try:
        trades = load_recent_trades()
        return jsonify(trades)
    except Exception as e:
        logging.error(f"Error loading trade data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predictions')
def api_predictions():
    """Return current model predictions as JSON"""
    try:
        predictions = load_predictions()
        return jsonify(predictions)
    except Exception as e:
        logging.error(f"Error loading predictions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/models')
def api_models():
    """Return active model information as JSON"""
    try:
        ml_config = load_ml_config()
        return jsonify(ml_config)
    except Exception as e:
        logging.error(f"Error loading model data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/activate_pair', methods=['POST'])
def activate_pair():
    """Activate or deactivate a trading pair"""
    try:
        pair = request.form.get('pair')
        action = request.form.get('action', 'activate')
        
        if not pair:
            return jsonify({'error': 'No pair specified'}), 400
        
        # Load ML config
        ml_config = load_ml_config()
        
        # Check if pair exists in config
        if pair not in ml_config.get('models', {}):
            return jsonify({'error': f'Pair {pair} not found in configuration'}), 404
        
        # Activate or deactivate pair
        ml_config['models'][pair]['enabled'] = (action == 'activate')
        
        # Save updated config
        save_ml_config(ml_config)
        
        return jsonify({'success': True, 'message': f'Pair {pair} {"activated" if action == "activate" else "deactivated"}'})
    except Exception as e:
        logging.error(f"Error activating pair: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/close_position', methods=['POST'])
def close_position():
    """Close an open trading position"""
    try:
        pair = request.form.get('pair')
        
        if not pair:
            return jsonify({'error': 'No pair specified'}), 400
        
        # In a real implementation, this would call the trading bot API
        # For now, we'll just return success
        return jsonify({'success': True, 'message': f'Position for {pair} closed'})
    except Exception as e:
        logging.error(f"Error closing position: {e}")
        return jsonify({'error': str(e)}), 500


def load_portfolio():
    """Load portfolio data from file"""
    try:
        portfolio_path = 'config/sandbox_portfolio.json'
        if os.path.exists(portfolio_path):
            with open(portfolio_path, 'r') as f:
                return json.load(f)
        return {
            'base_currency': 'USD',
            'starting_capital': 20000.0,
            'current_capital': 20000.0,
            'equity': 20000.0,
            'positions': {},
            'completed_trades': []
        }
    except Exception as e:
        logging.error(f"Error loading portfolio: {e}")
        raise


def load_ml_config():
    """Load ML configuration from file"""
    try:
        config_path = 'config/ml_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {
            'enabled': False,
            'sandbox': True,
            'models': {}
        }
    except Exception as e:
        logging.error(f"Error loading ML config: {e}")
        raise


def save_ml_config(config):
    """Save ML configuration to file"""
    try:
        config_path = 'config/ml_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving ML config: {e}")
        raise


def load_recent_trades():
    """Load recent trade history"""
    try:
        # In a real implementation, this would load from a database
        # For now, just return an empty list
        return []
    except Exception as e:
        logging.error(f"Error loading recent trades: {e}")
        raise


def load_predictions():
    """Load current model predictions"""
    try:
        # In a real implementation, this would load from the trading bot
        # For now, just return an empty dict
        return {}
    except Exception as e:
        logging.error(f"Error loading predictions: {e}")
        raise


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)