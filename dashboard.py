#!/usr/bin/env python3
"""
Real-time dashboard for Kraken Trading Bot

This module provides a web-based dashboard for visualizing
trading signals, performance metrics, and market data.
"""

import os
import sys
import json
import logging
import csv
from datetime import datetime, timedelta
import threading
import time
import argparse
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flask import Flask, render_template, jsonify, request, send_from_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)

# Constants
DEFAULT_PORT = 8050
DATA_REFRESH_INTERVAL = 5  # seconds
SIGNAL_MARKERS = ["【SIGNAL】", "【ACTION】", "【INTEGRATED】"]
STATIC_DIR = 'static'
TEMPLATES_DIR = 'templates'

# Ensure directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, 'js'), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, 'css'), exist_ok=True)

# Global data cache
data_cache = {
    'portfolio': {},
    'signals': [],
    'trades': [],
    'market_data': [],
    'indicators': {}
}

# Initialize Flask app
app = Flask(__name__,
            static_folder=STATIC_DIR,
            template_folder=TEMPLATES_DIR)


class DataManager:
    """Manager for data collection and processing"""
    
    def __init__(self):
        """Initialize the data manager"""
        self.trades_file = 'trades.csv'
        self.log_file = 'integrated_strategy_log.txt'
        self.last_update_time = datetime.now() - timedelta(minutes=10)
        self.running = False
        self.update_thread = None
    
    def start_background_updates(self):
        """Start background data updates"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
            logging.info("Started background data updates")
    
    def stop_background_updates(self):
        """Stop background data updates"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2)
            logging.info("Stopped background data updates")
    
    def _update_loop(self):
        """Background update loop"""
        while self.running:
            try:
                self.update_data()
            except Exception as e:
                logging.error(f"Error updating data: {str(e)}")
            
            # Sleep until next update
            time.sleep(DATA_REFRESH_INTERVAL)
    
    def update_data(self):
        """Update all data"""
        # Update trades
        self._update_trades()
        
        # Update signals
        self._update_signals()
        
        # Update market data
        self._update_market_data()
        
        # Update portfolio
        self._update_portfolio()
        
        # Update indicators
        self._update_indicators()
        
        # Update last update time
        self.last_update_time = datetime.now()
    
    def _update_trades(self):
        """Update trades data"""
        if not os.path.exists(self.trades_file):
            return
        
        try:
            # Check if file has been modified since last update
            mtime = os.path.getmtime(self.trades_file)
            if datetime.fromtimestamp(mtime) <= self.last_update_time:
                return
            
            # Load trades data
            trades = []
            with open(self.trades_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert types
                    for key in ['price', 'amount', 'fee', 'pnl']:
                        if key in row and row[key]:
                            try:
                                row[key] = float(row[key])
                            except:
                                pass
                    
                    # Parse timestamp
                    if 'timestamp' in row:
                        try:
                            row['timestamp'] = datetime.fromisoformat(row['timestamp'])
                        except:
                            pass
                    
                    trades.append(row)
            
            # Update cache
            data_cache['trades'] = trades
            logging.info(f"Updated trades data ({len(trades)} trades)")
        
        except Exception as e:
            logging.error(f"Error updating trades: {str(e)}")
    
    def _update_signals(self):
        """Update signals data"""
        if not os.path.exists(self.log_file):
            return
        
        try:
            # Check if file has been modified since last update
            mtime = os.path.getmtime(self.log_file)
            if datetime.fromtimestamp(mtime) <= self.last_update_time:
                return
            
            # Extract signal data from log
            signals = []
            with open(self.log_file, 'r') as f:
                current_signal = {}
                
                for line in f:
                    # Check if line contains signal information
                    if any(marker in line for marker in SIGNAL_MARKERS):
                        # Extract timestamp
                        try:
                            timestamp_str = line.split('[INFO]')[0].strip()
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                        except:
                            timestamp = None
                        
                        # Extract signal direction
                        if "【SIGNAL】" in line:
                            if "BULLISH" in line:
                                current_signal = {
                                    'timestamp': timestamp,
                                    'direction': 'BUY',
                                    'strength': None
                                }
                            elif "BEARISH" in line:
                                current_signal = {
                                    'timestamp': timestamp,
                                    'direction': 'SELL',
                                    'strength': None
                                }
                            elif "NEUTRAL" in line:
                                current_signal = {
                                    'timestamp': timestamp,
                                    'direction': 'NEUTRAL',
                                    'strength': None
                                }
                        
                        # Extract action taken
                        elif "【ACTION】" in line:
                            if timestamp and current_signal and 'timestamp' in current_signal:
                                if abs((timestamp - current_signal['timestamp']).total_seconds()) < 10:
                                    if "BUY" in line:
                                        current_signal['action'] = 'BUY'
                                    elif "SELL" in line:
                                        current_signal['action'] = 'SELL'
                        
                        # Extract signal strength
                        elif "【INTEGRATED】" in line and "Final Signal Strength" in line:
                            if current_signal:
                                try:
                                    # Extract strength value
                                    strength_text = line.split("Final Signal Strength:")[1].split("(")[0].strip()
                                    current_signal['strength'] = float(strength_text)
                                    
                                    # Add completed signal to list if not already present
                                    if 'action' in current_signal and not any(
                                        s.get('timestamp') == current_signal['timestamp'] 
                                        and s.get('direction') == current_signal['direction']
                                        for s in signals
                                    ):
                                        signals.append(current_signal.copy())
                                    current_signal = {}
                                except:
                                    pass
            
            # Update cache with most recent signals (up to 100)
            data_cache['signals'] = sorted(signals, key=lambda x: x.get('timestamp', datetime.min), reverse=True)[:100]
            logging.info(f"Updated signals data ({len(signals)} signals)")
        
        except Exception as e:
            logging.error(f"Error updating signals: {str(e)}")
    
    def _update_market_data(self):
        """Update market data"""
        try:
            # For now, use data from signals and trades
            # In a full implementation, we would fetch real-time market data from Kraken API
            
            market_data = []
            
            # Extract price data from trades
            for trade in data_cache.get('trades', []):
                if 'timestamp' in trade and 'price' in trade:
                    market_data.append({
                        'timestamp': trade['timestamp'],
                        'price': trade['price']
                    })
            
            # Update cache
            data_cache['market_data'] = sorted(market_data, key=lambda x: x.get('timestamp', datetime.min))
            
            # If we have enough data, log info
            if len(market_data) > 0:
                logging.info(f"Updated market data ({len(market_data)} data points)")
        
        except Exception as e:
            logging.error(f"Error updating market data: {str(e)}")
    
    def _update_portfolio(self):
        """Update portfolio data"""
        try:
            # Extract portfolio data from trades
            trades = data_cache.get('trades', [])
            
            if not trades:
                return
            
            # Calculate portfolio value over time
            initial_value = 10000.0  # Default initial value
            portfolio_values = []
            
            # Group trades by date
            trade_dates = {}
            for trade in trades:
                if 'timestamp' in trade:
                    date_str = trade['timestamp'].strftime('%Y-%m-%d')
                    if date_str not in trade_dates:
                        trade_dates[date_str] = []
                    trade_dates[date_str].append(trade)
            
            # Calculate portfolio value for each day
            current_value = initial_value
            for date_str in sorted(trade_dates.keys()):
                day_trades = trade_dates[date_str]
                day_pnl = sum(trade.get('pnl', 0) for trade in day_trades)
                current_value += day_pnl
                
                portfolio_values.append({
                    'date': datetime.strptime(date_str, '%Y-%m-%d'),
                    'value': current_value
                })
            
            # Calculate overall metrics
            total_pnl = sum(trade.get('pnl', 0) for trade in trades)
            winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
            losing_trades = sum(1 for trade in trades if trade.get('pnl', 0) < 0)
            win_rate = winning_trades / len(trades) if trades else 0
            
            # Update cache
            data_cache['portfolio'] = {
                'initial_value': initial_value,
                'current_value': current_value,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'total_trades': len(trades),
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'values_over_time': portfolio_values
            }
            
            logging.info("Updated portfolio data")
        
        except Exception as e:
            logging.error(f"Error updating portfolio: {str(e)}")
    
    def _update_indicators(self):
        """Update technical indicators"""
        try:
            # Extract indicators from signal logs
            signals = data_cache.get('signals', [])
            
            if not signals:
                return
            
            # Find indicator values in log file
            indicators = {
                'timestamps': [],
                'rsi': [],
                'adx': [],
                'volatility': [],
                'ema9': [],
                'ema21': []
            }
            
            with open(self.log_file, 'r') as f:
                for line in f:
                    if "【BANDS】" in line:
                        try:
                            # Extract timestamp
                            timestamp_str = line.split('[INFO]')[0].strip()
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                            
                            # Extract RSI
                            if "RSI =" in line:
                                rsi_text = line.split("RSI =")[1].split("|")[0].strip()
                                rsi_value = float(rsi_text.split()[0])
                                
                                # Extract ADX
                                adx_text = line.split("ADX =")[1].split("|")[0].strip()
                                adx_value = float(adx_text.split()[0])
                                
                                # Extract Volatility
                                vol_text = line.split("Volatility =")[1].split("(")[0].strip()
                                vol_value = float(vol_text)
                                
                                # Extract EMA values
                                if "EMA9 < EMA21" in line or "EMA9 > EMA21" in line or "EMA9 = EMA21" in line:
                                    ema_text = line.split("(")[1].split(")")[0].strip()
                                    ema_values = ema_text.split("vs")
                                    ema9 = float(ema_values[0].strip())
                                    ema21 = float(ema_values[1].strip())
                                    
                                    # Add to indicators
                                    indicators['timestamps'].append(timestamp)
                                    indicators['rsi'].append(rsi_value)
                                    indicators['adx'].append(adx_value)
                                    indicators['volatility'].append(vol_value)
                                    indicators['ema9'].append(ema9)
                                    indicators['ema21'].append(ema21)
                        except Exception as e:
                            pass
            
            # Update cache
            data_cache['indicators'] = indicators
            
            if indicators['timestamps']:
                logging.info(f"Updated indicators data ({len(indicators['timestamps'])} data points)")
        
        except Exception as e:
            logging.error(f"Error updating indicators: {str(e)}")


# Initialize data manager
data_manager = DataManager()


@app.route('/')
def index():
    """Render dashboard homepage"""
    return render_template('index.html')


@app.route('/api/data')
def get_data():
    """API endpoint for dashboard data"""
    # Update data
    data_manager.update_data()
    
    # Prepare data for frontend
    return jsonify({
        'portfolio': data_cache['portfolio'],
        'signals': [
            {
                'timestamp': s.get('timestamp').isoformat() if s.get('timestamp') else None,
                'direction': s.get('direction'),
                'action': s.get('action'),
                'strength': s.get('strength')
            }
            for s in data_cache['signals']
        ],
        'trades': [
            {
                'timestamp': t.get('timestamp').isoformat() if s.get('timestamp') else None,
                'side': t.get('side'),
                'price': t.get('price'),
                'amount': t.get('amount'),
                'pnl': t.get('pnl')
            }
            for t in data_cache['trades']
        ],
        'market_data': [
            {
                'timestamp': m.get('timestamp').isoformat() if m.get('timestamp') else None,
                'price': m.get('price')
            }
            for m in data_cache['market_data']
        ],
        'indicators': {
            'timestamps': [t.isoformat() for t in data_cache['indicators'].get('timestamps', [])],
            'rsi': data_cache['indicators'].get('rsi', []),
            'adx': data_cache['indicators'].get('adx', []),
            'volatility': data_cache['indicators'].get('volatility', []),
            'ema9': data_cache['indicators'].get('ema9', []),
            'ema21': data_cache['indicators'].get('ema21', [])
        }
    })


def create_dashboard_files():
    """Create necessary dashboard files"""
    # Create index.html
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Bot Dashboard</title>
    <link rel="stylesheet" href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css">
    <link rel="stylesheet" href="/static/css/dashboard.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container-fluid">
        <div class="row mb-4">
            <div class="col-12">
                <h1 class="text-center mt-3">Trading Bot Dashboard</h1>
                <p class="text-center text-muted">Real-time visualization of trading signals and performance</p>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Portfolio Summary</h5>
                    </div>
                    <div class="card-body">
                        <div class="d-flex justify-content-between mb-2">
                            <span>Initial Value:</span>
                            <span id="initial-value">$10,000.00</span>
                        </div>
                        <div class="d-flex justify-content-between mb-2">
                            <span>Current Value:</span>
                            <span id="current-value">$10,000.00</span>
                        </div>
                        <div class="d-flex justify-content-between mb-2">
                            <span>Total P&L:</span>
                            <span id="total-pnl">$0.00</span>
                        </div>
                        <div class="d-flex justify-content-between mb-2">
                            <span>Win Rate:</span>
                            <span id="win-rate">0.00%</span>
                        </div>
                        <div class="d-flex justify-content-between mb-2">
                            <span>Total Trades:</span>
                            <span id="total-trades">0</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-md-9">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Portfolio Performance</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="portfolio-chart" height="200"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Price Chart</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="price-chart" height="250"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Technical Indicators</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="indicators-chart" height="250"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Recent Signals</h5>
                    </div>
                    <div class="card-body">
                        <table class="table table-sm table-striped">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Direction</th>
                                    <th>Action</th>
                                    <th>Strength</th>
                                </tr>
                            </thead>
                            <tbody id="signals-table-body">
                                <!-- Signals will be loaded here -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Recent Trades</h5>
                    </div>
                    <div class="card-body">
                        <table class="table table-sm table-striped">
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Side</th>
                                    <th>Price</th>
                                    <th>Amount</th>
                                    <th>P&L</th>
                                </tr>
                            </thead>
                            <tbody id="trades-table-body">
                                <!-- Trades will be loaded here -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="card-title mb-0">Signal Strength Distribution</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="signal-strength-chart" height="150"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="/static/js/dashboard.js"></script>
</body>
</html>
"""
    
    # Create dashboard.css
    dashboard_css = """/* Custom styles for dashboard */
body {
    background-color: var(--bs-dark);
    color: var(--bs-light);
}

.card {
    background-color: var(--bs-gray-800);
    border: 1px solid var(--bs-gray-700);
    margin-bottom: 1rem;
}

.card-header {
    background-color: var(--bs-gray-900);
    border-bottom: 1px solid var(--bs-gray-700);
}

.table {
    color: var(--bs-light);
}

/* Signal and trade styles */
.signal-buy {
    color: #28a745;
}

.signal-sell {
    color: #dc3545;
}

.signal-neutral {
    color: #ffc107;
}

.pnl-positive {
    color: #28a745;
}

.pnl-negative {
    color: #dc3545;
}
"""
    
    # Create dashboard.js
    dashboard_js = """// Dashboard JavaScript

// Charts
let portfolioChart = null;
let priceChart = null;
let indicatorsChart = null;
let signalStrengthChart = null;

// Data refresh interval (milliseconds)
const REFRESH_INTERVAL = 5000;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function () {
    initCharts();
    updateDashboard();
    
    // Set up periodic updates
    setInterval(updateDashboard, REFRESH_INTERVAL);
});

// Initialize charts
function initCharts() {
    // Portfolio chart
    const portfolioCtx = document.getElementById('portfolio-chart').getContext('2d');
    portfolioChart = new Chart(portfolioCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: '#0dcaf0',
                backgroundColor: 'rgba(13, 202, 240, 0.1)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day'
                    }
                },
                y: {
                    beginAtZero: false
                }
            }
        }
    });
    
    // Price chart
    const priceCtx = document.getElementById('price-chart').getContext('2d');
    priceChart = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Price',
                data: [],
                borderColor: '#6c757d',
                tension: 0.1,
                borderWidth: 2
            }, {
                label: 'EMA9',
                data: [],
                borderColor: '#0d6efd',
                borderWidth: 1,
                pointRadius: 0
            }, {
                label: 'EMA21',
                data: [],
                borderColor: '#fd7e14',
                borderWidth: 1,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'hour'
                    }
                }
            }
        }
    });
    
    // Indicators chart
    const indicatorsCtx = document.getElementById('indicators-chart').getContext('2d');
    indicatorsChart = new Chart(indicatorsCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'RSI',
                data: [],
                borderColor: '#20c997',
                yAxisID: 'y',
                tension: 0.1
            }, {
                label: 'ADX',
                data: [],
                borderColor: '#ffc107',
                yAxisID: 'y',
                tension: 0.1
            }, {
                label: 'Volatility',
                data: [],
                borderColor: '#dc3545',
                yAxisID: 'y1',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'hour'
                    }
                },
                y: {
                    position: 'left',
                    min: 0,
                    max: 100,
                    title: {
                        display: true,
                        text: 'RSI / ADX'
                    }
                },
                y1: {
                    position: 'right',
                    min: 0,
                    title: {
                        display: true,
                        text: 'Volatility'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            }
        }
    });
    
    // Signal strength chart
    const signalStrengthCtx = document.getElementById('signal-strength-chart').getContext('2d');
    signalStrengthChart = new Chart(signalStrengthCtx, {
        type: 'bar',
        data: {
            labels: ['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'],
            datasets: [{
                label: 'Buy Signals',
                data: [0, 0, 0, 0, 0],
                backgroundColor: 'rgba(40, 167, 69, 0.7)'
            }, {
                label: 'Sell Signals',
                data: [0, 0, 0, 0, 0],
                backgroundColor: 'rgba(220, 53, 69, 0.7)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Count'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Signal Strength'
                    }
                }
            }
        }
    });
}

// Update dashboard data
function updateDashboard() {
    fetch('/api/data')
        .then(response => response.json())
        .then(data => {
            updatePortfolioSummary(data.portfolio);
            updatePortfolioChart(data.portfolio);
            updatePriceChart(data.market_data, data.indicators);
            updateIndicatorsChart(data.indicators);
            updateSignalsTable(data.signals);
            updateTradesTable(data.trades);
            updateSignalStrengthChart(data.signals);
        })
        .catch(error => {
            console.error('Error fetching dashboard data:', error);
        });
}

// Update portfolio summary
function updatePortfolioSummary(portfolio) {
    document.getElementById('initial-value').textContent = formatCurrency(portfolio.initial_value || 0);
    document.getElementById('current-value').textContent = formatCurrency(portfolio.current_value || 0);
    
    const pnl = portfolio.total_pnl || 0;
    const pnlElement = document.getElementById('total-pnl');
    pnlElement.textContent = formatCurrency(pnl);
    pnlElement.className = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
    
    document.getElementById('win-rate').textContent = formatPercentage(portfolio.win_rate || 0);
    document.getElementById('total-trades').textContent = portfolio.total_trades || 0;
}

// Update portfolio chart
function updatePortfolioChart(portfolio) {
    if (!portfolio.values_over_time || portfolio.values_over_time.length === 0) {
        return;
    }
    
    const labels = portfolio.values_over_time.map(v => new Date(v.date));
    const values = portfolio.values_over_time.map(v => v.value);
    
    portfolioChart.data.labels = labels;
    portfolioChart.data.datasets[0].data = values;
    portfolioChart.update();
}

// Update price chart
function updatePriceChart(marketData, indicators) {
    if (!marketData || marketData.length === 0) {
        return;
    }
    
    // Sort market data by timestamp
    marketData.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    
    // Get most recent data (last 100 points)
    const recentData = marketData.slice(-100);
    
    const labels = recentData.map(d => new Date(d.timestamp));
    const prices = recentData.map(d => d.price);
    
    priceChart.data.labels = labels;
    priceChart.data.datasets[0].data = prices;
    
    // Add EMA data if available
    if (indicators && indicators.timestamps && indicators.timestamps.length > 0) {
        const ema9Data = indicators.timestamps.map((time, i) => ({
            x: new Date(time),
            y: indicators.ema9[i]
        }));
        
        const ema21Data = indicators.timestamps.map((time, i) => ({
            x: new Date(time),
            y: indicators.ema21[i]
        }));
        
        priceChart.data.datasets[1].data = ema9Data;
        priceChart.data.datasets[2].data = ema21Data;
    }
    
    priceChart.update();
}

// Update indicators chart
function updateIndicatorsChart(indicators) {
    if (!indicators || !indicators.timestamps || indicators.timestamps.length === 0) {
        return;
    }
    
    const labels = indicators.timestamps.map(t => new Date(t));
    
    // RSI data
    const rsiData = indicators.timestamps.map((time, i) => ({
        x: new Date(time),
        y: indicators.rsi[i]
    }));
    
    // ADX data
    const adxData = indicators.timestamps.map((time, i) => ({
        x: new Date(time),
        y: indicators.adx[i]
    }));
    
    // Volatility data
    const volatilityData = indicators.timestamps.map((time, i) => ({
        x: new Date(time),
        y: indicators.volatility[i]
    }));
    
    indicatorsChart.data.labels = labels;
    indicatorsChart.data.datasets[0].data = rsiData;
    indicatorsChart.data.datasets[1].data = adxData;
    indicatorsChart.data.datasets[2].data = volatilityData;
    
    indicatorsChart.update();
}

// Update signals table
function updateSignalsTable(signals) {
    if (!signals || signals.length === 0) {
        return;
    }
    
    const tableBody = document.getElementById('signals-table-body');
    tableBody.innerHTML = '';
    
    // Display most recent 10 signals
    const recentSignals = signals.slice(0, 10);
    
    recentSignals.forEach(signal => {
        const row = document.createElement('tr');
        
        // Time column
        const timeCell = document.createElement('td');
        timeCell.textContent = formatDateTime(signal.timestamp);
        row.appendChild(timeCell);
        
        // Direction column
        const directionCell = document.createElement('td');
        directionCell.textContent = signal.direction || 'N/A';
        directionCell.className = `signal-${signal.direction?.toLowerCase() || 'neutral'}`;
        row.appendChild(directionCell);
        
        // Action column
        const actionCell = document.createElement('td');
        actionCell.textContent = signal.action || 'N/A';
        row.appendChild(actionCell);
        
        // Strength column
        const strengthCell = document.createElement('td');
        strengthCell.textContent = signal.strength?.toFixed(2) || 'N/A';
        row.appendChild(strengthCell);
        
        tableBody.appendChild(row);
    });
}

// Update trades table
function updateTradesTable(trades) {
    if (!trades || trades.length === 0) {
        return;
    }
    
    const tableBody = document.getElementById('trades-table-body');
    tableBody.innerHTML = '';
    
    // Display most recent 10 trades
    const recentTrades = trades.slice(0, 10);
    
    recentTrades.forEach(trade => {
        const row = document.createElement('tr');
        
        // Time column
        const timeCell = document.createElement('td');
        timeCell.textContent = formatDateTime(trade.timestamp);
        row.appendChild(timeCell);
        
        // Side column
        const sideCell = document.createElement('td');
        sideCell.textContent = trade.side || 'N/A';
        sideCell.className = `signal-${trade.side === 'buy' ? 'buy' : 'sell'}`;
        row.appendChild(sideCell);
        
        // Price column
        const priceCell = document.createElement('td');
        priceCell.textContent = formatCurrency(trade.price || 0);
        row.appendChild(priceCell);
        
        // Amount column
        const amountCell = document.createElement('td');
        amountCell.textContent = trade.amount?.toFixed(4) || 'N/A';
        row.appendChild(amountCell);
        
        // PnL column
        const pnlCell = document.createElement('td');
        const pnl = trade.pnl || 0;
        pnlCell.textContent = formatCurrency(pnl);
        pnlCell.className = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
        row.appendChild(pnlCell);
        
        tableBody.appendChild(row);
    });
}

// Update signal strength chart
function updateSignalStrengthChart(signals) {
    if (!signals || signals.length === 0) {
        return;
    }
    
    // Count signals by strength range and direction
    const buyStrengthCounts = [0, 0, 0, 0, 0];
    const sellStrengthCounts = [0, 0, 0, 0, 0];
    
    signals.forEach(signal => {
        if (signal.strength === null || signal.strength === undefined) {
            return;
        }
        
        // Determine strength range index
        let rangeIndex = Math.floor((signal.strength - 0.5) * 10);
        rangeIndex = Math.max(0, Math.min(4, rangeIndex));
        
        // Update counts
        if (signal.direction === 'BUY') {
            buyStrengthCounts[rangeIndex]++;
        } else if (signal.direction === 'SELL') {
            sellStrengthCounts[rangeIndex]++;
        }
    });
    
    signalStrengthChart.data.datasets[0].data = buyStrengthCounts;
    signalStrengthChart.data.datasets[1].data = sellStrengthCounts;
    signalStrengthChart.update();
}

// Helper functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', { 
        style: 'currency', 
        currency: 'USD'
    }).format(value);
}

function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', { 
        style: 'percent', 
        minimumFractionDigits: 2, 
        maximumFractionDigits: 2
    }).format(value);
}

function formatDateTime(timestamp) {
    if (!timestamp) return 'N/A';
    
    const date = new Date(timestamp);
    return date.toLocaleString('en-US', { 
        month: 'short', 
        day: 'numeric', 
        hour: '2-digit', 
        minute: '2-digit'
    });
}
"""
    
    # Write files
    with open(os.path.join(TEMPLATES_DIR, 'index.html'), 'w') as f:
        f.write(index_html)
    
    with open(os.path.join(STATIC_DIR, 'css', 'dashboard.css'), 'w') as f:
        f.write(dashboard_css)
    
    with open(os.path.join(STATIC_DIR, 'js', 'dashboard.js'), 'w') as f:
        f.write(dashboard_js)
    
    logging.info("Created dashboard files")


def main():
    parser = argparse.ArgumentParser(description='Trading Bot Dashboard')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Port to run the dashboard on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Create dashboard files
    create_dashboard_files()
    
    # Start data manager
    data_manager.start_background_updates()
    
    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        # Stop data manager on exit
        data_manager.stop_background_updates()
        logging.info("Dashboard stopped")


if __name__ == '__main__':
    main()