from flask import Flask, jsonify, render_template_string
import logging
import time
import csv
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# HTML template for the main page
HTML_TEMPLATE = """
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
        .position-long { color: #28a745; }
        .position-short { color: #dc3545; }
        .position-none { color: #6c757d; }
        .profit-positive { color: #28a745; }
        .profit-negative { color: #dc3545; }
        .trade-buy { background-color: rgba(40, 167, 69, 0.1); }
        .trade-sell { background-color: rgba(220, 53, 69, 0.1); }
    </style>
</head>
<body>
    <div class="container mt-4">
        <div class="row mb-4">
            <div class="col">
                <h1>Kraken Multi-Strategy Trading Bot</h1>
                <p class="lead">Advanced algorithmic trading system for cryptocurrency markets</p>
            </div>
        </div>
        
        <div class="row mb-4">
            <!-- Portfolio Card -->
            <div class="col-md-4 mb-3">
                <div class="card stats-card">
                    <div class="card-header">Portfolio</div>
                    <div class="card-body">
                        <h5 class="card-title">Balance: <span id="portfolio-value">${{ current_value }}</span></h5>
                        <p class="card-text">Initial Capital: ${{ initial_capital }}</p>
                        <p class="card-text">
                            Total P&L: 
                            <span class="{% if profit_loss|float > 0 %}profit-positive{% elif profit_loss|float < 0 %}profit-negative{% endif %}">
                                ${{ profit_loss }} ({{ profit_loss_percent }}%)
                            </span>
                        </p>
                        <p class="card-text">Available Funds: ${{ available_funds }}</p>
                    </div>
                </div>
            </div>
            
            <!-- Market Data Card -->
            <div class="col-md-4 mb-3">
                <div class="card stats-card">
                    <div class="card-header">Market Data</div>
                    <div class="card-body">
                        <h5 class="card-title">{{ trading_pair }}</h5>
                        <p class="card-text">Current Price: ${{ current_price }}</p>
                        <p class="card-text">ATR: ${{ atr }}</p>
                        <p class="card-text">Forecast: <span class="badge {% if forecast == 'BULLISH' %}bg-success{% elif forecast == 'BEARISH' %}bg-danger{% else %}bg-secondary{% endif %}">{{ forecast }}</span></p>
                    </div>
                </div>
            </div>
            
            <!-- Performance Card -->
            <div class="col-md-4 mb-3">
                <div class="card stats-card">
                    <div class="card-header">Trading Performance</div>
                    <div class="card-body">
                        <h5 class="card-title">Total Trades: {{ total_trades }}</h5>
                        <p class="card-text">Win Rate: {{ win_rate }}% ({{ winning_trades }}/{{ total_trades }})</p>
                        <p class="card-text">Average P&L: ${{ avg_profit_per_trade }}</p>
                        <p class="card-text">Last Update: {{ last_update }}</p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Active Positions -->
        <div class="row mb-4">
            <div class="col">
                <div class="card">
                    <div class="card-header">Active Positions</div>
                    <div class="card-body">
                        {% if active_positions %}
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Strategy</th>
                                        <th>Pair</th>
                                        <th>Position</th>
                                        <th>Entry Price</th>
                                        <th>Current Price</th>
                                        <th>Leverage</th>
                                        <th>Unrealized P&L</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for position in active_positions %}
                                    <tr>
                                        <td>{{ position.strategy }}</td>
                                        <td>{{ position.pair }}</td>
                                        <td class="position-{{ position.type|lower }}">{{ position.type }}</td>
                                        <td>${{ position.entry_price }}</td>
                                        <td>${{ position.current_price }}</td>
                                        <td>{{ position.leverage }}x</td>
                                        <td class="{% if position.unrealized_pnl|float > 0 %}profit-positive{% elif position.unrealized_pnl|float < 0 %}profit-negative{% endif %}">
                                            ${{ position.unrealized_pnl }} ({{ position.unrealized_pnl_percent }}%)
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% else %}
                        <div class="alert alert-secondary" role="alert">
                            No active positions at the moment.
                        </div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Recent Trades -->
        <div class="row mb-4">
            <div class="col">
                <div class="card">
                    <div class="card-header">Recent Trades</div>
                    <div class="card-body">
                        {% if recent_trades %}
                        <div class="table-responsive">
                            <table class="table table-striped">
                                <thead>
                                    <tr>
                                        <th>Date</th>
                                        <th>Strategy</th>
                                        <th>Action</th>
                                        <th>Price</th>
                                        <th>Size</th>
                                        <th>P&L</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for trade in recent_trades %}
                                    <tr class="{% if trade.type == 'BUY' %}trade-buy{% elif trade.type == 'SELL' %}trade-sell{% endif %}">
                                        <td>{{ trade.date }}</td>
                                        <td>{{ trade.strategy }}</td>
                                        <td>{{ trade.type }}</td>
                                        <td>${{ trade.price }}</td>
                                        <td>{{ trade.size }}</td>
                                        <td class="{% if trade.pnl|float > 0 %}profit-positive{% elif trade.pnl|float < 0 %}profit-negative{% endif %}">
                                            ${{ trade.pnl }} ({{ trade.pnl_percent }}%)
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% else %}
                        <div class="alert alert-secondary" role="alert">
                            No trades recorded yet.
                        </div>
                        {% endif %}
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

        <!-- Strategy Performance -->
        <div class="row mb-4">
            <div class="col">
                <div class="card">
                    <div class="card-header">Strategy Performance</div>
                    <div class="card-body">
                        <div class="row">
                            {% for strategy in strategies %}
                            <div class="col-md-6 mb-3">
                                <div class="card">
                                    <div class="card-header">{{ strategy.name }}</div>
                                    <div class="card-body">
                                        <p>Trading Pair: {{ strategy.pair }}</p>
                                        <p>Current Position: <span class="position-{{ strategy.position|lower }}">{{ strategy.position }}</span></p>
                                        <p>P&L: <span class="{% if strategy.pnl|float > 0 %}profit-positive{% elif strategy.pnl|float < 0 %}profit-negative{% endif %}">
                                            ${{ strategy.pnl }} ({{ strategy.pnl_percent }}%)
                                        </span></p>
                                        <p>Win Rate: {{ strategy.win_rate }}%</p>
                                    </div>
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Sample data for the price chart
            const ctx = document.getElementById('price-chart').getContext('2d');
            const priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: {{ chart_labels|safe }},
                    datasets: [{
                        label: '{{ trading_pair }} Price',
                        data: {{ chart_data|safe }},
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1,
                        tension: 0.1
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
            
            // Auto-refresh the page every 30 seconds
            setTimeout(function() {
                window.location.reload();
            }, 30000);
        });
    </script>
</body>
</html>
"""

def parse_trades_csv():
    """Parse trades from trades.csv file"""
    trades = []
    try:
        if os.path.exists('trades.csv'):
            with open('trades.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trades.append(row)
    except Exception as e:
        logger.error(f"Error parsing trades CSV: {e}")
    return trades

def calculate_stats():
    """Calculate trading statistics"""
    trades = parse_trades_csv()
    
    # Default values
    stats = {
        "initial_capital": "20,000.00",
        "current_value": "20,000.00",
        "profit_loss": "0.00",
        "profit_loss_percent": "0.00",
        "available_funds": "20,000.00",
        "trading_pair": "SOL/USD",
        "current_price": "114.05",
        "atr": "0.12",
        "forecast": "NEUTRAL",
        "total_trades": len(trades),
        "winning_trades": 0,
        "win_rate": "0.00",
        "avg_profit_per_trade": "0.00",
        "recent_trades": [],
        "active_positions": [],
        "strategies": [
            {
                "name": "ARIMA Strategy",
                "pair": "SOL/USD",
                "position": "None",
                "pnl": "0.00",
                "pnl_percent": "0.00",
                "win_rate": "0.00"
            },
            {
                "name": "Adaptive Strategy",
                "pair": "SOL/USD",
                "position": "None",
                "pnl": "0.00",
                "pnl_percent": "0.00",
                "win_rate": "0.00"
            }
        ],
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "chart_labels": json.dumps([f"Time {i}" for i in range(1, 21)]),
        "chart_data": json.dumps([114.05 + (i * 0.01) for i in range(20)])
    }
    
    if not trades:
        return stats
    
    # Calculate winning trades
    winning_trades = sum(1 for trade in trades if float(trade.get('profit_usd', 0)) > 0)
    stats["winning_trades"] = winning_trades
    stats["win_rate"] = f"{(winning_trades / len(trades) * 100):.2f}" if trades else "0.00"
    
    # Calculate total profit/loss
    total_profit = sum(float(trade.get('profit_usd', 0)) for trade in trades)
    stats["profit_loss"] = f"{total_profit:.2f}"
    
    # Calculate average profit per trade
    stats["avg_profit_per_trade"] = f"{(total_profit / len(trades)):.2f}" if trades else "0.00"
    
    # Current portfolio value
    stats["current_value"] = f"{(20000 + total_profit):.2f}"
    stats["profit_loss_percent"] = f"{(total_profit / 20000 * 100):.2f}"
    
    # Format recent trades for display (last 5)
    recent_trades = []
    for trade in trades[-5:]:
        pnl = float(trade.get('profit_usd', 0))
        pnl_percent = float(trade.get('profit_percent', 0))
        recent_trades.append({
            "date": trade.get('timestamp', ''),
            "strategy": trade.get('strategy', 'ARIMA'),
            "type": trade.get('action', ''),
            "price": trade.get('price', '0.00'),
            "size": trade.get('quantity', '0.00'),
            "pnl": f"{pnl:.2f}",
            "pnl_percent": f"{pnl_percent:.2f}"
        })
    stats["recent_trades"] = list(reversed(recent_trades))
    
    # Try to get current price and position
    try:
        # This is a simple proxy for the last price
        if trades:
            stats["current_price"] = trades[-1].get('price', '114.05')
            
            # Set an example active position
            if trades[-1].get('action', '') == 'BUY':
                stats["active_positions"] = [{
                    "strategy": "ARIMA Strategy",
                    "pair": "SOL/USD",
                    "type": "LONG",
                    "entry_price": trades[-1].get('price', '0.00'),
                    "current_price": stats["current_price"],
                    "leverage": "30",
                    "unrealized_pnl": "0.00",
                    "unrealized_pnl_percent": "0.00"
                }]
                
                # Update strategy status
                stats["strategies"][0]["position"] = "LONG"
            elif trades[-1].get('action', '') == 'SELL':
                stats["active_positions"] = [{
                    "strategy": "ARIMA Strategy",
                    "pair": "SOL/USD",
                    "type": "SHORT",
                    "entry_price": trades[-1].get('price', '0.00'),
                    "current_price": stats["current_price"],
                    "leverage": "30",
                    "unrealized_pnl": "0.00",
                    "unrealized_pnl_percent": "0.00"
                }]
                
                # Update strategy status
                stats["strategies"][0]["position"] = "SHORT"
    except Exception as e:
        logger.error(f"Error getting current price: {e}")
    
    # Generate chart data
    # For simplicity, we're using the last 20 trades to build a price chart
    chart_data = []
    chart_labels = []
    
    if len(trades) >= 20:
        for trade in trades[-20:]:
            chart_data.append(float(trade.get('price', '0.00')))
            timestamp = trade.get('timestamp', '')
            if len(timestamp) > 16:  # Get only time part HH:MM:SS
                timestamp = timestamp[-8:]
            chart_labels.append(timestamp)
    else:
        # Fill with sample data if not enough trades
        base_price = float(stats["current_price"])
        for i in range(20):
            chart_data.append(base_price + (i * 0.01))
            chart_labels.append(f"Time {i}")
    
    stats["chart_data"] = json.dumps(chart_data)
    stats["chart_labels"] = json.dumps(chart_labels)
    
    return stats

@app.route('/')
def index():
    """Main page for trading bot web interface"""
    stats = calculate_stats()
    return render_template_string(HTML_TEMPLATE, **stats)

@app.route('/api/stats')
def api_stats():
    """API endpoint for getting trading statistics"""
    return jsonify(calculate_stats())

@app.route('/api/trades')
def api_trades():
    """API endpoint for getting trade history"""
    trades = parse_trades_csv()
    return jsonify(trades)

if __name__ == '__main__':
    # Run Flask on port 5001 to avoid conflict with the trading bot
    app.run(host='0.0.0.0', port=5001, debug=True)