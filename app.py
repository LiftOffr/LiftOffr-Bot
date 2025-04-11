import os
import json
import time
from datetime import datetime, timedelta

from flask import Flask, render_template, jsonify, request
from models import db, Trade, PortfolioSnapshot
from sqlalchemy import desc
import pandas as pd
import csv

# Create the flask app
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.secret_key = os.environ.get("SESSION_SECRET", "dev_key_please_change")

# Initialize the database
db.init_app(app)

# Create all tables
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    """Main page for trading bot web interface"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """API endpoint for getting bot status"""
    # First try to get the most recent portfolio snapshot from our database
    with app.app_context():
        latest_snapshot = PortfolioSnapshot.query.order_by(desc(PortfolioSnapshot.timestamp)).first()
        
        if latest_snapshot and (datetime.utcnow() - latest_snapshot.timestamp).total_seconds() < 60:
            # We have recent data in the database
            data = {
                'value': latest_snapshot.total_value,
                'profit': latest_snapshot.total_profit,
                'profit_percent': latest_snapshot.total_profit_percent,
                'trade_count': latest_snapshot.trade_count,
                'position': latest_snapshot.current_position,
                'entry_price': latest_snapshot.entry_price,
                'current_price': latest_snapshot.current_price,
                'data_source': 'database',
                'timestamp': latest_snapshot.timestamp.isoformat()
            }
            return jsonify(data)
    
    # If we don't have database data, try to get the data from the bot directly
    try:
        # Check if there is a recent portfolio value response
        response_file = "portfolio_value.txt"
        if os.path.exists(response_file):
            file_mtime = os.path.getmtime(response_file)
            if time.time() - file_mtime < 60:  # Only use if less than 60 seconds old
                with open(response_file, "r") as f:
                    try:
                        data = json.loads(f.read())
                        data['data_source'] = 'file'
                        return jsonify(data)
                    except json.JSONDecodeError:
                        pass
        
        # Create a new request for portfolio data
        request_file = "portfolio_request.txt"
        with open(request_file, "w") as f:
            f.write(str(time.time()))
        
        # Wait briefly for a response
        time.sleep(2)
        
        # Check if we got a response
        if os.path.exists(response_file):
            file_mtime = os.path.getmtime(response_file)
            if time.time() - file_mtime < 3:  # Only use if it was just updated
                with open(response_file, "r") as f:
                    try:
                        data = json.loads(f.read())
                        data['data_source'] = 'fresh'
                        return jsonify(data)
                    except json.JSONDecodeError:
                        pass
        
        # Fallback to checking the trades.csv file for portfolio value
        # This is a legacy approach and may be less accurate
        try:
            # Try to get portfolio data from trades CSV
            with open('trades.csv', 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)
                
                if rows:
                    # Get the last trade which should have final portfolio value
                    latest_trade = rows[-1]
                    if 'portfolio_value' in latest_trade:
                        portfolio_value = float(latest_trade['portfolio_value'])
                        # Estimate profit based on initial capital of $20,000
                        profit = portfolio_value - 20000
                        profit_percent = (profit / 20000) * 100
                        
                        data = {
                            'value': portfolio_value,
                            'profit': profit,
                            'profit_percent': profit_percent,
                            'trade_count': len(rows),
                            'position': None,
                            'entry_price': 0,
                            'current_price': 0,
                            'data_source': 'csv',
                            'timestamp': datetime.utcnow().isoformat()
                        }
                        return jsonify(data)
        except Exception as e:
            print(f"Error reading trades.csv: {e}")
    
    except Exception as e:
        print(f"Error retrieving portfolio data: {e}")
    
    # Default response if all else fails
    return jsonify({
        'value': 20000.00,
        'profit': 0.00,
        'profit_percent': 0.00,
        'trade_count': 0,
        'position': None,
        'entry_price': 0,
        'current_price': 0,
        'data_source': 'default',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/trades')
def get_trades():
    """API endpoint for getting trade history"""
    try:
        # First try to get data from our database
        with app.app_context():
            trades = Trade.query.order_by(desc(Trade.timestamp)).limit(100).all()
            
            if trades:
                trade_list = [{
                    'id': trade.id,
                    'timestamp': trade.timestamp.isoformat(),
                    'trade_type': trade.trade_type,
                    'symbol': trade.symbol,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'position': trade.position,
                    'profit': trade.profit,
                    'profit_percent': trade.profit_percent,
                    'exit_price': trade.exit_price,
                    'exit_timestamp': trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
                    'is_open': trade.is_open,
                    'strategy': trade.strategy
                } for trade in trades]
                
                return jsonify({
                    'trades': trade_list,
                    'data_source': 'database'
                })
        
        # Fallback to CSV file if database is empty
        try:
            # Read from CSV file
            trades = []
            with open('trades.csv', 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Convert relevant fields to numbers
                    for field in ['price', 'quantity', 'profit', 'profit_percent']:
                        if field in row and row[field]:
                            try:
                                row[field] = float(row[field])
                            except ValueError:
                                row[field] = None
                    
                    trades.append(row)
            
            return jsonify({
                'trades': trades,
                'data_source': 'csv'
            })
        
        except Exception as e:
            print(f"Error reading trades from CSV: {e}")
            return jsonify({
                'trades': [],
                'error': str(e),
                'data_source': 'error'
            })
    
    except Exception as e:
        print(f"Error retrieving trades: {e}")
        return jsonify({
            'trades': [],
            'error': str(e),
            'data_source': 'error'
        })

@app.route('/api/current_position')
def get_current_position():
    """API endpoint for getting current position"""
    try:
        # First try to get the most recent portfolio snapshot from our database
        with app.app_context():
            latest_snapshot = PortfolioSnapshot.query.order_by(desc(PortfolioSnapshot.timestamp)).first()
            
            if latest_snapshot and (datetime.utcnow() - latest_snapshot.timestamp).total_seconds() < 60:
                # We have recent data in the database
                data = {
                    'position': latest_snapshot.current_position,
                    'entry_price': latest_snapshot.entry_price,
                    'current_price': latest_snapshot.current_price,
                    'timestamp': latest_snapshot.timestamp.isoformat(),
                    'data_source': 'database'
                }
                
                # Calculate unrealized profit if we have a position
                if latest_snapshot.current_position and latest_snapshot.entry_price and latest_snapshot.current_price:
                    if latest_snapshot.current_position == 'long':
                        unrealized_profit = latest_snapshot.current_price - latest_snapshot.entry_price
                    else:  # short
                        unrealized_profit = latest_snapshot.entry_price - latest_snapshot.current_price
                    
                    # Calculate percentage based on a standard position size (this is an approximation)
                    position_value = latest_snapshot.entry_price * 50  # Assuming quantity of 50
                    unrealized_profit_percent = (unrealized_profit * 50 / position_value) * 100
                    
                    data['unrealized_profit'] = unrealized_profit * 50  # Multiply by quantity
                    data['unrealized_profit_percent'] = unrealized_profit_percent
                
                return jsonify(data)
        
        # Fallback to checking the portfolio_value.txt file
        response_file = "portfolio_value.txt"
        if os.path.exists(response_file):
            file_mtime = os.path.getmtime(response_file)
            if time.time() - file_mtime < 60:  # Only use if less than 60 seconds old
                with open(response_file, "r") as f:
                    try:
                        data = json.loads(f.read())
                        
                        position_data = {
                            'position': data.get('position'),
                            'entry_price': data.get('entry_price'),
                            'current_price': data.get('current_price'),
                            'timestamp': datetime.utcnow().isoformat(),
                            'data_source': 'file'
                        }
                        
                        # Calculate unrealized profit if we have a position
                        if position_data['position'] and position_data['entry_price'] and position_data['current_price']:
                            entry_price = float(position_data['entry_price'])
                            current_price = float(position_data['current_price'])
                            
                            if position_data['position'] == 'long':
                                unrealized_profit = current_price - entry_price
                            else:  # short
                                unrealized_profit = entry_price - current_price
                            
                            # Calculate percentage based on a standard position size (this is an approximation)
                            position_value = entry_price * 50  # Assuming quantity of 50
                            unrealized_profit_percent = (unrealized_profit * 50 / position_value) * 100
                            
                            position_data['unrealized_profit'] = unrealized_profit * 50  # Multiply by quantity
                            position_data['unrealized_profit_percent'] = unrealized_profit_percent
                        
                        return jsonify(position_data)
                    except json.JSONDecodeError:
                        pass
    
    except Exception as e:
        print(f"Error retrieving current position: {e}")
    
    # Default response if all else fails
    return jsonify({
        'position': None,
        'entry_price': 0,
        'current_price': 0,
        'timestamp': datetime.utcnow().isoformat(),
        'data_source': 'default'
    })

@app.route('/api/portfolio_history')
def get_portfolio_history():
    """API endpoint for getting historical portfolio values"""
    try:
        # First try to get data from our database
        with app.app_context():
            # Get snapshots from the last 7 days
            start_date = datetime.utcnow() - timedelta(days=7)
            snapshots = PortfolioSnapshot.query.filter(
                PortfolioSnapshot.timestamp >= start_date
            ).order_by(PortfolioSnapshot.timestamp).all()
            
            if snapshots:
                history = [{
                    'timestamp': snapshot.timestamp.isoformat(),
                    'total_value': snapshot.total_value,
                    'cash_value': snapshot.cash_value,
                    'holdings_value': snapshot.holdings_value,
                    'total_profit': snapshot.total_profit,
                    'total_profit_percent': snapshot.total_profit_percent,
                    'trade_count': snapshot.trade_count
                } for snapshot in snapshots]
                
                return jsonify({
                    'history': history,
                    'data_source': 'database'
                })
    
    except Exception as e:
        print(f"Error retrieving portfolio history: {e}")
    
    # Default response if all else fails - create synthetic data
    # Note: This is just for UI rendering until real data is available
    history = []
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)
    
    # Create a default history with the starting value of $20,000
    history.append({
        'timestamp': start_date.isoformat(),
        'total_value': 20000.00,
        'cash_value': 20000.00,
        'holdings_value': 0.00,
        'total_profit': 0.00,
        'total_profit_percent': 0.00,
        'trade_count': 0
    })
    
    # Add current value based on status API
    try:
        response_file = "portfolio_value.txt"
        if os.path.exists(response_file):
            with open(response_file, "r") as f:
                data = json.loads(f.read())
                
                history.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'total_value': data.get('value', 20000.00),
                    'cash_value': data.get('value', 20000.00) - (data.get('position_value', 0) or 0),
                    'holdings_value': data.get('position_value', 0) or 0,
                    'total_profit': data.get('profit', 0.00),
                    'total_profit_percent': data.get('profit_percent', 0.00),
                    'trade_count': data.get('trade_count', 0)
                })
    except Exception:
        # Add current with default values if we couldn't get real data
        history.append({
            'timestamp': end_date.isoformat(),
            'total_value': 20000.00,
            'cash_value': 20000.00,
            'holdings_value': 0.00,
            'total_profit': 0.00,
            'total_profit_percent': 0.00,
            'trade_count': 0
        })
    
    return jsonify({
        'history': history,
        'data_source': 'default'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)