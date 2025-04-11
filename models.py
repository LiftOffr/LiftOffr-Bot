from datetime import datetime
import os
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

class Trade(db.Model):
    """Model for storing trade information"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Trade parameters
    trade_type = db.Column(db.String(10))  # "buy" or "sell"
    symbol = db.Column(db.String(20))
    quantity = db.Column(db.Float)
    price = db.Column(db.Float)
    position = db.Column(db.String(10))  # "long" or "short"
    
    # Trade results (for closed trades)
    profit = db.Column(db.Float, nullable=True)
    profit_percent = db.Column(db.Float, nullable=True)
    exit_price = db.Column(db.Float, nullable=True)
    exit_timestamp = db.Column(db.DateTime, nullable=True)
    
    # Strategy information
    strategy = db.Column(db.String(50), default="adaptive")
    
    # Status
    is_open = db.Column(db.Boolean, default=True)
    
    def __repr__(self):
        return f"<Trade {self.id} {self.trade_type} {self.symbol} @ {self.price}>"

class PortfolioSnapshot(db.Model):
    """Model for storing portfolio snapshots"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Portfolio value
    total_value = db.Column(db.Float)
    cash_value = db.Column(db.Float)
    holdings_value = db.Column(db.Float)
    
    # Performance metrics
    total_profit = db.Column(db.Float)
    total_profit_percent = db.Column(db.Float)
    trade_count = db.Column(db.Integer)
    
    # Additional information
    current_position = db.Column(db.String(10), nullable=True)
    entry_price = db.Column(db.Float, nullable=True)
    current_price = db.Column(db.Float, nullable=True)
    
    def __repr__(self):
        return f"<PortfolioSnapshot {self.id} Value: ${self.total_value:.2f} Profit: ${self.total_profit:.2f} ({self.total_profit_percent:.2f}%)>"