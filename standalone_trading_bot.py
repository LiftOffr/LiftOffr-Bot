#!/usr/bin/env python3
"""
Standalone Trading Bot

This script provides a completely isolated trading bot implementation
that avoids any Flask imports or port conflicts.
"""
import os
import sys
import json
import time
import asyncio
import logging
import random
import argparse
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

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

# Default trading pairs
DEFAULT_PAIRS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD", 
    "LINK/USD", "AVAX/USD", "MATIC/USD", "UNI/USD", "ATOM/USD"
]

# Sandbox data for simulation
SANDBOX_PRICES = {
    "BTC/USD": 29735.50,
    "ETH/USD": 1865.25,
    "SOL/USD": 22.68,
    "ADA/USD": 0.381,
    "DOT/USD": 5.42,
    "LINK/USD": 6.51,
    "AVAX/USD": 14.28,
    "MATIC/USD": 0.665,
    "UNI/USD": 4.75,
    "ATOM/USD": 8.92
}

class MLRiskManager:
    """Simple risk manager that calculates optimal leverage based on confidence"""
    
    def __init__(self):
        """Initialize risk manager"""
        self.confidence_map = {
            0.5: 5,    # 50% confidence -> 5x leverage
            0.6: 10,   # 60% confidence -> 10x leverage
            0.7: 20,   # 70% confidence -> 20x leverage
            0.8: 40,   # 80% confidence -> 40x leverage
            0.9: 80,   # 90% confidence -> 80x leverage
            0.95: 100, # 95% confidence -> 100x leverage
            1.0: 125   # 100% confidence -> 125x leverage
        }
    
    def get_optimal_leverage(self, confidence: float) -> float:
        """Calculate optimal leverage based on confidence"""
        # Find closest confidence threshold in map
        thresholds = sorted(self.confidence_map.keys())
        
        for threshold in thresholds:
            if confidence <= threshold:
                return float(self.confidence_map[threshold])
        
        # If above highest threshold, return max leverage
        return 125.0
    
    def get_position_size(self, confidence: float, account_balance: float, 
                         current_price: float, leverage: float = None) -> Tuple[float, float]:
        """Calculate optimal position size based on confidence"""
        if leverage is None:
            leverage = self.get_optimal_leverage(confidence)
        
        # Scale risk percentage based on confidence
        # Higher confidence -> higher risk percentage
        base_risk = 0.01  # 1% base risk
        max_risk = 0.20   # 20% max risk
        
        risk_percentage = base_risk + (confidence * (max_risk - base_risk))
        
        # Calculate position size
        risk_amount = account_balance * risk_percentage
        margin = risk_amount
        
        # Calculate position size in units
        position_size = (margin * leverage) / current_price
        
        return position_size, risk_percentage
    
    def get_entry_parameters(self, pair: str, confidence: float, account_balance: float,
                           current_price: float) -> Dict[str, Any]:
        """Get complete set of entry parameters for a trade"""
        leverage = self.get_optimal_leverage(confidence)
        position_size, risk_percentage = self.get_position_size(
            confidence, account_balance, current_price, leverage
        )
        
        # Calculate liquidation prices (simplified)
        # For long position: entry_price * (1 - 1/leverage)
        # For short position: entry_price * (1 + 1/leverage)
        liquidation_price_long = current_price * (1 - 0.9/leverage)
        liquidation_price_short = current_price * (1 + 0.9/leverage)
        
        # Calculate margin
        margin = position_size * current_price / leverage
        
        return {
            "leverage": leverage,
            "position_size": position_size,
            "risk_percentage": risk_percentage,
            "margin": margin,
            "liquidation_price_long": liquidation_price_long,
            "liquidation_price_short": liquidation_price_short
        }
    
    def simulate_ml_prediction(self, pair: str) -> Dict[str, Any]:
        """Simulate ML model prediction for testing"""
        # Simulate different confidence levels for different pairs
        pair_confidence_map = {
            "BTC/USD": 0.95,
            "ETH/USD": 0.93,
            "SOL/USD": 0.92,
            "ADA/USD": 0.88,
            "DOT/USD": 0.87,
            "LINK/USD": 0.89,
            "AVAX/USD": 0.86,
            "MATIC/USD": 0.85,
            "UNI/USD": 0.84,
            "ATOM/USD": 0.88
        }
        
        # Get confidence for this pair or use a default
        confidence = pair_confidence_map.get(pair, 0.8)
        
        # Add some random variation
        confidence = max(0.6, min(0.98, confidence + (random.random() - 0.5) * 0.1))
        
        # Randomly choose direction with slight bias towards long for most pairs
        direction = "long" if random.random() > 0.4 else "short"
        
        time_horizon = random.choice([15, 30, 60, 120, 240])
        
        return {
            "pair": pair,
            "direction": direction,
            "confidence": confidence,
            "time_horizon": time_horizon,
            "strategy": random.choice(["ARIMA", "Adaptive"]),
            "category": random.choice(["those dudes", "him all along"]),
            "models": {
                "tcn": confidence * (1.0 + random.random() * 0.1 - 0.05),
                "lstm": confidence * (1.0 + random.random() * 0.1 - 0.05),
                "attention_gru": confidence * (1.0 + random.random() * 0.1 - 0.05),
                "transformer": confidence * (1.0 + random.random() * 0.1 - 0.05),
                "ensemble": confidence
            }
        }

class StandaloneTradingBot:
    """
    Standalone trading bot with simulated ML-based risk management
    """
    def __init__(self, trading_pairs: List[str] = None, sandbox: bool = True):
        """
        Initialize the trading bot
        
        Args:
            trading_pairs: List of trading pairs to trade
            sandbox: Whether to run in sandbox mode
        """
        self.trading_pairs = trading_pairs or DEFAULT_PAIRS
        self.sandbox = sandbox
        self.running = False
        self.current_prices = {}
        self.portfolio = {}
        self.positions = []
        self.risk_manager = MLRiskManager()
        self.price_update_thread = None
        
        # Initialize data directories
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Initialize data files if they don't exist
        self._init_data_files()
        
        # Load initial data
        self._load_data()
        
        logger.info(f"Initialized trading bot with {len(self.trading_pairs)} pairs in {'sandbox' if sandbox else 'live'} mode")
    
    def _init_data_files(self):
        """Initialize data files if they don't exist"""
        # Initialize portfolio file
        if not os.path.exists(PORTFOLIO_FILE):
            self._save_json(PORTFOLIO_FILE, {
                "balance": 20000.0,
                "equity": 20000.0,
                "total_value": 20000.0,
                "unrealized_pnl_usd": 0.0,
                "unrealized_pnl_pct": 0.0,
                "last_updated": datetime.now().isoformat()
            })
        
        # Initialize positions file
        if not os.path.exists(POSITIONS_FILE):
            self._save_json(POSITIONS_FILE, [])
        
        # Initialize portfolio history file
        if not os.path.exists(PORTFOLIO_HISTORY_FILE):
            self._save_json(PORTFOLIO_HISTORY_FILE, [
                {
                    "timestamp": datetime.now().isoformat(),
                    "portfolio_value": 20000.0
                }
            ])
        
        # Initialize trades file
        if not os.path.exists(TRADES_FILE):
            self._save_json(TRADES_FILE, [])
    
    def _load_data(self):
        """Load data from files"""
        self.portfolio = self._load_json(PORTFOLIO_FILE, {
            "balance": 20000.0,
            "equity": 20000.0,
            "total_value": 20000.0,
            "unrealized_pnl_usd": 0.0,
            "unrealized_pnl_pct": 0.0,
            "last_updated": datetime.now().isoformat()
        })
        
        self.positions = self._load_json(POSITIONS_FILE, [])
    
    def _load_json(self, filepath: str, default: Any = None) -> Any:
        """
        Load a JSON file or return default if not found
        
        Args:
            filepath: Path to the JSON file
            default: Default value if file not found or error
            
        Returns:
            Loaded data or default
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
            return default
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return default
    
    def _save_json(self, filepath: str, data: Any):
        """
        Save data to a JSON file
        
        Args:
            filepath: Path to the JSON file
            data: Data to save
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving {filepath}: {e}")
    
    def _update_prices(self):
        """Update price data periodically"""
        while self.running:
            try:
                for pair in self.trading_pairs:
                    # Use sandbox prices with small variations
                    if pair in SANDBOX_PRICES:
                        base_price = SANDBOX_PRICES[pair]
                        # Add small random price movement (±0.5%)
                        price_change_pct = (random.random() - 0.5) * 0.01
                        price = base_price * (1 + price_change_pct)
                        self.current_prices[pair] = price
                
                # Log price updates occasionally
                if random.random() < 0.01:  # Log about 1% of updates
                    pair = random.choice(self.trading_pairs)
                    logger.debug(f"Price update for {pair}: ${self.current_prices.get(pair, 0):.2f}")
                
                # Update portfolio with new prices
                self.update_portfolio()
                
                # Check for liquidations
                liquidated = self.check_liquidations()
                if liquidated > 0:
                    logger.warning(f"Liquidated {liquidated} positions")
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error updating prices: {e}")
                time.sleep(5)
    
    def update_portfolio(self):
        """Update portfolio with current prices"""
        # Make sure we have positions and prices
        if not self.positions:
            return
        
        # Get current time
        now = datetime.now()
        
        # Calculate total unrealized PnL
        total_pnl = 0.0
        for position in self.positions:
            pair = position.get("pair")
            if not pair or pair not in self.current_prices:
                continue
            
            current_price = self.current_prices[pair]
            entry_price = position.get("entry_price", current_price)
            size = position.get("size", 0)
            leverage = position.get("leverage", 1)
            direction = position.get("direction", "Long")
            
            # Update current price in position
            position["current_price"] = current_price
            
            # Calculate PnL
            if direction.lower() == "long":
                pnl_percentage = (current_price - entry_price) / entry_price * 100 * leverage
                pnl_amount = (current_price - entry_price) * size
            else:  # Short
                pnl_percentage = (entry_price - current_price) / entry_price * 100 * leverage
                pnl_amount = (entry_price - current_price) * size
            
            position["unrealized_pnl"] = pnl_percentage
            position["unrealized_pnl_amount"] = pnl_amount
            position["unrealized_pnl_pct"] = pnl_percentage
            
            # Calculate duration
            entry_time = position.get("entry_time")
            if entry_time:
                try:
                    entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    duration = now - entry_dt
                    hours, remainder = divmod(duration.total_seconds(), 3600)
                    minutes, _ = divmod(remainder, 60)
                    position["duration"] = f"{int(hours)}h {int(minutes)}m"
                except Exception as e:
                    logger.error(f"Error calculating duration: {e}")
            
            total_pnl += pnl_amount
        
        # Update portfolio
        self.portfolio["unrealized_pnl_usd"] = total_pnl
        self.portfolio["unrealized_pnl_pct"] = (total_pnl / self.portfolio.get("balance", 20000.0)) * 100 if self.portfolio.get("balance", 20000.0) > 0 else 0
        self.portfolio["total_value"] = self.portfolio.get("balance", 20000.0) + total_pnl
        self.portfolio["last_updated"] = now.isoformat()
        self.portfolio["equity"] = self.portfolio["total_value"]
        
        # Save updated data
        self._save_json(POSITIONS_FILE, self.positions)
        self._save_json(PORTFOLIO_FILE, self.portfolio)
        
        # Update portfolio history
        history = self._load_json(PORTFOLIO_HISTORY_FILE, [])
        history.append({
            "timestamp": now.isoformat(),
            "portfolio_value": self.portfolio["total_value"]
        })
        
        # Keep history to a reasonable size
        if len(history) > 1000:
            history = history[-1000:]
        
        self._save_json(PORTFOLIO_HISTORY_FILE, history)
    
    def check_liquidations(self):
        """Check positions for liquidation conditions"""
        if not self.positions:
            return 0
        
        active_positions = []
        liquidated_positions = []
        
        for position in self.positions:
            pair = position.get("pair")
            if not pair or pair not in self.current_prices:
                active_positions.append(position)
                continue
            
            current_price = self.current_prices[pair]
            liquidation_price = position.get("liquidation_price", 0)
            direction = position.get("direction", "")
            
            is_liquidated = False
            
            if direction.lower() == "long" and current_price <= liquidation_price:
                is_liquidated = True
                logger.warning(f"Long position liquidated: {pair} at {current_price}")
            elif direction.lower() == "short" and current_price >= liquidation_price:
                is_liquidated = True
                logger.warning(f"Short position liquidated: {pair} at {current_price}")
            
            if is_liquidated:
                # Mark as liquidated
                position["exit_price"] = current_price
                position["exit_time"] = datetime.now().isoformat()
                position["exit_reason"] = "LIQUIDATED"
                liquidated_positions.append(position)
            else:
                active_positions.append(position)
        
        # Record liquidated positions in trades history
        if liquidated_positions:
            trades = self._load_json(TRADES_FILE, [])
            for position in liquidated_positions:
                trade = position.copy()
                trade["pnl_percentage"] = position.get("unrealized_pnl", -100)
                trade["pnl_amount"] = position.get("unrealized_pnl_amount", 0)
                trades.append(trade)
            self._save_json(TRADES_FILE, trades)
        
        # Save updated positions
        self.positions = active_positions
        self._save_json(POSITIONS_FILE, self.positions)
        
        return len(liquidated_positions)
    
    def generate_trade(self):
        """Generate a trade based on ML predictions"""
        # Skip if portfolio isn't loaded yet
        if not self.portfolio:
            return
        
        # Skip if we don't have prices yet
        if not self.current_prices:
            logger.warning("No price data available for trading")
            return
        
        # Calculate available pairs (ones without open positions)
        existing_pairs = [p.get("pair") for p in self.positions if p.get("pair")]
        available_pairs = [p for p in self.trading_pairs if p not in existing_pairs]
        
        if not available_pairs:
            logger.debug("No available pairs for new trades")
            return
        
        # Pick a random pair from available ones
        pair = random.choice(available_pairs)
        
        # Make sure we have a price for this pair
        if pair not in self.current_prices:
            logger.warning(f"No price data available for {pair}")
            return
        
        current_price = self.current_prices[pair]
        account_balance = self.portfolio.get("balance", 20000.0)
        
        # Get ML prediction
        prediction = self.risk_manager.simulate_ml_prediction(pair)
        confidence = prediction.get("confidence", 0.7)
        direction = prediction.get("direction", "long")
        
        # Calculate entry parameters
        entry_params = self.risk_manager.get_entry_parameters(
            pair=pair,
            confidence=confidence,
            account_balance=account_balance,
            current_price=current_price
        )
        
        # Determine which liquidation price to use based on direction
        liquidation_price = entry_params["liquidation_price_long"] if direction.lower() == "long" else entry_params["liquidation_price_short"]
        
        # Create position
        position = {
            "pair": pair,
            "direction": direction.capitalize(),
            "size": entry_params["position_size"],
            "entry_price": current_price,
            "current_price": current_price,
            "leverage": entry_params["leverage"],
            "strategy": prediction.get("strategy", "ARIMA" if random.random() > 0.5 else "Adaptive"),
            "confidence": confidence,
            "entry_time": datetime.now().isoformat(),
            "unrealized_pnl": 0.0,
            "unrealized_pnl_amount": 0.0,
            "unrealized_pnl_pct": 0.0,
            "liquidation_price": liquidation_price,
            "risk_percentage": entry_params["risk_percentage"]
        }
        
        # Update balance (deduct margin)
        margin = entry_params["margin"]
        self.portfolio["balance"] = self.portfolio.get("balance", 20000.0) - margin
        
        # Add position
        self.positions.append(position)
        
        # Save updated data
        self._save_json(POSITIONS_FILE, self.positions)
        self._save_json(PORTFOLIO_FILE, self.portfolio)
        
        logger.info(f"Generated {direction} trade for {pair} with leverage {entry_params['leverage']}x")
        logger.info(f"Position size: {position['size']:.6f}, Entry price: ${current_price:.2f}")
        return position
    
    def close_position(self, position_index: int, reason: str = "MANUAL"):
        """
        Close a specific position
        
        Args:
            position_index: Index of position to close
            reason: Reason for closing
        """
        if position_index < 0 or position_index >= len(self.positions):
            logger.error(f"Invalid position index: {position_index}")
            return
        
        position = self.positions[position_index]
        pair = position.get("pair")
        
        if not pair or pair not in self.current_prices:
            logger.error(f"No price data for position {position_index}: {pair}")
            return
        
        current_price = self.current_prices[pair]
        entry_price = position.get("entry_price", current_price)
        size = position.get("size", 0)
        leverage = position.get("leverage", 1)
        direction = position.get("direction", "Long")
        
        # Calculate PnL
        if direction.lower() == "long":
            pnl_percentage = (current_price - entry_price) / entry_price * 100 * leverage
            pnl_amount = (current_price - entry_price) * size
        else:  # Short
            pnl_percentage = (entry_price - current_price) / entry_price * 100 * leverage
            pnl_amount = (entry_price - current_price) * size
        
        # Update position
        position["exit_price"] = current_price
        position["exit_time"] = datetime.now().isoformat()
        position["exit_reason"] = reason
        position["pnl_percentage"] = pnl_percentage
        position["pnl_amount"] = pnl_amount
        
        # Update balance (add initial margin + PnL)
        margin = size * entry_price / leverage
        self.portfolio["balance"] = self.portfolio.get("balance", 20000.0) + margin + pnl_amount
        
        # Add to trades history
        trades = self._load_json(TRADES_FILE, [])
        trades.append(position)
        self._save_json(TRADES_FILE, trades)
        
        # Remove from positions list
        self.positions.pop(position_index)
        
        # Save updated data
        self._save_json(POSITIONS_FILE, self.positions)
        self._save_json(PORTFOLIO_FILE, self.portfolio)
        
        logger.info(f"Closed {direction} position for {pair} with PnL: {pnl_percentage:.2f}% (${pnl_amount:.2f})")
        return position
    
    def auto_manage_positions(self):
        """
        Automatically manage open positions
        
        This function evaluates open positions and decides whether to:
        1. Take profit
        2. Cut losses
        3. Adjust stop loss
        4. Hold position
        """
        if not self.positions:
            return
        
        for i in range(len(self.positions) - 1, -1, -1):
            position = self.positions[i]
            unrealized_pnl = position.get("unrealized_pnl", 0)
            
            # Take profit at high levels
            if unrealized_pnl > 50:
                self.close_position(i, "TAKE_PROFIT")
            
            # Cut losses at low levels
            elif unrealized_pnl < -20:
                self.close_position(i, "STOP_LOSS")
    
    def print_status(self):
        """Print current status of the bot"""
        print("\n" + "=" * 80)
        print(f"TRADING BOT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        print(f"\nPortfolio Value: ${self.portfolio.get('total_value', 0):.2f}")
        print(f"Balance: ${self.portfolio.get('balance', 0):.2f}")
        print(f"Unrealized PnL: ${self.portfolio.get('unrealized_pnl_usd', 0):.2f} " +
              f"({self.portfolio.get('unrealized_pnl_pct', 0):.2f}%)")
        
        print("\nOpen Positions:")
        if not self.positions:
            print("No open positions")
        else:
            for i, pos in enumerate(self.positions):
                print(f"  {i+1}. {pos.get('pair')}: {pos.get('direction')} {pos.get('size'):.6f} @ " +
                      f"${pos.get('entry_price', 0):.2f} (Current: ${pos.get('current_price', 0):.2f}) " +
                      f"[PnL: {pos.get('unrealized_pnl', 0):.2f}%]")
        
        print("\nCurrent Prices:")
        for pair, price in self.current_prices.items():
            if pair in [p.get("pair") for p in self.positions]:
                print(f"  * {pair}: ${price:.2f}")
            else:
                print(f"  {pair}: ${price:.2f}")
        
        print("\nTrading with:")
        print(f"  Sandbox Mode: {self.sandbox}")
        print(f"  Trading Pairs: {', '.join(self.trading_pairs)}")
        print(f"  Open Positions: {len(self.positions)}")
        
        print("=" * 80)
    
    def start(self):
        """Start the trading bot"""
        logger.info("Starting trading bot...")
        self.running = True
        
        # Start price update thread
        self.price_update_thread = threading.Thread(target=self._update_prices)
        self.price_update_thread.daemon = True
        self.price_update_thread.start()
        
        # Main trading loop
        trade_interval = 15  # seconds between trade evaluations
        manage_interval = 30  # seconds between position management checks
        status_interval = 60  # seconds between status prints
        
        last_trade_time = 0
        last_manage_time = 0
        last_status_time = 0
        
        try:
            while self.running:
                current_time = time.time()
                
                # Generate trades
                if current_time - last_trade_time >= trade_interval:
                    # Only trade if we have capacity
                    if len(self.positions) < len(self.trading_pairs):
                        self.generate_trade()
                    last_trade_time = current_time
                
                # Manage positions
                if current_time - last_manage_time >= manage_interval:
                    self.auto_manage_positions()
                    last_manage_time = current_time
                
                # Print status
                if current_time - last_status_time >= status_interval:
                    self.print_status()
                    last_status_time = current_time
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self.running = False
        
        # Wait for price update thread to finish
        if self.price_update_thread and self.price_update_thread.is_alive():
            self.price_update_thread.join(2)
        
        # Save final state
        self._save_json(POSITIONS_FILE, self.positions)
        self._save_json(PORTFOLIO_FILE, self.portfolio)
        
        logger.info("Trading bot stopped")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start the trading bot")
    
    parser.add_argument("--sandbox", action="store_true", default=True,
                        help="Run in sandbox mode (default: True)")
    
    parser.add_argument("--pairs", type=str, nargs="+", 
                        default=DEFAULT_PAIRS,
                        help="Trading pairs to use")
    
    return parser.parse_args()

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print(" KRAKEN TRADING BOT WITH ML-BASED RISK MANAGEMENT")
    print("=" * 60)
    
    # Parse arguments
    args = parse_arguments()
    
    # Check for Kraken API credentials
    if not args.sandbox and (not os.environ.get("KRAKEN_API_KEY") or not os.environ.get("KRAKEN_API_SECRET")):
        logger.warning("Kraken API credentials not found, forcing sandbox mode")
        args.sandbox = True
    
    # Start trading bot
    print(f"\nStarting trading bot in {'sandbox' if args.sandbox else 'live'} mode")
    print(f"Trading pairs: {', '.join(args.pairs)}")
    print("\nPress Ctrl+C to stop the bot at any time\n")
    
    trading_bot = StandaloneTradingBot(
        trading_pairs=args.pairs,
        sandbox=args.sandbox
    )
    
    try:
        trading_bot.start()
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user")
    except Exception as e:
        logger.error(f"Error running trading bot: {e}", exc_info=True)
        return 1
    finally:
        if trading_bot:
            trading_bot.cleanup()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())