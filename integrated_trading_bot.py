#!/usr/bin/env python3
"""
Integrated Trading Bot with Real-time Data

This script runs the trading bot with Kraken API integration for real-time
market data and dynamic ML-based risk management.
"""
import os
import json
import time
import asyncio
import logging
import threading
import argparse
import random
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Import our modules
import kraken_api_client as kraken
import ml_risk_manager as ml_risk

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

class IntegratedTradingBot:
    """
    Integrated trading bot with real-time market data and ML-based risk management.
    
    This class combines the Kraken API client for real-time data with
    ML-based risk management to execute trading strategies.
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
        self.risk_manager = ml_risk.MLRiskManager()
        self.price_monitor_thread = None
        
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
    
    def _price_update_callback(self, ticker_data: Dict[str, Any]):
        """
        Process price updates from WebSocket
        
        Args:
            ticker_data: Ticker data from WebSocket
        """
        pair = ticker_data.get("pair")
        if not pair:
            return
        
        # Update current price
        if "price" in ticker_data:
            self.current_prices[pair] = ticker_data["price"]
        elif "last" in ticker_data:
            self.current_prices[pair] = ticker_data["last"]
        
        # Log price updates occasionally
        if random.random() < 0.01:  # Log about 1% of updates to avoid spam
            logger.debug(f"Price update for {pair}: ${self.current_prices.get(pair, 0):.2f}")
    
    async def _start_price_monitoring(self):
        """Start WebSocket price monitoring"""
        client = kraken.KrakenWebSocketClient(sandbox=self.sandbox)
        
        try:
            await client.connect()
            
            # Register callbacks for each pair
            for pair in self.trading_pairs:
                client.register_ticker_callback(pair, self._price_update_callback)
            
            # Subscribe to ticker updates
            await client.subscribe_ticker(self.trading_pairs)
            
            # Process messages until stopped
            while self.running:
                if not client.running:
                    logger.warning("WebSocket connection lost, reconnecting...")
                    await client.connect()
                    await client.subscribe_ticker(self.trading_pairs)
                
                await asyncio.sleep(1)
            
            await client.disconnect()
        
        except Exception as e:
            logger.error(f"Error in price monitoring: {e}")
        
        finally:
            if client:
                await client.disconnect()
    
    def _start_price_monitoring_thread(self):
        """Start price monitoring in a separate thread"""
        async def run():
            await self._start_price_monitoring()
        
        def thread_target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run())
        
        self.price_monitor_thread = threading.Thread(target=thread_target)
        self.price_monitor_thread.daemon = True
        self.price_monitor_thread.start()
        logger.info("Started price monitoring thread")
    
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
    
    def simulate_ml_prediction(self, pair: str) -> Dict[str, Any]:
        """
        Simulate ML model prediction for testing
        
        In a real implementation, this would call the actual ML model
        or load a prediction from a file/database.
        
        Args:
            pair: Trading pair
            
        Returns:
            Prediction dictionary
        """
        return self.risk_manager.simulate_ml_prediction(pair)
    
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
        prediction = self.simulate_ml_prediction(pair)
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
            "strategy": "ARIMA" if random.random() > 0.5 else "Adaptive",
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
        
        logger.info(f"Opened {direction} position on {pair} with "
                  f"leverage: {entry_params['leverage']:.2f}x, "
                  f"size: {entry_params['position_size']:.6f}, "
                  f"price: ${current_price:.2f}, "
                  f"confidence: {confidence:.2f}")
        
        return position
    
    def close_position(self, position_index: int, reason: str = "MANUAL"):
        """
        Close a specific position
        
        Args:
            position_index: Index of position to close
            reason: Reason for closing
        """
        if position_index >= len(self.positions):
            logger.error(f"Invalid position index: {position_index}")
            return
        
        position = self.positions[position_index]
        pair = position.get("pair")
        
        if not pair or pair not in self.current_prices:
            logger.error(f"Cannot close position {position_index}: price data not available")
            return
        
        current_price = self.current_prices[pair]
        
        # Calculate final PnL
        entry_price = position.get("entry_price", current_price)
        size = position.get("size", 0)
        leverage = position.get("leverage", 1)
        direction = position.get("direction", "Long")
        
        if direction.lower() == "long":
            pnl_percentage = (current_price - entry_price) / entry_price * 100 * leverage
            pnl_amount = (current_price - entry_price) * size
        else:  # Short
            pnl_percentage = (entry_price - current_price) / entry_price * 100 * leverage
            pnl_amount = (entry_price - current_price) * size
        
        # Record closed trade
        trade = position.copy()
        trade["exit_price"] = current_price
        trade["exit_time"] = datetime.now().isoformat()
        trade["exit_reason"] = reason
        trade["pnl_percentage"] = pnl_percentage
        trade["pnl_amount"] = pnl_amount
        
        # Add to trades history
        trades = self._load_json(TRADES_FILE, [])
        trades.append(trade)
        self._save_json(TRADES_FILE, trades)
        
        # Remove from positions
        del self.positions[position_index]
        self._save_json(POSITIONS_FILE, self.positions)
        
        # Update balance (add back margin + PnL)
        margin_used = (size * entry_price) / leverage
        self.portfolio["balance"] = self.portfolio.get("balance", 20000.0) + margin_used + pnl_amount
        self._save_json(PORTFOLIO_FILE, self.portfolio)
        
        logger.info(f"Closed {direction} position on {pair} with PnL: "
                  f"{pnl_percentage:.2f}% (${pnl_amount:.2f}), "
                  f"reason: {reason}")
        
        return trade
    
    def auto_manage_positions(self):
        """
        Automatically manage open positions
        
        This function evaluates open positions and decides whether to:
        1. Take profit
        2. Cut losses
        3. Adjust stop loss
        4. Hold position
        """
        for i, position in enumerate(self.positions[:]):  # Copy to avoid issues with deletion
            pair = position.get("pair")
            if not pair or pair not in self.current_prices:
                continue
            
            current_price = self.current_prices[pair]
            direction = position.get("direction", "Long").lower()
            entry_price = position.get("entry_price", current_price)
            unrealized_pnl_pct = position.get("unrealized_pnl_pct", 0)
            
            # Simple take profit logic
            take_profit_threshold = 20.0  # 20% profit
            if unrealized_pnl_pct >= take_profit_threshold:
                self.close_position(i, "TAKE_PROFIT")
                continue
            
            # Simple stop loss logic
            stop_loss_threshold = -10.0  # 10% loss
            if unrealized_pnl_pct <= stop_loss_threshold:
                self.close_position(i, "STOP_LOSS")
                continue
            
            # More sophisticated logic could be added here:
            # - Trailing stops
            # - Re-evaluate position with new ML predictions
            # - Adjust position size or leverage
    
    def print_status(self):
        """Print current status of the bot"""
        print("\n" + "-" * 60)
        print(f"STATUS UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Portfolio Value: ${self.portfolio.get('total_value', 0):.2f}")
        print(f"Available Balance: ${self.portfolio.get('balance', 0):.2f}")
        print(f"Unrealized PnL: ${self.portfolio.get('unrealized_pnl_usd', 0):.2f} "
             f"({self.portfolio.get('unrealized_pnl_pct', 0):.2f}%)")
        print(f"Open Positions: {len(self.positions)}")
        
        if self.positions:
            print("\nOpen Positions:")
            print("-" * 60)
            print(f"{'Pair':<10} {'Dir':<6} {'Leverage':<10} {'Entry':<10} {'Current':<10} {'PnL %':<10} {'PnL $':<12}")
            print("-" * 60)
            
            for position in self.positions:
                pair = position.get("pair", "")
                direction = position.get("direction", "")
                leverage = position.get("leverage", 0)
                entry_price = position.get("entry_price", 0)
                current_price = position.get("current_price", 0)
                unrealized_pnl_pct = position.get("unrealized_pnl_pct", 0)
                unrealized_pnl_amount = position.get("unrealized_pnl_amount", 0)
                
                print(f"{pair:<10} {direction:<6} {leverage:<10.2f}x ${entry_price:<8.2f} ${current_price:<8.2f} "
                     f"{unrealized_pnl_pct:<8.2f}% ${unrealized_pnl_amount:<10.2f}")
        
        print("-" * 60)
    
    def start(self):
        """Start the trading bot"""
        print("\n" + "=" * 60)
        print(" INTEGRATED TRADING BOT WITH DYNAMIC LEVERAGE")
        print("=" * 60)
        print("\nTrading pairs in real-time:")
        for i, pair in enumerate(self.trading_pairs):
            print(f"{i+1}. {pair}")
        
        print("\nFeatures:")
        print("- Real-time market data from Kraken API")
        print("- ML-based risk management with dynamic leverage (5x-125x)")
        print("- Automatic liquidation protection")
        print("- Comprehensive portfolio tracking")
        
        print(f"\nStarting trading bot in {'sandbox' if self.sandbox else 'live'} mode...")
        
        try:
            # Set running flag
            self.running = True
            
            # Initialize current prices from API
            self.current_prices = kraken.get_current_prices(self.trading_pairs)
            logger.info(f"Initialized prices for {len(self.current_prices)} pairs")
            
            # Start price monitoring thread
            self._start_price_monitoring_thread()
            
            # Main loop
            counter = 0
            while self.running:
                # Update portfolio with current prices
                self.update_portfolio()
                
                # Check for liquidations
                liquidated = self.check_liquidations()
                if liquidated:
                    logger.warning(f"Liquidated {liquidated} positions")
                
                # Automatically manage positions
                self.auto_manage_positions()
                
                # Generate new trades occasionally (about 10% of iterations)
                counter += 1
                if counter % 30 == 0 and random.random() < 0.1:
                    self.generate_trade()
                
                # Print status every 30 iterations
                if counter % 30 == 0:
                    self.print_status()
                
                # Sleep to avoid high CPU usage
                time.sleep(2)
        
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Error in trading bot: {e}", exc_info=True)
        finally:
            self.running = False
            logger.info("Trading bot stopped")
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        # Wait for price monitoring thread to finish
        if self.price_monitor_thread and self.price_monitor_thread.is_alive():
            self.price_monitor_thread.join(timeout=5)
        
        logger.info("Trading bot resources cleaned up")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Integrated Trading Bot")
    
    parser.add_argument("--pairs", type=str, nargs="+", default=DEFAULT_PAIRS,
                        help="Trading pairs to use")
    
    parser.add_argument("--sandbox", action="store_true", default=True,
                        help="Run in sandbox mode (default: True)")
    
    parser.add_argument("--live", action="store_true", default=False,
                        help="Run in live mode (overrides --sandbox)")
    
    args = parser.parse_args()
    
    # Override sandbox if live is specified
    if args.live:
        args.sandbox = False
    
    return args

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Check API credentials if not in sandbox mode
    if not args.sandbox:
        if not os.environ.get("KRAKEN_API_KEY") or not os.environ.get("KRAKEN_API_SECRET"):
            logger.error("Kraken API credentials not found in environment variables")
            logger.error("Set KRAKEN_API_KEY and KRAKEN_API_SECRET or use --sandbox mode")
            return 1
    
    # Create and start the bot
    bot = IntegratedTradingBot(trading_pairs=args.pairs, sandbox=args.sandbox)
    
    try:
        bot.start()
    except KeyboardInterrupt:
        print("\nTrading bot stopped by user")
    except Exception as e:
        logger.error(f"Error running trading bot: {e}", exc_info=True)
        return 1
    finally:
        bot.cleanup()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())