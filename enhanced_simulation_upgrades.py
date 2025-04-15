#!/usr/bin/env python3
"""
Enhanced Simulation Upgrades

This script adds realistic market simulation features to the trading bot:
1. Partial fills for large orders
2. Variable slippage based on order book depth
3. Realistic order book simulation
4. Market impact modeling
5. Latency simulation for order execution
6. Improved fee structure with tier-based fees
7. Flash crash stress testing
8. Order book visualization
"""

import os
import time
import json
import random
import logging
import datetime
import numpy as np
import threading
from typing import Dict, List, Tuple, Optional

# Import the risk-aware sandbox trader
from risk_aware_sandbox_trader import RiskAwareSandboxTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = "config"
DATA_DIR = "data"
MARKET_DATA_DIR = f"{DATA_DIR}/market_data"
ORDER_BOOK_FILE = f"{MARKET_DATA_DIR}/order_book_snapshot.json"
LATENCY_CONFIG_FILE = f"{CONFIG_DIR}/latency_config.json"
ADVANCED_FEE_CONFIG_FILE = f"{CONFIG_DIR}/advanced_fee_config.json"
TIER_BASED_FEES_FILE = f"{CONFIG_DIR}/tier_based_fees.json"
MARKET_IMPACT_CONFIG_FILE = f"{CONFIG_DIR}/market_impact_config.json"

class OrderBookEntry:
    """Represents an entry in the order book"""
    def __init__(self, price: float, size: float, order_id: str = None):
        self.price = price
        self.size = size
        self.order_id = order_id or f"order_{random.randint(10000, 99999)}"
    
    def to_dict(self):
        return {
            "price": self.price,
            "size": self.size,
            "order_id": self.order_id
        }

class OrderBook:
    """Simulated order book for a trading pair"""
    def __init__(self, pair: str, base_price: float, depth: int = 20, spread_pct: float = 0.001):
        self.pair = pair
        self.base_price = base_price
        self.depth = depth
        self.spread_pct = spread_pct
        self.bids = []  # Buy orders (price descending)
        self.asks = []  # Sell orders (price ascending)
        self.last_updated = datetime.datetime.now().replace(tzinfo=None)
        
        # Initialize order book
        self._generate_order_book()
    
    def _generate_order_book(self):
        """Generate a realistic order book based on base price"""
        # Clear current order book
        self.bids = []
        self.asks = []
        
        # Calculate bid-ask spread
        spread = self.base_price * self.spread_pct
        bid_price = self.base_price - (spread / 2)
        ask_price = self.base_price + (spread / 2)
        
        # Generate bids (buy orders)
        for i in range(self.depth):
            # Price gets lower as we go down the book
            price_decrease = (i * 0.0002 + (i**2) * 0.00001) * self.base_price
            price = bid_price - price_decrease
            
            # Size increases as we go down (larger orders at worse prices)
            size_factor = 1 + i * 0.1 + random.uniform(0, 0.05)
            size = (0.1 + random.uniform(0, 0.1)) * size_factor
            
            # Add some randomness
            price *= (1 - random.uniform(0, 0.0002))
            
            self.bids.append(OrderBookEntry(price, size))
        
        # Generate asks (sell orders)
        for i in range(self.depth):
            # Price gets higher as we go up the book
            price_increase = (i * 0.0002 + (i**2) * 0.00001) * self.base_price
            price = ask_price + price_increase
            
            # Size increases as we go up (larger orders at worse prices)
            size_factor = 1 + i * 0.1 + random.uniform(0, 0.05)
            size = (0.1 + random.uniform(0, 0.1)) * size_factor
            
            # Add some randomness
            price *= (1 + random.uniform(0, 0.0002))
            
            self.asks.append(OrderBookEntry(price, size))
        
        # Sort bids (descending) and asks (ascending)
        self.bids.sort(key=lambda x: x.price, reverse=True)
        self.asks.sort(key=lambda x: x.price)
        
        self.last_updated = datetime.datetime.now().replace(tzinfo=None)
    
    def update(self, new_base_price: float = None, volatility: float = 0.0001):
        """Update the order book with market movements"""
        if new_base_price is not None:
            self.base_price = new_base_price
        else:
            # Simulate small price movements
            self.base_price *= (1 + random.uniform(-volatility, volatility))
        
        # Update order book with new base price
        self._generate_order_book()
    
    def get_best_bid(self) -> float:
        """Get the best (highest) bid price"""
        return self.bids[0].price if self.bids else self.base_price * 0.999
    
    def get_best_ask(self) -> float:
        """Get the best (lowest) ask price"""
        return self.asks[0].price if self.asks else self.base_price * 1.001
    
    def get_mid_price(self) -> float:
        """Get the mid price (between best bid and ask)"""
        return (self.get_best_bid() + self.get_best_ask()) / 2
    
    def get_market_buy_price(self, size: float) -> Tuple[float, float]:
        """
        Get the average price for a market buy order of given size
        
        Args:
            size: The size of the order
            
        Returns:
            (average_price, filled_size)
        """
        if not self.asks:
            return self.base_price * 1.001, size
        
        total_cost = 0
        filled_size = 0
        avg_price = 0
        
        for ask in self.asks:
            if filled_size >= size:
                break
            
            fill_amount = min(ask.size, size - filled_size)
            total_cost += fill_amount * ask.price
            filled_size += fill_amount
        
        if filled_size > 0:
            avg_price = total_cost / filled_size
        else:
            avg_price = self.base_price * 1.001
        
        return avg_price, filled_size
    
    def get_market_sell_price(self, size: float) -> Tuple[float, float]:
        """
        Get the average price for a market sell order of given size
        
        Args:
            size: The size of the order
            
        Returns:
            (average_price, filled_size)
        """
        if not self.bids:
            return self.base_price * 0.999, size
        
        total_revenue = 0
        filled_size = 0
        avg_price = 0
        
        for bid in self.bids:
            if filled_size >= size:
                break
            
            fill_amount = min(bid.size, size - filled_size)
            total_revenue += fill_amount * bid.price
            filled_size += fill_amount
        
        if filled_size > 0:
            avg_price = total_revenue / filled_size
        else:
            avg_price = self.base_price * 0.999
        
        return avg_price, filled_size
    
    def get_slippage(self, size: float, direction: str) -> float:
        """
        Calculate slippage for a given order size and direction
        
        Args:
            size: Order size
            direction: 'buy' or 'sell'
            
        Returns:
            slippage percentage
        """
        if direction.lower() == 'buy':
            avg_price, _ = self.get_market_buy_price(size)
            slippage = (avg_price / self.get_best_ask()) - 1
        else:  # 'sell'
            avg_price, _ = self.get_market_sell_price(size)
            slippage = 1 - (avg_price / self.get_best_bid())
        
        return max(0, slippage)  # Ensure non-negative
    
    def to_dict(self) -> Dict:
        """Convert order book to a dictionary"""
        return {
            "pair": self.pair,
            "base_price": self.base_price,
            "best_bid": self.get_best_bid(),
            "best_ask": self.get_best_ask(),
            "mid_price": self.get_mid_price(),
            "spread_pct": (self.get_best_ask() - self.get_best_bid()) / self.get_mid_price(),
            "bids": [bid.to_dict() for bid in self.bids],
            "asks": [ask.to_dict() for ask in self.asks],
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OrderBook':
        """Create an order book from a dictionary"""
        order_book = cls(data["pair"], data["base_price"])
        
        # Load bids
        order_book.bids = []
        for bid_data in data["bids"]:
            order_book.bids.append(OrderBookEntry(
                bid_data["price"],
                bid_data["size"],
                bid_data.get("order_id")
            ))
        
        # Load asks
        order_book.asks = []
        for ask_data in data["asks"]:
            order_book.asks.append(OrderBookEntry(
                ask_data["price"],
                ask_data["size"],
                ask_data.get("order_id")
            ))
        
        # Set last updated time
        order_book.last_updated = datetime.datetime.fromisoformat(data["last_updated"])
        
        return order_book

class MarketImpactModel:
    """Model for simulating market impact of trades"""
    def __init__(self, config_file: str = MARKET_IMPACT_CONFIG_FILE):
        """Initialize market impact model"""
        self.config = self._load_config(config_file)
        self.pair_volatility = {}
        self.pair_liquidity = {}
    
    def _load_config(self, config_file: str) -> Dict:
        """Load market impact configuration"""
        default_config = {
            "impact_factor": 0.2,  # Lower = less impact
            "decay_factor": 0.5,   # Higher = faster decay
            "impact_window": 3600,  # Impact window in seconds
            "pair_factors": {
                "BTC/USD": 0.5,    # Lower = less impact
                "ETH/USD": 0.7,
                "SOL/USD": 1.2,
                "ADA/USD": 1.5,
                "DOT/USD": 1.4,
                "LINK/USD": 1.3
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading market impact config: {e}")
        
        # Save default config if not exists
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def update_volatility(self, pair: str, volatility: float):
        """Update volatility for a pair"""
        self.pair_volatility[pair] = volatility
    
    def update_liquidity(self, pair: str, liquidity: float):
        """Update liquidity for a pair"""
        self.pair_liquidity[pair] = liquidity
    
    def calculate_impact(self, pair: str, size: float, price: float) -> float:
        """
        Calculate market impact for a trade
        
        Args:
            pair: Trading pair
            size: Order size in base currency
            price: Current price
            
        Returns:
            price_impact_percentage
        """
        # Get pair-specific impact factor
        pair_factor = self.config["pair_factors"].get(pair, 1.0)
        
        # Get/calculate pair volatility
        volatility = self.pair_volatility.get(pair, 0.01)
        
        # Get/calculate pair liquidity
        liquidity = self.pair_liquidity.get(pair, 1000000)
        
        # Calculate notional value
        notional = size * price
        
        # Calculate impact as a percentage of price
        # Higher volatility = higher impact
        # Higher liquidity = lower impact
        impact_pct = self.config["impact_factor"] * pair_factor * (notional / liquidity) * (volatility * 100)
        
        # Cap impact to reasonable levels
        return min(0.05, max(0, impact_pct))  # Between 0% and 5%
    
    def apply_impact(self, pair: str, size: float, price: float, direction: str) -> float:
        """
        Apply market impact to price
        
        Args:
            pair: Trading pair
            size: Order size
            price: Current price
            direction: 'buy' or 'sell'
            
        Returns:
            impacted_price
        """
        impact_pct = self.calculate_impact(pair, size, price)
        
        if direction.lower() == 'buy':
            # Buy orders push price up
            return price * (1 + impact_pct)
        else:
            # Sell orders push price down
            return price * (1 - impact_pct)

class LatencySimulator:
    """Simulate network and exchange latency for orders"""
    def __init__(self, config_file: str = LATENCY_CONFIG_FILE):
        """Initialize latency simulator"""
        self.config = self._load_config(config_file)
    
    def _load_config(self, config_file: str) -> Dict:
        """Load latency configuration"""
        default_config = {
            "base_latency_ms": 150,  # Base latency in milliseconds
            "latency_variance_ms": 50,  # Variance in latency
            "market_rush_factor": 2.5,  # Factor for high volatility periods
            "peak_hours_factor": 1.5,  # Factor during peak trading hours
            "connection_failure_prob": 0.001,  # Probability of connection failure
            "timeout_prob": 0.002,  # Probability of timeout
            "retry_delay_ms": 500  # Delay before retry in milliseconds
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading latency config: {e}")
        
        # Save default config if not exists
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def get_execution_delay(self, is_market_rush: bool = False, is_peak_hours: bool = False) -> float:
        """
        Get execution delay in seconds
        
        Args:
            is_market_rush: Whether it's during high volatility
            is_peak_hours: Whether it's during peak trading hours
            
        Returns:
            delay in seconds
        """
        base_delay = self.config["base_latency_ms"] / 1000.0  # Convert to seconds
        variance = self.config["latency_variance_ms"] / 1000.0
        
        # Apply factors
        factor = 1.0
        if is_market_rush:
            factor *= self.config["market_rush_factor"]
        if is_peak_hours:
            factor *= self.config["peak_hours_factor"]
        
        # Calculate delay with randomness
        delay = base_delay * factor + random.uniform(-variance, variance)
        
        # Ensure positive delay
        return max(0.01, delay)
    
    def simulate_execution_delay(self, is_market_rush: bool = False, is_peak_hours: bool = False):
        """
        Simulate execution delay by sleeping
        
        Args:
            is_market_rush: Whether it's during high volatility
            is_peak_hours: Whether it's during peak trading hours
        """
        delay = self.get_execution_delay(is_market_rush, is_peak_hours)
        time.sleep(delay)
    
    def should_fail(self) -> bool:
        """Check if connection should fail based on probability"""
        return random.random() < self.config["connection_failure_prob"]
    
    def should_timeout(self) -> bool:
        """Check if request should timeout based on probability"""
        return random.random() < self.config["timeout_prob"]

class AdvancedFeeStructure:
    """Advanced fee structure with tier-based fees"""
    def __init__(self, advanced_fee_config_file: str = ADVANCED_FEE_CONFIG_FILE, 
                tier_based_fees_file: str = TIER_BASED_FEES_FILE):
        """Initialize advanced fee structure"""
        self.fee_config = self._load_fee_config(advanced_fee_config_file)
        self.tier_config = self._load_tier_config(tier_based_fees_file)
        self.trading_volume = 0  # 30-day trading volume
    
    def _load_fee_config(self, config_file: str) -> Dict:
        """Load advanced fee configuration"""
        default_config = {
            "base_maker_fee": 0.0002,  # 0.02%
            "base_taker_fee": 0.0005,  # 0.05%
            "funding_fee_8h": 0.0001,  # 0.01%
            "liquidation_fee": 0.0075,  # 0.75%
            "min_margin_ratio": 0.0125,  # 1.25%
            "maintenance_margin": 0.04,  # 4%
            "withdrawal_fee": {
                "BTC": 0.0005,
                "ETH": 0.005,
                "SOL": 0.01,
                "ADA": 1,
                "DOT": 0.1,
                "LINK": 0.1,
                "USD": 5
            },
            "deposit_fee": {
                "crypto": 0,
                "bank_transfer": 0,
                "credit_card": 0.035  # 3.5%
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading advanced fee config: {e}")
        
        # Save default config if not exists
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _load_tier_config(self, config_file: str) -> Dict:
        """Load tier-based fee configuration"""
        default_config = {
            "tiers": [
                {
                    "min_volume": 0,
                    "max_volume": 50000,
                    "maker_fee": 0.0002,  # 0.02%
                    "taker_fee": 0.0005   # 0.05%
                },
                {
                    "min_volume": 50000,
                    "max_volume": 100000,
                    "maker_fee": 0.00016,  # 0.016%
                    "taker_fee": 0.0004    # 0.04%
                },
                {
                    "min_volume": 100000,
                    "max_volume": 250000,
                    "maker_fee": 0.00014,  # 0.014%
                    "taker_fee": 0.00035   # 0.035%
                },
                {
                    "min_volume": 250000,
                    "max_volume": 500000,
                    "maker_fee": 0.00012,  # 0.012%
                    "taker_fee": 0.0003    # 0.03%
                },
                {
                    "min_volume": 500000,
                    "max_volume": 1000000,
                    "maker_fee": 0.0001,   # 0.01%
                    "taker_fee": 0.00025   # 0.025%
                },
                {
                    "min_volume": 1000000,
                    "max_volume": float('inf'),
                    "maker_fee": 0.00008,  # 0.008%
                    "taker_fee": 0.0002    # 0.02%
                }
            ]
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading tier-based fee config: {e}")
        
        # Save default config if not exists
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def update_trading_volume(self, volume: float):
        """Update 30-day trading volume"""
        self.trading_volume = volume
    
    def get_fee_tier(self) -> Dict:
        """Get current fee tier based on trading volume"""
        for tier in self.tier_config["tiers"]:
            if tier["min_volume"] <= self.trading_volume < tier["max_volume"]:
                return tier
        
        # Default to lowest tier if not found
        return self.tier_config["tiers"][0]
    
    def get_maker_fee(self) -> float:
        """Get maker fee based on current tier"""
        tier = self.get_fee_tier()
        return tier["maker_fee"]
    
    def get_taker_fee(self) -> float:
        """Get taker fee based on current tier"""
        tier = self.get_fee_tier()
        return tier["taker_fee"]
    
    def get_funding_fee(self) -> float:
        """Get funding fee (8h)"""
        return self.fee_config["funding_fee_8h"]
    
    def get_liquidation_fee(self) -> float:
        """Get liquidation fee"""
        return self.fee_config["liquidation_fee"]
    
    def get_withdrawal_fee(self, currency: str) -> float:
        """Get withdrawal fee for a currency"""
        return self.fee_config["withdrawal_fee"].get(currency, 0)
    
    def get_deposit_fee(self, method: str) -> float:
        """Get deposit fee for a method"""
        return self.fee_config["deposit_fee"].get(method, 0)

class FlashCrashSimulator:
    """Simulate flash crashes for stress testing"""
    def __init__(self):
        """Initialize flash crash simulator"""
        self.crash_probability = 0.001  # 0.1% chance per day
        self.recovery_probability = 0.8  # 80% chance of recovery
        self.min_crash_size = 0.05      # 5% minimum drop
        self.max_crash_size = 0.3       # 30% maximum drop
        self.crash_duration_min = 10    # 10 seconds minimum
        self.crash_duration_max = 300   # 5 minutes maximum
        self.active_crashes = {}        # Pair: (end_time, recovery_price, bottom_price)
    
    def should_crash(self, pair: str) -> bool:
        """Check if a flash crash should occur"""
        # Only consider if no active crash for this pair
        if pair in self.active_crashes:
            return False
        
        # Convert daily probability to per-minute probability
        per_minute_prob = self.crash_probability / (24 * 60)
        
        return random.random() < per_minute_prob
    
    def start_crash(self, pair: str, current_price: float) -> Tuple[float, float, datetime.datetime]:
        """
        Start a flash crash
        
        Args:
            pair: Trading pair
            current_price: Current price
            
        Returns:
            (target_price, bottom_price, end_time)
        """
        # Calculate crash size
        crash_size = random.uniform(self.min_crash_size, self.max_crash_size)
        
        # Calculate bottom price
        bottom_price = current_price * (1 - crash_size)
        
        # Determine if crash will recover
        will_recover = random.random() < self.recovery_probability
        
        # Calculate target price (recovery or not)
        if will_recover:
            recovery_factor = random.uniform(0.9, 1.0)  # Recover 90-100%
            target_price = current_price * recovery_factor
        else:
            recovery_factor = random.uniform(0.4, 0.7)  # Recover 40-70%
            target_price = bottom_price + (current_price - bottom_price) * recovery_factor
        
        # Calculate crash duration
        duration_seconds = random.uniform(self.crash_duration_min, self.crash_duration_max)
        end_time = datetime.datetime.now().replace(tzinfo=None) + datetime.timedelta(seconds=duration_seconds)
        
        # Store crash info
        self.active_crashes[pair] = (end_time, target_price, bottom_price)
        
        return target_price, bottom_price, end_time
    
    def is_crashing(self, pair: str) -> bool:
        """Check if a pair is currently experiencing a flash crash"""
        if pair not in self.active_crashes:
            return False
        
        end_time, _, _ = self.active_crashes[pair]
        return datetime.datetime.now().replace(tzinfo=None) < end_time
    
    def get_crash_price(self, pair: str, current_price: float) -> float:
        """
        Get current price during a crash
        
        Args:
            pair: Trading pair
            current_price: Normal current price
            
        Returns:
            crash_adjusted_price
        """
        if not self.is_crashing(pair):
            return current_price
        
        end_time, target_price, bottom_price = self.active_crashes[pair]
        
        # Calculate progress through crash (0 to 1)
        start_time = end_time - datetime.timedelta(seconds=random.uniform(self.crash_duration_min, self.crash_duration_max))
        current_time = datetime.datetime.now().replace(tzinfo=None)
        
        total_duration = (end_time - start_time).total_seconds()
        elapsed = (current_time - start_time).total_seconds()
        progress = min(1, max(0, elapsed / total_duration))
        
        if progress < 0.3:
            # First 30%: Price drops quickly
            drop_progress = progress / 0.3
            return current_price * (1 - drop_progress * (current_price - bottom_price) / current_price)
        elif progress < 0.7:
            # Middle 40%: Price fluctuates around bottom
            fluctuation = random.uniform(-0.02, 0.02)  # ±2% fluctuation
            return bottom_price * (1 + fluctuation)
        else:
            # Last 30%: Price recovers
            recovery_progress = (progress - 0.7) / 0.3
            return bottom_price + recovery_progress * (target_price - bottom_price)
    
    def cleanup_finished_crashes(self):
        """Remove finished crashes from tracking"""
        current_time = datetime.datetime.now().replace(tzinfo=None)
        pairs_to_remove = []
        
        for pair, (end_time, _, _) in self.active_crashes.items():
            if current_time >= end_time:
                pairs_to_remove.append(pair)
        
        for pair in pairs_to_remove:
            del self.active_crashes[pair]

class EnhancedSimulation:
    """Enhanced trading simulation with realistic market behavior"""
    def __init__(self):
        """Initialize enhanced simulation"""
        self.sandbox_trader = RiskAwareSandboxTrader()
        self.order_books = {}
        self.market_impact = MarketImpactModel()
        self.latency = LatencySimulator()
        self.fees = AdvancedFeeStructure()
        self.flash_crashes = FlashCrashSimulator()
        
        # Initialize directories
        os.makedirs(CONFIG_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MARKET_DATA_DIR, exist_ok=True)
        
        # Load initial order books
        self._load_order_books()
    
    def _load_order_books(self):
        """Load order books from file or initialize new ones"""
        # Default initial prices
        initial_prices = {
            "SOL/USD": 160.0,
            "BTC/USD": 70000.0,
            "ETH/USD": 3500.0,
            "ADA/USD": 0.45,
            "DOT/USD": 7.0,
            "LINK/USD": 18.0
        }
        
        if os.path.exists(ORDER_BOOK_FILE):
            try:
                with open(ORDER_BOOK_FILE, 'r') as f:
                    order_books_data = json.load(f)
                
                for pair, data in order_books_data.items():
                    self.order_books[pair] = OrderBook.from_dict(data)
            except Exception as e:
                logger.error(f"Error loading order books: {e}")
                self._initialize_order_books(initial_prices)
        else:
            self._initialize_order_books(initial_prices)
    
    def _initialize_order_books(self, initial_prices: Dict[str, float]):
        """Initialize order books with initial prices"""
        for pair, price in initial_prices.items():
            self.order_books[pair] = OrderBook(pair, price)
        
        # Save order books
        self._save_order_books()
    
    def _save_order_books(self):
        """Save order books to file"""
        order_books_data = {}
        for pair, order_book in self.order_books.items():
            order_books_data[pair] = order_book.to_dict()
        
        with open(ORDER_BOOK_FILE, 'w') as f:
            json.dump(order_books_data, f, indent=2)
    
    def update_market_data(self, pair_price_updates: Optional[Dict[str, float]] = None):
        """
        Update market data including order books
        
        Args:
            pair_price_updates: Dictionary of {pair: new_price} (optional)
        """
        # Handle flash crashes first
        self.flash_crashes.cleanup_finished_crashes()
        
        # Update pricing data for each pair
        for pair, order_book in self.order_books.items():
            new_price = None
            
            # Check if we have a specific price update
            if pair_price_updates and pair in pair_price_updates:
                new_price = pair_price_updates[pair]
            
            # Check if pair is experiencing a flash crash
            if self.flash_crashes.is_crashing(pair):
                # Get crash-adjusted price
                normal_price = new_price or order_book.base_price
                new_price = self.flash_crashes.get_crash_price(pair, normal_price)
            
            # Check if a new flash crash should start
            elif self.flash_crashes.should_crash(pair):
                current_price = new_price or order_book.base_price
                target_price, bottom_price, end_time = self.flash_crashes.start_crash(pair, current_price)
                logger.warning(f"Flash crash started for {pair}: {current_price} -> {bottom_price} -> {target_price}")
                
                # Don't update price here, it will be handled in next cycle
            
            # Update the order book with new price
            if new_price is not None:
                # Add some small random noise to the price
                noise = random.uniform(-0.0005, 0.0005)  # ±0.05% noise
                noisy_price = new_price * (1 + noise)
                order_book.update(noisy_price)
            else:
                # Just update with current price and random walks
                order_book.update()
        
        # Save updated order books
        self._save_order_books()
    
    def get_execution_price(self, pair: str, size: float, direction: str) -> Tuple[float, float]:
        """
        Get execution price with slippage and market impact
        
        Args:
            pair: Trading pair
            size: Order size
            direction: 'buy' or 'sell'
            
        Returns:
            (execution_price, filled_size)
        """
        # Get order book
        if pair not in self.order_books:
            logger.warning(f"Order book not found for {pair}, creating new one")
            # Use a default price if not available
            default_price = 100.0
            if "BTC" in pair:
                default_price = 70000.0
            elif "ETH" in pair:
                default_price = 3500.0
            elif "SOL" in pair:
                default_price = 160.0
            
            self.order_books[pair] = OrderBook(pair, default_price)
        
        order_book = self.order_books[pair]
        
        # Apply latency simulation
        is_peak_hours = self._is_peak_hours()
        is_market_rush = self._is_market_rush(pair)
        self.latency.simulate_execution_delay(is_market_rush, is_peak_hours)
        
        # Check for connection failures
        if self.latency.should_fail():
            logger.warning(f"Connection failure simulated for {pair}")
            return 0, 0
        
        # Check for timeouts
        if self.latency.should_timeout():
            logger.warning(f"Timeout simulated for {pair}")
            return 0, 0
        
        # Get market price
        if direction.lower() in ['buy', 'long']:
            price, filled_size = order_book.get_market_buy_price(size)
        else:
            price, filled_size = order_book.get_market_sell_price(size)
        
        # Apply market impact
        impacted_price = self.market_impact.apply_impact(pair, size, price, direction)
        
        return impacted_price, filled_size
    
    def _is_peak_hours(self) -> bool:
        """Check if current time is during peak trading hours"""
        # Consider peak hours to be 9:30 AM - 4:00 PM ET (13:30 - 20:00 UTC)
        current_hour = datetime.datetime.now().hour
        return 13 <= current_hour < 20
    
    def _is_market_rush(self, pair: str) -> bool:
        """Check if the market is in a rush (high volatility) period"""
        # In a real implementation, this would analyze recent price movements
        # For simulation, we'll just return a random value with low probability
        return random.random() < 0.1
    
    def execute_trade(self, pair: str, direction: str, size: float, 
                     entry_price: float, leverage: float, strategy: str,
                     confidence: float = 0.7) -> Tuple[bool, Dict]:
        """
        Execute a trade with realistic simulation
        
        Args:
            pair: Trading pair
            direction: 'Long' or 'Short'
            size: Size of position
            entry_price: Entry price
            leverage: Leverage to use
            strategy: Trading strategy
            confidence: Confidence level (0.0-1.0)
            
        Returns:
            (success, position)
        """
        # Get execution price with slippage and market impact
        actual_price, filled_size = self.get_execution_price(pair, size, direction)
        
        # Handle partial fills
        if filled_size < size * 0.95:  # Less than 95% filled
            logger.warning(f"Partial fill for {pair}: {filled_size}/{size} ({filled_size/size:.2%})")
            
            if filled_size < size * 0.5:  # Less than 50% filled
                logger.warning(f"Fill too small, cancelling order for {pair}")
                return False, {}
            
            # Adjust size to what was filled
            size = filled_size
        
        # Get tier-based fees
        taker_fee_rate = self.fees.get_taker_fee()
        
        # Calculate liquidation price
        maintenance_margin = self.sandbox_trader.fee_config["maintenance_margin"]
        if direction == "Long":
            liquidation_price = actual_price * (1 - (1 / leverage) + maintenance_margin)
        else:  # Short
            liquidation_price = actual_price * (1 + (1 / leverage) - maintenance_margin)
        
        # Calculate stop loss and take profit
        stop_loss_pct = 0.04  # 4% max loss
        take_profit_pct = stop_loss_pct * 2.5  # 10% take profit (risk-reward 1:2.5)
        
        if confidence > 0.90:
            take_profit_pct = stop_loss_pct * 4.0  # Higher target for high confidence
        elif confidence > 0.80:
            take_profit_pct = stop_loss_pct * 3.0  # Adjusted target for medium-high confidence
        
        if direction == "Long":
            stop_loss = actual_price * (1 - stop_loss_pct / leverage)
            take_profit = actual_price * (1 + take_profit_pct / leverage)
        else:  # Short
            stop_loss = actual_price * (1 + stop_loss_pct / leverage)
            take_profit = actual_price * (1 - take_profit_pct / leverage)
        
        # Call sandbox trader to execute the trade
        success, position = self.sandbox_trader.open_position(
            pair=pair,
            direction=direction,
            size=size,
            entry_price=actual_price,
            leverage=leverage,
            strategy=strategy,
            confidence=confidence
        )
        
        if success:
            # Add liquidation price to position
            position["liquidation_price"] = liquidation_price
            # Update position with enhanced stop loss and take profit
            position["stop_loss"] = stop_loss
            position["take_profit"] = take_profit
            # Store actual fees paid
            position["taker_fee_rate"] = taker_fee_rate
            position["taker_fee_paid"] = size * actual_price * taker_fee_rate
        
        return success, position
    
    def close_trade(self, pair: str, strategy: str, exit_price: float, 
                   close_reason: str = "Manual") -> Tuple[bool, Dict]:
        """
        Close a trade with realistic simulation
        
        Args:
            pair: Trading pair
            strategy: Trading strategy
            exit_price: Exit price
            close_reason: Reason for closing
            
        Returns:
            (success, trade)
        """
        # Find position
        position = None
        for pos in self.sandbox_trader.positions:
            if pos["pair"] == pair and pos["strategy"] == strategy:
                position = pos
                break
        
        if not position:
            logger.warning(f"Position not found for {pair} with {strategy} strategy")
            return False, {}
        
        # Get position size and direction
        size = position["size"]
        direction = position["direction"]
        
        # Get execution price with slippage and market impact
        # For closing, direction is opposite of position
        close_direction = "sell" if direction == "Long" else "buy"
        actual_price, filled_size = self.get_execution_price(pair, size, close_direction)
        
        # Handle partial fills
        if filled_size < size * 0.95:  # Less than 95% filled
            logger.warning(f"Partial fill on close for {pair}: {filled_size}/{size} ({filled_size/size:.2%})")
            
            if filled_size < size * 0.5:  # Less than 50% filled
                logger.warning(f"Fill too small, retrying later for {pair}")
                return False, {}
            
            # Adjust size to what was filled
            size = filled_size
        
        # Get tier-based fees
        taker_fee_rate = self.fees.get_taker_fee()
        
        # Call sandbox trader to close the trade
        success, trade = self.sandbox_trader.close_position(
            pair=pair,
            strategy=strategy,
            exit_price=actual_price,
            close_reason=close_reason
        )
        
        if success:
            # Store actual fees paid
            trade["taker_fee_rate"] = taker_fee_rate
            trade["taker_fee_paid"] = size * actual_price * taker_fee_rate
        
        return success, trade
    
    def update_positions(self):
        """Update positions with current prices and check for liquidations"""
        # Get current prices from order books
        price_updates = {}
        for pair, order_book in self.order_books.items():
            price_updates[pair] = order_book.get_mid_price()
        
        # Update positions with current prices
        self.sandbox_trader.update_position_prices(price_updates)
        
        # Apply funding fees every 8 hours
        current_time = datetime.datetime.now().replace(tzinfo=None)
        last_funding_time = getattr(self, '_last_funding_time', None)
        
        if last_funding_time is None:
            self._last_funding_time = current_time
        elif (current_time - last_funding_time).total_seconds() >= 28800:  # 8 hours
            self.sandbox_trader.apply_funding_fees()
            self._last_funding_time = current_time

def simulate_enhanced_trading():
    """Run enhanced trading simulation"""
    logger.info("Starting enhanced trading simulation...")
    
    # Initialize enhanced simulation
    simulation = EnhancedSimulation()
    
    # Sample initial prices
    initial_prices = {
        "SOL/USD": 160.0,
        "BTC/USD": 70000.0,
        "ETH/USD": 3500.0,
        "ADA/USD": 0.45,
        "DOT/USD": 7.0,
        "LINK/USD": 18.0
    }
    
    # Update market data with initial prices
    simulation.update_market_data(initial_prices)
    
    # Print current positions
    logger.info(f"Current positions: {len(simulation.sandbox_trader.positions)}")
    
    # Print portfolio value
    portfolio_value = simulation.sandbox_trader.get_current_portfolio_value()
    logger.info(f"Current portfolio value: ${portfolio_value:.2f}")
    
    # Run a trading cycle
    logger.info("Running trading cycle...")
    
    # Example: Open a position
    if len(simulation.sandbox_trader.positions) < 6:  # Maximum 6 positions
        # Select a random pair that doesn't have an open position
        available_pairs = [
            pair for pair in initial_prices.keys()
            if not any(pos["pair"] == pair for pos in simulation.sandbox_trader.positions)
        ]
        
        if available_pairs:
            pair = random.choice(available_pairs)
            direction = "Long" if random.random() > 0.3 else "Short"  # 70% Long bias
            confidence = random.uniform(0.65, 0.95)
            
            # Get current price
            current_price = simulation.order_books[pair].get_mid_price()
            
            # Calculate leverage based on confidence
            base_leverage = 20.0
            max_leverage = 125.0
            leverage = base_leverage + (confidence - 0.65) * (max_leverage - base_leverage) / 0.3
            leverage = min(max_leverage, max(base_leverage, leverage))
            
            # Calculate position size
            portfolio_value = simulation.sandbox_trader.get_current_portfolio_value()
            risk_percentage = 0.2  # 20% risk per trade
            margin = portfolio_value * risk_percentage
            notional_value = margin * leverage
            size = notional_value / current_price
            
            # Strategy selection
            strategy = random.choice(["Adaptive", "ARIMA"])
            
            logger.info(f"Opening {direction} position for {pair} with {strategy} strategy")
            logger.info(f"  Price: ${current_price:.2f}, Size: {size:.6f}, Leverage: {leverage:.1f}x")
            logger.info(f"  Confidence: {confidence:.2f}")
            
            success, position = simulation.execute_trade(
                pair=pair,
                direction=direction,
                size=size,
                entry_price=current_price,
                leverage=leverage,
                strategy=strategy,
                confidence=confidence
            )
            
            if success:
                logger.info(f"Successfully opened position for {pair}")
                
                # Look for take profit opportunities on existing positions
                for pos in simulation.sandbox_trader.positions:
                    # 10% chance to close a position for testing
                    if random.random() < 0.1:
                        close_pair = pos["pair"]
                        close_strategy = pos["strategy"]
                        current_price = simulation.order_books[close_pair].get_mid_price()
                        
                        logger.info(f"Closing position for {close_pair} with {close_strategy} strategy")
                        success, trade = simulation.close_trade(
                            pair=close_pair,
                            strategy=close_strategy,
                            exit_price=current_price,
                            close_reason="Manual"
                        )
                        
                        if success:
                            logger.info(f"Successfully closed position for {close_pair}")
                            logger.info(f"  P/L: {trade['pnl_percentage']:.2%}")
            else:
                logger.warning(f"Failed to open position for {pair}")
    
    # Update positions with current prices
    simulation.update_positions()
    
    # Print current positions again
    logger.info(f"Current positions after cycle: {len(simulation.sandbox_trader.positions)}")
    for pos in simulation.sandbox_trader.positions:
        unrealized_pnl = pos["unrealized_pnl"] * 100
        logger.info(f"  {pos['pair']} {pos['direction']} {pos['leverage']}x: {unrealized_pnl:.2f}%")
    
    # Print portfolio value again
    portfolio_value = simulation.sandbox_trader.get_current_portfolio_value()
    logger.info(f"Portfolio value after cycle: ${portfolio_value:.2f}")

if __name__ == "__main__":
    simulate_enhanced_trading()