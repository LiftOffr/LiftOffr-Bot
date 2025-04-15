#!/usr/bin/env python3
"""
Risk-Aware Sandbox Trader

This module enhances the sandbox trading simulation to accurately account for:
1. Trading fees (maker/taker)
2. Funding fees for leveraged positions
3. Liquidation risks based on maintenance margin requirements
4. Slippage based on trade size and market liquidity
5. Realistic fill execution

It ensures the sandbox simulation closely matches real-world trading conditions.
"""

import os
import json
import time
import logging
import datetime
import math
import random
from typing import Dict, List, Tuple, Optional, Union, Any

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
FEE_CONFIG_FILE = f"{CONFIG_DIR}/fee_config.json"
PORTFOLIO_FILE = f"{DATA_DIR}/sandbox_portfolio_history.json"
POSITIONS_FILE = f"{DATA_DIR}/sandbox_positions.json"
TRADES_FILE = f"{DATA_DIR}/sandbox_trades.json"
RISK_METRICS_FILE = f"{DATA_DIR}/risk_metrics.json"
DEFAULT_STARTING_CAPITAL = 20000.0

class RiskAwareSandboxTrader:
    """Risk-aware sandbox trading simulation that accounts for realistic trading conditions"""
    
    def __init__(self):
        """Initialize the risk-aware sandbox trader"""
        self.fee_config = self._load_fee_config()
        self.portfolio = self._load_portfolio()
        self.positions = self._load_positions()
        self.trades = self._load_trades()
        self.risk_metrics = self._load_risk_metrics()
        
        # Ensure we have starting capital
        if not self.portfolio:
            self._initialize_portfolio()
    
    def _load_fee_config(self) -> Dict:
        """Load fee configuration"""
        try:
            if os.path.exists(FEE_CONFIG_FILE):
                with open(FEE_CONFIG_FILE, 'r') as f:
                    return json.load(f)
            else:
                # Default fee config
                config = {
                    "maker_fee": 0.0002,  # 0.02%
                    "taker_fee": 0.0005,  # 0.05%
                    "funding_fee_8h": 0.0001,  # 0.01% per 8 hours
                    "liquidation_fee": 0.0075,  # 0.75%
                    "min_margin_ratio": 0.0125,  # 1.25%
                    "maintenance_margin": 0.04,  # 4%
                    "slippage": {
                        "low_volume": 0.001,  # 0.1%
                        "medium_volume": 0.0003,  # 0.03%
                        "high_volume": 0.0001  # 0.01%
                    },
                    "volume_thresholds": {
                        "low": 100000,  # $100k
                        "medium": 500000  # $500k
                    }
                }
                
                # Save default config
                os.makedirs(CONFIG_DIR, exist_ok=True)
                with open(FEE_CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=2)
                
                return config
        except Exception as e:
            logger.error(f"Error loading fee config: {e}")
            return {
                "maker_fee": 0.0002,
                "taker_fee": 0.0005,
                "funding_fee_8h": 0.0001,
                "liquidation_fee": 0.0075,
                "min_margin_ratio": 0.0125,
                "maintenance_margin": 0.04,
                "slippage": {"low_volume": 0.001, "medium_volume": 0.0003, "high_volume": 0.0001},
                "volume_thresholds": {"low": 100000, "medium": 500000}
            }
    
    def _load_portfolio(self) -> List:
        """Load portfolio history"""
        try:
            if os.path.exists(PORTFOLIO_FILE):
                with open(PORTFOLIO_FILE, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            return []
    
    def _load_positions(self) -> List:
        """Load open positions"""
        try:
            if os.path.exists(POSITIONS_FILE):
                with open(POSITIONS_FILE, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading positions: {e}")
            return []
    
    def _load_trades(self) -> List:
        """Load trade history"""
        try:
            if os.path.exists(TRADES_FILE):
                with open(TRADES_FILE, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading trades: {e}")
            return []
    
    def _load_risk_metrics(self) -> Dict:
        """Load risk metrics"""
        try:
            if os.path.exists(RISK_METRICS_FILE):
                with open(RISK_METRICS_FILE, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading risk metrics: {e}")
            return {}
    
    def _initialize_portfolio(self):
        """Initialize portfolio with starting capital"""
        self.portfolio = [{
            "timestamp": datetime.datetime.now().isoformat(),
            "portfolio_value": DEFAULT_STARTING_CAPITAL
        }]
        self._save_portfolio()
    
    def _save_portfolio(self):
        """Save portfolio history"""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(self.portfolio, f, indent=2)
    
    def _save_positions(self):
        """Save open positions"""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(POSITIONS_FILE, 'w') as f:
            json.dump(self.positions, f, indent=2)
    
    def _save_trades(self):
        """Save trade history"""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(TRADES_FILE, 'w') as f:
            json.dump(self.trades, f, indent=2)
    
    def _save_risk_metrics(self):
        """Save risk metrics"""
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(RISK_METRICS_FILE, 'w') as f:
            json.dump(self.risk_metrics, f, indent=2)
    
    def get_current_portfolio_value(self) -> float:
        """Get current portfolio value"""
        if not self.portfolio:
            return DEFAULT_STARTING_CAPITAL
        return self.portfolio[-1]["portfolio_value"]
    
    def calculate_liquidation_price(
        self, 
        entry_price: float, 
        leverage: float, 
        direction: str
    ) -> float:
        """
        Calculate liquidation price based on leverage and direction
        
        Args:
            entry_price: Entry price
            leverage: Leverage used
            direction: 'Long' or 'Short'
            
        Returns:
            Liquidation price
        """
        maintenance_margin = self.fee_config["maintenance_margin"]
        
        if direction == "Long":
            liquidation_price = entry_price * (1 - (1 / leverage) + maintenance_margin)
        else:  # Short
            liquidation_price = entry_price * (1 + (1 / leverage) - maintenance_margin)
        
        return liquidation_price
    
    def calculate_slippage(self, notional_value: float) -> float:
        """
        Calculate slippage based on trade size
        
        Args:
            notional_value: Notional value of trade
            
        Returns:
            Slippage percentage
        """
        if notional_value >= self.fee_config["volume_thresholds"]["medium"]:
            return self.fee_config["slippage"]["high_volume"]
        elif notional_value >= self.fee_config["volume_thresholds"]["low"]:
            return self.fee_config["slippage"]["medium_volume"]
        else:
            return self.fee_config["slippage"]["low_volume"]
    
    def open_position(
        self,
        pair: str,
        direction: str,
        size: float,
        entry_price: float,
        leverage: float,
        strategy: str,
        confidence: float = 0.7
    ) -> Tuple[bool, Dict]:
        """
        Open a trading position with realistic conditions
        
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
        # Check if we already have a position for this pair and strategy
        for pos in self.positions:
            if pos["pair"] == pair and pos["strategy"] == strategy:
                logger.warning(f"Already have a position for {pair} with {strategy} strategy")
                return False, {}
        
        # Calculate notional value and check position size
        notional_value = size * entry_price
        
        # Calculate margin required
        margin = notional_value / leverage
        
        # Get current portfolio value
        portfolio_value = self.get_current_portfolio_value()
        
        # Check if we have enough margin
        if margin > portfolio_value * 0.20:  # Max 20% per trade
            logger.warning(f"Margin required ({margin}) exceeds 20% of portfolio ({portfolio_value})")
            return False, {}
        
        # Apply slippage to entry price
        slippage = self.calculate_slippage(notional_value)
        if direction == "Long":
            actual_entry_price = entry_price * (1 + slippage)
        else:  # Short
            actual_entry_price = entry_price * (1 - slippage)
        
        # Calculate fees
        taker_fee = notional_value * self.fee_config["taker_fee"]
        
        # Calculate stop loss and take profit
        stop_loss_pct = 0.04  # 4% max loss
        take_profit_pct = stop_loss_pct * 2.5  # 10% take profit (risk-reward 1:2.5)
        
        if confidence > 0.90:
            take_profit_pct = stop_loss_pct * 4.0  # Higher target for high confidence
        elif confidence > 0.80:
            take_profit_pct = stop_loss_pct * 3.0  # Adjusted target for medium-high confidence
        
        if direction == "Long":
            stop_loss = actual_entry_price * (1 - stop_loss_pct / leverage)
            take_profit = actual_entry_price * (1 + take_profit_pct / leverage)
        else:  # Short
            stop_loss = actual_entry_price * (1 + stop_loss_pct / leverage)
            take_profit = actual_entry_price * (1 - take_profit_pct / leverage)
        
        # Calculate liquidation price
        liquidation_price = self.calculate_liquidation_price(actual_entry_price, leverage, direction)
        
        # Create position
        position = {
            "pair": pair,
            "strategy": strategy,
            "direction": direction,
            "entry_price": actual_entry_price,
            "current_price": actual_entry_price,
            "size": size,
            "leverage": leverage,
            "margin": margin,
            "liquidation_price": liquidation_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "unrealized_pnl": 0.0,
            "entry_time": datetime.datetime.now().isoformat(),
            "duration": "0h 0m",
            "confidence": confidence
        }
        
        # Create trade entry record
        trade = {
            "timestamp": datetime.datetime.now().isoformat(),
            "pair": pair,
            "strategy": strategy,
            "type": "Entry",
            "direction": direction,
            "entry_price": actual_entry_price,
            "exit_price": 0,
            "size": size,
            "leverage": leverage,
            "fees": taker_fee,
            "slippage": slippage * entry_price * size,
            "pnl_percentage": 0
        }
        
        # Update portfolio (deduct fees)
        new_portfolio_value = portfolio_value - taker_fee
        self.portfolio.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "portfolio_value": new_portfolio_value
        })
        
        # Add position and trade
        self.positions.append(position)
        self.trades.append(trade)
        
        # Save all data
        self._save_portfolio()
        self._save_positions()
        self._save_trades()
        
        logger.info(f"Opened {direction} position for {pair} at {actual_entry_price} with {leverage}x leverage")
        return True, position
    
    def close_position(
        self,
        pair: str,
        strategy: str,
        exit_price: float,
        close_reason: str = "Manual"
    ) -> Tuple[bool, Dict]:
        """
        Close a trading position with realistic conditions
        
        Args:
            pair: Trading pair
            strategy: Trading strategy
            exit_price: Exit price
            close_reason: Reason for closing
            
        Returns:
            (success, trade)
        """
        # Find the position
        position_idx = None
        position = None
        
        for idx, pos in enumerate(self.positions):
            if pos["pair"] == pair and pos["strategy"] == strategy:
                position_idx = idx
                position = pos
                break
        
        if position is None:
            logger.warning(f"No position found for {pair} with {strategy} strategy")
            return False, {}
        
        # Calculate notional value
        notional_value = position["size"] * exit_price
        
        # Apply slippage to exit price
        slippage = self.calculate_slippage(notional_value)
        if position["direction"] == "Long":
            actual_exit_price = exit_price * (1 - slippage)
        else:  # Short
            actual_exit_price = exit_price * (1 + slippage)
        
        # Calculate fees
        taker_fee = notional_value * self.fee_config["taker_fee"]
        
        # Calculate PnL
        entry_price = position["entry_price"]
        if position["direction"] == "Long":
            pnl_percentage = (actual_exit_price / entry_price) - 1
        else:  # Short
            pnl_percentage = 1 - (actual_exit_price / entry_price)
        
        # Apply leverage to PnL
        leveraged_pnl_percentage = pnl_percentage * position["leverage"]
        
        # Calculate PnL amount
        pnl_amount = position["margin"] * leveraged_pnl_percentage
        
        # Create trade exit record
        trade = {
            "timestamp": datetime.datetime.now().isoformat(),
            "pair": pair,
            "strategy": strategy,
            "type": "Exit",
            "direction": position["direction"],
            "entry_price": entry_price,
            "exit_price": actual_exit_price,
            "size": position["size"],
            "leverage": position["leverage"],
            "fees": taker_fee,
            "slippage": slippage * exit_price * position["size"],
            "pnl_percentage": leveraged_pnl_percentage,
            "pnl_amount": pnl_amount,
            "close_reason": close_reason
        }
        
        # Calculate duration
        entry_time = datetime.datetime.fromisoformat(position["entry_time"])
        exit_time = datetime.datetime.now()
        duration_seconds = (exit_time - entry_time).total_seconds()
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        duration = f"{hours}h {minutes}m"
        trade["duration"] = duration
        
        # Get current portfolio value
        portfolio_value = self.get_current_portfolio_value()
        
        # Calculate new portfolio value (add margin + PnL - fees)
        new_portfolio_value = portfolio_value + position["margin"] + pnl_amount - taker_fee
        
        # Add trade and update portfolio
        self.trades.append(trade)
        self.portfolio.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "portfolio_value": new_portfolio_value
        })
        
        # Remove the position
        if position_idx is not None:
            del self.positions[position_idx]
        
        # Update risk metrics
        self._update_risk_metrics()
        
        # Save all data
        self._save_trades()
        self._save_positions()
        self._save_portfolio()
        self._save_risk_metrics()
        
        logger.info(f"Closed {position['direction']} position for {pair} at {actual_exit_price} with {leveraged_pnl_percentage:.2%} PnL")
        return True, trade
    
    def update_position_prices(self, price_updates: Dict[str, float]):
        """
        Update current prices of open positions
        
        Args:
            price_updates: Dictionary of {pair: current_price}
        """
        positions_updated = False
        
        for position in self.positions:
            pair = position["pair"]
            if pair in price_updates:
                position["current_price"] = price_updates[pair]
                
                # Recalculate unrealized PnL
                entry_price = position["entry_price"]
                current_price = position["current_price"]
                
                if position["direction"] == "Long":
                    pnl_percentage = (current_price / entry_price) - 1
                else:  # Short
                    pnl_percentage = 1 - (current_price / entry_price)
                
                # Apply leverage to PnL
                leveraged_pnl_percentage = pnl_percentage * position["leverage"]
                position["unrealized_pnl"] = leveraged_pnl_percentage
                
                # Update duration
                entry_time = datetime.datetime.fromisoformat(position["entry_time"])
                current_time = datetime.datetime.now()
                duration_seconds = (current_time - entry_time).total_seconds()
                hours = int(duration_seconds // 3600)
                minutes = int((duration_seconds % 3600) // 60)
                position["duration"] = f"{hours}h {minutes}m"
                
                # Check for liquidation
                if position["direction"] == "Long" and current_price <= position["liquidation_price"]:
                    logger.warning(f"Position {pair} liquidated at {current_price}")
                    self.handle_liquidation(position)
                    positions_updated = True
                elif position["direction"] == "Short" and current_price >= position["liquidation_price"]:
                    logger.warning(f"Position {pair} liquidated at {current_price}")
                    self.handle_liquidation(position)
                    positions_updated = True
                # Check for stop loss
                elif position["direction"] == "Long" and current_price <= position["stop_loss"]:
                    logger.info(f"Stop loss triggered for {pair} at {current_price}")
                    self.close_position(pair, position["strategy"], current_price, "Stop Loss")
                    positions_updated = True
                elif position["direction"] == "Short" and current_price >= position["stop_loss"]:
                    logger.info(f"Stop loss triggered for {pair} at {current_price}")
                    self.close_position(pair, position["strategy"], current_price, "Stop Loss")
                    positions_updated = True
                # Check for take profit
                elif position["direction"] == "Long" and current_price >= position["take_profit"]:
                    logger.info(f"Take profit triggered for {pair} at {current_price}")
                    self.close_position(pair, position["strategy"], current_price, "Take Profit")
                    positions_updated = True
                elif position["direction"] == "Short" and current_price <= position["take_profit"]:
                    logger.info(f"Take profit triggered for {pair} at {current_price}")
                    self.close_position(pair, position["strategy"], current_price, "Take Profit")
                    positions_updated = True
                else:
                    positions_updated = True
        
        if positions_updated:
            self._save_positions()
    
    def handle_liquidation(self, position: Dict):
        """
        Handle liquidation of a position
        
        Args:
            position: Position being liquidated
        """
        pair = position["pair"]
        strategy = position["strategy"]
        
        # Calculate liquidation fee
        liquidation_fee = position["margin"] * self.fee_config["liquidation_fee"]
        
        # Create trade record for liquidation
        trade = {
            "timestamp": datetime.datetime.now().isoformat(),
            "pair": pair,
            "strategy": strategy,
            "type": "Liquidation",
            "direction": position["direction"],
            "entry_price": position["entry_price"],
            "exit_price": position["liquidation_price"],
            "size": position["size"],
            "leverage": position["leverage"],
            "fees": liquidation_fee,
            "slippage": 0,
            "pnl_percentage": -1.0,  # 100% loss
            "pnl_amount": -position["margin"],
            "close_reason": "Liquidation"
        }
        
        # Calculate duration
        entry_time = datetime.datetime.fromisoformat(position["entry_time"])
        exit_time = datetime.datetime.now()
        duration_seconds = (exit_time - entry_time).total_seconds()
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        trade["duration"] = f"{hours}h {minutes}m"
        
        # Get current portfolio value
        portfolio_value = self.get_current_portfolio_value()
        
        # Calculate new portfolio value (loss of margin plus liquidation fee)
        new_portfolio_value = portfolio_value - position["margin"] - liquidation_fee
        
        # Add trade and update portfolio
        self.trades.append(trade)
        self.portfolio.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "portfolio_value": new_portfolio_value
        })
        
        # Remove the position
        self.positions = [p for p in self.positions if not (p["pair"] == pair and p["strategy"] == strategy)]
        
        # Update risk metrics
        self._update_risk_metrics()
        
        # Save all data
        self._save_trades()
        self._save_positions()
        self._save_portfolio()
        self._save_risk_metrics()
        
        logger.warning(f"Liquidated {position['direction']} position for {pair} with 100% loss")
    
    def apply_funding_fees(self):
        """Apply funding fees to open positions (called every 8 hours)"""
        if not self.positions:
            return
        
        funding_fee_rate = self.fee_config["funding_fee_8h"]
        total_funding_fees = 0
        
        for position in self.positions:
            notional_value = position["size"] * position["current_price"]
            funding_fee = notional_value * funding_fee_rate
            total_funding_fees += funding_fee
        
        if total_funding_fees > 0:
            # Get current portfolio value
            portfolio_value = self.get_current_portfolio_value()
            
            # Apply funding fees
            new_portfolio_value = portfolio_value - total_funding_fees
            
            # Update portfolio
            self.portfolio.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "portfolio_value": new_portfolio_value
            })
            
            # Save portfolio
            self._save_portfolio()
            
            logger.info(f"Applied {total_funding_fees:.2f} in funding fees")
    
    def _update_risk_metrics(self):
        """Update risk metrics based on trade history and portfolio"""
        if not self.trades or not self.portfolio:
            return
        
        # Calculate max drawdown
        max_value = 0
        max_drawdown = 0
        for point in self.portfolio:
            value = point["portfolio_value"]
            max_value = max(max_value, value)
            drawdown = (max_value - value) / max_value if max_value > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Extract completed trades (Entry and Exit pairs)
        completed_trades = []
        trade_map = {}
        
        for trade in self.trades:
            if trade["type"] == "Entry":
                key = f"{trade['pair']}_{trade['strategy']}_{trade['timestamp']}"
                trade_map[key] = trade
            elif trade["type"] == "Exit" or trade["type"] == "Liquidation":
                # Find the corresponding entry
                for key, entry_trade in list(trade_map.items()):
                    if (entry_trade["pair"] == trade["pair"] and 
                        entry_trade["strategy"] == trade["strategy"] and
                        entry_trade["direction"] == trade["direction"]):
                        completed_trades.append({
                            "entry": entry_trade,
                            "exit": trade
                        })
                        del trade_map[key]
                        break
        
        # Calculate win rate and profit factor
        wins = 0
        losses = 0
        total_profit = 0
        total_loss = 0
        returns = []
        
        for completed in completed_trades:
            pnl = completed["exit"].get("pnl_percentage", 0)
            returns.append(pnl)
            
            if pnl > 0:
                wins += 1
                total_profit += pnl
            else:
                losses += 1
                total_loss += abs(pnl)
        
        num_trades = len(completed_trades)
        win_rate = wins / num_trades if num_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate average win and loss
        avg_win = total_profit / wins if wins > 0 else 0
        avg_loss = total_loss / losses if losses > 0 else 0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Calculate mean return
        mean_return = sum(returns) / len(returns) if returns else 0
        
        # Calculate Sharpe ratio (simplified)
        if len(returns) > 1:
            std_return = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate Sortino ratio (simplified)
        if len(returns) > 1:
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_deviation = (sum(r ** 2 for r in downside_returns) / len(returns)) ** 0.5
                sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = 0
        
        # Calculate value at risk (simplified 95% VaR)
        if len(returns) > 5:
            returns.sort()
            var_index = int(0.05 * len(returns))
            value_at_risk = abs(returns[var_index])
        else:
            value_at_risk = 0.05  # Default 5%
        
        # Calculate Kelly criterion
        if win_rate > 0 and avg_loss > 0:
            kelly_criterion = (win_rate * avg_win / avg_loss - (1 - win_rate)) / (avg_win / avg_loss)
            kelly_criterion = max(0, min(kelly_criterion, 0.5))  # Cap at 50%
        else:
            kelly_criterion = 0
        
        # Calculate optimal leverage based on Kelly and risk
        optimal_leverage = kelly_criterion * 50  # Scale to realistic leverage range
        optimal_leverage = max(1, min(optimal_leverage, 25))  # Cap between 1x and 25x
        
        # Determine current risk level
        current_risk_level = "Low"
        if max_drawdown > 0.15 or value_at_risk > 0.08:
            current_risk_level = "High"
        elif max_drawdown > 0.08 or value_at_risk > 0.05:
            current_risk_level = "Medium"
        
        # Update risk metrics
        self.risk_metrics = {
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win_loss_ratio": avg_win_loss_ratio,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "value_at_risk": value_at_risk,
            "kelly_criterion": kelly_criterion,
            "optimal_leverage": optimal_leverage,
            "current_risk_level": current_risk_level,
            "updated_at": datetime.datetime.now().isoformat()
        }
        
        # Save risk metrics
        self._save_risk_metrics()

# Test the module
if __name__ == "__main__":
    trader = RiskAwareSandboxTrader()
    
    # Print current status
    print(f"Portfolio Value: ${trader.get_current_portfolio_value():.2f}")
    print(f"Open Positions: {len(trader.positions)}")
    print(f"Completed Trades: {len([t for t in trader.trades if t['type'] in ['Exit', 'Liquidation']])}")
    
    # Apply funding fees (simulation)
    trader.apply_funding_fees()
    
    # Update risk metrics
    trader._update_risk_metrics()
    
    print("Risk metrics updated")