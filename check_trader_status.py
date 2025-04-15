#!/usr/bin/env python3
"""
Check Trader Status

This script checks the status of the risk-aware sandbox trader and its active positions.
It validates that the trader properly simulates all trading conditions including fees,
liquidation risks, and realistic trading constraints.
"""

import os
import json
import logging
import datetime
from typing import Dict, List

from risk_aware_sandbox_trader import RiskAwareSandboxTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def format_usd(value: float) -> str:
    """Format a value as USD currency"""
    return f"${value:,.2f}"

def format_percent(value: float) -> str:
    """Format a value as a percentage"""
    return f"{value * 100:.2f}%"

def format_leverage(value: float) -> str:
    """Format leverage value"""
    return f"{value:.1f}x"

def main():
    """Main function to check trader status"""
    logger.info("Checking risk-aware sandbox trader status...")
    
    # Initialize trader
    trader = RiskAwareSandboxTrader()
    
    # Get current portfolio value
    portfolio_value = trader.get_current_portfolio_value()
    logger.info(f"Current portfolio value: {format_usd(portfolio_value)}")
    
    # Load risk metrics
    risk_metrics = trader.risk_metrics
    
    # Print portfolio history
    if trader.portfolio:
        initial_value = trader.portfolio[0]["portfolio_value"]
        current_value = trader.portfolio[-1]["portfolio_value"]
        change = current_value - initial_value
        change_percent = (change / initial_value) if initial_value > 0 else 0
        
        logger.info(f"Portfolio starting value: {format_usd(initial_value)}")
        logger.info(f"Portfolio current value: {format_usd(current_value)}")
        
        if change >= 0:
            logger.info(f"Portfolio change: +{format_usd(change)} ({format_percent(change_percent)})")
        else:
            logger.info(f"Portfolio change: {format_usd(change)} ({format_percent(change_percent)})")
    
    # Print active positions
    if trader.positions:
        logger.info(f"\nActive positions ({len(trader.positions)}):")
        for pos in trader.positions:
            direction = pos["direction"]
            pair = pos["pair"]
            strategy = pos["strategy"]
            entry_price = pos["entry_price"]
            current_price = pos["current_price"]
            leverage = pos["leverage"]
            unrealized_pnl = pos["unrealized_pnl"]
            entry_time = datetime.datetime.fromisoformat(pos["entry_time"].replace('Z', '+00:00') if pos["entry_time"].endswith('Z') else pos["entry_time"])
            duration = pos["duration"]
            confidence = pos.get("confidence", 0.7)  # Default to 0.7 if not present
            
            logger.info(f"  {direction} {pair} ({strategy}):")
            logger.info(f"    Entry: {format_usd(entry_price)}, Current: {format_usd(current_price)}")
            logger.info(f"    Leverage: {format_leverage(leverage)}, Confidence: {format_percent(confidence)}")
            logger.info(f"    Unrealized P/L: {format_percent(unrealized_pnl)}")
            logger.info(f"    Duration: {duration}")
            if 'stop_loss' in pos and 'take_profit' in pos:
                logger.info(f"    Stop Loss: {format_usd(pos['stop_loss'])}, Take Profit: {format_usd(pos['take_profit'])}")
            if 'liquidation_price' in pos:
                logger.info(f"    Liquidation Price: {format_usd(pos['liquidation_price'])}")
    else:
        logger.info("\nNo active positions")
    
    # Print recent trades
    if trader.trades:
        recent_trades = [t for t in trader.trades if t["type"] == "Exit" or t["type"] == "Liquidation"]
        recent_trades = sorted(recent_trades, key=lambda x: x["timestamp"], reverse=True)
        recent_trades = recent_trades[:5]  # Show last 5 trades
        
        if recent_trades:
            logger.info(f"\nRecent completed trades ({len(recent_trades)}):")
            for trade in recent_trades:
                direction = trade["direction"]
                pair = trade["pair"]
                strategy = trade["strategy"]
                entry_price = trade["entry_price"]
                exit_price = trade["exit_price"]
                pnl_percentage = trade.get("pnl_percentage", 0)
                close_reason = trade.get("close_reason", "Manual")
                duration = trade.get("duration", "N/A")
                
                logger.info(f"  {direction} {pair} ({strategy}):")
                logger.info(f"    Entry: {format_usd(entry_price)}, Exit: {format_usd(exit_price)}")
                logger.info(f"    P/L: {format_percent(pnl_percentage)}")
                logger.info(f"    Reason: {close_reason}, Duration: {duration}")
    
    # Print risk metrics
    if risk_metrics:
        logger.info("\nRisk metrics:")
        if "win_rate" in risk_metrics:
            logger.info(f"  Win Rate: {format_percent(risk_metrics['win_rate'])}")
        if "profit_factor" in risk_metrics:
            profit_factor = risk_metrics["profit_factor"]
            if isinstance(profit_factor, float) and profit_factor != float('inf'):
                logger.info(f"  Profit Factor: {profit_factor:.2f}")
            else:
                logger.info(f"  Profit Factor: ∞")
        if "max_drawdown" in risk_metrics:
            logger.info(f"  Max Drawdown: {format_percent(risk_metrics['max_drawdown'])}")
        if "sharpe_ratio" in risk_metrics:
            logger.info(f"  Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
        if "sortino_ratio" in risk_metrics:
            logger.info(f"  Sortino Ratio: {risk_metrics['sortino_ratio']:.2f}")
        if "current_risk_level" in risk_metrics:
            logger.info(f"  Current Risk Level: {risk_metrics['current_risk_level']}")

    # Check if portfolio was reset to $20,000
    logger.info("\nChecking portfolio reset status:")
    if trader.portfolio and trader.portfolio[0]["portfolio_value"] == 20000.0:
        logger.info("  ✓ Portfolio was reset to $20,000 as requested")
    else:
        initial_value = trader.portfolio[0]["portfolio_value"] if trader.portfolio else 0
        logger.info(f"  ✗ Portfolio was not reset to $20,000, current initial value: {format_usd(initial_value)}")
    
    # Check if fees and liquidation risk are properly simulated
    logger.info("\nChecking simulation realism:")
    
    # Check fee configuration
    fee_config_path = "config/fee_config.json"
    if os.path.exists(fee_config_path):
        with open(fee_config_path, 'r') as f:
            fee_config = json.load(f)
        
        logger.info("  ✓ Fee configuration exists with the following settings:")
        logger.info(f"    • Maker Fee: {format_percent(fee_config.get('maker_fee', 0))}")
        logger.info(f"    • Taker Fee: {format_percent(fee_config.get('taker_fee', 0))}")
        logger.info(f"    • Funding Fee (8h): {format_percent(fee_config.get('funding_fee_8h', 0))}")
        logger.info(f"    • Liquidation Fee: {format_percent(fee_config.get('liquidation_fee', 0))}")
        logger.info(f"    • Maintenance Margin: {format_percent(fee_config.get('maintenance_margin', 0))}")
    else:
        logger.info("  ✗ Fee configuration file not found")
    
    # Check if any positions have proper liquidation prices calculated
    if trader.positions:
        has_liquidation = any("liquidation_price" in pos for pos in trader.positions)
        if has_liquidation:
            logger.info("  ✓ Positions have liquidation prices calculated")
        else:
            logger.info("  ✗ Positions do not have liquidation prices calculated")
    
    # Check if any trades have fees applied
    if trader.trades:
        has_fees = any("fees" in trade for trade in trader.trades)
        if has_fees:
            logger.info("  ✓ Trades have fees applied")
        else:
            logger.info("  ✗ Trades do not have fees applied")
    
    logger.info("\nTrade simulation status:")
    # Check if any liquidations have occurred
    if trader.trades:
        liquidations = [t for t in trader.trades if t.get("type") == "Liquidation"]
        if liquidations:
            logger.info(f"  ⚠ {len(liquidations)} liquidation(s) have occurred")
            for liq in liquidations[:3]:  # Show up to 3 liquidations
                logger.info(f"    • {liq['direction']} {liq['pair']} at {format_usd(liq['exit_price'])}")
        else:
            logger.info("  ✓ No liquidations have occurred")
    
    # Check stop losses and take profits
    if trader.trades:
        stop_losses = [t for t in trader.trades if t.get("close_reason") == "Stop Loss"]
        take_profits = [t for t in trader.trades if t.get("close_reason") == "Take Profit"]
        
        if stop_losses:
            logger.info(f"  ⚠ {len(stop_losses)} stop loss(es) have been triggered")
        if take_profits:
            logger.info(f"  ✓ {len(take_profits)} take profit(s) have been triggered")
    
    logger.info("\nCheck completed")

if __name__ == "__main__":
    main()