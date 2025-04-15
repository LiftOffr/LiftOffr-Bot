#!/usr/bin/env python3
"""
Calculate current PnL for all open positions using real-time market data.
"""
import os
import sys
import json
import requests
from datetime import datetime

# Load positions from file
def load_positions():
    try:
        with open('data/sandbox_positions.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading positions: {e}")
        return []

# Get current price from Kraken API
def get_current_price(pair):
    try:
        kraken_pair = pair.replace('/', '')
        url = f"https://api.kraken.com/0/public/Ticker?pair={kraken_pair}"
        
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if 'result' in data and data['result']:
                pair_data = next(iter(data['result'].values()))
                if 'c' in pair_data and pair_data['c']:
                    return float(pair_data['c'][0])
        
        print(f"Could not get price for {pair} from Kraken API")
        return None
    except Exception as e:
        print(f"Error getting price for {pair} from Kraken API: {e}")
        return None

# Calculate PnL for a position
def calculate_position_pnl(position):
    pair = position['pair']
    direction = position['direction']
    position_size = position['position_size']
    leverage = position['leverage']
    entry_price = position['entry_price']
    
    # Get current price
    current_price = get_current_price(pair)
    if not current_price:
        # Use entry price if we can't get current price
        current_price = entry_price
        
    # Calculate price change percentage
    if direction.lower() == 'long':
        price_change_pct = (current_price / entry_price) - 1
    else:  # short
        price_change_pct = (entry_price / current_price) - 1
    
    # Calculate PnL
    pnl_pct = price_change_pct * leverage
    pnl_amount = position_size * pnl_pct
    
    return {
        'pair': pair,
        'direction': direction,
        'position_size': position_size,
        'leverage': leverage,
        'entry_price': entry_price,
        'current_price': current_price,
        'pnl_amount': pnl_amount,
        'pnl_pct': pnl_pct * 100  # Convert to percentage
    }

# Calculate PnL for all positions
def main():
    positions = load_positions()
    
    if not positions:
        print("No positions found")
        return
    
    # Calculate PnL for each position
    pnl_results = []
    total_pnl = 0.0
    initial_capital = 20000.0
    
    print("\nCurrent PnL for Active Positions:")
    print("--------------------------------")
    print(f"{'Pair':10} | {'Direction':6} | {'Size':>10} | {'Leverage':>8} | {'Entry':>10} | {'Current':>10} | {'PnL $':>10} | {'PnL %':>8}")
    print("-" * 90)
    
    for position in positions:
        result = calculate_position_pnl(position)
        pnl_results.append(result)
        total_pnl += result['pnl_amount']
        
        # Print position details
        print(f"{result['pair']:10} | {result['direction']:6} | ${result['position_size']:>9,.2f} | {result['leverage']:>7,.1f}x | ${result['entry_price']:>9,.4f} | ${result['current_price']:>9,.4f} | ${result['pnl_amount']:>9,.2f} | {result['pnl_pct']:>7,.2f}%")
    
    # Calculate total PnL percentage of initial capital
    total_pnl_pct = (total_pnl / initial_capital) * 100
    
    print("-" * 90)
    print(f"Total PnL: ${total_pnl:,.2f} ({total_pnl_pct:.2f}% of initial capital)")
    print(f"Updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return total_pnl, total_pnl_pct

if __name__ == "__main__":
    main()