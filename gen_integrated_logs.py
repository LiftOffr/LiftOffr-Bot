#!/usr/bin/env python3
"""
Script to generate detailed test logs for the integrated strategy
to demonstrate the new detailed logging functionality.
"""
import random
import time
from datetime import datetime

def generate_integrated_logs(filename, entry_count=200):
    """
    Generate detailed logs for the integrated strategy with various signals
    
    Args:
        filename (str): Path to log file to create
        entry_count (int): Number of log entries to generate
    """
    # Price range and indicators
    current_price = 131.50
    
    # Signal counters
    buy_signals = 0
    sell_signals = 0
    neutral_signals = 0
    
    # Open log file
    with open(filename, 'w') as f:
        for i in range(entry_count):
            # Timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Randomly adjust price
            price_change = random.uniform(-0.5, 0.5)
            current_price += price_change
            
            # Generate technical indicators
            ema9 = current_price + random.uniform(-1.0, 1.0)
            ema21 = current_price + random.uniform(-1.5, 1.5)
            rsi = random.uniform(20, 80)
            adx = random.uniform(15, 45)
            volatility = random.uniform(0.001, 0.01)
            
            # BB and KC values
            upper_bb = current_price + random.uniform(1.0, 3.0)
            lower_bb = current_price - random.uniform(1.0, 3.0)
            kc_middle = current_price + random.uniform(-1.0, 1.0)
            
            # MACD setup
            macd = random.uniform(-0.1, 0.1)
            signal = random.uniform(-0.1, 0.1)
            
            # Target price
            if random.random() > 0.5:
                forecast_direction = "BULLISH"
                target_price = current_price + random.uniform(0.05, 0.3)
            else:
                forecast_direction = "BEARISH"
                target_price = current_price - random.uniform(0.05, 0.3)
            
            # Generate analysis log
            f.write(f"{timestamp} [INFO] ============================================================\n")
            f.write(f"{timestamp} [INFO] 【ANALYSIS】 Forecast: {forecast_direction} | Current: ${current_price:.2f} → Target: ${target_price:.2f}\n")
            
            # Generate indicators log
            ema_relation = ">" if ema9 > ema21 else "<"
            macd_relation = ">" if macd > signal else "<"
            rsi_check = "✓" if (rsi < 30 or rsi > 70) else ""
            adx_check = "✓" if adx > 20 else ""
            
            f.write(f"{timestamp} [INFO] 【INDICATORS】 EMA9 {ema_relation} EMA21 | RSI = {rsi:.2f} {rsi_check} | MACD {macd_relation} Signal | ADX = {adx:.2f} {adx_check}\n")
            
            # Generate volatility log
            vol_check = "✓" if volatility < 0.006 else ""
            f.write(f"{timestamp} [INFO] 【VOLATILITY】 Volatility = {volatility:.4f} {vol_check} (threshold: 0.006)\n")
            
            # Generate bands log
            f.write(f"{timestamp} [INFO] 【BANDS】 EMA9 {ema_relation} EMA21 ({ema9:.2f} vs {ema21:.2f}) | " +
                  f"RSI = {rsi:.2f} {rsi_check} | MACD {macd_relation} Signal ({macd:.4f} vs {signal:.4f}) | " +
                  f"ADX = {adx:.2f} {adx_check} | Volatility = {volatility:.4f} {vol_check} (threshold: 0.006) | " +
                  f"Price {'<' if current_price < upper_bb else '>'} Upper BB ({current_price:.2f} vs {upper_bb:.2f}) | " +
                  f"Price {'>' if current_price > lower_bb else '<'} Lower BB ({current_price:.2f} vs {lower_bb:.2f}) | " +
                  f"Price vs KC Middle: {current_price:.2f} vs {kc_middle:.2f}\n")
            
            # Generate signal based on conditions
            signal_type = ""
            if random.random() > 0.65:  # 35% chance of no signal
                signal_str = "NEUTRAL"
                symbol = "⚪"
                message = "Conditions not met for trading"
                neutral_signals += 1
            elif random.random() > 0.5:  # ~32.5% chance of buy signal
                signal_str = "BULLISH"
                symbol = "🟢"
                message = "Trade conditions met for LONG position"
                buy_signals += 1
            else:  # ~32.5% chance of sell signal
                signal_str = "BEARISH"
                symbol = "🔴"
                message = "Trade conditions met for SHORT position"
                sell_signals += 1
                
            f.write(f"{timestamp} [INFO] 【SIGNAL】 {symbol} {signal_str} - {message}\n")
            f.write(f"{timestamp} [INFO] ============================================================\n")
            
            # Generate action if signal is not neutral
            if signal_str != "NEUTRAL":
                atr = random.uniform(0.18, 0.25)
                stop_price = current_price - (1.5 * atr) if signal_str == "BULLISH" else current_price + (1.5 * atr)
                
                action = "BUY" if signal_str == "BULLISH" else "SELL"
                action_symbol = "🟢" if action == "BUY" else "🔴"
                
                f.write(f"{timestamp} [INFO] 【ACTION】 {action_symbol} {action} | ATR: ${atr:.4f} | Volatility Stop: ${stop_price:.2f}\n")
                
                # Add signal strength calculation
                f.write(f"{timestamp} [INFO] 【INTEGRATED】 Signal Strength: EMA={random.uniform(0.5, 1.0):.2f}, " +
                      f"RSI={random.uniform(0.5, 1.0):.2f}, MACD={random.uniform(0.5, 1.0):.2f}, " +
                      f"ADX={random.uniform(0.5, 1.0):.2f}, ARIMA={random.uniform(0.5, 1.0):.2f}\n")
                
                # Add final signal strength
                overall_strength = random.uniform(0.65, 0.95)
                f.write(f"{timestamp} [INFO] 【INTEGRATED】 Final Signal Strength: {overall_strength:.2f} ({action} {symbol})\n")
            
            # Simulate some time passing between log entries
            time.sleep(0.01)
            
    print(f"Generated {entry_count} integrated strategy log entries in '{filename}'")
    print(f"Signal distribution: Buy: {buy_signals}, Sell: {sell_signals}, Neutral: {neutral_signals}")

if __name__ == "__main__":
    filename = "integrated_strategy_log.txt"
    generate_integrated_logs(filename, entry_count=100)