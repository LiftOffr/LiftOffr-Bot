#!/usr/bin/env python3
"""
Script to analyze integrated strategy logs and provide a summary
of the strategy's decision-making process.
"""
import re
import sys
from collections import Counter
from datetime import datetime


def analyze_logs(log_file):
    """
    Analyze integrated strategy log file content
    
    Args:
        log_file (str): Path to log file
    """
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading log file: {e}")
        return
    
    # Initialize counters
    signals = Counter()
    indicators = {
        'ema': Counter(),
        'rsi': Counter(),
        'macd': Counter(),
        'adx': Counter(),
        'volatility': Counter()
    }
    
    # Extract signal patterns
    buy_signals = re.findall(r'【SIGNAL】 🟢 BULLISH', content)
    sell_signals = re.findall(r'【SIGNAL】 🔴 BEARISH', content)
    neutral_signals = re.findall(r'【SIGNAL】 ⚪ NEUTRAL', content)
    
    signals['buy'] = len(buy_signals)
    signals['sell'] = len(sell_signals)
    signals['neutral'] = len(neutral_signals)
    
    # Extract indicator patterns
    ema_bullish = re.findall(r'EMA9 > EMA21', content)
    ema_bearish = re.findall(r'EMA9 < EMA21', content)
    rsi_bullish = re.findall(r'RSI = (\d+\.\d+) ✓ .* RSI > 70', content)
    rsi_bearish = re.findall(r'RSI = (\d+\.\d+) ✓ .* RSI < 30', content)
    macd_bullish = re.findall(r'MACD > Signal', content)
    macd_bearish = re.findall(r'MACD < Signal', content)
    adx_trending = re.findall(r'ADX = (\d+\.\d+) ✓', content)
    volatility_pass = re.findall(r'Volatility = (\d+\.\d+) ✓', content)
    
    indicators['ema']['bullish'] = len(ema_bullish)
    indicators['ema']['bearish'] = len(ema_bearish)
    indicators['macd']['bullish'] = len(macd_bullish)
    indicators['macd']['bearish'] = len(macd_bearish)
    indicators['rsi']['bullish'] = len(rsi_bullish)
    indicators['rsi']['bearish'] = len(rsi_bearish)
    indicators['adx']['trending'] = len(adx_trending)
    indicators['volatility']['pass'] = len(volatility_pass)
    
    # Extract price trends
    arima_bullish = re.findall(r'Forecast: BULLISH \| Current: \$(\d+\.\d+) → Target: \$(\d+\.\d+)', content)
    arima_bearish = re.findall(r'Forecast: BEARISH \| Current: \$(\d+\.\d+) → Target: \$(\d+\.\d+)', content)
    
    # Extract actions
    buy_actions = re.findall(r'【ACTION】 🟢 BUY', content)
    sell_actions = re.findall(r'【ACTION】 🔴 SELL', content)
    
    # Print summary
    print("\n" + "="*80)
    print(f"INTEGRATED STRATEGY LOG ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    print("\nSIGNAL DISTRIBUTION:")
    print(f"  Bullish Signals:  {signals['buy']}")
    print(f"  Bearish Signals:  {signals['sell']}")
    print(f"  Neutral Signals:  {signals['neutral']}")
    
    print("\nINDICATOR BREAKDOWN:")
    print(f"  EMA:        Bullish: {indicators['ema']['bullish']}, Bearish: {indicators['ema']['bearish']}")
    print(f"  MACD:       Bullish: {indicators['macd']['bullish']}, Bearish: {indicators['macd']['bearish']}")
    print(f"  RSI:        Bullish: {indicators['rsi']['bullish']}, Bearish: {indicators['rsi']['bearish']}")
    print(f"  ADX:        Trending: {indicators['adx']['trending']}")
    print(f"  Volatility: Passed filter: {indicators['volatility']['pass']}")
    
    print("\nARIMA FORECASTS:")
    print(f"  Bullish Forecasts: {len(arima_bullish)}")
    print(f"  Bearish Forecasts: {len(arima_bearish)}")
    
    print("\nTRADE ACTIONS:")
    print(f"  Buy Actions:  {len(buy_actions)}")
    print(f"  Sell Actions: {len(sell_actions)}")
    
    # Calculate signal to action ratio (conversion rate)
    buy_ratio = len(buy_actions) / signals['buy'] if signals['buy'] > 0 else 0
    sell_ratio = len(sell_actions) / signals['sell'] if signals['sell'] > 0 else 0
    
    print(f"\nSignal to Action Conversion:")
    print(f"  Buy: {buy_ratio:.2%}")
    print(f"  Sell: {sell_ratio:.2%}")
    
    print("\nNote: This analysis helps understand how often different signals are generated and")
    print("which indicators are most active in the integrated strategy's decision process.")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = "integrated_strategy_log.txt"
    
    analyze_logs(log_file)