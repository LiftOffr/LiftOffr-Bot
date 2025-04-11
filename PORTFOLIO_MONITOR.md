# Portfolio Monitor Feature

The Kraken Trading Bot includes a Portfolio Monitor feature that provides a readable, real-time overview of your trading activities. This document explains how to use and configure this feature.

## Overview

The Portfolio Monitor displays the following information at regular intervals:

- **Overall Portfolio Status**
  - Portfolio value
  - Profit/loss metrics
  - Number of trades
  - Win rate

- **Active Bot Status**
  - Current position (long, short, or none)
  - Strategy type
  - Trading pair
  - Margin allocation and leverage
  - Unrealized profit/loss for open positions

- **Market Data**
  - Current price information

## Example Output

```
============================================================
📊 PORTFOLIO STATUS (Updated: 2025-04-11 02:13:50)
============================================================

💰 PORTFOLIO OVERVIEW:
  Current Value: $20,010.16 (+$10.16, +0.05%)
  Initial Value: $20,000.00
  
📈 TRADING PERFORMANCE:
  Total Trades: 13
  Win Rate: 7.69% (1 Wins, 12 Losses)

🤖 ACTIVE BOTS:
  🟢 adaptive-SOLUSD (SOL/USD):
      Strategy: adaptive | Position: LONG @ $113.77
      Margin: 25.0% | Leverage: 25x
      Current Price: $113.92 | Unrealized P&L: 0.13%

  ⚪ arima-SOLUSD (SOL/USD):
      Strategy: arima | Position: No Position
      Margin: 10.0% | Leverage: 30x

📈 RECENT MARKET DATA:
  Current Price (SOL/USD): $113.92
============================================================
```

## Configuring Update Interval

By default, the Portfolio Monitor updates every 60 seconds. You can change this interval using the `set_monitor_interval.py` script:

```bash
python set_monitor_interval.py 30  # Update every 30 seconds
```

Valid values range from 10 to 3600 seconds (1 hour).

## Implementation Details

The Portfolio Monitor runs in a separate thread to prevent blocking the main trading functionality. It fetches the current status from all active bots and formats the data into a clear, readable display.

Key files:
- `portfolio_monitor.py`: Contains the implementation of the PortfolioMonitor class
- `set_monitor_interval.py`: Utility script to adjust the update interval
- `bot_manager.py`: Integrates with the PortfolioMonitor to provide consolidated data

## API Access

The same portfolio data is available through the web API at:
```
http://localhost:5001/api/status
```

This allows you to build custom monitoring tools or dashboards that consume the same data shown in the terminal display.