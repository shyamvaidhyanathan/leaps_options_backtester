# LEAP Option Backtester

Simulates a simple **LEAP Call Strategy** on any stock — buying a 1-year (LEAP) call option, holding it for 30 trading days, then selling.  
Uses **Black–Scholes pricing** with **realized volatility** as a proxy for implied volatility.

---

## 🚀 Features

- Backtests over full available price history from Yahoo Finance
- Configurable LEAP duration, holding period, volatility lookback
- Black–Scholes modeled call pricing
- Summary statistics and visualizations:
  - Trade return distribution
  - Cumulative equity curve
- Minimal dependencies (uses `yfinance`, `numpy`, `pandas`, `matplotlib`, `scipy`)

---

## 🧩 Strategy Overview

1. On each trading day:
   - Assume you **buy a LEAP call** (ATM strike, 1-year expiry).
2. Hold for **30 trading days**.
3. Sell the option (re-priced with shorter time to expiry).
4. Repeat for all available historical dates.

The model assumes:
- Risk-free rate = 1% annualized
- Option volatility = 30-day realized volatility
- No dividends or slippage (can be added)
- One contract per trade (100 shares)
- Optional per-contract commission

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/leap_option_backtester.git
cd leap_option_backtester
pip install -r requirements.txt
