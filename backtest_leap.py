"""
LEAP Option Backtester
----------------------

Simulates a historical strategy where you buy a LEAP call option on a given stock,
hold it for 30 trading days, and sell it — repeating across the entire available
price history. Option prices are modeled with the Black-Scholes formula using
historical realized volatility as a proxy for implied vol.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ---------------------------
# Black-Scholes European Call
# ---------------------------
def bs_call_price(S, K, T, r, sigma, q=0.0):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# ---------------------------
# Annualized realized volatility
# ---------------------------
def realized_volatility(series, lookback=30, trading_days=252):
    logr = np.log(series / series.shift(1))
    return logr.rolling(window=lookback).std() * np.sqrt(trading_days)

# ---------------------------
# Backtest LEAP call strategy
# ---------------------------
def backtest_leap_call_strategy(
    ticker: str,
    leap_days=252,
    hold_days=30,
    vol_lookback=30,
    risk_free_rate=0.01,
    start=None,
    end=None,
    contracts_per_trade=1,
    shares_per_contract=100,
    commission_per_contract=0.0,
    trading_days_per_year=252
):
    """
    Simulates buying a LEAP call (1-year expiry) and holding it for 30 trading days.
    Returns (trades_df, summary)
    """

    # Fetch historical daily price data
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start, end=end, period="max", auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No historical data for {ticker}")

    prices = hist["Close"].copy().dropna().sort_index()
    rv = realized_volatility(prices, lookback=vol_lookback, trading_days=trading_days_per_year)

    trades = []
    trade_dates = prices.index.to_list()

    for i, buy_date in enumerate(trade_dates):
        sell_idx = i + hold_days
        if sell_idx >= len(trade_dates):
            break
        sell_date = trade_dates[sell_idx]

        S_buy = prices.iloc[i]
        S_sell = prices.iloc[sell_idx]

        sigma_buy = rv.iloc[i]
        sigma_sell = rv.iloc[sell_idx]

        # Fill missing vols with fallback
        if np.isnan(sigma_buy):
            window = min(vol_lookback, i)
            if window < 2:
                continue
            sigma_buy = np.log(prices.iloc[i-window+1:i+1] / prices.iloc[i-window:i]).std() * np.sqrt(trading_days_per_year)
        if np.isnan(sigma_sell):
            window = min(vol_lookback, sell_idx)
            if window < 2:
                continue
            sigma_sell = np.log(prices.iloc[sell_idx-window+1:sell_idx+1] / prices.iloc[sell_idx-window:sell_idx]).std() * np.sqrt(trading_days_per_year)

        # Time to expiry (in years)
        T_buy = leap_days / trading_days_per_year
        T_sell = max((leap_days - hold_days) / trading_days_per_year, 1 / trading_days_per_year)

        # ATM strike at buy
        K = round(S_buy)

        c_buy = bs_call_price(S=S_buy, K=K, T=T_buy, r=risk_free_rate, sigma=sigma_buy)
        c_sell = bs_call_price(S=S_sell, K=K, T=T_sell, r=risk_free_rate, sigma=sigma_sell)

        cost_buy = c_buy * shares_per_contract * contracts_per_trade
        proceeds_sell = c_sell * shares_per_contract * contracts_per_trade
        commission = commission_per_contract * contracts_per_trade * 2

        pnl = proceeds_sell - cost_buy - commission
        ret_pct = pnl / cost_buy if cost_buy > 0 else np.nan

        trades.append({
            "buy_date": buy_date,
            "sell_date": sell_date,
            "S_buy": S_buy,
            "S_sell": S_sell,
            "strike": K,
            "c_buy": c_buy,
            "c_sell": c_sell,
            "pnl": pnl,
            "return_pct": ret_pct,
            "sigma_buy": sigma_buy,
            "sigma_sell": sigma_sell
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        raise ValueError("No trades generated. Try adjusting parameters.")

    total_pnl = trades_df["pnl"].sum()
    avg_return = trades_df["return_pct"].mean()
    win_rate = (trades_df["pnl"] > 0).mean()
    annualized_return = (1 + avg_return)**(trading_days_per_year / hold_days) - 1

    summary = {
        "ticker": ticker,
        "num_trades": len(trades_df),
        "win_rate": round(win_rate * 100, 2),
        "avg_return_per_trade_%": round(avg_return * 100, 2),
        "annualized_return_est_%": round(annualized_return * 100, 2),
        "total_pnl_$": round(total_pnl, 2)
    }

    return trades_df, summary

# ---------------------------
# Visualization helpers
# ---------------------------
def plot_trade_returns(trades_df, title=None):
    plt.figure(figsize=(10, 4))
    plt.hist(trades_df["return_pct"].dropna() * 100, bins=50)
    plt.xlabel("Trade Return (%)")
    plt.ylabel("Count")
    plt.title(title or "Distribution of Trade Returns")
    plt.grid(alpha=0.3)
    plt.show()

def plot_equity_curve(trades_df, initial_capital=100000):
    eq = np.cumsum(np.insert(trades_df["pnl"].fillna(0).values, 0, 0)) + initial_capital
    plt.figure(figsize=(10, 4))
    plt.plot(eq)
    plt.title("Cumulative Equity Curve")
    plt.ylabel("USD")
    plt.xlabel("Trade #")
    plt.grid(alpha=0.3)
    plt.show()
