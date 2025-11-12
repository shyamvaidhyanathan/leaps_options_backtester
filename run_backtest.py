from backtest_leap import backtest_leap_call_strategy, plot_trade_returns, plot_equity_curve

if __name__ == "__main__":
    ticker = "GOOG"  # Change to any ticker, e.g., "MSFT", "NVDA", "GOOG"

    trades_df, summary = backtest_leap_call_strategy(
        ticker=ticker,
        leap_days=300,          # 1-year LEAP at entry
        hold_days=30,           # Hold for 30 trading days
        vol_lookback=30,        # Realized vol lookback window
        risk_free_rate=0.01,    # 1% annual risk-free rate
        commission_per_contract=1.0
    )

    print("\n=== Backtest Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nFirst few trades:")
    print(trades_df.head())

    # Visualization
    plot_trade_returns(trades_df, title=f"{ticker} LEAP 30-day Hold Returns")
    plot_equity_curve(trades_df, initial_capital=100000)
