#!/usr/bin/env python3
"""
Run basic indicators for a ticker (requires extra 'yf').

Usage:
  python examples/run_basic.py --ticker AAPL --period 6mo
  python examples/run_basic.py --ticker MSFT --start 2024-01-01 --end 2024-06-30
"""

import argparse
import sys
from typing import Optional

import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:
    print("yfinance not installed. Install with: pip install .[yf]", file=sys.stderr)
    sys.exit(1)

from quantjourney_ti import TechnicalIndicators


def fetch_yf(ticker: str, start: Optional[str], end: Optional[str], period: Optional[str]) -> pd.DataFrame:
    if period:
        df = yf.download(ticker, period=period, progress=False)
    else:
        df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise SystemExit("No data returned from yfinance")
    # Normalize columns
    cols = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}
    df = df.rename(columns=cols)
    return df


def main():
    ap = argparse.ArgumentParser(description="Compute basic indicators for a ticker")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--period", default="6mo", help="yfinance period, e.g., 1mo, 6mo, 1y")
    args = ap.parse_args()

    df = fetch_yf(args.ticker, args.start, args.end, args.period)
    ti = TechnicalIndicators()

    out = {}
    out["SMA_20"] = ti.SMA(df["close"], 20)
    out["EMA_20"] = ti.EMA(df["close"], 20)
    out["RSI_14"] = ti.RSI(df["close"], 14)
    macd = ti.MACD(df["close"], 12, 26, 9)

    print(f"=== {args.ticker} ===")
    print("SMA_20 tail:\n", out["SMA_20"].tail())
    print("EMA_20 tail:\n", out["EMA_20"].tail())
    print("RSI_14 tail:\n", out["RSI_14"].tail())
    print("MACD tail:\n", macd.tail())


if __name__ == "__main__":
    main()

