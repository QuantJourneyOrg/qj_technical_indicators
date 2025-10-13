#!/usr/bin/env python3
"""
Compute channels/bands for a ticker (BB, Keltner, Donchian).
Requires extra 'yf'.

Usage:
  python examples/run_channels.py --ticker AAPL --period 6mo
"""

import argparse
import sys
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:
    print("yfinance not installed. Install with: pip install .[yf]", file=sys.stderr)
    sys.exit(1)

from quantjourney_ti import TechnicalIndicators


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    cols = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}
    return df.rename(columns=cols)


def main():
    ap = argparse.ArgumentParser(description="Compute bands/channels")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--period", default="6mo")
    args = ap.parse_args()

    df = yf.download(args.ticker, period=args.period, progress=False)
    if df.empty:
        raise SystemExit("No data returned from yfinance")
    df = normalize(df)
    ti = TechnicalIndicators()

    bb = ti.BB(df["close"], 20, 2.0)
    kc = ti.KELTNER(df[["high", "low", "close"]], 20, 10, 2.0)
    dc = ti.DONCHIAN(df[["high", "low"]], 20)

    print("BB tail:\n", bb.tail())
    print("Keltner tail:\n", kc.tail())
    print("Donchian tail:\n", dc.tail())


if __name__ == "__main__":
    main()

