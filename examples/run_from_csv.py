#!/usr/bin/env python3
"""
Load OHLCV data from CSV and compute selected indicators.
CSV must contain columns: open, high, low, close, volume (case-insensitive).

Usage:
  python examples/run_from_csv.py --csv path/to/data.csv --sep ,
"""

import argparse
import sys
import pandas as pd

from quantjourney_ti import TechnicalIndicators


def load_csv(path: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    # normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        raise SystemExit(f"CSV must include columns: {required}")
    # parse date if present
    for c in ("date", "datetime", "timestamp"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            df = df.set_index(c)
            break
    return df


def main():
    ap = argparse.ArgumentParser(description="Compute indicators from CSV")
    ap.add_argument("--csv", required=True, help="Path to CSV file")
    ap.add_argument("--sep", default=",", help="CSV separator")
    args = ap.parse_args()

    df = load_csv(args.csv, args.sep)
    ti = TechnicalIndicators()

    sma = ti.SMA(df["close"], 20)
    ema = ti.EMA(df["close"], 20)
    atr = ti.ATR(df[["high", "low", "close"]], 14)
    print("SMA_20 tail:\n", sma.tail())
    print("EMA_20 tail:\n", ema.tail())
    print("ATR_14 tail:\n", atr.tail() if isinstance(atr, pd.Series) else atr.iloc[:, 0].tail())


if __name__ == "__main__":
    main()

