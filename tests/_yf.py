"""Helper to fetch data via yfinance for integration tests only."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)
_YF_ERR = "yfinance is required; install with `pip install yfinance`"


@lru_cache(maxsize=32)
def get_ohlcv(
    ticker: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = "1y",
    interval: Literal[
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    ] = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(_YF_ERR) from exc
    logger.info("Downloading %s %s (interval=%s)", ticker, period, interval)
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"No data for {ticker}")
        # Ensure consistent column names
        return df.rename(columns=str.lower)
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch data for {ticker}: {exc}") from exc


__all__ = ["get_ohlcv"]
