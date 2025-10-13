import os
import pytest


tickers = ["AAPL", "MSFT", "SPY"]


@pytest.mark.slow
def test_yf_multi_basic_indicators():
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        pytest.skip("yfinance not installed; install extra 'yf' to run")

    from quantjourney_ti import TechnicalIndicators

    ti = TechnicalIndicators()

    for t in tickers:
        df = yf.download(t, period="6mo", progress=False)
        if df.empty:
            continue
        # Normalize column names
        df = df.rename(columns={
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
        })
        # Sanity calculations
        sma = ti.SMA(df["close"], 20)
        ema = ti.EMA(df["close"], 20)
        macd = ti.MACD(df["close"], 12, 26, 9)
        atr = ti.ATR(df[["high", "low", "close"]], 14)

        assert len(sma) == len(df)
        assert len(ema) == len(df)
        assert set(["MACD", "Signal", "Histogram"]).issubset(set(macd.columns))
        if hasattr(atr, "__len__"):
            assert len(atr) == len(df)

