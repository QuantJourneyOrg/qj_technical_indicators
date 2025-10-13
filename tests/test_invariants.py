import numpy as np
import pandas as pd

from quantjourney_ti import TechnicalIndicators


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    base = np.cumsum(rng.normal(0, 1, n)) + 100
    close = base + rng.normal(0, 0.2, n)
    high = close + np.abs(rng.normal(0.5, 0.2, n))
    low = close - np.abs(rng.normal(0.5, 0.2, n))
    open_ = close + rng.normal(0, 0.1, n)
    volume = np.abs(rng.normal(1e6, 2e5, n)).astype(np.int64)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_bb_bands_ordering():
    df = _make_ohlcv()
    ti = TechnicalIndicators()
    bb = ti.BB(df["close"], 20, 2.0)
    up = bb["BB_Upper"]
    mid = bb["BB_Middle"]
    lo = bb["BB_Lower"]
    mask = ~(up.isna() | mid.isna() | lo.isna())
    assert (up[mask] >= mid[mask]).all()
    assert (mid[mask] >= lo[mask]).all()


def test_atr_nonnegative():
    df = _make_ohlcv()
    ti = TechnicalIndicators()
    atr = ti.ATR(df[["high", "low", "close"]], 14)
    if isinstance(atr, pd.DataFrame):
        series = atr.iloc[:, 0]
    else:
        series = atr
    series = series.dropna()
    assert (series >= 0).all()


def test_vwap_within_high_low_window():
    df = _make_ohlcv()
    ti = TechnicalIndicators()
    vwap = ti.VWAP(df[["high", "low", "close", "volume"]], 14)
    series = vwap.iloc[:, 0] if isinstance(vwap, pd.DataFrame) else vwap
    # Rolling bounds
    roll_high = df["high"].rolling(14).max()
    roll_low = df["low"].rolling(14).min()
    mask = ~(series.isna() | roll_high.isna() | roll_low.isna())
    assert (series[mask] <= roll_high[mask] + 1e-9).all()
    assert (series[mask] >= roll_low[mask] - 1e-9).all()


def test_macd_matches_ema_difference():
    df = _make_ohlcv()
    s = df["close"]
    ti = TechnicalIndicators()
    macd = ti.MACD(s, 12, 26, 9)["MACD"].dropna()
    # Reference EMAs using pandas
    ema_fast = s.ewm(span=12, adjust=False).mean()
    ema_slow = s.ewm(span=26, adjust=False).mean()
    ref = (ema_fast - ema_slow).dropna()
    aligned = macd.index.intersection(ref.index)
    diff = (macd.loc[aligned] - ref.loc[aligned]).abs()
    # Allow small numerical differences
    assert diff.max() < 1e-6
