import inspect
import os
import pprint
import time

import pandas as pd
import pytest
import numpy as np

if os.getenv("CI") or os.getenv("SKIP_ALL_INDICATORS"):
    pytest.skip("Skipping full indicator run on CI", allow_module_level=True)

try:
    from tests._yf import get_ohlcv
    import quantjourney_ti as qti
    from quantjourney_ti._decorators import timer
except Exception as exc:
    pytest.skip(f"Prerequisites missing: {exc}", allow_module_level=True)

TICKER = "AAPL"

def _feed_arg(method, df):
    """Return appropriate first argument (Series or DataFrame)."""
    sig = inspect.signature(method)
    first = next(iter(sig.parameters.values()))
    if first.annotation is pd.Series or first.annotation is inspect._empty:
        return pd.Series(df["adj_close"].values, index=df.index)
    return df

@pytest.fixture
def mock_ohlcv():
    data = {
        "open": [100, 101, 102] * 21,  # 63 rows
        "high": [102, 103, 104] * 21,
        "low": [99, 100, 101] * 21,
        "adj_close": [101, 102, 103] * 21,
        "volume": [1000, 1100, 1200] * 21,
    }
    return pd.DataFrame(data, index=pd.date_range("2025-01-01", periods=63))

def test_run_all_indicators(capsys, mock_ohlcv):
    """Execute every indicator once with defaults."""
    df = mock_ohlcv
    ti = qti.TechnicalIndicators()
    results, errors, timings = {}, {}, {}
    skip_indicators = {
        "BETA", "calculate_multiple_indicators", "plot_indicators", "BENFORD_LAW",
    }
    for name, meth in inspect.getmembers(ti, predicate=inspect.ismethod):
        if name.startswith("_") or name in skip_indicators:
            continue
        try:
            arg = _feed_arg(meth, df)
            @timer
            def _run():
                return meth(arg)
            start = time.perf_counter()
            out = _run()
            timings[name] = time.perf_counter() - start
            val = out.dropna().iloc[-1] if isinstance(out, (pd.Series, pd.DataFrame)) else out
            results[name] = val
        except Exception as exc:
            errors[name] = str(exc)
    print("\n=== Indicator outputs (last non-NaN value) ===")
    pprint.pprint(results, compact=True)
    print("\n=== Execution times (seconds) ===")
    pprint.pprint(timings, compact=True)
    if errors:
        print("\n=== Errors ===")
        pprint.pprint(errors, compact=True)
    total = len(results) + len(errors)
    assert len(results) / total >= 0.5, f"Too many indicator failures: {len(errors)}/{total}"