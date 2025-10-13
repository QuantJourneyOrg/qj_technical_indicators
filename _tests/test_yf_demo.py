import pytest


@pytest.mark.slow
def test_yf_demo_plot_optional():
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        pytest.skip("yfinance not installed; install extra 'yf' to run")

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception:
        pytest.skip("matplotlib not installed; install extra 'plot' to run")

    from quantjourney_ti import TechnicalIndicators

    ti = TechnicalIndicators()
    df = yf.download("AAPL", period="3mo", progress=False)
    if df.empty:
        pytest.skip("no data returned from yfinance")

    df = df.rename(columns={
        "Close": "close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Volume": "volume",
    })
    ind = {
        "SMA_20": ti.SMA(df["close"], 20),
        "RSI_14": ti.RSI(df["close"], 14),
    }
    # Plot (optional; test ensures call doesn't raise)
    ti.plot_indicators(df, ind, title="AAPL demo", overlay=False)

