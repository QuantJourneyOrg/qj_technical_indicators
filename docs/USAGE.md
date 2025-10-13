# Usage Guide

## Installation

Base installation:

```
pip install .
```

Optional extras:

```
pip install .[yf]    # yfinance integration
pip install .[plot]  # matplotlib plotting
```

## Basics

```python
import pandas as pd, numpy as np
from quantjourney_ti import TechnicalIndicators

idx = pd.date_range('2024-01-01', periods=200, freq='D')
close = pd.Series(np.cumsum(np.random.normal(0,1,200)) + 100, index=idx)
ti = TechnicalIndicators()  # or TechnicalIndicators(warmup=True)

sma = ti.SMA(close, 20)
ema = ti.EMA(close, 20)
rsi = ti.RSI(close, 14)
macd = ti.MACD(close, 12, 26, 9)
```

## Channels and Bands

```python
high = close + np.abs(np.random.normal(0.5,0.2,200))
low  = close - np.abs(np.random.normal(0.5,0.2,200))
df = pd.DataFrame({'high':high,'low':low,'close':close})

bb = ti.BB(close, 20, 2.0)
kc = ti.KELTNER(df[['high','low','close']], 20, 10, 2.0)
dc = ti.DONCHIAN(df[['high','low']], 20)
```

## Streaming

```python
from quantjourney_ti._streaming import StreamingIndicators

si = StreamingIndicators(max_buffer_size=1000)
for ts, val in close.items():
    si.update_tick('SYM', ts, close=float(val))

last_sma = si.states['SYM'].last_values['sma_20']
```

## Logging Queue (opt-in)

```python
from quantjourney_ti import start_logging_queue
start_logging_queue()
```

