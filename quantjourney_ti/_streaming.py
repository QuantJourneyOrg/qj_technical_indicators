"""
QuantJourney Technical-Indicators - Streaming Interface
======================================================
Real-time streaming data processing for technical indicators with incremental updates.

Author: Jakub Polec <jakub@quantjourney.pro>
License: MIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Callable
from collections import deque
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

from ._indicator_kernels import *
from ._utils import validate_data
from ._errors import InvalidInputError

@dataclass
class StreamingState:
    """State container for streaming indicators."""
    buffer: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_values: Dict[str, float] = field(default_factory=dict)
    periods: Dict[str, int] = field(default_factory=dict)
    ema_multipliers: Dict[str, float] = field(default_factory=dict)
    sma_sums: Dict[str, float] = field(default_factory=dict)
    rsi_gains: deque = field(default_factory=lambda: deque(maxlen=100))
    rsi_losses: deque = field(default_factory=lambda: deque(maxlen=100))
    macd_fast_ema: float = 0.0
    macd_slow_ema: float = 0.0
    macd_signal_ema: float = 0.0
    atr_values: deque = field(default_factory=lambda: deque(maxlen=100))
    bb_values: deque = field(default_factory=lambda: deque(maxlen=100))
    initialized: bool = False
    tick_count: int = 0

class StreamingIndicators:
    """
    High-performance streaming technical indicators with incremental updates.
    
    Designed for real-time data feeds where full recalculation would be too slow.
    Maintains internal state and updates indicators incrementally as new data arrives.
    """
    
    def __init__(self, max_buffer_size: int = 1000):
        """
        Initialize streaming indicators.
        
        Args:
            max_buffer_size: Maximum number of historical values to keep in memory
        """
        self.max_buffer_size = max_buffer_size
        self.states: Dict[str, StreamingState] = {}
        self.lock = threading.RLock()
        self.callbacks: Dict[str, Callable] = {}
        
    def register_callback(self, indicator: str, callback: Callable[[str, float, pd.Timestamp], None]):
        """Register callback for indicator updates."""
        self.callbacks[indicator] = callback
        
    def _get_or_create_state(self, symbol: str) -> StreamingState:
        """Get or create streaming state for a symbol."""
        if symbol not in self.states:
            self.states[symbol] = StreamingState()
        return self.states[symbol]
    
    def _notify_callback(self, symbol: str, indicator: str, value: float, timestamp: pd.Timestamp):
        """Notify registered callbacks of indicator updates."""
        callback_key = f"{symbol}_{indicator}"
        if callback_key in self.callbacks:
            try:
                self.callbacks[callback_key](symbol, value, timestamp)
            except Exception as e:
                # Log error but don't break the stream
                pass
    
    def update_tick(
        self, 
        symbol: str, 
        timestamp: pd.Timestamp,
        open_price: float = None,
        high: float = None, 
        low: float = None, 
        close: float = None,
        volume: float = None
    ) -> Dict[str, float]:
        """
        Update all indicators with new tick data.
        
        Args:
            symbol: Symbol identifier
            timestamp: Timestamp of the tick
            open_price: Open price
            high: High price
            low: Low price  
            close: Close price
            volume: Volume
            
        Returns:
            Dictionary of updated indicator values
        """
        with self.lock:
            state = self._get_or_create_state(symbol)
            
            # Add to buffer
            tick_data = {
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            }
            state.buffer.append(tick_data)
            state.tick_count += 1
            
            results = {}
            
            if close is not None:
                # Update SMA
                if 'sma_20' not in state.periods:
                    state.periods['sma_20'] = 20
                    state.sma_sums['sma_20'] = 0.0
                
                results['sma_20'] = self._update_sma(state, close, 'sma_20', 20)
                
                # Update EMA
                results['ema_20'] = self._update_ema(state, close, 'ema_20', 20)
                
                # Update RSI
                results['rsi_14'] = self._update_rsi(state, close, 14)
                
                # Update MACD
                macd_results = self._update_macd(state, close, 12, 26, 9)
                results.update(macd_results)
            
            if high is not None and low is not None and close is not None:
                # Update ATR
                results['atr_14'] = self._update_atr(state, high, low, close, 14)
                
                # Update Bollinger Bands
                bb_results = self._update_bollinger_bands(state, close, 20, 2.0)
                results.update(bb_results)
            
            # Notify callbacks
            for indicator, value in results.items():
                if not np.isnan(value):
                    self._notify_callback(symbol, indicator, value, timestamp)
            
            return results
    
    def _update_sma(self, state: StreamingState, price: float, key: str, period: int) -> float:
        """Update Simple Moving Average incrementally."""
        if key not in state.last_values:
            state.last_values[key] = np.nan
            state.sma_sums[key] = 0.0
        
        # Get recent prices
        recent_prices = [tick['close'] for tick in list(state.buffer)[-period:] if tick['close'] is not None]
        
        if len(recent_prices) < period:
            return np.nan
        
        # Calculate SMA
        sma_value = sum(recent_prices) / period
        state.last_values[key] = sma_value
        return sma_value
    
    def _update_ema(self, state: StreamingState, price: float, key: str, period: int) -> float:
        """Update Exponential Moving Average incrementally."""
        multiplier_key = f"{key}_mult"
        
        if multiplier_key not in state.ema_multipliers:
            state.ema_multipliers[multiplier_key] = 2.0 / (period + 1)
        
        if key not in state.last_values or np.isnan(state.last_values[key]):
            state.last_values[key] = price
            return price
        
        multiplier = state.ema_multipliers[multiplier_key]
        ema_value = (price - state.last_values[key]) * multiplier + state.last_values[key]
        state.last_values[key] = ema_value
        return ema_value
    
    def _update_rsi(self, state: StreamingState, price: float, period: int = 14) -> float:
        """Update RSI incrementally."""
        if len(state.buffer) < 2:
            return np.nan
        
        # Get previous price
        prev_tick = list(state.buffer)[-2]
        prev_price = prev_tick['close']
        
        if prev_price is None:
            return np.nan
        
        # Calculate price change
        change = price - prev_price
        gain = max(change, 0)
        loss = max(-change, 0)
        
        # Add to gain/loss buffers
        state.rsi_gains.append(gain)
        state.rsi_losses.append(loss)
        
        # Need at least period values
        if len(state.rsi_gains) < period:
            return np.nan
        
        # Calculate average gain and loss
        recent_gains = list(state.rsi_gains)[-period:]
        recent_losses = list(state.rsi_losses)[-period:]
        
        avg_gain = sum(recent_gains) / period
        avg_loss = sum(recent_losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _update_macd(self, state: StreamingState, price: float, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Update MACD incrementally."""
        fast_mult = 2.0 / (fast + 1)
        slow_mult = 2.0 / (slow + 1)
        signal_mult = 2.0 / (signal + 1)
        
        # Initialize if needed
        if state.macd_fast_ema == 0:
            state.macd_fast_ema = price
            state.macd_slow_ema = price
            return {'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0}
        
        # Update EMAs
        state.macd_fast_ema = (price - state.macd_fast_ema) * fast_mult + state.macd_fast_ema
        state.macd_slow_ema = (price - state.macd_slow_ema) * slow_mult + state.macd_slow_ema
        
        # Calculate MACD line
        macd_line = state.macd_fast_ema - state.macd_slow_ema
        
        # Update signal line
        if state.macd_signal_ema == 0:
            state.macd_signal_ema = macd_line
        else:
            state.macd_signal_ema = (macd_line - state.macd_signal_ema) * signal_mult + state.macd_signal_ema
        
        # Calculate histogram
        histogram = macd_line - state.macd_signal_ema
        
        return {
            'macd': macd_line,
            'macd_signal': state.macd_signal_ema,
            'macd_histogram': histogram
        }
    
    def _update_atr(self, state: StreamingState, high: float, low: float, close: float, period: int = 14) -> float:
        """Update ATR incrementally."""
        if len(state.buffer) < 2:
            return np.nan
        
        # Get previous close
        prev_tick = list(state.buffer)[-2]
        prev_close = prev_tick['close']
        
        if prev_close is None:
            return np.nan
        
        # Calculate True Range
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        
        # Add to ATR buffer
        state.atr_values.append(tr)
        
        # Need at least period values
        if len(state.atr_values) < period:
            return np.nan
        
        # Calculate ATR as average of recent TR values
        recent_tr = list(state.atr_values)[-period:]
        atr_value = sum(recent_tr) / period
        
        return atr_value
    
    def _update_bollinger_bands(self, state: StreamingState, price: float, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Update Bollinger Bands incrementally."""
        # Add price to buffer
        state.bb_values.append(price)
        
        # Need at least period values
        if len(state.bb_values) < period:
            return {'bb_upper': np.nan, 'bb_middle': np.nan, 'bb_lower': np.nan}
        
        # Get recent prices
        recent_prices = list(state.bb_values)[-period:]
        
        # Calculate middle band (SMA)
        middle = sum(recent_prices) / period
        
        # Calculate standard deviation
        variance = sum((p - middle) ** 2 for p in recent_prices) / period
        std = variance ** 0.5
        
        # Calculate bands
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return {
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower
        }
    
    def get_current_values(self, symbol: str) -> Dict[str, float]:
        """Get current indicator values for a symbol."""
        with self.lock:
            if symbol not in self.states:
                return {}
            
            state = self.states[symbol]
            return state.last_values.copy()
    
    def reset_symbol(self, symbol: str):
        """Reset all indicators for a symbol."""
        with self.lock:
            if symbol in self.states:
                del self.states[symbol]
    
    def get_buffer_data(self, symbol: str, limit: int = None) -> pd.DataFrame:
        """Get historical buffer data as DataFrame."""
        with self.lock:
            if symbol not in self.states:
                return pd.DataFrame()
            
            state = self.states[symbol]
            buffer_list = list(state.buffer)
            
            if limit:
                buffer_list = buffer_list[-limit:]
            
            if not buffer_list:
                return pd.DataFrame()
            
            df = pd.DataFrame(buffer_list)
            df.set_index('timestamp', inplace=True)
            return df

class StreamingDataFeed:
    """
    Mock streaming data feed for testing and development.
    """
    
    def __init__(self, data: pd.DataFrame, speed_multiplier: float = 1.0):
        """
        Initialize streaming data feed.
        
        Args:
            data: Historical data to stream
            speed_multiplier: Speed multiplier (1.0 = real-time, 10.0 = 10x faster)
        """
        self.data = data.copy()
        self.speed_multiplier = speed_multiplier
        self.current_index = 0
        self.is_running = False
        self.subscribers = []
        self.thread = None
        
    def subscribe(self, callback: Callable):
        """Subscribe to data feed updates."""
        self.subscribers.append(callback)
        
    def start(self):
        """Start streaming data."""
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._stream_data, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop streaming data."""
        self.is_running = False
        if self.thread:
            self.thread.join()
            
    def _stream_data(self):
        """Internal method to stream data."""
        while self.is_running and self.current_index < len(self.data):
            row = self.data.iloc[self.current_index]
            timestamp = self.data.index[self.current_index]
            
            # Notify subscribers
            for callback in self.subscribers:
                try:
                    callback(timestamp, row)
                except Exception as e:
                    # Log error but continue streaming
                    pass
            
            self.current_index += 1
            
            # Sleep based on speed multiplier
            if self.speed_multiplier > 0:
                time.sleep(1.0 / self.speed_multiplier)

# Utility functions for streaming setup
def create_streaming_setup(historical_data: pd.DataFrame, symbol: str = "DEFAULT") -> tuple[StreamingIndicators, StreamingDataFeed]:
    """
    Create a complete streaming setup with indicators and data feed.
    
    Args:
        historical_data: Historical OHLCV data
        symbol: Symbol identifier
        
    Returns:
        Tuple of (StreamingIndicators, StreamingDataFeed)
    """
    indicators = StreamingIndicators()
    
    def data_callback(timestamp, row):
        indicators.update_tick(
            symbol=symbol,
            timestamp=timestamp,
            open_price=row.get('open'),
            high=row.get('high'),
            low=row.get('low'),
            close=row.get('close'),
            volume=row.get('volume')
        )
    
    feed = StreamingDataFeed(historical_data)
    feed.subscribe(data_callback)
    
    return indicators, feed

__all__ = [
    'StreamingIndicators',
    'StreamingDataFeed', 
    'StreamingState',
    'create_streaming_setup'
]