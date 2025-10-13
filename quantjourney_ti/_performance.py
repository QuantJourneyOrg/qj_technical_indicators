"""
QuantJourney Technical-Indicators - Performance Optimizations
============================================================
Caching, memory management, and performance utilities for hedge fund production use.

Author: Jakub Polec <jakub@quantjourney.pro>
License: MIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable, Tuple, Union
from functools import lru_cache, wraps
import hashlib
import pickle
import threading
import time
import psutil
import gc
from dataclasses import dataclass
from collections import defaultdict

# Performance monitoring
@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    function_name: str
    execution_time: float
    memory_usage_mb: float
    cache_hit: bool
    input_size: int
    timestamp: float

class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.metrics_history = []
        self.max_history = max_history
        self.lock = threading.Lock()
        
    def record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric."""
        with self.lock:
            self.metrics_history.append(metric)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
    
    def get_stats(self, function_name: str = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self.lock:
            if function_name:
                metrics = [m for m in self.metrics_history if m.function_name == function_name]
            else:
                metrics = self.metrics_history
            
            if not metrics:
                return {}
            
            execution_times = [m.execution_time for m in metrics]
            memory_usage = [m.memory_usage_mb for m in metrics]
            cache_hits = sum(1 for m in metrics if m.cache_hit)
            
            return {
                'count': len(metrics),
                'avg_execution_time': np.mean(execution_times),
                'max_execution_time': np.max(execution_times),
                'min_execution_time': np.min(execution_times),
                'avg_memory_usage_mb': np.mean(memory_usage),
                'cache_hit_rate': cache_hits / len(metrics) if metrics else 0,
                'total_cache_hits': cache_hits
            }

# Global performance monitor
_performance_monitor = PerformanceMonitor()

def get_performance_stats(function_name: str = None) -> Dict[str, Any]:
    """Get performance statistics for monitoring."""
    return _performance_monitor.get_stats(function_name)

# Advanced caching system
class IndicatorCache:
    """Advanced caching system for technical indicators."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function arguments."""
        # Convert pandas objects to hashable representations
        hashable_args = []
        for arg in args:
            if isinstance(arg, (pd.Series, pd.DataFrame)):
                # Use shape, dtype, and hash of first/last few values
                if len(arg) > 10:
                    sample = pd.concat([arg.head(5), arg.tail(5)])
                else:
                    sample = arg
                hashable_args.append((arg.shape, str(arg.dtype), hash(tuple(sample.values.flatten()))))
            else:
                hashable_args.append(arg)
        
        # Create hash from function name, args, and kwargs
        key_data = (func_name, tuple(hashable_args), tuple(sorted(kwargs.items())))
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            # Check TTL
            if time.time() - self.access_times[key] > self.ttl_seconds:
                del self.cache[key]
                del self.access_times[key]
                self.miss_count += 1
                return None
            
            self.access_times[key] = time.time()
            self.hit_count += 1
            return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        with self.lock:
            # Evict oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': self.hit_count / total_requests if total_requests > 0 else 0,
                'ttl_seconds': self.ttl_seconds
            }

# Global cache instance
_indicator_cache = IndicatorCache()

def cached_indicator(ttl_seconds: int = 3600):
    """
    Decorator for caching indicator calculations.
    
    Args:
        ttl_seconds: Time to live for cached results
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _indicator_cache._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = _indicator_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Calculate and cache result
            start_time = time.time()
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            # Determine input size
            input_size = 0
            for arg in args:
                if isinstance(arg, (pd.Series, pd.DataFrame)):
                    input_size = max(input_size, len(arg))
            
            # Record performance metrics
            metric = PerformanceMetrics(
                function_name=func.__name__,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cache_hit=False,
                input_size=input_size,
                timestamp=time.time()
            )
            _performance_monitor.record_metric(metric)
            
            # Cache result
            _indicator_cache.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator

def clear_indicator_cache():
    """Clear the indicator cache."""
    _indicator_cache.clear()

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return _indicator_cache.get_stats()

# Memory optimization utilities
class MemoryManager:
    """Memory management utilities for large datasets."""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage.
        
        Args:
            df: Input DataFrame
            aggressive: Use more aggressive optimization (may lose precision)
        """
        optimized = df.copy()
        
        for col in optimized.columns:
            col_type = optimized[col].dtype
            
            if col_type == 'object':
                # Try to convert to category if many repeats
                if optimized[col].nunique() / len(optimized) < 0.5:
                    optimized[col] = optimized[col].astype('category')
            
            elif 'int' in str(col_type):
                c_min = optimized[col].min()
                c_max = optimized[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized[col] = optimized[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized[col] = optimized[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized[col] = optimized[col].astype(np.int32)
            
            elif 'float' in str(col_type):
                c_min = optimized[col].min()
                c_max = optimized[col].max()
                
                if aggressive:
                    # More aggressive - use float16 if possible
                    if (c_min > np.finfo(np.float16).min and 
                        c_max < np.finfo(np.float16).max):
                        optimized[col] = optimized[col].astype(np.float16)
                    elif (c_min > np.finfo(np.float32).min and 
                          c_max < np.finfo(np.float32).max):
                        optimized[col] = optimized[col].astype(np.float32)
                else:
                    # Conservative - only use float32
                    if (c_min > np.finfo(np.float32).min and 
                        c_max < np.finfo(np.float32).max):
                        optimized[col] = optimized[col].astype(np.float32)
        
        return optimized
    
    @staticmethod
    def get_memory_usage(obj) -> Dict[str, float]:
        """Get memory usage statistics for an object."""
        if isinstance(obj, pd.DataFrame):
            memory_usage = obj.memory_usage(deep=True)
            return {
                'total_mb': memory_usage.sum() / 1024 / 1024,
                'per_column_mb': (memory_usage / 1024 / 1024).to_dict()
            }
        elif isinstance(obj, pd.Series):
            memory_usage = obj.memory_usage(deep=True)
            return {
                'total_mb': memory_usage / 1024 / 1024
            }
        else:
            import sys
            return {
                'total_mb': sys.getsizeof(obj) / 1024 / 1024
            }
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection and return memory freed."""
        import gc
        before = psutil.Process().memory_info().rss / 1024 / 1024
        gc.collect()
        after = psutil.Process().memory_info().rss / 1024 / 1024
        return before - after

# Batch processing utilities
class BatchProcessor:
    """Process multiple symbols/datasets efficiently."""
    
    def __init__(self, batch_size: int = 100, n_workers: int = None):
        self.batch_size = batch_size
        self.n_workers = n_workers or min(32, (psutil.cpu_count() or 1) + 4)
        
    def process_symbols(
        self, 
        data_dict: Dict[str, pd.DataFrame], 
        indicator_func: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process indicators for multiple symbols efficiently.
        
        Args:
            data_dict: Dictionary of symbol -> DataFrame
            indicator_func: Function to calculate indicator
            **kwargs: Arguments for indicator function
            
        Returns:
            Dictionary of symbol -> indicator results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        def process_single(symbol_data_pair):
            symbol, data = symbol_data_pair
            try:
                result = indicator_func(data, **kwargs)
                return symbol, result
            except Exception as e:
                return symbol, None
        
        # Process in batches to manage memory
        symbols = list(data_dict.keys())
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            for i in range(0, len(symbols), self.batch_size):
                batch_symbols = symbols[i:i + self.batch_size]
                batch_data = [(s, data_dict[s]) for s in batch_symbols]
                
                # Submit batch
                futures = {
                    executor.submit(process_single, item): item[0] 
                    for item in batch_data
                }
                
                # Collect results
                for future in as_completed(futures):
                    symbol, result = future.result()
                    if result is not None:
                        results[symbol] = result
                
                # Force garbage collection between batches
                if i + self.batch_size < len(symbols):
                    MemoryManager.force_garbage_collection()
        
        return results

# Performance profiling decorator
def profile_performance(include_memory: bool = True):
    """
    Decorator to profile function performance.
    
    Args:
        include_memory: Whether to include memory profiling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            if include_memory:
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024
            
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            if include_memory:
                memory_after = process.memory_info().rss / 1024 / 1024
                memory_usage = memory_after - memory_before
            else:
                memory_usage = 0
            
            # Determine input size
            input_size = 0
            for arg in args:
                if isinstance(arg, (pd.Series, pd.DataFrame)):
                    input_size = max(input_size, len(arg))
            
            # Record metrics
            metric = PerformanceMetrics(
                function_name=func.__name__,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                cache_hit=False,
                input_size=input_size,
                timestamp=time.time()
            )
            _performance_monitor.record_metric(metric)
            
            return result
        
        return wrapper
    return decorator

# System resource monitoring
def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage."""
    process = psutil.Process()
    
    return {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
        'process_memory_mb': process.memory_info().rss / 1024 / 1024,
        'process_cpu_percent': process.cpu_percent(),
        'disk_usage_percent': psutil.disk_usage('/').percent,
        'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
    }

__all__ = [
    'PerformanceMetrics',
    'PerformanceMonitor', 
    'IndicatorCache',
    'MemoryManager',
    'BatchProcessor',
    'cached_indicator',
    'profile_performance',
    'clear_indicator_cache',
    'get_cache_stats',
    'get_performance_stats',
    'get_system_resources'
]