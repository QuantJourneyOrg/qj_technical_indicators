"""
Hedge Fund Production Example
============================

This example demonstrates the enhanced features for hedge fund production use:
- Risk metrics calculation
- Streaming data processing
- Performance optimization
- Multi-asset batch processing
- Memory management

Author: Jakub Polec <jakub@quantjourney.pro>
License: MIT
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time

# Import the enhanced library
import quantjourney_ti as qti
from quantjourney_ti import (
    TechnicalIndicators, 
    StreamingIndicators,
    calculate_risk_metrics,
    get_performance_stats,
    get_cache_stats,
    MemoryManager,
    BatchProcessor
)

def fetch_portfolio_data(symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'], period='2y'):
    """Fetch data for multiple symbols."""
    print(f"Fetching data for {len(symbols)} symbols...")
    data_dict = {}
    
    for symbol in symbols:
        try:
            df = yf.download(symbol, period=period, progress=False)
            if not df.empty:
                # Clean column names
                df.columns = df.columns.str.lower().str.replace(' ', '_')
                # Handle volume zeros
                df['volume'] = df['volume'].replace(0, np.nan).ffill()
                data_dict[symbol] = df
                print(f"✓ {symbol}: {len(df)} rows")
            else:
                print(f"✗ {symbol}: No data")
        except Exception as e:
            print(f"✗ {symbol}: Error - {e}")
    
    return data_dict

def demonstrate_risk_metrics():
    """Demonstrate comprehensive risk metrics calculation."""
    print("\n" + "="*60)
    print("RISK METRICS DEMONSTRATION")
    print("="*60)
    
    # Fetch sample data
    aapl = yf.download('AAPL', period='2y', progress=False)
    spy = yf.download('SPY', period='2y', progress=False)  # Benchmark
    
    # Clean data
    aapl.columns = aapl.columns.str.lower().str.replace(' ', '_')
    spy.columns = spy.columns.str.lower().str.replace(' ', '_')
    
    # Calculate comprehensive risk metrics
    print("Calculating risk metrics for AAPL vs SPY benchmark...")
    
    ti = TechnicalIndicators()
    risk_metrics = ti.RISK_METRICS(
        data=aapl['close'],
        benchmark=spy['close'],
        risk_free_rate=0.05,  # 5% risk-free rate
        periods_per_year=252,
        confidence_level=0.05
    )
    
    print("\nRisk Metrics Results:")
    print("-" * 40)
    for metric, value in risk_metrics.items():
        if isinstance(value, (int, float)):
            if 'ratio' in metric.lower() or 'return' in metric.lower():
                print(f"{metric:.<25} {value:>8.3f}")
            elif 'drawdown' in metric.lower() or 'var' in metric.lower():
                print(f"{metric:.<25} {value:>8.2%}")
            else:
                print(f"{metric:.<25} {value:>8.4f}")
        else:
            print(f"{metric:.<25} {str(value):>15}")

def demonstrate_streaming_indicators():
    """Demonstrate real-time streaming indicators."""
    print("\n" + "="*60)
    print("STREAMING INDICATORS DEMONSTRATION")
    print("="*60)
    
    # Get historical data for simulation
    df = yf.download('AAPL', period='5d', interval='1m', progress=False)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Create streaming setup
    streaming_indicators, data_feed = qti.create_streaming_setup(df, symbol='AAPL')
    
    # Set up callback to track updates
    results = []
    
    def indicator_callback(symbol, value, timestamp):
        results.append({
            'timestamp': timestamp,
            'symbol': symbol, 
            'value': value
        })
    
    # Register callbacks for key indicators
    streaming_indicators.register_callback('AAPL_sma_20', indicator_callback)
    streaming_indicators.register_callback('AAPL_rsi_14', indicator_callback)
    
    print("Starting streaming simulation (processing 100 ticks)...")
    
    # Simulate streaming data
    for i, (timestamp, row) in enumerate(df.head(100).iterrows()):
        indicators = streaming_indicators.update_tick(
            symbol='AAPL',
            timestamp=timestamp,
            open_price=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume']
        )
        
        if i % 20 == 0:  # Print every 20th update
            print(f"Tick {i:3d}: SMA={indicators.get('sma_20', np.nan):7.2f}, "
                  f"RSI={indicators.get('rsi_14', np.nan):6.2f}, "
                  f"MACD={indicators.get('macd', np.nan):7.4f}")
    
    print(f"\nStreaming complete. Processed {len(results)} indicator updates.")
    
    # Get final values
    final_values = streaming_indicators.get_current_values('AAPL')
    print("\nFinal Indicator Values:")
    for indicator, value in final_values.items():
        if not np.isnan(value):
            print(f"  {indicator}: {value:.4f}")

def demonstrate_batch_processing():
    """Demonstrate efficient multi-asset processing."""
    print("\n" + "="*60)
    print("BATCH PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Fetch portfolio data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    data_dict = fetch_portfolio_data(symbols, period='1y')
    
    if not data_dict:
        print("No data available for batch processing demo")
        return
    
    ti = TechnicalIndicators()
    
    # Demonstrate batch RSI calculation
    print(f"\nCalculating RSI for {len(data_dict)} symbols...")
    start_time = time.time()
    
    rsi_results = ti.batch_calculate(
        data_dict=data_dict,
        indicator_name='RSI',
        period=14
    )
    
    batch_time = time.time() - start_time
    print(f"Batch processing completed in {batch_time:.2f} seconds")
    
    # Show results
    print("\nRSI Results (last 5 values):")
    for symbol, rsi_series in rsi_results.items():
        if rsi_series is not None:
            last_values = rsi_series.dropna().tail(5)
            print(f"{symbol}: {last_values.iloc[-1]:.2f} (avg: {last_values.mean():.2f})")

def demonstrate_performance_optimization():
    """Demonstrate performance monitoring and caching."""
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATION DEMONSTRATION")
    print("="*60)
    
    # Get sample data
    df = yf.download('AAPL', period='2y', progress=False)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    ti = TechnicalIndicators()
    
    # First calculation (no cache)
    print("First SMA calculation (no cache)...")
    start_time = time.time()
    sma1 = ti.SMA(df['close'], period=20)
    first_time = time.time() - start_time
    
    # Second calculation (should use cache)
    print("Second SMA calculation (with cache)...")
    start_time = time.time()
    sma2 = ti.SMA(df['close'], period=20)
    second_time = time.time() - start_time
    
    print(f"First calculation: {first_time:.4f} seconds")
    print(f"Second calculation: {second_time:.4f} seconds")
    print(f"Speedup: {first_time/second_time:.1f}x")
    
    # Show cache statistics
    cache_stats = qti.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")
    print(f"  Cache Size: {cache_stats['size']}/{cache_stats['max_size']}")
    
    # Show performance statistics
    perf_stats = qti.get_performance_stats('SMA')
    if perf_stats:
        print(f"\nPerformance Statistics for SMA:")
        print(f"  Average execution time: {perf_stats['avg_execution_time']:.4f}s")
        print(f"  Cache hit rate: {perf_stats['cache_hit_rate']:.1%}")
        print(f"  Total calls: {perf_stats['count']}")

def demonstrate_memory_management():
    """Demonstrate memory optimization features."""
    print("\n" + "="*60)
    print("MEMORY MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    # Create large dataset
    print("Creating large dataset for memory optimization demo...")
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='1min')
    large_df = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 102,
        'low': np.random.randn(len(dates)).cumsum() + 98,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Show original memory usage
    original_memory = MemoryManager.get_memory_usage(large_df)
    print(f"Original memory usage: {original_memory['total_mb']:.2f} MB")
    
    # Optimize memory
    optimized_df = MemoryManager.optimize_dataframe(large_df, aggressive=False)
    optimized_memory = MemoryManager.get_memory_usage(optimized_df)
    
    print(f"Optimized memory usage: {optimized_memory['total_mb']:.2f} MB")
    print(f"Memory savings: {(1 - optimized_memory['total_mb']/original_memory['total_mb']):.1%}")
    
    # Show system resources
    resources = qti.get_system_resources()
    print(f"\nSystem Resources:")
    print(f"  CPU Usage: {resources['cpu_percent']:.1f}%")
    print(f"  Memory Usage: {resources['memory_percent']:.1f}%")
    print(f"  Available Memory: {resources['memory_available_gb']:.1f} GB")
    print(f"  Process Memory: {resources['process_memory_mb']:.1f} MB")

def demonstrate_market_data_validation():
    """Demonstrate enhanced market data validation."""
    print("\n" + "="*60)
    print("MARKET DATA VALIDATION DEMONSTRATION")
    print("="*60)
    
    # Create problematic market data
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
    problematic_df = pd.DataFrame({
        'open': [100, 101, np.nan, 103, 104, 105, np.nan, 107, 108, 109],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close': [101, 102, np.nan, 104, 105, 106, np.nan, 108, 109, 110],
        'volume': [1000, 0, 1200, 0, 1400, 1500, 1600, 0, 1800, 1900]  # Zero volumes
    }, index=dates)
    
    print("Original data issues:")
    print(f"  NaN values: {problematic_df.isnull().sum().sum()}")
    print(f"  Zero volumes: {(problematic_df['volume'] == 0).sum()}")
    
    ti = TechnicalIndicators()
    
    # Validate and fix data
    fixed_df = ti.validate_market_data(
        problematic_df, 
        allow_gaps=True, 
        fix_common_issues=True
    )
    
    print("\nAfter validation and fixing:")
    print(f"  NaN values: {fixed_df.isnull().sum().sum()}")
    print(f"  Zero volumes: {(fixed_df['volume'] == 0).sum()}")
    print("✓ Data validation and fixing completed successfully")

def main():
    """Run all demonstrations."""
    print("QUANTJOURNEY TECHNICAL INDICATORS")
    print("Hedge Fund Production Features Demo")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        demonstrate_risk_metrics()
        demonstrate_streaming_indicators()
        demonstrate_batch_processing()
        demonstrate_performance_optimization()
        demonstrate_memory_management()
        demonstrate_market_data_validation()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Final performance summary
        print("\nFinal Performance Summary:")
        cache_stats = qti.get_cache_stats()
        print(f"  Total cache hits: {cache_stats['hit_count']}")
        print(f"  Overall hit rate: {cache_stats['hit_rate']:.1%}")
        
        # Clear cache for clean exit
        qti.clear_indicator_cache()
        print("  Cache cleared for clean exit")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()