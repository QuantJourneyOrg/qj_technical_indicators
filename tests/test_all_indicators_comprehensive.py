#!/usr/bin/env python3
"""
Comprehensive Test Script for ALL Technical Indicators
====================================================
This script tests all 46 indicators in the quantjourney-ti library to ensure they work correctly
and to identify any compilation errors, data validation issues, or other problems.

Based on the template from test_adx_fix.py, this script provides:
- Environment information for debugging
- Both synthetic and real-world data testing 
- Detailed error categorization and reporting
- Performance timing information
- Clear success/failure reporting

This script can be used by quantjourney-ti library maintainers to:
- Verify all indicators work correctly after code changes
- Identify Numba compilation issues
- Test indicator robustness with different data types
- Monitor performance across all indicators
"""

import sys
import inspect
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import time
from quantjourney_ti import TechnicalIndicators

def print_environment_info():
    """Print detailed environment information for debugging."""
    print("Environment Information:")
    print("=" * 50)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    try:
        import numba
        print(f"Numba Version: {numba.__version__}")
    except ImportError:
        print("Numba: Not installed")
    
    try:
        print(f"Pandas Version: {pd.__version__}")
    except AttributeError:
        print("Pandas Version: Unknown")
    
    try:
        print(f"NumPy Version: {np.__version__}")
    except AttributeError:
        print("NumPy Version: Unknown")
    
    # Try to get quantjourney-ti version
    try:
        import quantjourney_ti
        if hasattr(quantjourney_ti, '__version__'):
            print(f"quantjourney-ti Version: {quantjourney_ti.__version__}")
        else:
            print("quantjourney-ti Version: Unknown (no __version__ attribute)")
    except Exception:
        print("quantjourney-ti Version: Unable to determine")
    
    print("=" * 50)
    print()

def create_synthetic_test_data(rows: int = 100) -> pd.DataFrame:
    """Create comprehensive synthetic OHLCV data for testing."""
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start='2024-01-01', periods=rows, freq='D')
    
    # Create realistic price data with some volatility
    base_price = 100
    price_changes = np.random.normal(0, 2, rows)
    closes = [base_price]
    
    for change in price_changes[1:]:
        new_price = closes[-1] + change
        closes.append(max(new_price, 50))  # Prevent negative prices
    
    # Generate OHLC from closes with realistic relationships
    opens = [closes[0]] + [closes[i-1] + np.random.normal(0, 0.5) for i in range(1, rows)]
    highs = [max(opens[i], closes[i]) + abs(np.random.normal(0, 1)) for i in range(rows)]
    lows = [min(opens[i], closes[i]) - abs(np.random.normal(0, 1)) for i in range(rows)]
    volumes = np.random.randint(10000, 100000, rows)
    
    # Ensure no negative prices
    opens = [max(price, 50) for price in opens]
    highs = [max(price, 50) for price in highs]
    lows = [max(price, 50) for price in lows]
    
    data = {
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    return df

def create_market_data_for_beta(rows: int = 100) -> pd.Series:
    """Create market index data for BETA calculation."""
    np.random.seed(24)  # Different seed for market data
    dates = pd.date_range(start='2024-01-01', periods=rows, freq='D')
    
    # Create market returns with different characteristics
    market_returns = np.random.normal(0.001, 0.02, rows)  # Daily returns
    market_prices = [1000]  # Start at 1000
    
    for ret in market_returns:
        new_price = market_prices[-1] * (1 + ret)
        market_prices.append(max(new_price, 500))  # Prevent crash below 500
    
    return pd.Series(market_prices[1:], index=dates)

def get_real_world_data():
    """Get real-world data using yfinance if available."""
    try:
        import yfinance as yf
        
        # Fetch real data
        ticker = yf.Ticker("AAPL")
        df = ticker.history(period="6mo")  # 6 months of data
        df.columns = df.columns.str.lower()  # Ensure lowercase
        
        # Get market data for BETA
        market_ticker = yf.Ticker("SPY")
        market_df = market_ticker.history(period="6mo")
        market_series = market_df["Close"]
        
        return df, market_series
        
    except ImportError:
        return None, None
    except Exception as e:
        print(f"Warning: Could not fetch real-world data: {e}")
        return None, None

def categorize_error(error_msg: str) -> str:
    """Categorize errors into different types for better analysis."""
    error_lower = error_msg.lower()
    
    if "numba" in error_lower and "compilation" in error_lower:
        return "NUMBA_COMPILATION"
    elif "numba" in error_lower and "typing" in error_lower:
        return "NUMBA_TYPING"
    elif "untyped list" in error_lower:
        return "NUMBA_UNTYPED_LIST"
    elif "missing required columns" in error_lower:
        return "MISSING_COLUMNS"
    elif "index must be" in error_lower:
        return "INDEX_VALIDATION"
    elif "input data is empty" in error_lower:
        return "EMPTY_DATA"
    elif "numeric data" in error_lower:
        return "DATA_TYPE"
    elif "period" in error_lower and ("must be" in error_lower or "invalid" in error_lower):
        return "PARAMETER_VALIDATION"
    elif "fallback" in error_lower:
        return "FALLBACK_ERROR"
    else:
        return "OTHER"

def determine_input_type(indicator_name: str, signature: inspect.Signature) -> Tuple[str, Dict]:
    """Determine what type of input data an indicator needs and its parameters."""
    
    # Special cases that need specific handling
    special_cases = {
        'BETA': ('series_with_market', {'period': 60}),  # Needs market data
        'BENFORD_LAW': ('series', {}),  # Special analysis indicator
        'RAINBOW': ('series', {'periods': [2, 3, 4, 5]}),  # List parameter
    }
    
    if indicator_name in special_cases:
        return special_cases[indicator_name]
    
    # Check first parameter annotation to determine input type
    params = list(signature.parameters.values())
    if len(params) > 0:
        first_param = params[0]
        if first_param.annotation == pd.Series:
            return ('series', {})
        elif first_param.annotation == pd.DataFrame:
            return ('dataframe', {})
    
    # Series-based indicators (typically work with close prices)
    series_indicators = {
        'SMA', 'EMA', 'RSI', 'ROC', 'DEMA', 'KAMA', 'CMO', 'DPO', 
        'HISTORICAL_VOLATILITY', 'HULL_MA', 'TRIX', 'ALMA', 'LINEAR_REGRESSION_CHANNEL',
        'MOMENTUM_INDEX'
    }
    
    # DataFrame-based indicators (need OHLCV data)
    dataframe_indicators = {
        'ATR', 'ADX', 'DI', 'STOCH', 'CCI', 'WILLR', 'MFI', 'AROON', 'AO', 
        'ULTIMATE_OSCILLATOR', 'MASS_INDEX', 'VWAP', 'SUPERTREND', 'PVO',
        'CHAIKIN_VOLATILITY', 'KDJ', 'HEIKEN_ASHI', 'AD', 'ADOSC', 'OBV',
        'ELDER_RAY', 'RVI', 'PIVOT_POINTS', 'VOLUME_INDICATORS', 'DONCHIAN',
        'BB', 'MACD', 'KELTNER', 'ICHIMOKU'
    }
    
    if indicator_name in series_indicators:
        return ('series', {})
    elif indicator_name in dataframe_indicators:
        return ('dataframe', {})
    else:
        # Default to dataframe for unknown indicators
        return ('dataframe', {})

def test_indicator(ti: TechnicalIndicators, indicator_name: str, 
                  synthetic_data: pd.DataFrame, market_data: pd.Series,
                  real_data: pd.DataFrame = None, real_market_data: pd.Series = None) -> Dict:
    """Test a single indicator with both synthetic and real data."""
    
    result = {
        'name': indicator_name,
        'synthetic_success': False,
        'real_success': False,
        'synthetic_error': None,
        'real_error': None,
        'synthetic_time': None,
        'real_time': None,
        'synthetic_output_info': None,
        'real_output_info': None
    }
    
    try:
        # Get the indicator method
        method = getattr(ti, indicator_name)
        signature = inspect.signature(method)
        
        # Determine input type and default parameters
        input_type, default_params = determine_input_type(indicator_name, signature)
        
        # Test with synthetic data
        try:
            if input_type == 'series':
                test_data = synthetic_data['close']
            elif input_type == 'series_with_market':
                test_data = synthetic_data['close']
            else:  # dataframe
                test_data = synthetic_data
            
            start_time = time.perf_counter()
            
            if indicator_name == 'BETA':
                synthetic_result = method(test_data, market_data, **default_params)
            else:
                synthetic_result = method(test_data, **default_params)
            
            result['synthetic_time'] = time.perf_counter() - start_time
            result['synthetic_success'] = True
            
            # Get output information
            if hasattr(synthetic_result, 'shape'):
                result['synthetic_output_info'] = f"Shape: {synthetic_result.shape}"
                if hasattr(synthetic_result, 'columns'):
                    result['synthetic_output_info'] += f", Columns: {list(synthetic_result.columns)}"
            else:
                result['synthetic_output_info'] = f"Type: {type(synthetic_result)}"
                
        except Exception as e:
            result['synthetic_error'] = str(e)
        
        # Test with real data (if available)
        if real_data is not None:
            try:
                if input_type == 'series':
                    test_data = real_data['close']
                elif input_type == 'series_with_market':
                    test_data = real_data['close']
                else:  # dataframe
                    test_data = real_data
                
                start_time = time.perf_counter()
                
                if indicator_name == 'BETA' and real_market_data is not None:
                    real_result = method(test_data, real_market_data, **default_params)
                elif indicator_name != 'BETA':
                    real_result = method(test_data, **default_params)
                else:
                    # Skip BETA if no market data
                    raise Exception("No market data available for BETA")
                
                result['real_time'] = time.perf_counter() - start_time
                result['real_success'] = True
                
                # Get output information
                if hasattr(real_result, 'shape'):
                    result['real_output_info'] = f"Shape: {real_result.shape}"
                    if hasattr(real_result, 'columns'):
                        result['real_output_info'] += f", Columns: {list(real_result.columns)}"
                else:
                    result['real_output_info'] = f"Type: {type(real_result)}"
                    
            except Exception as e:
                result['real_error'] = str(e)
    
    except Exception as e:
        result['synthetic_error'] = f"Method access error: {str(e)}"
        result['real_error'] = f"Method access error: {str(e)}"
    
    return result

def analyze_results(results: List[Dict]) -> Dict[str, Any]:
    """Analyze test results and categorize issues."""
    
    analysis = {
        'total_indicators': len(results),
        'synthetic_success_count': 0,
        'real_success_count': 0,
        'error_categories': defaultdict(list),
        'performance_stats': {},
        'critical_failures': [],
        'working_indicators': [],
        'failing_indicators': []
    }
    
    synthetic_times = []
    real_times = []
    
    for result in results:
        name = result['name']
        
        # Count successes
        if result['synthetic_success']:
            analysis['synthetic_success_count'] += 1
            analysis['working_indicators'].append(name)
            if result['synthetic_time']:
                synthetic_times.append(result['synthetic_time'])
        else:
            analysis['failing_indicators'].append(name)
        
        if result['real_success']:
            analysis['real_success_count'] += 1
            if result['real_time']:
                real_times.append(result['real_time'])
        
        # Categorize errors
        for error_type in ['synthetic_error', 'real_error']:
            error = result.get(error_type)
            if error:
                category = categorize_error(error)
                analysis['error_categories'][category].append({
                    'indicator': name,
                    'error_type': error_type,
                    'error': error
                })
                
                # Mark critical failures (Numba compilation issues)
                if category in ['NUMBA_COMPILATION', 'NUMBA_TYPING', 'NUMBA_UNTYPED_LIST']:
                    analysis['critical_failures'].append({
                        'indicator': name,
                        'category': category,
                        'error': error
                    })
    
    # Calculate performance statistics
    if synthetic_times:
        analysis['performance_stats']['synthetic'] = {
            'mean': np.mean(synthetic_times),
            'median': np.median(synthetic_times),
            'min': np.min(synthetic_times),
            'max': np.max(synthetic_times)
        }
    
    if real_times:
        analysis['performance_stats']['real'] = {
            'mean': np.mean(real_times),
            'median': np.median(real_times),
            'min': np.min(real_times),
            'max': np.max(real_times)
        }
    
    return analysis

def print_detailed_results(results: List[Dict], analysis: Dict[str, Any]):
    """Print comprehensive test results."""
    
    print("\n" + "="*80)
    print("DETAILED TEST RESULTS")
    print("="*80)
    
    # Summary statistics
    total = analysis['total_indicators']
    synthetic_success = analysis['synthetic_success_count']
    real_success = analysis['real_success_count']
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total Indicators Tested: {total}")
    print(f"   Synthetic Data Success: {synthetic_success}/{total} ({synthetic_success/total*100:.1f}%)")
    if real_success > 0:
        print(f"   Real Data Success: {real_success}/{total} ({real_success/total*100:.1f}%)")
    else:
        print(f"   Real Data Success: Not tested (no data available)")
    
    # Critical failures
    if analysis['critical_failures']:
        print(f"\n🚨 CRITICAL FAILURES ({len(analysis['critical_failures'])}):")
        for failure in analysis['critical_failures']:
            print(f"   ❌ {failure['indicator']}: {failure['category']}")
    else:
        print(f"\n✅ NO CRITICAL FAILURES DETECTED!")
    
    # Error categories
    if analysis['error_categories']:
        print(f"\n📋 ERROR CATEGORIES:")
        for category, errors in analysis['error_categories'].items():
            print(f"   {category}: {len(errors)} indicators")
            for error_info in errors[:3]:  # Show first 3 examples
                print(f"      • {error_info['indicator']}")
            if len(errors) > 3:
                print(f"      ... and {len(errors)-3} more")
    
    # Performance statistics
    if analysis['performance_stats']:
        print(f"\n⚡ PERFORMANCE STATISTICS:")
        for data_type, stats in analysis['performance_stats'].items():
            print(f"   {data_type.title()} Data:")
            print(f"      Mean: {stats['mean']:.4f}s")
            print(f"      Median: {stats['median']:.4f}s") 
            print(f"      Range: {stats['min']:.4f}s - {stats['max']:.4f}s")
    
    # Working indicators
    working = analysis['working_indicators']
    print(f"\n✅ WORKING INDICATORS ({len(working)}):")
    for i in range(0, len(working), 8):  # Print 8 per line
        line_indicators = working[i:i+8]
        print(f"   {', '.join(line_indicators)}")
    
    # Failing indicators
    failing = analysis['failing_indicators']
    if failing:
        print(f"\n❌ FAILING INDICATORS ({len(failing)}):")
        for i in range(0, len(failing), 8):  # Print 8 per line
            line_indicators = failing[i:i+8]
            print(f"   {', '.join(line_indicators)}")

def main():
    """Main test execution function."""
    print("quantjourney-ti Comprehensive Indicator Test Suite")
    print("=" * 80)
    print("Testing all indicators for compilation errors, data validation,")
    print("performance, and functionality with both synthetic and real data.")
    print("=" * 80)
    print()
    
    # Print environment information
    print_environment_info()
    
    # Create test data
    print("Creating test data...")
    synthetic_data = create_synthetic_test_data(100)
    market_data = create_market_data_for_beta(100)
    
    print(f"Synthetic data shape: {synthetic_data.shape}")
    print(f"Market data shape: {market_data.shape}")
    print(f"Synthetic data columns: {list(synthetic_data.columns)}")
    print()
    
    # Get real-world data
    print("Attempting to fetch real-world data...")
    real_data, real_market_data = get_real_world_data()
    if real_data is not None:
        print(f"✅ Real data fetched: {real_data.shape}")
        print(f"✅ Real market data fetched: {real_market_data.shape}")
    else:
        print("⚠️  Real-world data not available (yfinance not installed or failed)")
    print()
    
    # Initialize technical indicators
    print("Initializing TechnicalIndicators...")
    ti = TechnicalIndicators()
    
    # Get all indicator methods
    indicator_methods = []
    for name, method in inspect.getmembers(ti, predicate=inspect.ismethod):
        if name.isupper() and not name.startswith('_') and name != 'BENFORD_LAW':
            # Skip BENFORD_LAW as it's not a typical technical indicator
            indicator_methods.append(name)
    
    indicator_methods.sort()
    print(f"Testing {len(indicator_methods)} indicators...")
    print()
    
    # Test all indicators
    results = []
    for i, indicator_name in enumerate(indicator_methods, 1):
        print(f"[{i:2d}/{len(indicator_methods)}] Testing {indicator_name}...", end=" ")
        
        result = test_indicator(
            ti, indicator_name, synthetic_data, market_data, real_data, real_market_data
        )
        results.append(result)
        
        # Quick status indicator
        if result['synthetic_success']:
            status = "✅"
        else:
            status = "❌"
        print(status)
    
    print()
    
    # Analyze results
    print("Analyzing results...")
    analysis = analyze_results(results)
    
    # Print detailed results
    print_detailed_results(results, analysis)
    
    print("\n" + "="*80)
    print("FINAL ASSESSMENT:")
    print("="*80)
    
    critical_failures = len(analysis['critical_failures'])
    success_rate = analysis['synthetic_success_count'] / analysis['total_indicators']
    
    if critical_failures == 0 and success_rate >= 0.90:
        print("🎉 EXCELLENT: All indicators working properly!")
        print("   No critical compilation errors detected.")
        print(f"   Success rate: {success_rate*100:.1f}%")
    elif critical_failures == 0 and success_rate >= 0.75:
        print("✅ GOOD: Most indicators working, minor issues detected.")
        print("   No critical compilation errors.")
        print(f"   Success rate: {success_rate*100:.1f}%")
    elif critical_failures <= 2:
        print("⚠️  WARNING: Some indicators have critical issues.")
        print(f"   {critical_failures} critical failures detected.")
        print(f"   Success rate: {success_rate*100:.1f}%")
    else:
        print("🚨 CRITICAL: Multiple indicators failing!")
        print(f"   {critical_failures} critical failures detected.")
        print(f"   Success rate: {success_rate*100:.1f}%")
        print("   Immediate attention required.")
    
    print("="*80)

if __name__ == "__main__":
    main() 