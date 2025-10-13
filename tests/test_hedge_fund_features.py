"""
Tests for hedge fund specific features.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

import quantjourney_ti as qti
from quantjourney_ti import (
    TechnicalIndicators,
    StreamingIndicators, 
    calculate_risk_metrics,
    MemoryManager,
    BatchProcessor,
    get_cache_stats,
    clear_indicator_cache
)

@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
    prices = 100 * (1 + returns).cumprod()
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 252)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 252))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 252))),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, 252)
    }, index=dates)
    
    return df

@pytest.fixture
def benchmark_data():
    """Create benchmark data for testing."""
    np.random.seed(123)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate benchmark returns (slightly lower volatility)
    returns = np.random.normal(0.0008, 0.015, 252)
    prices = 100 * (1 + returns).cumprod()
    
    return pd.Series(prices, index=dates)

class TestRiskMetrics:
    """Test risk metrics calculations."""
    
    def test_risk_metrics_basic(self, sample_price_data):
        """Test basic risk metrics calculation."""
        ti = TechnicalIndicators()
        
        risk_metrics = ti.RISK_METRICS(sample_price_data['close'])
        
        # Check that all expected metrics are present
        expected_metrics = [
            'Total Return', 'Annualized Return', 'Volatility',
            'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown',
            'VaR (5%)', 'CVaR (5%)', 'Calmar Ratio'
        ]
        
        for metric in expected_metrics:
            assert metric in risk_metrics.index
            assert not pd.isna(risk_metrics[metric])
    
    def test_risk_metrics_with_benchmark(self, sample_price_data, benchmark_data):
        """Test risk metrics with benchmark."""
        ti = TechnicalIndicators()
        
        risk_metrics = ti.RISK_METRICS(
            sample_price_data['close'],
            benchmark=benchmark_data
        )
        
        # Check benchmark-specific metrics
        benchmark_metrics = ['Information Ratio', 'Treynor Ratio', 'Beta']
        
        for metric in benchmark_metrics:
            assert metric in risk_metrics.index
            assert not pd.isna(risk_metrics[metric])
    
    def test_risk_metrics_edge_cases(self):
        """Test risk metrics with edge cases."""
        ti = TechnicalIndicators()
        
        # Test with constant prices (zero volatility)
        constant_prices = pd.Series([100] * 100, 
                                  index=pd.date_range('2023-01-01', periods=100))
        
        risk_metrics = ti.RISK_METRICS(constant_prices)
        
        # Sharpe ratio should be NaN or inf for zero volatility
        assert pd.isna(risk_metrics['Sharpe Ratio']) or np.isinf(risk_metrics['Sharpe Ratio'])
        assert risk_metrics['Volatility'] == 0
        assert risk_metrics['Max Drawdown'] == 0

class TestStreamingIndicators:
    """Test streaming indicators functionality."""
    
    def test_streaming_basic(self, sample_price_data):
        """Test basic streaming functionality."""
        streaming = StreamingIndicators()
        
        # Process first few rows
        results = []
        for i, (timestamp, row) in enumerate(sample_price_data.head(50).iterrows()):
            indicators = streaming.update_tick(
                symbol='TEST',
                timestamp=timestamp,
                close=row['close'],
                high=row['high'],
                low=row['low'],
                volume=row['volume']
            )
            results.append(indicators)
            
            if i >= 20:  # After enough data points
                assert 'sma_20' in indicators
                assert not np.isnan(indicators['sma_20'])
        
        # Check final values
        final_values = streaming.get_current_values('TEST')
        assert len(final_values) > 0
    
    def test_streaming_callbacks(self, sample_price_data):
        """Test streaming callbacks."""
        streaming = StreamingIndicators()
        callback_results = []
        
        def test_callback(symbol, value, timestamp):
            callback_results.append((symbol, value, timestamp))
        
        streaming.register_callback('TEST_sma_20', test_callback)
        
        # Process data
        for timestamp, row in sample_price_data.head(30).iterrows():
            streaming.update_tick(
                symbol='TEST',
                timestamp=timestamp,
                close=row['close'],
                high=row['high'],
                low=row['low']
            )
        
        # Should have received callbacks after SMA period
        assert len(callback_results) > 0
    
    def test_streaming_buffer(self, sample_price_data):
        """Test streaming buffer functionality."""
        streaming = StreamingIndicators(max_buffer_size=100)
        
        # Process more data than buffer size
        for timestamp, row in sample_price_data.iterrows():
            streaming.update_tick(
                symbol='TEST',
                timestamp=timestamp,
                close=row['close']
            )
        
        # Get buffer data
        buffer_df = streaming.get_buffer_data('TEST')
        
        # Should not exceed max buffer size
        assert len(buffer_df) <= 100
        assert not buffer_df.empty

class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    def test_caching(self, sample_price_data):
        """Test indicator caching."""
        clear_indicator_cache()  # Start fresh
        
        ti = TechnicalIndicators()
        
        # First calculation
        start_time = time.time()
        sma1 = ti.SMA(sample_price_data['close'], period=20)
        first_time = time.time() - start_time
        
        # Second calculation (should be faster due to caching)
        start_time = time.time()
        sma2 = ti.SMA(sample_price_data['close'], period=20)
        second_time = time.time() - start_time
        
        # Results should be identical
        pd.testing.assert_series_equal(sma1, sma2)
        
        # Second should be faster (though this might be flaky in fast systems)
        # Just check that caching is working
        cache_stats = get_cache_stats()
        assert cache_stats['hit_count'] > 0
    
    def test_batch_processing(self, sample_price_data):
        """Test batch processing functionality."""
        ti = TechnicalIndicators()
        
        # Create multiple symbol datasets
        data_dict = {
            'SYMBOL1': sample_price_data,
            'SYMBOL2': sample_price_data * 1.1,  # Slightly different data
            'SYMBOL3': sample_price_data * 0.9
        }
        
        # Batch calculate RSI
        results = ti.batch_calculate(
            data_dict=data_dict,
            indicator_name='RSI',
            period=14
        )
        
        assert len(results) == 3
        for symbol, rsi_series in results.items():
            assert isinstance(rsi_series, pd.Series)
            assert len(rsi_series) == len(sample_price_data)
            assert rsi_series.name == 'RSI_14'

class TestMemoryManagement:
    """Test memory management features."""
    
    def test_dataframe_optimization(self, sample_price_data):
        """Test DataFrame memory optimization."""
        # Create DataFrame with inefficient dtypes
        inefficient_df = sample_price_data.copy()
        inefficient_df = inefficient_df.astype(np.float64)  # Force float64
        
        original_memory = MemoryManager.get_memory_usage(inefficient_df)
        
        # Optimize
        optimized_df = MemoryManager.optimize_dataframe(inefficient_df)
        optimized_memory = MemoryManager.get_memory_usage(optimized_df)
        
        # Should use less memory (or at least not more)
        assert optimized_memory['total_mb'] <= original_memory['total_mb']
        
        # Data should be preserved
        pd.testing.assert_frame_equal(
            inefficient_df.astype(optimized_df.dtypes), 
            optimized_df,
            check_dtype=False
        )
    
    def test_memory_usage_calculation(self, sample_price_data):
        """Test memory usage calculation."""
        memory_stats = MemoryManager.get_memory_usage(sample_price_data)
        
        assert 'total_mb' in memory_stats
        assert 'per_column_mb' in memory_stats
        assert memory_stats['total_mb'] > 0
        assert len(memory_stats['per_column_mb']) == len(sample_price_data.columns)

class TestDataValidation:
    """Test enhanced data validation."""
    
    def test_market_data_validation(self):
        """Test market data validation with gaps."""
        # Create data with gaps (missing weekends)
        business_days = pd.bdate_range('2023-01-01', '2023-01-31')
        df = pd.DataFrame({
            'close': np.random.randn(len(business_days)).cumsum() + 100
        }, index=business_days)
        
        ti = TechnicalIndicators()
        
        # Should not raise error with allow_gaps=True
        validated_df = ti.validate_market_data(df, allow_gaps=True)
        assert len(validated_df) == len(df)
    
    def test_data_fixing(self):
        """Test automatic data issue fixing."""
        # Create problematic data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        problematic_df = pd.DataFrame({
            'open': [100, 101, np.nan, 103, 104, 105, np.nan, 107, 108, 109],
            'close': [101, 102, np.nan, 104, 105, 106, np.nan, 108, 109, 110],
            'volume': [1000, 0, 1200, 0, 1400, 1500, 1600, 0, 1800, 1900]
        }, index=dates)
        
        ti = TechnicalIndicators()
        
        # Fix issues
        fixed_df = ti.validate_market_data(
            problematic_df,
            allow_gaps=True,
            fix_common_issues=True
        )
        
        # Should have fewer NaNs and no zero volumes
        assert fixed_df.isnull().sum().sum() <= problematic_df.isnull().sum().sum()
        assert (fixed_df['volume'] == 0).sum() == 0

class TestIntegration:
    """Integration tests for hedge fund features."""
    
    def test_complete_workflow(self, sample_price_data, benchmark_data):
        """Test complete hedge fund workflow."""
        ti = TechnicalIndicators()
        
        # 1. Validate and fix data
        clean_data = ti.validate_market_data(
            sample_price_data, 
            allow_gaps=True, 
            fix_common_issues=True
        )
        
        # 2. Calculate technical indicators
        sma = ti.SMA(clean_data['close'], period=20)
        rsi = ti.RSI(clean_data['close'], period=14)
        
        # 3. Calculate risk metrics
        risk_metrics = ti.RISK_METRICS(
            clean_data['close'],
            benchmark=benchmark_data
        )
        
        # 4. Test streaming
        streaming = ti.create_streaming_indicators()
        
        for timestamp, row in clean_data.head(50).iterrows():
            streaming.update_tick(
                symbol='TEST',
                timestamp=timestamp,
                close=row['close'],
                high=row['high'],
                low=row['low']
            )
        
        # All should complete without errors
        assert len(sma.dropna()) > 0
        assert len(rsi.dropna()) > 0
        assert len(risk_metrics) > 0
        assert len(streaming.get_current_values('TEST')) > 0
    
    def test_error_handling(self):
        """Test error handling in edge cases."""
        ti = TechnicalIndicators()
        
        # Empty data
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            ti.SMA(empty_df, period=20)
        
        # Invalid data types
        invalid_data = pd.Series(['a', 'b', 'c'])
        with pytest.raises(Exception):
            ti.RSI(invalid_data, period=14)
        
        # Insufficient data
        short_data = pd.Series([1, 2], index=pd.date_range('2023-01-01', periods=2))
        result = ti.SMA(short_data, period=20)
        assert result.isna().all()  # Should return all NaN

if __name__ == "__main__":
    pytest.main([__file__, "-v"])