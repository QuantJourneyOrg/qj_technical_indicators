"""
QuantJourney Technical-Indicators - Risk Metrics
===============================================
Hedge fund specific risk and performance metrics with Numba optimization.

Author: Jakub Polec <jakub@quantjourney.pro>
License: MIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit
from typing import Union, Tuple, Optional
import warnings

@njit(parallel=False, fastmath=True)
def _calculate_returns_numba(prices: np.ndarray) -> np.ndarray:
    """Calculate returns from price series."""
    returns = np.full(len(prices), np.nan, dtype=np.float64)
    for i in range(1, len(prices)):
        if not np.isnan(prices[i]) and not np.isnan(prices[i-1]) and prices[i-1] != 0:
            returns[i] = (prices[i] - prices[i-1]) / prices[i-1]
    return returns

@njit(parallel=False, fastmath=True)
def _calculate_sharpe_ratio_numba(returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Calculate annualized Sharpe ratio."""
    valid_returns = returns[~np.isnan(returns)]
    if len(valid_returns) < 2:
        return np.nan
    
    mean_return = np.mean(valid_returns)
    std_return = np.std(valid_returns)
    
    if std_return == 0:
        return np.nan
    
    excess_return = mean_return - (risk_free_rate / periods_per_year)
    sharpe = (excess_return * np.sqrt(periods_per_year)) / std_return
    return sharpe

@njit(parallel=False, fastmath=True)
def _calculate_sortino_ratio_numba(returns: np.ndarray, target_return: float = 0.0, periods_per_year: int = 252) -> float:
    """Calculate annualized Sortino ratio."""
    valid_returns = returns[~np.isnan(returns)]
    if len(valid_returns) < 2:
        return np.nan
    
    mean_return = np.mean(valid_returns)
    downside_returns = valid_returns[valid_returns < target_return]
    
    if len(downside_returns) == 0:
        return np.inf if mean_return > target_return else np.nan
    
    downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))
    
    if downside_deviation == 0:
        return np.nan
    
    excess_return = mean_return - (target_return / periods_per_year)
    sortino = (excess_return * np.sqrt(periods_per_year)) / downside_deviation
    return sortino

@njit(parallel=False, fastmath=True)
def _calculate_max_drawdown_numba(prices: np.ndarray) -> Tuple[float, int, int]:
    """Calculate maximum drawdown and its start/end indices."""
    if len(prices) == 0:
        return np.nan, -1, -1
    
    max_dd = 0.0
    peak = prices[0]
    peak_idx = 0
    start_idx = 0
    end_idx = 0
    temp_start = 0
    
    for i in range(1, len(prices)):
        if np.isnan(prices[i]):
            continue
            
        if prices[i] > peak:
            peak = prices[i]
            peak_idx = i
            temp_start = i
        else:
            drawdown = (peak - prices[i]) / peak
            if drawdown > max_dd:
                max_dd = drawdown
                start_idx = temp_start
                end_idx = i
    
    return max_dd, start_idx, end_idx

@njit(parallel=False, fastmath=True)
def _calculate_var_numba(returns: np.ndarray, confidence: float = 0.05) -> float:
    """Calculate Value at Risk using historical simulation."""
    valid_returns = returns[~np.isnan(returns)]
    if len(valid_returns) == 0:
        return np.nan
    
    # Sort returns in ascending order
    sorted_returns = np.sort(valid_returns)
    index = int(confidence * len(sorted_returns))
    
    if index >= len(sorted_returns):
        return sorted_returns[-1]
    
    return sorted_returns[index]

@njit(parallel=False, fastmath=True)
def _calculate_cvar_numba(returns: np.ndarray, confidence: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)."""
    valid_returns = returns[~np.isnan(returns)]
    if len(valid_returns) == 0:
        return np.nan
    
    sorted_returns = np.sort(valid_returns)
    index = int(confidence * len(sorted_returns))
    
    if index == 0:
        return sorted_returns[0]
    
    return np.mean(sorted_returns[:index])

@njit(parallel=False, fastmath=True)
def _calculate_calmar_ratio_numba(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calculate Calmar ratio (annualized return / max drawdown)."""
    valid_returns = returns[~np.isnan(returns)]
    if len(valid_returns) < 2:
        return np.nan
    
    # Calculate annualized return
    mean_return = np.mean(valid_returns) * periods_per_year
    
    # Calculate cumulative returns to get prices
    cum_returns = np.ones(len(valid_returns) + 1)
    for i in range(len(valid_returns)):
        cum_returns[i + 1] = cum_returns[i] * (1 + valid_returns[i])
    
    # Calculate max drawdown
    max_dd = 0.0
    peak = cum_returns[0]
    
    for i in range(1, len(cum_returns)):
        if cum_returns[i] > peak:
            peak = cum_returns[i]
        else:
            drawdown = (peak - cum_returns[i]) / peak
            if drawdown > max_dd:
                max_dd = drawdown
    
    if max_dd == 0:
        return np.inf if mean_return > 0 else np.nan
    
    return mean_return / max_dd

@njit(parallel=False, fastmath=True)
def _calculate_omega_ratio_numba(returns: np.ndarray, threshold: float = 0.0) -> float:
    """Calculate Omega ratio."""
    valid_returns = returns[~np.isnan(returns)]
    if len(valid_returns) == 0:
        return np.nan
    
    gains = 0.0
    losses = 0.0
    
    for ret in valid_returns:
        if ret > threshold:
            gains += (ret - threshold)
        else:
            losses += (threshold - ret)
    
    if losses == 0:
        return np.inf if gains > 0 else 1.0
    
    return gains / losses

@njit(parallel=False, fastmath=True)
def _calculate_information_ratio_numba(portfolio_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """Calculate Information Ratio."""
    if len(portfolio_returns) != len(benchmark_returns):
        return np.nan
    
    active_returns = portfolio_returns - benchmark_returns
    valid_active = active_returns[~np.isnan(active_returns)]
    
    if len(valid_active) < 2:
        return np.nan
    
    mean_active = np.mean(valid_active)
    std_active = np.std(valid_active)
    
    if std_active == 0:
        return np.nan
    
    return mean_active / std_active

@njit(parallel=False, fastmath=True)
def _calculate_treynor_ratio_numba(returns: np.ndarray, market_returns: np.ndarray, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Calculate Treynor ratio."""
    if len(returns) != len(market_returns):
        return np.nan
    
    # Calculate beta
    valid_mask = ~(np.isnan(returns) | np.isnan(market_returns))
    valid_returns = returns[valid_mask]
    valid_market = market_returns[valid_mask]
    
    if len(valid_returns) < 2:
        return np.nan
    
    # Calculate covariance and variance
    mean_ret = np.mean(valid_returns)
    mean_mkt = np.mean(valid_market)
    
    covariance = np.mean((valid_returns - mean_ret) * (valid_market - mean_mkt))
    market_variance = np.mean((valid_market - mean_mkt) ** 2)
    
    if market_variance == 0:
        return np.nan
    
    beta = covariance / market_variance
    
    if beta == 0:
        return np.nan
    
    # Calculate annualized excess return
    excess_return = (mean_ret - risk_free_rate / periods_per_year) * periods_per_year
    
    return excess_return / beta

def calculate_risk_metrics(
    data: Union[pd.Series, pd.DataFrame],
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    confidence_level: float = 0.05
) -> pd.Series:
    """
    Calculate comprehensive risk metrics for a return series or price series.
    
    Args:
        data: Price or return series
        benchmark: Benchmark series for relative metrics
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Trading periods per year (default: 252)
        confidence_level: Confidence level for VaR/CVaR (default: 5%)
    
    Returns:
        Series with calculated risk metrics
    """
    if isinstance(data, pd.DataFrame):
        if 'close' in data.columns:
            prices = data['close']
        elif 'adj_close' in data.columns:
            prices = data['adj_close']
        else:
            raise ValueError("DataFrame must contain 'close' or 'adj_close' column")
    else:
        prices = data
    
    # Determine if data is prices or returns
    if prices.min() > 0 and prices.max() / prices.min() > 2:
        # Likely prices, calculate returns
        returns = prices.pct_change().dropna()
        is_prices = True
    else:
        # Likely returns
        returns = prices.dropna()
        is_prices = False
        # Reconstruct prices for drawdown calculation
        prices = (1 + returns).cumprod()
    
    returns_np = returns.values.astype(np.float64)
    prices_np = prices.values.astype(np.float64)
    
    metrics = {}
    
    # Basic statistics
    metrics['Total Return'] = (prices.iloc[-1] / prices.iloc[0] - 1) if is_prices else returns.sum()
    metrics['Annualized Return'] = (1 + metrics['Total Return']) ** (periods_per_year / len(returns)) - 1
    metrics['Volatility'] = returns.std() * np.sqrt(periods_per_year)
    metrics['Skewness'] = returns.skew()
    metrics['Kurtosis'] = returns.kurtosis()
    
    # Risk-adjusted returns
    metrics['Sharpe Ratio'] = _calculate_sharpe_ratio_numba(returns_np, risk_free_rate, periods_per_year)
    metrics['Sortino Ratio'] = _calculate_sortino_ratio_numba(returns_np, 0.0, periods_per_year)
    metrics['Calmar Ratio'] = _calculate_calmar_ratio_numba(returns_np, periods_per_year)
    
    # Drawdown metrics
    max_dd, start_idx, end_idx = _calculate_max_drawdown_numba(prices_np)
    metrics['Max Drawdown'] = max_dd
    metrics['Max Drawdown Start'] = prices.index[start_idx] if start_idx >= 0 else None
    metrics['Max Drawdown End'] = prices.index[end_idx] if end_idx >= 0 else None
    
    # Risk metrics
    metrics['VaR (5%)'] = _calculate_var_numba(returns_np, confidence_level)
    metrics['CVaR (5%)'] = _calculate_cvar_numba(returns_np, confidence_level)
    metrics['Omega Ratio'] = _calculate_omega_ratio_numba(returns_np, 0.0)
    
    # Benchmark relative metrics
    if benchmark is not None:
        if isinstance(benchmark, pd.Series):
            bench_returns = benchmark.pct_change().dropna() if benchmark.min() > 0 else benchmark.dropna()
            # Align indices
            aligned_returns, aligned_bench = returns.align(bench_returns, join='inner')
            
            if len(aligned_returns) > 1:
                metrics['Information Ratio'] = _calculate_information_ratio_numba(
                    aligned_returns.values, aligned_bench.values
                )
                metrics['Treynor Ratio'] = _calculate_treynor_ratio_numba(
                    aligned_returns.values, aligned_bench.values, risk_free_rate, periods_per_year
                )
                
                # Beta calculation
                covariance = np.cov(aligned_returns.values, aligned_bench.values)[0, 1]
                benchmark_variance = np.var(aligned_bench.values)
                metrics['Beta'] = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
    
    return pd.Series(metrics)

__all__ = [
    'calculate_risk_metrics',
    '_calculate_sharpe_ratio_numba',
    '_calculate_sortino_ratio_numba',
    '_calculate_max_drawdown_numba',
    '_calculate_var_numba',
    '_calculate_cvar_numba',
    '_calculate_calmar_ratio_numba',
    '_calculate_omega_ratio_numba',
    '_calculate_information_ratio_numba',
    '_calculate_treynor_ratio_numba'
]