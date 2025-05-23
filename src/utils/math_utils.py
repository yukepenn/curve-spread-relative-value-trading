"""
Module for mathematical utility functions.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple

def calculate_sharpe(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        periods_per_year: Number of periods in a year (default: 252 for daily)
        
    Returns:
        Sharpe ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    excess_returns = returns - risk_free_rate / periods_per_year
    if len(excess_returns) < 2:
        return 0.0
    
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)

def max_drawdown(
    returns: Union[pd.Series, np.ndarray]
) -> Tuple[float, int, int]:
    """
    Calculate the maximum drawdown and its start/end indices.
    
    Args:
        returns: Series of returns
        
    Returns:
        Tuple of (max_drawdown, start_idx, end_idx)
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)
    
    # Calculate drawdowns
    drawdowns = (cum_returns - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = np.min(drawdowns)
    end_idx = np.argmin(drawdowns)
    start_idx = np.argmax(cum_returns[:end_idx])
    
    return max_dd, start_idx, end_idx

def annualize_return(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Annualize a series of returns.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year (default: 252 for daily)
        
    Returns:
        Annualized return
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    return (1 + np.mean(returns)) ** periods_per_year - 1

def annualize_vol(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Annualize volatility of returns.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year (default: 252 for daily)
        
    Returns:
        Annualized volatility
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    return np.std(returns) * np.sqrt(periods_per_year)

def compute_dv01(
    price: float,
    yield_value: float,
    maturity: float
) -> float:
    """
    Compute DV01 (dollar value of 1 basis point) for a bond.
    
    Args:
        price: Bond price
        yield_value: Yield to maturity (in decimal)
        maturity: Time to maturity in years
        
    Returns:
        DV01 in dollars
    """
    # Simple approximation using modified duration
    duration = maturity / (1 + yield_value)
    return -duration * price / 100

def bp_to_decimal(bp: float) -> float:
    """
    Convert basis points to decimal.
    
    Args:
        bp: Value in basis points
        
    Returns:
        Value in decimal
    """
    return bp / 10000

def decimal_to_bp(decimal: float) -> float:
    """
    Convert decimal to basis points.
    
    Args:
        decimal: Value in decimal
        
    Returns:
        Value in basis points
    """
    return decimal * 10000

def zscore(
    series: Union[pd.Series, np.ndarray],
    window: int = 20
) -> Union[pd.Series, np.ndarray]:
    """
    Calculate z-score of a series using a rolling window.
    
    Args:
        series: Input series
        window: Rolling window size
        
    Returns:
        Z-scored series
    """
    if isinstance(series, pd.Series):
        return (series - series.rolling(window=window).mean()) / series.rolling(window=window).std()
    else:
        mean = pd.Series(series).rolling(window=window).mean().values
        std = pd.Series(series).rolling(window=window).std().values
        return (series - mean) / std

def calculate_carry(
    front_yield: float,
    back_yield: float,
    position: dict,
    dv01_front: float,
    dv01_back: float
) -> float:
    """
    Calculate daily carry for a spread position.
    
    Args:
        front_yield: Front leg yield (in decimal)
        back_yield: Back leg yield (in decimal)
        position: Dictionary with 'front_leg' and 'back_leg' positions
        dv01_front: DV01 of front leg
        dv01_back: DV01 of back leg
        
    Returns:
        Daily carry in dollars
    """
    # Calculate yield difference
    yield_diff = front_yield - back_yield
    
    # Calculate notional for each leg
    notional_front = abs(position['front_leg']) * dv01_front * 10000
    notional_back = abs(position['back_leg']) * dv01_back * 10000
    
    # Calculate carry
    carry = (
        (position['front_leg'] * front_yield * notional_front) +
        (position['back_leg'] * back_yield * notional_back)
    ) / 252  # Daily carry
    
    return carry

def calculate_transaction_cost(
    position_change: dict,
    cost_per_contract: float
) -> float:
    """
    Calculate transaction cost for a position change.
    
    Args:
        position_change: Dictionary with changes in 'front_leg' and 'back_leg' positions
        cost_per_contract: Cost per contract traded
        
    Returns:
        Total transaction cost in dollars
    """
    return (
        abs(position_change['front_leg']) +
        abs(position_change['back_leg'])
    ) * cost_per_contract 