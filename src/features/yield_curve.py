"""
Yield curve feature engineering module.

This module handles the computation of yield spreads and generation of yield curve
related features for predictive modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def compute_spread(
    front_yield_series: pd.Series,
    back_yield_series: pd.Series
) -> pd.Series:
    """
    Calculate the yield spread time series (back leg minus front leg).
    
    Args:
        front_yield_series: Front leg yield series (e.g., 2-year)
        back_yield_series: Back leg yield series (e.g., 10-year)
        
    Returns:
        Series containing the yield spread
    """
    try:
        # Ensure both series have the same index
        common_index = front_yield_series.index.intersection(back_yield_series.index)
        front_yield = front_yield_series.loc[common_index]
        back_yield = back_yield_series.loc[common_index]
        
        # Compute spread (back - front)
        spread = back_yield - front_yield
        
        logger.info(f"Computed spread with {len(spread)} data points")
        return spread
        
    except Exception as e:
        logger.error(f"Error computing spread: {str(e)}")
        return pd.Series()

def generate_yield_curve_features(
    spread_series: pd.Series,
    window_sizes: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Generate features from the yield spread series.
    
    Args:
        spread_series: The yield spread time series
        window_sizes: Dictionary of window sizes for different features
            Default: {'short': 20, 'medium': 60, 'long': 120}
            
    Returns:
        DataFrame containing the generated features
    """
    try:
        if window_sizes is None:
            window_sizes = {
                'short': 20,
                'medium': 60,
                'long': 120
            }
        
        features = pd.DataFrame(index=spread_series.index)
        
        # Current spread value
        features['spread'] = spread_series
        
        # Moving averages
        for window_name, window_size in window_sizes.items():
            # Simple moving average
            features[f'ma_{window_name}'] = spread_series.rolling(window=window_size).mean()
            
            # Exponential moving average
            features[f'ema_{window_name}'] = spread_series.ewm(span=window_size).mean()
            
            # Spread momentum (change over window)
            features[f'momentum_{window_name}'] = spread_series - spread_series.shift(window_size)
            
            # Volatility (rolling standard deviation)
            features[f'volatility_{window_name}'] = spread_series.rolling(window=window_size).std()
            
            # Z-score relative to window
            features[f'zscore_{window_name}'] = (
                (spread_series - features[f'ma_{window_name}']) / 
                features[f'volatility_{window_name}']
            )
        
        # Rate of change
        features['roc_1d'] = spread_series.pct_change()
        features['roc_5d'] = spread_series.pct_change(periods=5)
        
        # Additional technical indicators
        # RSI (Relative Strength Index)
        delta = spread_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for window_name, window_size in window_sizes.items():
            ma = features[f'ma_{window_name}']
            std = features[f'volatility_{window_name}']
            features[f'bb_upper_{window_name}'] = ma + (std * 2)
            features[f'bb_lower_{window_name}'] = ma - (std * 2)
            features[f'bb_width_{window_name}'] = (
                (features[f'bb_upper_{window_name}'] - features[f'bb_lower_{window_name}']) / 
                ma
            )
        
        # Drop any NaN values
        features = features.dropna()
        
        logger.info(f"Generated {len(features.columns)} yield curve features")
        return features
        
    except Exception as e:
        logger.error(f"Error generating yield curve features: {str(e)}")
        return pd.DataFrame()

def get_target(
    spread_series: pd.Series,
    horizons: Optional[Dict[str, int]] = None
) -> pd.DataFrame:
    """
    Compute prediction targets (future spread changes).
    
    Args:
        spread_series: The yield spread time series
        horizons: Dictionary of prediction horizons in days
            Default: {'1d': 1, '5d': 5, '20d': 20}
            
    Returns:
        DataFrame containing the target variables
    """
    try:
        if horizons is None:
            horizons = {
                '1d': 1,
                '5d': 5,
                '20d': 20
            }
        
        targets = pd.DataFrame(index=spread_series.index)
        
        # Compute future spread changes for each horizon
        for horizon_name, horizon in horizons.items():
            # Future spread change
            targets[f'spread_change_{horizon_name}'] = (
                spread_series.shift(-horizon) - spread_series
            )
            
            # Future spread change percentage
            targets[f'spread_change_pct_{horizon_name}'] = (
                targets[f'spread_change_{horizon_name}'] / spread_series
            )
        
        # Drop any NaN values
        targets = targets.dropna()
        
        logger.info(f"Generated {len(targets.columns)} target variables")
        return targets
        
    except Exception as e:
        logger.error(f"Error computing targets: {str(e)}")
        return pd.DataFrame()

def save_yield_curve_data(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    spread: pd.Series,
    output_dir: Path,
    spread_name: str
) -> None:
    """
    Save yield curve features, targets, and spread to files.
    
    Args:
        features: DataFrame of yield curve features
        targets: DataFrame of target variables
        spread: Series of the yield spread
        output_dir: Directory to save the files
        spread_name: Name of the spread (e.g., '2s10s')
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features_path = output_dir / f"{spread_name}_yield_features.csv"
        features.to_csv(features_path)
        logger.info(f"Saved yield curve features to {features_path}")
        
        # Save targets
        targets_path = output_dir / f"{spread_name}_yield_targets.csv"
        targets.to_csv(targets_path)
        logger.info(f"Saved yield curve targets to {targets_path}")
        
        # Save spread
        spread_path = output_dir / f"{spread_name}_spread.csv"
        spread.to_csv(spread_path)
        logger.info(f"Saved yield spread to {spread_path}")
        
    except Exception as e:
        logger.error(f"Error saving yield curve data: {str(e)}")
        raise
