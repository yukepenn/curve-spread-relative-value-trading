"""
Macroeconomic feature engineering module.

This module generates features from macroeconomic and market data to complement
yield curve features for predictive modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_macro_features(
    cleaned_data: pd.DataFrame,
    window_sizes: Optional[Dict[str, int]] = None,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate macroeconomic features from cleaned data.
    
    Args:
        cleaned_data: DataFrame containing cleaned market and economic data
        window_sizes: Dictionary of window sizes for different features
            Default: {'short': 20, 'medium': 60, 'long': 120}
        config: Configuration dictionary with feature settings
            
    Returns:
        DataFrame containing the generated macro features
    """
    try:
        if window_sizes is None:
            window_sizes = {
                'short': 20,
                'medium': 60,
                'long': 120
            }
        
        features = pd.DataFrame(index=cleaned_data.index)
        
        # 1. Interest Rate Features
        if 'DGS3MO' in cleaned_data.columns:  # 3-month T-bill rate
            features['short_rate'] = cleaned_data['DGS3MO']
            features['short_rate_change'] = cleaned_data['DGS3MO'].pct_change()
            
            # Short rate momentum
            for window_name, window_size in window_sizes.items():
                features[f'short_rate_momentum_{window_name}'] = (
                    cleaned_data['DGS3MO'] - cleaned_data['DGS3MO'].shift(window_size)
                )
        
        if 'DFF' in cleaned_data.columns:  # Fed Funds Rate
            features['fed_funds'] = cleaned_data['DFF']
            features['fed_funds_change'] = cleaned_data['DFF'].pct_change()
            
            # Fed Funds momentum
            for window_name, window_size in window_sizes.items():
                features[f'fed_funds_momentum_{window_name}'] = (
                    cleaned_data['DFF'] - cleaned_data['DFF'].shift(window_size)
                )
        
        # 2. Economic Indicators
        # Inflation
        if 'CPIAUCSL' in cleaned_data.columns:  # CPI
            features['cpi_yoy'] = cleaned_data['CPIAUCSL'].pct_change(periods=12)
            features['cpi_mom'] = cleaned_data['CPIAUCSL'].pct_change()
        
        if 'PCEPI' in cleaned_data.columns:  # PCE Price Index
            features['pce_yoy'] = cleaned_data['PCEPI'].pct_change(periods=12)
            features['pce_mom'] = cleaned_data['PCEPI'].pct_change()
        
        # Unemployment
        if 'UNRATE' in cleaned_data.columns:  # Unemployment Rate
            features['unemployment'] = cleaned_data['UNRATE']
            features['unemployment_change'] = cleaned_data['UNRATE'].diff()
        
        # PMI
        if 'PMI' in cleaned_data.columns:  # Manufacturing PMI
            features['pmi'] = cleaned_data['PMI']
            features['pmi_change'] = cleaned_data['PMI'].diff()
        
        # 3. Risk Indicators
        # VIX
        if '^VIX' in cleaned_data.columns:
            features['vix'] = cleaned_data['^VIX']
            features['vix_change'] = cleaned_data['^VIX'].pct_change()
            
            # VIX momentum and volatility
            for window_name, window_size in window_sizes.items():
                features[f'vix_momentum_{window_name}'] = (
                    cleaned_data['^VIX'] - cleaned_data['^VIX'].shift(window_size)
                )
                features[f'vix_volatility_{window_name}'] = (
                    cleaned_data['^VIX'].rolling(window=window_size).std()
                )
        
        # Equity Market
        if '^GSPC' in cleaned_data.columns:  # S&P 500
            # Returns
            features['sp500_return'] = cleaned_data['^GSPC'].pct_change()
            features['sp500_return_5d'] = cleaned_data['^GSPC'].pct_change(periods=5)
            
            # Volatility
            for window_name, window_size in window_sizes.items():
                features[f'sp500_volatility_{window_name}'] = (
                    cleaned_data['^GSPC'].pct_change().rolling(window=window_size).std()
                )
        
        # 4. Global Factors
        # FX Rates
        if 'DEXUSEU' in cleaned_data.columns:  # EUR/USD
            features['eur_usd'] = cleaned_data['DEXUSEU']
            features['eur_usd_change'] = cleaned_data['DEXUSEU'].pct_change()
        
        if 'DEXJPUS' in cleaned_data.columns:  # JPY/USD
            features['jpy_usd'] = cleaned_data['DEXJPUS']
            features['jpy_usd_change'] = cleaned_data['DEXJPUS'].pct_change()
        
        # Foreign Yields
        if 'IRLTLT02JPM156N' in cleaned_data.columns:  # Japan 10Y
            features['japan_10y'] = cleaned_data['IRLTLT02JPM156N']
            features['japan_10y_change'] = cleaned_data['IRLTLT02JPM156N'].diff()
        
        if 'IRLTLT10DEM156N' in cleaned_data.columns:  # Germany 10Y
            features['germany_10y'] = cleaned_data['IRLTLT10DEM156N']
            features['germany_10y_change'] = cleaned_data['IRLTLT10DEM156N'].diff()
        
        # 5. Technical Indicators
        # For each numeric column, add technical indicators
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Moving averages
            for window_name, window_size in window_sizes.items():
                features[f'{col}_ma_{window_name}'] = (
                    cleaned_data[col].rolling(window=window_size).mean()
                )
                features[f'{col}_ema_{window_name}'] = (
                    cleaned_data[col].ewm(span=window_size).mean()
                )
            
            # Rate of change
            features[f'{col}_roc'] = cleaned_data[col].pct_change()
            features[f'{col}_roc_5d'] = cleaned_data[col].pct_change(periods=5)
        
        # 6. Cross-Asset Correlations
        if '^GSPC' in cleaned_data.columns and '^VIX' in cleaned_data.columns:
            for window_name, window_size in window_sizes.items():
                features[f'sp500_vix_corr_{window_name}'] = (
                    cleaned_data['^GSPC'].pct_change()
                    .rolling(window=window_size)
                    .corr(cleaned_data['^VIX'].pct_change())
                )
        
        # Drop any NaN values
        features = features.dropna()
        
        logger.info(f"Generated {len(features.columns)} macro features")
        return features
        
    except Exception as e:
        logger.error(f"Error generating macro features: {str(e)}")
        return pd.DataFrame()

def save_macro_features(
    features: pd.DataFrame,
    output_dir: Path,
    spread_name: str
) -> None:
    """
    Save macro features to file.
    
    Args:
        features: DataFrame of macro features
        output_dir: Directory to save the file
        spread_name: Name of the spread (e.g., '2s10s')
    """
    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features_path = output_dir / f"{spread_name}_macro_features.csv"
        features.to_csv(features_path)
        logger.info(f"Saved macro features to {features_path}")
        
    except Exception as e:
        logger.error(f"Error saving macro features: {str(e)}")
        raise
