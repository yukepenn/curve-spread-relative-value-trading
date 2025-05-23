"""
Module for cleaning and preprocessing data.
Handles data alignment, missing values, and transformations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import yaml
from datetime import datetime
import sys
import argparse
import os

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config, get_data_dir, load_curves
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

class DataCleaner:
    """Class for cleaning and preprocessing data."""
    
    def __init__(self, curves_config: Optional[Dict] = None):
        """
        Initialize the data cleaner.
        
        Args:
            curves_config: Optional curves configuration dictionary
        """
        self.config = load_config()
        self.curves_config = curves_config or load_curves()
        self.data_dir = get_data_dir()
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        
        # Create necessary directories
        for directory in [self.raw_dir, self.interim_dir, self.processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw data from FRED and Yahoo Finance.
        
        Returns:
            Tuple of (fred_data, yahoo_data) DataFrames
        """
        try:
            # Load FRED data
            fred_path = self.raw_dir / "fred_combined.csv"
            if not fred_path.exists():
                raise FileNotFoundError(f"FRED data not found at {fred_path}")
            fred_data = pd.read_csv(fred_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded FRED data with {len(fred_data)} rows")
            
            # Load Yahoo data
            yahoo_path = self.raw_dir / "yahoo_combined.csv"
            if not yahoo_path.exists():
                raise FileNotFoundError(f"Yahoo data not found at {yahoo_path}")
            yahoo_data = pd.read_csv(yahoo_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded Yahoo data with {len(yahoo_data)} rows")
            
            return fred_data, yahoo_data
            
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def merge_data(
        self,
        fred_data: pd.DataFrame,
        yahoo_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge data from multiple sources into a single DataFrame.
        
        Args:
            fred_data: DataFrame with FRED series
            yahoo_data: DataFrame with Yahoo Finance data
            
        Returns:
            Merged DataFrame with aligned dates
        """
        try:
            logger.info("Merging data from multiple sources")
            
            # Ensure both DataFrames have datetime index
            fred_data.index = pd.to_datetime(fred_data.index)
            yahoo_data.index = pd.to_datetime(yahoo_data.index)
            
            # Merge on date index
            merged_data = pd.merge(
                fred_data,
                yahoo_data,
                left_index=True,
                right_index=True,
                how='outer'
            )
            
            # Sort by date
            merged_data.sort_index(inplace=True)
            
            # Save intermediate result
            interim_path = self.interim_dir / "merged_data.csv"
            merged_data.to_csv(interim_path)
            logger.info(f"Saved merged data to {interim_path}")
            
            return merged_data
            
        except Exception as e:
            logger.error(f"Error merging data: {str(e)}")
            return pd.DataFrame()
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df: Raw merged DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info("Cleaning and preprocessing data")
            
            # Create a copy to avoid modifying the original
            cleaned = df.copy()
            
            # Handle missing values
            if self.config.get('data_cleaning', {}).get('fill_method') == 'ffill':
                # Forward fill missing values
                cleaned.fillna(method='ffill', inplace=True)
            else:
                # Interpolate missing values
                cleaned.interpolate(method='time', inplace=True)
            
            # Remove outliers using z-score
            z_score_threshold = self.config.get('data_cleaning', {}).get('z_score_threshold', 3)
            for column in cleaned.columns:
                if column != 'Symbol':  # Skip non-numeric columns
                    z_scores = np.abs((cleaned[column] - cleaned[column].mean()) / cleaned[column].std())
                    cleaned.loc[z_scores > z_score_threshold, column] = np.nan
            
            # Fill remaining missing values after outlier removal
            cleaned.interpolate(method='time', inplace=True)
            
            # Add derived features
            self._add_derived_features(cleaned)
            
            # Save processed data
            processed_path = self.processed_dir / "cleaned_data.csv"
            cleaned.to_csv(processed_path)
            logger.info(f"Saved cleaned data to {processed_path}")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> None:
        """
        Add derived features to the DataFrame.
        
        Args:
            df: DataFrame to add features to
        """
        try:
            # Calculate spreads for each configured spread
            for spread_name, spread_config in self.curves_config['spreads'].items():
                front_leg_id = spread_config['front_leg']['fred_id']
                back_leg_id = spread_config['back_leg']['fred_id']
                if front_leg_id in df.columns and back_leg_id in df.columns:
                    spread_col = f"{spread_name}_spread"
                    df[spread_col] = df[back_leg_id] - df[front_leg_id]
                    logger.info(f"Added {spread_col}")
            # Add other derived features as needed
        except Exception as e:
            logger.error(f"Error adding derived features: {str(e)}")
    
    def process_spreads(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process spread data for each configured spread.
        
        Args:
            df: DataFrame with cleaned data
            
        Returns:
            DataFrame with processed spreads
        """
        try:
            logger.info("Processing spreads")
            spreads = {}
            # Process each spread
            for spread_name, spread_config in self.curves_config['spreads'].items():
                front_leg_id = spread_config['front_leg']['fred_id']
                back_leg_id = spread_config['back_leg']['fred_id']
                if front_leg_id in df.columns and back_leg_id in df.columns:
                    # Calculate spread
                    spread = df[back_leg_id] - df[front_leg_id]
                    spreads[f"{spread_name}_spread"] = spread
                    # Save individual spread
                    spread_path = self.processed_dir / f"{spread_name}_spread.csv"
                    spread.to_frame().to_csv(spread_path)
                    logger.info(f"Saved {spread_name} spread to {spread_path}")
            # Combine all spreads
            spreads_df = pd.DataFrame(spreads)
            # Save combined spreads
            spreads_path = self.processed_dir / "all_spreads.csv"
            spreads_df.to_csv(spreads_path)
            logger.info(f"Saved all spreads to {spreads_path}")
            return spreads_df
        except Exception as e:
            logger.error(f"Error processing spreads: {str(e)}")
            return pd.DataFrame()

def main():
    """Main function to clean and process data."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Clean and process data')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--curves', type=str, default='config/curves.yaml',
                      help='Path to curves configuration file')
    args = parser.parse_args()
    
    # Set config path
    os.environ['CONFIG_PATH'] = args.config
    
    # Load config and setup logging
    config = load_config()
    setup_logging(
        level=config.get('logging', {}).get('level', 'INFO'),
        log_file=config.get('logging', {}).get('file'),
        fmt=config.get('logging', {}).get('format')
    )
    
    # Load curves config
    curves_config = None
    if args.curves:
        try:
            with open(args.curves, 'r') as f:
                curves_config = yaml.safe_load(f)
            logger.info(f"Loaded curves configuration from {args.curves}")
        except Exception as e:
            logger.error(f"Error loading curves configuration: {str(e)}")
            return
    
    cleaner = DataCleaner(curves_config=curves_config)
    
    # Load raw data
    fred_data, yahoo_data = cleaner.load_raw_data()
    
    if fred_data.empty or yahoo_data.empty:
        logger.error("Failed to load raw data")
        return
    
    # Merge data
    merged_data = cleaner.merge_data(fred_data, yahoo_data)
    
    if merged_data.empty:
        logger.error("Failed to merge data")
        return
    
    # Clean data
    cleaned_data = cleaner.clean_data(merged_data)
    
    if cleaned_data.empty:
        logger.error("Failed to clean data")
        return
    
    # Process spreads
    spreads_data = cleaner.process_spreads(cleaned_data)
    
    if spreads_data.empty:
        logger.error("Failed to process spreads")
        return
    
    logger.info("Data cleaning and processing completed successfully")

if __name__ == "__main__":
    main() 