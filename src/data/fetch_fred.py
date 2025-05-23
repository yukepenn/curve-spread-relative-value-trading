"""
Module for fetching data from FRED (Federal Reserve Economic Data).
Retrieves historical yield and macroeconomic data.
"""

import pandas as pd
from fredapi import Fred
from typing import List, Dict, Optional
from datetime import datetime
import logging
from pathlib import Path
import argparse
import os
import sys
import time
from functools import wraps

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config_loader import load_config, get_data_dir

logger = logging.getLogger(__name__)

def retry_on_error(max_retries: int = 3, delay: int = 1):
    """
    Decorator to retry functions on error.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class FredDataFetcher:
    """Class for fetching data from FRED."""
    
    def __init__(self):
        """Initialize the data fetcher."""
        self.config = load_config()
        self.data_dir = get_data_dir()
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory if needed
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Initialize FRED API client
        self.fred = Fred(api_key=self.config['fred']['api_key'])
    
    @retry_on_error(max_retries=3, delay=2)
    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "d"
    ) -> pd.Series:
        """
        Fetch a single FRED series.
        
        Args:
            series_id: FRED series identifier (e.g., "DGS10" for 10-year Treasury)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency (d=daily, w=weekly, m=monthly)
            
        Returns:
            Series with date index
        """
        try:
            logger.info(f"Fetching FRED series {series_id}")
            
            # Use config dates if not provided
            if start_date is None:
                start_date = self.config['data']['start_date']
            if end_date is None:
                end_date = self.config['data']['end_date']
            
            # Fetch data
            series = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
                frequency=frequency
            )
            
            if series.empty:
                logger.warning(f"No data found for {series_id}")
                return pd.Series()
            
            # Save to CSV
            file_path = self.raw_dir / f"{series_id}.csv"
            series.to_csv(file_path)
            logger.info(f"Saved data to {file_path}")
            
            return series
            
        except Exception as e:
            logger.error(f"Error fetching series {series_id}: {str(e)}")
            return pd.Series()
    
    @retry_on_error(max_retries=3, delay=2)
    def fetch_multiple(
        self,
        series_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: str = "d"
    ) -> pd.DataFrame:
        """
        Fetch multiple FRED series efficiently.
        
        Args:
            series_ids: List of FRED series identifiers
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency (d=daily, w=weekly, m=monthly)
            
        Returns:
            DataFrame with series as columns
        """
        try:
            logger.info(f"Fetching {len(series_ids)} FRED series")
            
            # Use config dates if not provided
            if start_date is None:
                start_date = self.config['data']['start_date']
            if end_date is None:
                end_date = self.config['data']['end_date']
            
            # Fetch all series
            data = {}
            for series_id in series_ids:
                series = self.fetch_series(series_id, start_date, end_date, frequency)
                if not series.empty:
                    data[series_id] = series
            
            # Combine into DataFrame
            df = pd.DataFrame(data)
            
            if not df.empty:
                # Save combined data
                file_path = self.raw_dir / "fred_combined.csv"
                df.to_csv(file_path)
                logger.info(f"Saved combined data to {file_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in multiple series fetch: {str(e)}")
            return pd.DataFrame()
    
    def get_series_info(self, series_id: str) -> Dict:
        """
        Get metadata for a FRED series.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            Dictionary with series metadata
        """
        try:
            return self.fred.get_series_info(series_id)
        except Exception as e:
            logger.error(f"Error getting info for {series_id}: {str(e)}")
            return {}
    
    def get_series_tags(self, series_id: str) -> List[Dict]:
        """
        Get tags associated with a FRED series.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            List of tag dictionaries
        """
        try:
            return self.fred.get_series_tags(series_id)
        except Exception as e:
            logger.error(f"Error getting tags for {series_id}: {str(e)}")
            return []
    
    def get_related_series(self, series_id: str) -> List[str]:
        """
        Get related series IDs for a given series.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            List of related series IDs
        """
        try:
            return self.fred.get_related_series(series_id)
        except Exception as e:
            logger.error(f"Error getting related series for {series_id}: {str(e)}")
            return []

def main():
    """Main function to fetch FRED data for configured series."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fetch data from FRED')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()
    
    # Set config path
    os.environ['CONFIG_PATH'] = args.config
    
    # Load config and setup logging
    config = load_config()
    from src.utils.logging import setup_logging
    setup_logging(
        level=config.get('logging', {}).get('level', 'INFO'),
        log_file=config.get('logging', {}).get('file'),
        fmt=config.get('logging', {}).get('format')
    )
    
    fetcher = FredDataFetcher()
    
    # Get all series IDs from config
    series_ids = []
    
    # Add yield series from spreads
    for spread_name, spread_config in config['spreads'].items():
        logger.info(f"Processing {spread_name}")
        series_ids.extend([
            spread_config['front_leg']['fred_series'],
            spread_config['back_leg']['fred_series']
        ])
    
    # Add macro series
    if 'fred_series' in config:
        series_ids.extend(config['fred_series'])
    
    # Remove duplicates
    series_ids = list(set(series_ids))
    
    # Fetch all series
    fetcher.fetch_multiple(series_ids)

if __name__ == "__main__":
    main() 