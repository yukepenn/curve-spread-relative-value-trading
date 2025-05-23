"""
Module for fetching data from Yahoo Finance.
Retrieves historical market data for given tickers.
"""

import pandas as pd
import yfinance as yf
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

class YahooDataFetcher:
    """Class for fetching data from Yahoo Finance."""
    
    def __init__(self):
        """Initialize the data fetcher."""
        self.config = load_config()
        self.data_dir = get_data_dir()
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs directory if needed
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
    
    @retry_on_error(max_retries=3, delay=2)
    def fetch_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data for a single symbol.
        
        Args:
            symbol: Yahoo Finance ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1wk, 1mo)
            
        Returns:
            DataFrame with OHLC data
        """
        try:
            logger.info(f"Fetching data for {symbol}")
            
            # Use config dates if not provided
            if start_date is None:
                start_date = self.config['data']['start_date']
            if end_date is None:
                end_date = self.config['data']['end_date']
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval
            )
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Add symbol column
            df['Symbol'] = symbol
            
            # Save to CSV
            file_path = self.raw_dir / f"{symbol.replace('=', '_')}.csv"
            df.to_csv(file_path)
            logger.info(f"Saved data to {file_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    @retry_on_error(max_retries=3, delay=2)
    def fetch_batch(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols efficiently.
        
        Args:
            symbols: List of Yahoo Finance ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1wk, 1mo)
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        try:
            logger.info(f"Fetching batch data for {len(symbols)} symbols")
            
            # Use config dates if not provided
            if start_date is None:
                start_date = self.config['data']['start_date']
            if end_date is None:
                end_date = self.config['data']['end_date']
            
            # Fetch data for all symbols
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by='ticker'
            )
            
            # Process each symbol's data
            result = {}
            for symbol in symbols:
                if symbol in data.columns.levels[0]:
                    # Extract data for this symbol
                    symbol_data = data[symbol].copy()
                    
                    # Add symbol column
                    symbol_data['Symbol'] = symbol
                    
                    # Save to CSV
                    file_path = self.raw_dir / f"{symbol.replace('=', '_')}.csv"
                    symbol_data.to_csv(file_path)
                    logger.info(f"Saved data to {file_path}")
                    
                    result[symbol] = symbol_data
                else:
                    logger.warning(f"No data found for {symbol}")
                    result[symbol] = pd.DataFrame()
            
            # Save combined data
            if result:
                combined_path = self.raw_dir / "yahoo_combined.csv"
                pd.concat(result.values()).to_csv(combined_path)
                logger.info(f"Saved combined data to {combined_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in batch fetch: {str(e)}")
            return {symbol: pd.DataFrame() for symbol in symbols}
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: Yahoo Finance ticker symbol
            
        Returns:
            Latest closing price or None if not available
        """
        try:
            ticker = yf.Ticker(symbol)
            latest = ticker.history(period='1d')
            if not latest.empty:
                return latest['Close'].iloc[-1]
            return None
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {str(e)}")
            return None

def main():
    """Main function to fetch data for configured symbols."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fetch data from Yahoo Finance')
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
    
    fetcher = YahooDataFetcher()
    
    # Get symbols from config
    symbols = []
    
    # Add symbols from spreads
    for spread_name, spread_config in config['spreads'].items():
        logger.info(f"Processing {spread_name}")
        symbols.extend([
            spread_config['front_leg']['symbol'],
            spread_config['back_leg']['symbol']
        ])
    
    # Add any additional symbols from macro config
    if 'macro_symbols' in config:
        symbols.extend(config['macro_symbols'])
    
    # Remove duplicates
    symbols = list(set(symbols))
    
    # Fetch data
    fetcher.fetch_batch(symbols)

if __name__ == "__main__":
    main() 