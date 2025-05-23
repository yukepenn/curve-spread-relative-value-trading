"""
Feature preparation pipeline module.

This module orchestrates the end-to-end feature preparation process for modeling,
including data fetching, cleaning, feature engineering, and dataset preparation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import logging
import yaml

from src.data.fetch_fred import FredDataFetcher
from src.data.fetch_yahoo import fetch_yahoo_data
from src.data.data_cleaner import merge_data, clean_data
from src.features.yield_curve import (
    compute_spread,
    generate_yield_curve_features,
    get_target,
    save_yield_curve_data
)
from src.features.macro_features import (
    generate_macro_features,
    save_macro_features
)
from src.utils.config_loader import load_config, load_yaml

logger = logging.getLogger(__name__)

class FeaturePipeline:
    """
    Feature preparation pipeline for yield curve relative value trading.
    """
    
    def __init__(self, config_path: str, curves_path: str):
        """
        Initialize the feature pipeline.
        
        Args:
            config_path: Path to the main configuration file
            curves_path: Path to the curves configuration file
        """
        self.config = load_config()
        self.curves = load_yaml(Path(curves_path))
        self.data_dir = Path(self.config['data']['data_dir'])
        
        # Initialize data fetchers
        self.fred_fetcher = FredDataFetcher()
        
    def prepare_features(
        self,
        spread_name: str,
        save: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """
        Prepare features and target for a given spread.
        
        Args:
            spread_name: Name of the spread (e.g., '2s10s')
            save: Whether to save the prepared features
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series]: Features, targets, and spread series
        """
        try:
            logger.info(f"Preparing features for {spread_name}")
            
            # 1. Get spread configuration
            spread_config = self.curves['spreads'][spread_name]
            front_leg = spread_config['front_leg']['fred_id']
            back_leg = spread_config['back_leg']['fred_id']
            
            # 2. Fetch data
            logger.info("Fetching FRED data")
            fred_data = self.fred_fetcher.fetch_multiple(
                series_ids=[front_leg, back_leg],
                start_date=self.config['data']['start_date'],
                end_date=self.config['data']['end_date']
            )
            
            logger.info("Fetching Yahoo Finance data")
            yahoo_data = fetch_yahoo_data(
                symbols=[spread_config['front_leg']['ticker'], spread_config['back_leg']['ticker']],
                config=self.config
            )
            
            # 3. Merge and clean data
            logger.info("Merging and cleaning data")
            merged_data = merge_data(fred_data, yahoo_data)
            cleaned_data = clean_data(merged_data)
            
            # 4. Compute spread
            logger.info("Computing spread")
            spread = compute_spread(
                cleaned_data[front_leg],
                cleaned_data[back_leg]
            )
            
            # 5. Generate yield curve features
            logger.info("Generating yield curve features")
            yield_features = generate_yield_curve_features(
                spread,
                window_sizes=self.config.get('feature_engineering', {}).get('window_sizes', {
                    'short': 20,
                    'medium': 60,
                    'long': 120
                })
            )
            
            # 6. Generate macro features
            logger.info("Generating macro features")
            macro_features = generate_macro_features(
                cleaned_data,
                window_sizes=self.config.get('feature_engineering', {}).get('window_sizes', {
                    'short': 20,
                    'medium': 60,
                    'long': 120
                }),
                config=self.config
            )
            
            # 7. Generate targets
            logger.info("Generating targets")
            targets = get_target(
                spread,
                horizons=self.config.get('feature_engineering', {}).get('horizons', {
                    '1d': 1,
                    '5d': 5,
                    '20d': 20
                })
            )
            
            # 8. Merge features
            logger.info("Merging features")
            features = pd.concat([yield_features, macro_features], axis=1)
            
            # 9. Align features and targets
            max_horizon = max(self.config.get('feature_engineering', {}).get('horizons', {'1d': 1}).values())
            features = features.iloc[:-max_horizon]
            targets = targets.iloc[:-max_horizon]
            spread = spread.iloc[:-max_horizon]
            
            # 10. Save if requested
            if save:
                self._save_features(features, targets, spread, spread_name)
            
            logger.info(f"Successfully prepared features for {spread_name}")
            return features, targets, spread
            
        except Exception as e:
            logger.error(f"Error preparing features for {spread_name}: {str(e)}")
            raise
    
    def prepare_all_features(self, save: bool = True) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]:
        """
        Prepare features for all spreads defined in curves.yaml.
        
        Args:
            save: Whether to save the prepared features
            
        Returns:
            Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]]: Dictionary of features, targets, and spreads
        """
        results = {}
        
        try:
            for spread_name in self.curves['spreads'].keys():
                features, targets, spread = self.prepare_features(spread_name, save)
                results[spread_name] = (features, targets, spread)
            
            logger.info(f"Successfully prepared features for {len(results)} spreads")
            return results
            
        except Exception as e:
            logger.error(f"Error preparing all features: {str(e)}")
            raise
    
    def _save_features(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        spread: pd.Series,
        spread_name: str
    ) -> None:
        """
        Save prepared features, targets, and spread to files.
        
        Args:
            features: Features DataFrame
            targets: Targets DataFrame
            spread: Spread Series
            spread_name: Name of the spread
        """
        try:
            # Create output directory
            output_dir = self.data_dir / 'processed' / 'features'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save features
            features_path = output_dir / f"{spread_name}_features.csv"
            features.to_csv(features_path)
            logger.info(f"Saved features to {features_path}")
            
            # Save targets
            targets_path = output_dir / f"{spread_name}_targets.csv"
            targets.to_csv(targets_path)
            logger.info(f"Saved targets to {targets_path}")
            
            # Save spread
            spread_path = output_dir / f"{spread_name}_spread.csv"
            spread.to_csv(spread_path)
            logger.info(f"Saved spread to {spread_path}")
            
        except Exception as e:
            logger.error(f"Error saving features for {spread_name}: {str(e)}")
            raise

def main():
    """
    Main function to run the feature pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the feature preparation pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--curves', type=str, required=True, help='Path to curves.yaml')
    parser.add_argument('--spread', type=str, help='Specific spread to process (optional)')
    args = parser.parse_args()
    
    try:
        pipeline = FeaturePipeline(args.config, args.curves)
        
        if args.spread:
            features, targets, spread = pipeline.prepare_features(args.spread)
            logger.info(f"Processed {args.spread} spread")
        else:
            results = pipeline.prepare_all_features()
            logger.info(f"Processed {len(results)} spreads")
            
    except Exception as e:
        logger.error(f"Error running feature pipeline: {str(e)}")
        raise

if __name__ == '__main__':
    main()
