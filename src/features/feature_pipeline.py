"""
Module for feature engineering pipeline.
Orchestrates the computation of features from yield curve and macro data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from datetime import datetime

from utils.config_loader import load_config
from utils.logging import get_logger
from features.yield_curve import YieldCurveFeatures
from features.macro_features import MacroFeatures

logger = get_logger(__name__)

class FeaturePipeline:
    """Class for orchestrating feature engineering."""
    
    def __init__(self, yield_data: pd.DataFrame, macro_data: pd.DataFrame):
        """Initialize with yield and macro data."""
        self.yield_data = yield_data
        self.macro_data = macro_data
        self.config = load_config()
        
        # Initialize feature generators
        self.yield_features = YieldCurveFeatures(yield_data)
        self.macro_features = MacroFeatures(macro_data)
    
    def compute_features(self) -> pd.DataFrame:
        """Compute all features."""
        try:
            logger.info("Starting feature computation")
            
            # Compute yield curve features
            yield_features = self.yield_features.compute_all_features()
            logger.info(f"Computed {len(yield_features.columns)} yield curve features")
            
            # Compute macro features
            macro_features = self.macro_features.compute_all_features()
            logger.info(f"Computed {len(macro_features.columns)} macro features")
            
            # Combine features
            features = pd.concat([yield_features, macro_features], axis=1)
            
            # Handle missing values
            features = self._handle_missing_values(features)
            
            # Add feature metadata
            features = self._add_feature_metadata(features)
            
            logger.info(f"Completed feature computation with {len(features.columns)} total features")
            return features
            
        except Exception as e:
            logger.error(f"Error computing features: {str(e)}")
            raise
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features."""
        # Forward fill missing values up to 5 periods
        features = features.ffill(limit=5)
        
        # Backward fill any remaining missing values at the start
        features = features.bfill(limit=5)
        
        # Drop any remaining rows with missing values
        features = features.dropna()
        
        return features
    
    def _add_feature_metadata(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add metadata to features."""
        # Add computation timestamp
        features.attrs['computed_at'] = datetime.now().isoformat()
        
        # Add feature categories
        feature_categories = {}
        for col in features.columns:
            if any(prefix in col for prefix in ['yield_', 'pc']):
                feature_categories[col] = 'yield_curve'
            else:
                feature_categories[col] = 'macro'
        
        features.attrs['feature_categories'] = feature_categories
        
        return features 