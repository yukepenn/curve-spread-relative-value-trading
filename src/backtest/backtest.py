"""
Backtester for the yield curve relative value trading system.
Evaluates trading strategies using historical data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import joblib

from ..utils.config_loader import load_config, get_data_dir
from ..utils.logging import get_logger
from ..utils.math_utils import (
    calculate_sharpe,
    max_drawdown,
    annualize_return,
    annualize_vol,
    calculate_carry,
    calculate_transaction_cost
)

logger = get_logger(__name__)

class Backtester:
    """Class for backtesting trading strategies."""
    
    def __init__(self):
        """Initialize backtester."""
        self.config = load_config()
        self.data_dir = get_data_dir()
        self.model_dir = self.data_dir / 'models'
        self.results_dir = self.data_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
    
    def load_data(self, spread_name: str) -> pd.DataFrame:
        """
        Load processed data for a spread.
        
        Args:
            spread_name: Name of the spread
            
        Returns:
            DataFrame with processed data
        """
        filepath = self.data_dir / f"{spread_name}_processed.csv"
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(data)} records from {filepath}")
        return data
    
    def load_model(self, spread_name: str) -> Tuple[object, object]:
        """
        Load trained model and scaler.
        
        Args:
            spread_name: Name of the spread
            
        Returns:
            Tuple of (model, scaler)
        """
        model_path = self.model_dir / f"{spread_name}_model.joblib"
        scaler_path = self.model_dir / f"{spread_name}_scaler.joblib"
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    
    def generate_signals(
        self,
        data: pd.DataFrame,
        model: object,
        scaler: object,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate trading signals using model predictions.
        
        Args:
            data: DataFrame with processed data
            model: Trained model
            scaler: Fitted scaler
            threshold: Signal threshold
            
        Returns:
            DataFrame with trading signals
        """
        # Prepare features
        feature_cols = [col for col in data.columns if col != 'spread']
        X = data[feature_cols].values
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[predictions > threshold] = 1  # Long signal
        signals[predictions < -threshold] = -1  # Short signal
        
        return signals
    
    def calculate_returns(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        spread_config: Dict
    ) -> pd.DataFrame:
        """
        Calculate strategy returns.
        
        Args:
            data: DataFrame with processed data
            signals: Series with trading signals
            spread_config: Spread configuration
            
        Returns:
            DataFrame with returns
        """
        # Calculate spread returns
        spread_returns = data['spread'].pct_change()
        
        # Calculate carry
        carry = calculate_carry(
            data['front_yield'],
            data['back_yield'],
            spread_config['front_leg']['dv01'],
            spread_config['back_leg']['dv01']
        )
        
        # Calculate transaction costs
        position_changes = signals.diff().abs()
        costs = calculate_transaction_cost(
            position_changes,
            spread_config['front_leg']['dv01'],
            spread_config['back_leg']['dv01']
        )
        
        # Calculate strategy returns
        strategy_returns = signals.shift(1) * spread_returns + carry - costs
        
        # Create returns DataFrame
        returns = pd.DataFrame({
            'spread_returns': spread_returns,
            'carry': carry,
            'costs': costs,
            'strategy_returns': strategy_returns
        })
        
        return returns
    
    def calculate_metrics(
        self,
        returns: pd.DataFrame
    ) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            returns: DataFrame with returns
            
        Returns:
            Dictionary with performance metrics
        """
        strategy_returns = returns['strategy_returns']
        
        metrics = {
            'total_return': (1 + strategy_returns).prod() - 1,
            'annualized_return': annualize_return(strategy_returns),
            'annualized_vol': annualize_vol(strategy_returns),
            'sharpe_ratio': calculate_sharpe(strategy_returns),
            'max_drawdown': max_drawdown(strategy_returns)[0],
            'win_rate': (strategy_returns > 0).mean(),
            'avg_win': strategy_returns[strategy_returns > 0].mean(),
            'avg_loss': strategy_returns[strategy_returns < 0].mean(),
            'profit_factor': abs(
                strategy_returns[strategy_returns > 0].sum() /
                strategy_returns[strategy_returns < 0].sum()
            )
        }
        
        return metrics
    
    def run_backtest(
        self,
        spread_name: str,
        spread_config: Dict,
        threshold: float = 0.5
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run backtest for a single spread.
        
        Args:
            spread_name: Name of the spread
            spread_config: Spread configuration
            threshold: Signal threshold
            
        Returns:
            Tuple of (returns DataFrame, metrics dictionary)
        """
        # Load data and model
        data = self.load_data(spread_name)
        model, scaler = self.load_model(spread_name)
        
        # Generate signals
        signals = self.generate_signals(data, model, scaler, threshold)
        
        # Calculate returns
        returns = self.calculate_returns(data, signals, spread_config)
        
        # Calculate metrics
        metrics = self.calculate_metrics(returns)
        
        return returns, metrics
    
    def save_results(
        self,
        returns: pd.DataFrame,
        metrics: Dict,
        spread_name: str
    ) -> None:
        """
        Save backtest results.
        
        Args:
            returns: DataFrame with returns
            metrics: Dictionary with metrics
            spread_name: Name of the spread
        """
        # Save returns
        returns_path = self.results_dir / f"{spread_name}_returns.csv"
        returns.to_csv(returns_path)
        
        # Save metrics
        metrics_path = self.results_dir / f"{spread_name}_backtest_metrics.csv"
        pd.Series(metrics).to_csv(metrics_path)
        
        logger.info(f"Saved backtest results for {spread_name}")

def main():
    """Main function to run backtests for all spreads."""
    # Load configuration
    config = load_config()
    curves = load_curves()
    
    # Initialize backtester
    backtester = Backtester()
    
    # Run backtest for each spread
    for spread_name, spread_config in curves['spreads'].items():
        logger.info(f"Running backtest for {spread_name}")
        
        # Run backtest
        returns, metrics = backtester.run_backtest(
            spread_name,
            spread_config,
            threshold=config['backtest']['signal_threshold']
        )
        
        # Save results
        backtester.save_results(returns, metrics, spread_name)

if __name__ == "__main__":
    main() 