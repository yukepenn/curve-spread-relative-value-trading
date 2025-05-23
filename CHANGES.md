# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2024-03-19

### Added
- Configuration loading with YAML support
- Centralized logging system
- Financial math utilities for calculations
- FRED API integration for yield data
- Data processing pipeline with feature generation
- Random Forest model training with cross-validation
- Backtesting framework with performance metrics
- Macro market data fetching (equity, VIX, FX)
- Integration of macro features into spread analysis
- Command-line interface for pipeline execution
- Comprehensive FRED macroeconomic indicators:
  - Interest Rates and Monetary Policy (Fed Funds, Target Rates, etc.)
  - Inflation Metrics (CPI, PCE, Breakeven Rates)
  - Credit Spreads (Corporate, High Yield, CMBS)
  - Economic Indicators (GDP, Unemployment, Production)
  - Money Supply and Bank Credit
  - Market Indicators (Equities, FX, Commodities)
  - Volatility and Risk Measures
  - Bond Market Indicators
  - Market Liquidity Measures
  - Business Cycle Indicators
- Comprehensive feature engineering module:
  - Calendar features (day of week, holidays, end-of-period indicators)
  - Trend features (moving averages, momentum, RSI, volatility)
  - Yield curve features (PCA decomposition for level, slope, curvature)
  - Carry features (spread level, carry returns, historical performance)
  - Macro features (levels, changes, z-scores, distance from mean)
- Feature analysis module:
  - Correlation analysis with visualization
  - Feature importance using Random Forest
  - Feature selection using mutual information and F-regression
  - Correlation heatmaps for selected features
  - Automated generation of analysis reports and plots

### Changed
- Enhanced data processing to incorporate both FRED and yfinance macro data
- Improved feature generation with comprehensive macro indicators
- Updated main pipeline to fetch and process all macro data sources
- Enhanced feature engineering with more sophisticated calculations
- Improved feature analysis with automated visualization

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None 