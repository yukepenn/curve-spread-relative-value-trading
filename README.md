# Yield Curve Relative Value Trading System

A quantitative trading system for yield curve relative value strategies using bond futures. The system forecasts changes in yield spreads between different maturities and generates trading signals based on model predictions.

## Overview

This system implements a modular framework for:
- Data collection and processing of yield curve data
- Feature engineering for yield spreads and macro indicators
- Multiple predictive models (XGBoost, ARIMA, LSTM)
- Signal generation and position sizing
- Backtesting with transaction costs and carry
- Performance analysis and risk metrics
- Model interpretability via SHAP analysis

## Project Structure

```
├── README.md
├── config/
│ ├── config.yaml
│ └── curves.yaml
├── data/
│ ├── raw/
│ ├── interim/
│ └── processed/
├── src/
│ ├── data/
│ │ ├── fetch_yahoo.py
│ │ ├── fetch_fred.py
│ │ └── data_cleaner.py
│ ├── features/
│ │ ├── yield_curve.py
│ │ ├── macro_features.py
│ │ └── feature_pipeline.py
│ ├── models/
│ │ ├── xgboost_model.py
│ │ ├── arima_model.py
│ │ ├── lstm_model.py
│ │ └── training_pipeline.py
│ ├── signals/
│ │ ├── signal_generation.py
│ │ └── position_sizing.py
│ ├── backtest/
│ │ ├── backtester.py
│ │ ├── transaction_costs.py
│ │ ├── carry_calculator.py
│ │ └── performance.py
│ ├── analysis/
│ │ ├── shap_analysis.py
│ │ ├── risk_analysis.py
│ │ └── visualization.py
│ └── utils/
│ ├── config_loader.py
│ ├── math_utils.py
│ └── logging_utils.py
├── reports/
│ └── figures/
├── models/
│ └── 2s10s_xgboost.pkl
├── requirements.txt
└── .cursor/
└── project-rules.mdc
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Set up your API keys in `config/config.yaml`:
   - FRED API key for yield data
   - Yahoo Finance API settings (if needed)

2. Configure spreads in `config/curves.yaml`:
   - Define yield curve spreads to trade
   - Set DV01 values and hedge ratios
   - Specify model types for each spread

## Usage

1. Data Collection:
   ```bash
   python src/data/fetch_fred.py
   python src/data/fetch_yahoo.py
   ```

2. Feature Generation:
   ```bash
   python src/features/feature_pipeline.py
   ```

3. Model Training:
   ```bash
   python src/models/training_pipeline.py
   ```

4. Backtesting:
   ```bash
   python src/backtest/backtester.py
   ```

5. Analysis:
   ```bash
   python src/analysis/shap_analysis.py
   python src/analysis/risk_analysis.py
   ```

## Key Features

- **DV01-Neutral Trading**: All trades are balanced for interest rate risk
- **Multiple Models**: XGBoost, ARIMA, and LSTM for spread prediction
- **Comprehensive Backtesting**: Includes transaction costs and carry
- **Risk Analysis**: VaR, CVaR, and scenario testing
- **Model Interpretability**: SHAP analysis for feature importance
- **Extensible Design**: Easy to add new spreads or models

## Outputs

- Trained models saved in `models/`
- Performance metrics and visualizations in `reports/figures/`
- Backtest results including equity curves and risk metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 