"""
Centralized logging setup for the project.
Provides consistent logging configuration across all modules.
"""

import logging
from pathlib import Path
from typing import Optional

def setup_logging(level: str = "INFO", log_file: str = None, fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
    """
    Configure logging for the project.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        fmt: Log message format string
    """
    # Create log directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers
    )
    
    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level={level}")
    if log_file:
        logger.info(f"Log file: {log_file}")

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name of the logger (defaults to module name)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name or __name__)

# Create a default logger for the utils module
logger = get_logger(__name__)

def log_config_loaded() -> None:
    """Log that configuration has been loaded successfully."""
    logger.info("Configuration loaded successfully")

def log_data_fetched(source: str, count: int) -> None:
    """
    Log that data has been fetched from a source.
    
    Args:
        source: Data source name
        count: Number of records fetched
    """
    logger.info(f"Fetched {count} records from {source}")

def log_model_trained(model_type: str, spread: str) -> None:
    """
    Log that a model has been trained.
    
    Args:
        model_type: Type of model trained
        spread: Spread the model was trained on
    """
    logger.info(f"Trained {model_type} model for {spread} spread")

def log_backtest_completed(spread: str, performance: dict) -> None:
    """
    Log that a backtest has been completed.
    
    Args:
        spread: Spread that was backtested
        performance: Dictionary of performance metrics
    """
    logger.info(f"Completed backtest for {spread} spread")
    logger.info(f"Performance metrics: {performance}")

def log_error(message: str, exc: Optional[Exception] = None) -> None:
    """
    Log an error message.
    
    Args:
        message: Error message
        exc: Optional exception that caused the error
    """
    if exc:
        logger.error(f"{message}: {str(exc)}", exc_info=True)
    else:
        logger.error(message) 