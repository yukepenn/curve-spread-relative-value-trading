"""
Module for loading and managing configuration.
Handles loading of YAML configuration files and environment variables.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

def get_config_dir() -> Path:
    """Get the configuration directory."""
    return get_project_root() / "config"

def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"

def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Returns:
        Dictionary containing configuration
    """
    try:
        # Get config path from environment or use default
        config_path = os.environ.get('CONFIG_PATH')
        if config_path:
            config_path = Path(config_path)
        else:
            config_path = get_config_dir() / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        config = load_yaml(config_path)
        
        # Override with environment variables if they exist
        if 'FRED_API_KEY' in os.environ:
            config['fred']['api_key'] = os.environ['FRED_API_KEY']
        
        return config
        
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {str(e)}")

def get_fred_api_key() -> str:
    """
    Get FRED API key from configuration.
    
    Returns:
        FRED API key string
    """
    config = load_config()
    return config['fred']['api_key']

def get_spread_config(spread_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific spread.
    
    Args:
        spread_name: Name of the spread (e.g., "2s10s")
        
    Returns:
        Dictionary containing spread configuration
    """
    config = load_config()
    return config['spreads'][spread_name]

def get_feature_config() -> Dict[str, Any]:
    """
    Get feature engineering configuration.
    
    Returns:
        Dictionary containing feature configuration
    """
    config = load_config()
    return config['features']

def get_model_config() -> Dict[str, Any]:
    """
    Get model configuration.
    
    Returns:
        Dictionary containing model configuration
    """
    config = load_config()
    return config['model']

def load_curves() -> Dict[str, Any]:
    """
    Load the curves configuration file.
    
    Returns:
        Dictionary containing the curves configuration
    """
    curves_path = get_config_dir() / "curves.yaml"
    if not curves_path.exists():
        raise FileNotFoundError(f"Curves configuration file not found: {curves_path}")
    
    return load_yaml(curves_path)

def get_spread_list() -> list:
    """
    Get the list of configured spreads.
    
    Returns:
        List of spread names
    """
    curves = load_curves()
    return list(curves['spreads'].keys())

def get_output_dir() -> Path:
    """
    Get the output directory path.
    
    Returns:
        Path object pointing to output directory
    """
    config = load_config()
    return get_project_root() / config['output']['figures_dir'] 