"""
Configuration loader for experiment setup.
Loads method endpoints and evaluator configurations.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentConfig:
    """Manages configuration for experiments including method endpoints and evaluators."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to config YAML file. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_method_config(self, method_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific method.
        
        Args:
            method_name: Name of the method (e.g., 'method1', 'method2')
            
        Returns:
            Dictionary containing method configuration
        """
        methods = self.config.get('methods', {})
        if method_name not in methods:
            raise ValueError(f"Method '{method_name}' not found in config")
        return methods[method_name]
    
    def get_all_methods(self) -> Dict[str, Dict[str, Any]]:
        """Get all method configurations."""
        return self.config.get('methods', {})
    
    def get_method_url(self, method_name: str, endpoint: str) -> str:
        """
        Get full URL for a method endpoint.
        
        Args:
            method_name: Name of the method
            endpoint: Endpoint name (e.g., 'generate', 'process')
            
        Returns:
            Full URL string
        """
        method_config = self.get_method_config(method_name)
        base_url = method_config['base_url'].rstrip('/')
        endpoint_path = method_config['endpoints'].get(endpoint)
        
        if endpoint_path is None:
            raise ValueError(f"Endpoint '{endpoint}' not found for method '{method_name}'")
        
        return f"{base_url}{endpoint_path}"
    
    def get_vlm_config(self, vlm_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific VLM evaluator.
        
        Args:
            vlm_name: Name of the VLM (e.g., 'gpt4v', 'claude_vision')
            
        Returns:
            Dictionary containing VLM configuration
        """
        vlms = self.config.get('vlm_evaluators', {})
        if vlm_name not in vlms:
            raise ValueError(f"VLM '{vlm_name}' not found in config")
        return vlms[vlm_name]
    
    def get_all_vlms(self) -> Dict[str, Dict[str, Any]]:
        """Get all VLM configurations."""
        return self.config.get('vlm_evaluators', {})
    
    def get_metrics(self) -> list:
        """Get list of evaluation metrics."""
        return self.config.get('metrics', [])
    
    def get_experiment_settings(self) -> Dict[str, Any]:
        """Get general experiment settings."""
        return self.config.get('experiment', {})
    
    def update_method_url(self, method_name: str, new_base_url: str) -> None:
        """
        Update base URL for a method and save to config file.
        
        Args:
            method_name: Name of the method
            new_base_url: New base URL
        """
        if method_name not in self.config['methods']:
            raise ValueError(f"Method '{method_name}' not found in config")
        
        self.config['methods'][method_name]['base_url'] = new_base_url
        self._save_config()
    
    def _save_config(self) -> None:
        """Save current configuration to YAML file."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)


# Singleton instance
_config_instance = None


def get_config(config_path: Optional[str] = None) -> ExperimentConfig:
    """
    Get or create the global configuration instance.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        ExperimentConfig instance
    """
    global _config_instance
    if _config_instance is None or config_path is not None:
        _config_instance = ExperimentConfig(config_path)
    return _config_instance
