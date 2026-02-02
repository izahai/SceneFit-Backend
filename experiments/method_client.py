"""
Client utility for making HTTP requests to method endpoints.
Handles communication with different research methods hosted on separate servers.
"""

import requests
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from config_loader import get_config


class MethodClient:
    """Client for interacting with a specific method's API."""
    
    def __init__(
        self,
        method_name: str,
        timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize method client.
        
        Args:
            method_name: Name of the method (e.g., 'method1', 'method2')
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.method_name = method_name
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Load configuration
        config = get_config()
        self.method_config = config.get_method_config(method_name)
        self.base_url = self.method_config['base_url'].rstrip('/')
        self.endpoints = self.method_config['endpoints']
        
        # Track request statistics
        self.request_count = 0
        self.error_count = 0
    
    def _make_request(
        self,
        endpoint_name: str,
        method: str = "POST",
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to a specific endpoint.
        
        Args:
            endpoint_name: Name of the endpoint (e.g., 'generate', 'process')
            method: HTTP method (GET, POST, etc.)
            data: JSON data to send
            files: Files to upload
            params: URL parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            requests.exceptions.RequestException: If request fails after retries
        """
        endpoint_path = self.endpoints.get(endpoint_name)
        if endpoint_path is None:
            raise ValueError(f"Endpoint '{endpoint_name}' not found for {self.method_name}")
        
        url = f"{self.base_url}{endpoint_path}"
        
        for attempt in range(self.retry_attempts):
            try:
                self.request_count += 1
                
                if method.upper() == "GET":
                    response = requests.get(
                        url,
                        params=params,
                        timeout=self.timeout
                    )
                elif method.upper() == "POST":
                    if files:
                        response = requests.post(
                            url,
                            data=data,
                            files=files,
                            timeout=self.timeout
                        )
                    else:
                        response = requests.post(
                            url,
                            json=data,
                            params=params,
                            timeout=self.timeout
                        )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                
                # Try to parse JSON, fallback to text
                try:
                    return response.json()
                except json.JSONDecodeError:
                    return {"response": response.text}
                
            except requests.exceptions.RequestException as e:
                self.error_count += 1
                
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    raise RuntimeError(
                        f"Request to {self.method_name}/{endpoint_name} failed after "
                        f"{self.retry_attempts} attempts: {str(e)}"
                    )
    
    def generate(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Call the generate endpoint.
        
        Args:
            input_data: Input data for generation
            **kwargs: Additional parameters
            
        Returns:
            Generation result
        """
        return self._make_request("generate", method="POST", data=input_data, **kwargs)
    
    def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Call the process endpoint.
        
        Args:
            input_data: Input data for processing
            **kwargs: Additional parameters
            
        Returns:
            Processing result
        """
        return self._make_request("process", method="POST", data=input_data, **kwargs)
    
    def health_check(self) -> bool:
        """
        Check if the method server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = self._make_request("health", method="GET")
            return True
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "method_name": self.method_name,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0
        }


class MethodOrchestrator:
    """Orchestrator for managing multiple method clients."""
    
    def __init__(self, method_names: Optional[List[str]] = None):
        """
        Initialize orchestrator.
        
        Args:
            method_names: List of method names to initialize. If None, loads all from config.
        """
        config = get_config()
        
        if method_names is None:
            method_names = list(config.get_all_methods().keys())
        
        self.clients = {
            name: MethodClient(name) for name in method_names
        }
    
    def get_client(self, method_name: str) -> MethodClient:
        """
        Get client for a specific method.
        
        Args:
            method_name: Name of the method
            
        Returns:
            MethodClient instance
        """
        if method_name not in self.clients:
            raise ValueError(f"Method '{method_name}' not initialized")
        return self.clients[method_name]
    
    def health_check_all(self) -> Dict[str, bool]:
        """
        Check health of all methods.
        
        Returns:
            Dictionary mapping method names to health status
        """
        return {
            name: client.health_check()
            for name, client in self.clients.items()
        }
    
    def run_on_all_methods(
        self,
        endpoint: str,
        input_data: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the same input on all methods.
        
        Args:
            endpoint: Endpoint name ('generate' or 'process')
            input_data: Input data to send to all methods
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping method names to their outputs
        """
        results = {}
        
        for name, client in self.clients.items():
            try:
                if endpoint == "generate":
                    results[name] = client.generate(input_data, **kwargs)
                elif endpoint == "process":
                    results[name] = client.process(input_data, **kwargs)
                else:
                    raise ValueError(f"Unsupported endpoint: {endpoint}")
            except Exception as e:
                results[name] = {"error": str(e), "success": False}
        
        return results
    
    def run_batch_on_method(
        self,
        method_name: str,
        endpoint: str,
        batch_data: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run a batch of inputs on a single method.
        
        Args:
            method_name: Name of the method
            endpoint: Endpoint name
            batch_data: List of input data dictionaries
            **kwargs: Additional parameters
            
        Returns:
            List of output dictionaries
        """
        client = self.get_client(method_name)
        results = []
        
        for input_data in batch_data:
            try:
                if endpoint == "generate":
                    result = client.generate(input_data, **kwargs)
                elif endpoint == "process":
                    result = client.process(input_data, **kwargs)
                else:
                    raise ValueError(f"Unsupported endpoint: {endpoint}")
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "success": False})
        
        return results
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all clients.
        
        Returns:
            Dictionary mapping method names to their statistics
        """
        return {
            name: client.get_stats()
            for name, client in self.clients.items()
        }
