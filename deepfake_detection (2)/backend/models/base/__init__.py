"""
DeepFake Detection System - Base Models Package
Created: 2025-06-07
Author: ninjacode911

This module provides base model abstractions and interfaces with comprehensive 
error handling and resource management.
"""

import logging
from typing import Dict, Any, Type, Optional
from pathlib import Path

from .model_base import (
    BaseModel,
    VideoModel,
    AudioModel,
    MultiModalModel,
    ModelConfig,
    ModelMetrics
)
from ..logger import get_model_logger
from ...core.exceptions.backend_exceptions import ModelError

# Configure module logger
logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for model implementations."""
    
    _models: Dict[str, Type[BaseModel]] = {}
    _initialized: bool = False
    _instance = None
    
    def __new__(cls) -> 'ModelRegistry':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize model registry."""
        if self._initialized:
            return
            
        try:
            # Set up model cache directory
            cache_dir = Path(__file__).parent.parent.parent / "data" / "cache" / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_dir = cache_dir
            
            logger.info("Model registry initialized successfully")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize model registry: {e}")
            raise ModelError(
                message="Model registry initialization failed",
                error_code=3000,
                operation="init",
                details={'error': str(e)}
            )

    @classmethod
    def register(cls, model_type: str) -> callable:
        """
        Register model class decorator.
        
        Args:
            model_type: Type identifier for model
            
        Returns:
            Decorator function
        """
        def decorator(model_class: Type[BaseModel]) -> Type[BaseModel]:
            if not issubclass(model_class, BaseModel):
                raise ModelError(
                    message=f"Class {model_class.__name__} is not a BaseModel subclass",
                    error_code=3001,
                    operation="register",
                    details={'model_type': model_type}
                )
            
            cls._models[model_type] = model_class
            logger.debug(f"Registered model type: {model_type}")
            return model_class
            
        return decorator

    @classmethod
    def get_model_class(cls, model_type: str) -> Type[BaseModel]:
        """
        Get model class by type.
        
        Args:
            model_type: Type identifier for model
            
        Returns:
            Model class
        """
        try:
            return cls._models[model_type]
        except KeyError:
            raise ModelError(
                message=f"Model type '{model_type}' not registered",
                error_code=3002,
                operation="get_model_class",
                details={'available_types': list(cls._models.keys())}
            )

    @classmethod
    def list_models(cls) -> Dict[str, Type[BaseModel]]:
        """Get registered model types."""
        return cls._models.copy()

    def get_cache_dir(self) -> Path:
        """Get model cache directory."""
        return self._cache_dir

    def cleanup(self) -> None:
        """Clean up registry resources."""
        try:
            self._models.clear()
            logger.info("Model registry cleaned up")
            
        except Exception as e:
            logger.error(f"Model registry cleanup failed: {e}")
            raise ModelError(
                message="Model registry cleanup failed",
                error_code=3003,
                operation="cleanup",
                details={'error': str(e)}
            )

# Initialize model registry
model_registry = ModelRegistry()

# Export public interfaces
__all__ = [
    'BaseModel',
    'VideoModel', 
    'AudioModel',
    'MultiModalModel',
    'ModelConfig',
    'ModelMetrics',
    'ModelRegistry',
    'model_registry'
]

def get_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
    """
    Factory function to get model instance.

    Args:
        model_type: Type identifier for model
        config: Optional model configuration

    Returns:
        Model instance
    """
    try:
        model_class = model_registry.get_model_class(model_type)
        return model_class(config or {})
        
    except Exception as e:
        logger.error(f"Failed to get model instance: {e}")
        raise ModelError(
            message=f"Failed to instantiate model type '{model_type}'",
            error_code=3004,
            operation="get_model",
            details={
                'model_type': model_type,
                'error': str(e)
            }
        )