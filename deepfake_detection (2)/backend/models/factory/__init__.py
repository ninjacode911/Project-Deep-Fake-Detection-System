"""
DeepFake Detection System - Model Factory Initialization
Created: 2025-06-07
Author: ninjacode911

This module initializes the model factory components and exports factory classes.
"""

# filepath: /media/ssd/deepfake_detection/backend/models/factory/__init__.py

from typing import Dict, Type, Optional
import logging
from pathlib import Path
import gc
import weakref

from ..base import BaseModel
from ...core.exceptions.backend_exceptions import FactoryError
from .model_factory import ModelFactory

# Configure logger
logger = logging.getLogger(__name__)

# Global model registry and cache
_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}
_MODEL_CACHE = weakref.WeakValueDictionary()

def register_model(name: str, model_class: Type[BaseModel]) -> None:
    """
    Register a model class with the factory.
    
    Args:
        name: Model identifier
        model_class: Model class to register
    """
    try:
        if name in _MODEL_REGISTRY:
            logger.warning(f"Model {name} already registered, overwriting")
        _MODEL_REGISTRY[name] = model_class
        logger.info(f"Registered model: {name}")
        
    except Exception as e:
        logger.error(f"Failed to register model {name}: {e}")
        raise FactoryError(
            message="Model registration failed",
            error_code=5000,
            operation="register_model",
            details={'model': name, 'error': str(e)}
        )

def get_model(
    name: str,
    cache: bool = True,
    **kwargs
) -> BaseModel:
    """
    Get model instance from registry.
    
    Args:
        name: Model identifier
        cache: Whether to cache model instance
        **kwargs: Additional model arguments
        
    Returns:
        Model instance
    """
    try:
        # Check cache first
        if cache and name in _MODEL_CACHE:
            logger.debug(f"Retrieved {name} from cache")
            return _MODEL_CACHE[name]
            
        # Get model class and instantiate
        if name not in _MODEL_REGISTRY:
            raise KeyError(f"Model {name} not found in registry")
            
        model_class = _MODEL_REGISTRY[name]
        model = model_class(**kwargs)
        
        # Add to cache if enabled
        if cache:
            _MODEL_CACHE[name] = model
            logger.debug(f"Cached model instance: {name}")
            
        return model
        
    except Exception as e:
        logger.error(f"Failed to get model {name}: {e}")
        raise FactoryError(
            message="Model instantiation failed",
            error_code=5001,
            operation="get_model",
            details={'model': name, 'error': str(e)}
        )

def clear_cache() -> None:
    """Clear model cache and run garbage collection."""
    try:
        _MODEL_CACHE.clear()
        gc.collect()
        logger.info("Cleared model cache")
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise FactoryError(
            message="Cache clearing failed",
            error_code=5002,
            operation="clear_cache",
            details={'error': str(e)}
        )

def cleanup() -> None:
    """Clean up factory resources."""
    try:
        # Clear caches
        clear_cache()
        
        # Clear registry
        _MODEL_REGISTRY.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Cleaned up factory resources")
        
    except Exception as e:
        logger.error(f"Factory cleanup failed: {e}")
        raise FactoryError(
            message="Factory cleanup failed",
            error_code=5003,
            operation="cleanup",
            details={'error': str(e)}
        )

# Export factory class
__all__ = ['ModelFactory', 'register_model', 'get_model', 'clear_cache', 'cleanup']