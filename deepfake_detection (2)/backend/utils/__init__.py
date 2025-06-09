"""
DeepFake Detection System - Utilities Module
Created: 2025-06-07
Author: ninjacode911

This module provides core utility functions and helpers with proper resource
management and error handling.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from functools import lru_cache
import torch

from .logger import (
    setup_logger,
    get_logger,
    LogLevel,
    configure_logging,
)

from .model_utils import (
    save_model_weights,
    optimize_model,
    get_device,
    clear_gpu_memory,
    calculate_model_size,
    enable_gpu_optimization,
    ModelOptimizationConfig,
    load_model_weights
)

# Configure base logger
logger = get_logger(__name__)

class ResourceManager:
    """Manages system resources and cleanup."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.resources: Dict[str, Any] = {}
            self.cache_enabled = True
            self._initialized = True
            logger.info("Resource manager initialized")
    
    def register(self, resource_id: str, resource: Any) -> None:
        """Register a resource for management."""
        self.resources[resource_id] = resource
        logger.debug(f"Registered resource: {resource_id}")
    
    def release(self, resource_id: str) -> None:
        """Release a specific resource."""
        try:
            if resource_id in self.resources:
                resource = self.resources.pop(resource_id)
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'close'):
                    resource.close()
                del resource
                torch.cuda.empty_cache()
                logger.debug(f"Released resource: {resource_id}")
        except Exception as e:
            logger.error(f"Error releasing resource {resource_id}: {str(e)}")

    def cleanup(self) -> None:
        """Cleanup all managed resources."""
        try:
            for resource_id in list(self.resources.keys()):
                self.release(resource_id)
            torch.cuda.empty_cache()
            logger.info("All resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

class CacheManager:
    """Manages caching with automatic invalidation."""
    
    def __init__(self, max_size: int = 1024 * 1024 * 100):  # 100MB default
        self.max_size = max_size
        self.current_size = 0
        self.cache: Dict[str, Any] = {}
        logger.info("Cache manager initialized")
    
    @lru_cache(maxsize=128)
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with LRU policy."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set cache item with size tracking."""
        try:
            item_size = sys.getsizeof(value)
            while self.current_size + item_size > self.max_size:
                if not self.invalidate_oldest():
                    break
            self.cache[key] = value
            self.current_size += item_size
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
    
    def invalidate_oldest(self) -> bool:
        """Invalidate oldest cache entry."""
        try:
            if self.cache:
                oldest_key = next(iter(self.cache))
                self.invalidate(oldest_key)
                return True
            return False
        except Exception as e:
            logger.error(f"Cache invalidation error: {str(e)}")
            return False
    
    def invalidate(self, key: str) -> None:
        """Invalidate specific cache entry."""
        try:
            if key in self.cache:
                value = self.cache.pop(key)
                self.current_size -= sys.getsizeof(value)
                del value
        except Exception as e:
            logger.error(f"Error invalidating cache key {key}: {str(e)}")

    def clear(self) -> None:
        """Clear entire cache."""
        try:
            self.cache.clear()
            self.current_size = 0
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

# Initialize managers
resource_manager = ResourceManager()
cache_manager = CacheManager()

# Configure logging
configure_logging()

__all__ = [
    # Resource Management
    'ResourceManager',
    'resource_manager',
    'CacheManager', 
    'cache_manager',
    
    # Logging
    'setup_logger',
    'get_logger',
    'LogLevel',
    'configure_logging',
    
    # Model Utilities
    'load_model_weights',
    'save_model_weights', 
    'optimize_model',
    'get_device',
    'clear_gpu_memory',
    'calculate_model_size',
    'enable_gpu_optimization',
    'ModelOptimizationConfig'
]

def cleanup():
    """Cleanup all resources on module unload."""
    try:
        resource_manager.cleanup()
        cache_manager.clear()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")

def verify_system_setup():
    checks = {
        "CUDA Available": torch.cuda.is_available(),
        "GPU Memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        "Model Files": all(Path(f).exists() for f in required_files),
        "Directories": all(Path(d).exists() for d in required_dirs),
        "Database": Path("data/deepfake_detection.db").exists(),
    }
    return checks