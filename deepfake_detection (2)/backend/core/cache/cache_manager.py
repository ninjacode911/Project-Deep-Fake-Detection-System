"""
DeepFake Detection System - Cache Manager
Created: 2025-06-07
Author: ninjacode911

This module implements a hybrid memory/disk cache system with LRU eviction policy
and automatic cleanup for the deepfake detection system.
"""

import logging
import time
import traceback
import threading
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import json
import hashlib
from collections import OrderedDict
import diskcache
from diskcache import Cache as DiskCache
from contextlib import contextmanager

from ..exceptions.backend_exceptions import CacheError, BackendError, ResourceError
from ...config import config_manager

logger = logging.getLogger(__name__)

class LRUCache:
    """Thread-safe LRU cache implementation with size limits."""

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize LRU cache with size limit."""
        self._max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._size = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU update."""
        with self._lock:
            if key in self._cache:
                value = self._cache.pop(key)
                self._cache[key] = value
                return value
            return None

    def put(self, key: str, value: Any) -> None:
        """Put value in cache with size management."""
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value
            self._size += 1

    def remove(self, key: str) -> None:
        """Remove value from cache."""
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                self._size -= 1

    def clear(self) -> None:
        """Clear all values from cache."""
        with self._lock:
            self._cache.clear()
            self._size = 0

    @property
    def size(self) -> int:
        """Get current cache size."""
        return self._size

class CacheManager:
    """Hybrid memory/disk cache manager with automatic cleanup."""

    def __init__(self, cache_dir: Path) -> None:
        """Initialize cache manager with configuration."""
        try:
            # Validate cache directory
            self._validate_cache_dir(cache_dir)
            self._cache_dir = cache_dir
            
            # Initialize caches
            self._memory_cache = LRUCache(
                max_size=config_manager.get("cache.memory_size", 1000)
            )
            self._disk_cache = DiskCache(
                directory=str(cache_dir),
                size_limit=config_manager.get("cache.disk_size", 1024 * 1024 * 1024),  # 1GB
                eviction_policy='least-recently-used'
            )
            
            # Initialize cleanup settings
            self._cleanup_interval = config_manager.get("cache.cleanup_interval", 3600)  # 1 hour
            self._max_age = config_manager.get("cache.max_age", 86400)  # 24 hours
            self._last_cleanup = time.time()
            
            # Initialize resource tracking
            self._lock = threading.RLock()
            self._active_operations = 0
            self._max_concurrent = config_manager.get("cache.max_concurrent", 10)
            
            logger.info(f"Cache manager initialized at {cache_dir}")
            
        except Exception as e:
            logger.error(f"Cache manager initialization failed: {e}")
            raise CacheError(
                message="Failed to initialize cache manager",
                error_code=4000,
                operation="init",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_cache_dir(self, cache_dir: Path) -> None:
        """Validate and prepare cache directory."""
        try:
            # Create directory if it doesn't exist
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Check permissions
            if not os.access(cache_dir, os.W_OK):
                raise PermissionError(f"No write permission for cache directory: {cache_dir}")
                
            # Check disk space
            free_space = shutil.disk_usage(cache_dir).free
            min_space = config_manager.get("cache.min_disk_space", 1024 * 1024 * 1024)  # 1GB
            if free_space < min_space:
                raise OSError(f"Insufficient disk space for cache: {free_space} < {min_space}")
                
        except Exception as e:
            logger.error(f"Cache directory validation failed: {e}")
            raise CacheError(
                message="Failed to validate cache directory",
                error_code=4001,
                operation="validate_cache_dir",
                details={'error': str(e)}
            )

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with validation."""
        try:
            # Check memory cache first
            value = self._memory_cache.get(key)
            if value is not None:
                return value
                
            # Check disk cache
            with self._operation_slot():
                value = self._disk_cache.get(key)
                if value is not None:
                    # Validate cached data
                    if self._validate_cached_data(value):
                        # Update memory cache
                        self._memory_cache.put(key, value)
                        return value
                    else:
                        # Remove invalid data
                        self._disk_cache.delete(key)
                        
                    return None
                
        except Exception as e:
            logger.error(f"Cache get operation failed: {e}")
            raise CacheError(
                message="Failed to get value from cache",
                error_code=4002,
                operation="get",
                details={'key': key, 'error': str(e)}
            )

    def put(self, key: str, value: Any) -> None:
        """Put value in cache with validation."""
        try:
            # Validate value
            if not self._validate_value(value):
                raise ValueError("Invalid value for caching")
                
            # Update both caches
            with self._operation_slot():
                self._memory_cache.put(key, value)
                self._disk_cache.set(key, value)
                
            # Check cleanup
            self._check_cleanup()
                
        except Exception as e:
            logger.error(f"Cache put operation failed: {e}")
            raise CacheError(
                message="Failed to put value in cache",
                error_code=4003,
                operation="put",
                details={'key': key, 'error': str(e)}
            )

    def remove(self, key: str) -> None:
        """Remove value from cache."""
        try:
            with self._operation_slot():
                self._memory_cache.remove(key)
                self._disk_cache.delete(key)
                
        except Exception as e:
            logger.error(f"Cache remove operation failed: {e}")
            raise CacheError(
                message="Failed to remove value from cache",
                error_code=4004,
                operation="remove",
                details={'key': key, 'error': str(e)}
            )

    def clear(self) -> None:
        """Clear all values from cache."""
        try:
            with self._operation_slot():
                self._memory_cache.clear()
                self._disk_cache.clear()
                
        except Exception as e:
            logger.error(f"Cache clear operation failed: {e}")
            raise CacheError(
                message="Failed to clear cache",
                error_code=4005,
                operation="clear",
                details={'error': str(e)}
            )

    def _validate_value(self, value: Any) -> bool:
        """Validate value before caching."""
        try:
            # Check if value is JSON serializable
            json.dumps(value)
            return True
        except Exception:
            return False

    def _validate_cached_data(self, value: Any) -> bool:
        """Validate cached data integrity."""
        try:
            # Check if value is valid
            if not self._validate_value(value):
                return False
                
            # Check if value has expired
            if isinstance(value, dict) and 'timestamp' in value:
                age = time.time() - value['timestamp']
                if age > self._max_age:
                    return False
                    
            return True
            
        except Exception as e:
            logger.warning(f"Cache data validation failed: {e}")
            return False

    def _check_cleanup(self) -> None:
        """Check if cleanup is needed."""
        try:
            current_time = time.time()
            if current_time - self._last_cleanup >= self._cleanup_interval:
                self._perform_cleanup()
            self._last_cleanup = current_time
        except Exception as e:
            logger.error(f"Cache cleanup check failed: {e}")
            raise CacheError(
                message="Failed to check cache cleanup",
                error_code=4006,
                operation="check_cleanup",
                details={'error': str(e)}
            )

    def _perform_cleanup(self) -> None:
        """Perform cache cleanup."""
        try:
            with self._operation_slot():
                # Clean up expired items
                for key in list(self._disk_cache.iterkeys()):
                    value = self._disk_cache.get(key)
                    if not self._validate_cached_data(value):
                        self._disk_cache.delete(key)
                        self._memory_cache.remove(key)
                
                # Clean up memory cache
                self._memory_cache.clear()
                
                # Compact disk cache
                self._disk_cache.expire()
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            raise CacheError(
                message="Failed to perform cache cleanup",
                error_code=4007,
                operation="perform_cleanup",
                details={'error': str(e)}
            )

    @contextmanager
    def _operation_slot(self):
        """Manage concurrent operations."""
        try:
            with self._lock:
                while self._active_operations >= self._max_concurrent:
                    time.sleep(0.1)
                self._active_operations += 1
                
            try:
                yield
            finally:
                with self._lock:
                    self._active_operations -= 1
                    
        except Exception as e:
            logger.error(f"Cache operation slot management failed: {e}")
            raise CacheError(
                message="Failed to manage cache operation slot",
                error_code=4008,
                operation="operation_slot",
                details={'error': str(e)}
            )

    @property
    def size(self) -> int:
        """Get total cache size."""
        return self._memory_cache.size + len(self._disk_cache)

    def cleanup(self) -> None:
        """Clean up cache resources."""
        try:
            # Clear caches
            self.clear()
            
            # Close disk cache
            self._disk_cache.close()
            
            logger.info("Cache manager resources cleaned up")
            
        except Exception as e:
            logger.error(f"Cache manager cleanup failed: {e}")
            raise CacheError(
                message="Failed to cleanup cache manager resources",
                error_code=4009,
                operation="cleanup",
                details={'error': str(e)}
            )

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Cache manager cleanup in destructor failed: {e}")