"""
DeepFake Detection System - Cache Manager
Created: 2025-06-07
Author: ninjacode911

This module implements caching functionality for deepfake detection results
with proper resource management and error handling.
"""

import logging
import time
import traceback
import threading
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np
import torch
from contextlib import contextmanager

from .exceptions.backend_exceptions import CacheError, ValidationError, ResourceError
from ..config import config_manager

logger = logging.getLogger(__name__)

class CacheManager:
    """Manager class for caching detection results."""

    def __init__(self) -> None:
        """Initialize cache manager with configuration."""
        try:
            self._lock = threading.RLock()
            self._memory_limit = config_manager.get("cache.memory_limit", 1 * 1024 * 1024 * 1024)  # 1GB
            self._max_entries = config_manager.get("cache.max_entries", 1000)
            self._cache_dir = Path(config_manager.get("cache.directory", "cache"))
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize cache
            self._cache: Dict[str, Dict[str, Any]] = {}
            self._access_times: Dict[str, float] = {}
            
            logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Cache manager initialization failed: {e}")
            raise CacheError(
                message="Failed to initialize cache manager",
                error_code=4000,
                operation="init",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_key(self, key: str) -> None:
        """
        Validate cache key.
        
        Args:
            key: Cache key
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if not key or not isinstance(key, str):
                raise ValidationError(
                    message="Invalid cache key",
                    error_code=4001,
                    details={'key': key}
                )
                
            if len(key) > 256:  # Reasonable key length limit
                raise ValidationError(
                    message="Cache key too long",
                    error_code=4002,
                    details={'key': key, 'length': len(key)}
                )
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Cache key validation failed: {e}")
            raise ValidationError(
                message="Cache key validation failed",
                error_code=4003,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_value(self, value: Any) -> None:
        """
        Validate cache value.
        
        Args:
            value: Cache value
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if value is None:
                raise ValidationError(
                    message="Cache value cannot be None",
                    error_code=4004,
                    details={'value': value}
                )
                
            # Check if value is too large
            value_size = self._estimate_size(value)
            max_size = config_manager.get("cache.max_value_size", 100 * 1024 * 1024)  # 100MB
            if value_size > max_size:
                raise ValidationError(
                    message="Cache value too large",
                    error_code=4005,
                    details={
                        'size': value_size,
                        'max_size': max_size
                    }
                )
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Cache value validation failed: {e}")
            raise ValidationError(
                message="Cache value validation failed",
                error_code=4006,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _estimate_size(self, value: Any) -> int:
        """
        Estimate size of cache value in bytes.
        
        Args:
            value: Cache value
            
        Returns:
            Estimated size in bytes
        """
        try:
            if isinstance(value, (str, bytes)):
                return len(value)
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in value.items()
                )
            elif isinstance(value, np.ndarray):
                return value.nbytes
            elif isinstance(value, torch.Tensor):
                return value.element_size() * value.nelement()
            else:
                return len(str(value))
                
        except Exception as e:
            logger.error(f"Size estimation failed: {e}")
            return 0

    @contextmanager
    def _memory_context(self):
        """Context manager for memory monitoring."""
        try:
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            yield
            current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            if current_memory - start_memory > self._memory_limit:
                raise ResourceError(
                    message="Memory limit exceeded",
                    error_code=4007,
                    details={
                        'used': current_memory - start_memory,
                        'limit': self._memory_limit
                    }
                )
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            raise ResourceError(
                message="Memory monitoring failed",
                error_code=4008,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            # Validate key
            self._validate_key(key)
            
            # Check memory cache
            with self._lock:
                if key in self._cache:
                    self._access_times[key] = time.time()
                    return self._cache[key]['value']
                    
            # Check disk cache
            cache_file = self._cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        value = self._deserialize_value(data['value'])
                        
                        # Update memory cache
                        with self._lock:
                            self._cache[key] = {
                                'value': value,
                                'size': self._estimate_size(value)
                            }
                            self._access_times[key] = time.time()
                            
                        return value
                        
                except Exception as e:
                    logger.error(f"Failed to load cache file: {e}")
                    return None
                    
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            raise CacheError(
                message="Failed to get from cache",
                error_code=4009,
                operation="get",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def set(
        self,
        key: str,
        value: Any,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Cache value
            progress_callback: Optional callback for progress updates
        """
        try:
            # Validate inputs
            self._validate_key(key)
            self._validate_value(value)
            
            # Check cache size
            value_size = self._estimate_size(value)
            with self._lock:
                if len(self._cache) >= self._max_entries:
                    self._evict_entries()
                    
            # Update memory cache
            with self._lock:
                self._cache[key] = {
                    'value': value,
                    'size': value_size
                }
                self._access_times[key] = time.time()
                
            if progress_callback:
                progress_callback(0.5)  # Memory cache updated
                
            # Update disk cache
            cache_file = self._cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
            try:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'value': self._serialize_value(value),
                        'timestamp': time.time()
                    }, f)
                    
                if progress_callback:
                    progress_callback(1.0)  # Disk cache updated
                    
            except Exception as e:
                logger.error(f"Failed to write cache file: {e}")
                # Remove from memory cache if disk write fails
                with self._lock:
                    self._cache.pop(key, None)
                    self._access_times.pop(key, None)
                raise
                
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            raise CacheError(
                message="Failed to set cache",
                error_code=4010,
                operation="set",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _evict_entries(self) -> None:
        """Evict least recently used entries from cache."""
        try:
            # Sort by access time
            sorted_entries = sorted(
                self._access_times.items(),
                key=lambda x: x[1]
            )
            
            # Remove oldest entries
            num_to_remove = max(1, len(sorted_entries) // 10)  # Remove 10% or at least 1
            for key, _ in sorted_entries[:num_to_remove]:
                self._cache.pop(key, None)
                self._access_times.pop(key, None)
                
                # Remove from disk
                cache_file = self._cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.json"
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.error(f"Failed to remove cache file: {e}")
                        
        except Exception as e:
            logger.error(f"Cache eviction failed: {e}")
            raise CacheError(
                message="Failed to evict cache entries",
                error_code=4011,
                operation="evict_entries",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _serialize_value(self, value: Any) -> Any:
        """
        Serialize cache value for storage.
        
        Args:
            value: Cache value
            
        Returns:
            Serialized value
        """
        try:
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            elif isinstance(value, np.ndarray):
                return {
                    'type': 'ndarray',
                    'data': value.tolist(),
                    'dtype': str(value.dtype)
                }
            elif isinstance(value, torch.Tensor):
                return {
                    'type': 'tensor',
                    'data': value.cpu().numpy().tolist(),
                    'dtype': str(value.dtype)
                }
            elif isinstance(value, (list, tuple)):
                return [self._serialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {
                    k: self._serialize_value(v)
                    for k, v in value.items()
                }
            else:
                return str(value)
                
        except Exception as e:
            logger.error(f"Value serialization failed: {e}")
            raise CacheError(
                message="Failed to serialize value",
                error_code=4012,
                operation="serialize_value",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _deserialize_value(self, value: Any) -> Any:
        """
        Deserialize cache value from storage.
        
        Args:
            value: Serialized value
            
        Returns:
            Deserialized value
        """
        try:
            if isinstance(value, dict) and 'type' in value:
                if value['type'] == 'ndarray':
                    return np.array(value['data'], dtype=value['dtype'])
                elif value['type'] == 'tensor':
                    return torch.tensor(value['data'], dtype=value['dtype'])
            elif isinstance(value, (list, tuple)):
                return [self._deserialize_value(item) for item in value]
            elif isinstance(value, dict):
                return {
                    k: self._deserialize_value(v)
                    for k, v in value.items()
                }
            return value
            
        except Exception as e:
            logger.error(f"Value deserialization failed: {e}")
            raise CacheError(
                message="Failed to deserialize value",
                error_code=4013,
                operation="deserialize_value",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            with self._lock:
                self._cache.clear()
                self._access_times.clear()
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise CacheError(
                message="Failed to cleanup resources",
                error_code=4014,
                operation="cleanup",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def __del__(self) -> None:
        """Cleanup in destructor."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Cache manager cleanup in destructor failed: {e}") 