"""
DeepFake Detection System - Core Cache Management
Created: 2025-06-07
Author: ninjacode911

This module initializes the caching system for frames, audio, and analysis results.
It provides a centralized cache management system with memory optimization and 
automatic cleanup.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from .cache_manager import CacheManager

# Configure logger
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_CONFIG = {
    'frames': {
        'max_size_mb': 512,
        'ttl_seconds': 3600,
        'cleanup_interval': 300
    },
    'audio': {
        'max_size_mb': 256,
        'ttl_seconds': 3600,
        'cleanup_interval': 300
    },
    'results': {
        'max_size_mb': 128,
        'ttl_seconds': 7200,
        'cleanup_interval': 600
    }
}

# Initialize cache managers
_frame_cache: Optional[CacheManager] = None
_audio_cache: Optional[CacheManager] = None
_result_cache: Optional[CacheManager] = None

def initialize_caches() -> None:
    """Initialize all cache managers with their respective configurations."""
    global _frame_cache, _audio_cache, _result_cache
    
    try:
        cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        _frame_cache = CacheManager(
            cache_dir / "frames",
            **CACHE_CONFIG['frames']
        )
        _audio_cache = CacheManager(
            cache_dir / "audio",
            **CACHE_CONFIG['audio']
        )
        _result_cache = CacheManager(
            cache_dir / "results",
            **CACHE_CONFIG['results']
        )
        
        logger.info("Cache managers initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize cache managers: {e}")
        raise

def get_frame_cache() -> CacheManager:
    """Get the frame cache manager instance."""
    if _frame_cache is None:
        initialize_caches()
    return _frame_cache

def get_audio_cache() -> CacheManager:
    """Get the audio cache manager instance."""
    if _audio_cache is None:
        initialize_caches()
    return _audio_cache

def get_result_cache() -> CacheManager:
    """Get the analysis results cache manager instance."""
    if _result_cache is None:
        initialize_caches()
    return _result_cache

def cleanup_all_caches() -> None:
    """Clean up all cache managers and their resources."""
    try:
        for cache in [_frame_cache, _audio_cache, _result_cache]:
            if cache is not None:
                cache.cleanup()
        logger.info("All caches cleaned up successfully")
    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}")
        raise

# Register cleanup on module unload
import atexit
atexit.register(cleanup_all_caches)

__all__ = [
    'get_frame_cache',
    'get_audio_cache',
    'get_result_cache',
    'cleanup_all_caches'
]