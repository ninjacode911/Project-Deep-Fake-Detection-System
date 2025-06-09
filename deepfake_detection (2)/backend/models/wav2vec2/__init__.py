"""
DeepFake Detection System - Wav2Vec2 Model Package
Created: 2025-06-07
Author: ninjacode911

This module provides Wav2Vec2 model implementations with optimized inference
and resource management.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, Any
import weakref
import gc

from .model import Wav2Vec2Model
from .utils import load_wav2vec2, infer_wav2vec2
from ..base import BaseModel, ModelConfig
from ..logger import get_model_logger
from ...core.exceptions.backend_exceptions import ModelError

# Configure module logger
logger = logging.getLogger(__name__)

class Wav2Vec2Wrapper:
    """Wrapper class for Wav2Vec2 model with resource management."""
    
    # Singleton pattern
    _instance = None
    
    # Model cache using weak references
    _model_cache = weakref.WeakValueDictionary()
    
    # Configuration cache
    _config_cache: Dict[str, Any] = {}
    
    def __new__(cls) -> 'Wav2Vec2Wrapper':
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize wrapper."""
        if hasattr(self, '_initialized'):
            return
            
        try:
            # Set up cache directory
            self._cache_dir = Path(__file__).parent / "cache"
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load default configuration
            self._default_config = {
                'model_path': str(Path(__file__).parent / "Deepfake-audio-detection-V2"),
                'device': "cuda",
                'sample_rate': 16000,
                'cache_size': 100
            }
            
            logger.info("Wav2Vec2 wrapper initialized successfully")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Wav2Vec2 wrapper: {e}")
            raise ModelError(
                message="Wav2Vec2 wrapper initialization failed",
                error_code=6000,
                operation="init",
                details={'error': str(e)}
            )

    def get_model(
        self,
        config: Optional[Dict[str, Any]] = None,
        cache: bool = True
    ) -> Wav2Vec2Model:
        """
        Get Wav2Vec2 model instance.
        
        Args:
            config: Optional model configuration
            cache: Whether to cache model instance
            
        Returns:
            Model instance
        """
        try:
            # Generate cache key from config
            cfg = {**self._default_config, **(config or {})}
            cache_key = str(hash(str(cfg)))
            
            # Check cache
            if cache and cache_key in self._model_cache:
                logger.debug("Retrieved model from cache")
                return self._model_cache[cache_key]
            
            # Create new model instance
            model = Wav2Vec2Model(
                weights_path=cfg['model_path'],
                device=cfg['device']
            )
            
            # Add to cache if enabled
            if cache:
                self._model_cache[cache_key] = model
                logger.debug("Added model to cache")
                
            return model
            
        except Exception as e:
            logger.error(f"Failed to get Wav2Vec2 model: {e}")
            raise ModelError(
                message="Failed to get Wav2Vec2 model",
                error_code=6001,
                operation="get_model",
                details={
                    'config': str(config),
                    'error': str(e)
                }
            )

    def clear_cache(self) -> None:
        """Clear model cache."""
        try:
            self._model_cache.clear()
            self._config_cache.clear()
            gc.collect()
            logger.info("Cleared Wav2Vec2 caches")
            
        except Exception as e:
            logger.error(f"Failed to clear caches: {e}")
            raise ModelError(
                message="Cache clearing failed",
                error_code=6002,
                operation="clear_cache",
                details={'error': str(e)}
            )

    def cleanup(self) -> None:
        """Clean up wrapper resources."""
        try:
            # Clear caches
            self.clear_cache()
            
            # Clear configuration
            self._config_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Cleaned up Wav2Vec2 resources")
            
        except Exception as e:
            logger.error(f"Wav2Vec2 cleanup failed: {e}")
            raise ModelError(
                message="Wav2Vec2 cleanup failed",
                error_code=6003,
                operation="cleanup",
                details={'error': str(e)}
            )

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error in Wav2Vec2 destructor: {e}")

# Initialize global wrapper
wav2vec2_wrapper = Wav2Vec2Wrapper()

# Export public interfaces
__all__ = [
    'Wav2Vec2Model',
    'Wav2Vec2Wrapper',
    'wav2vec2_wrapper',
    'load_wav2vec2',
    'infer_wav2vec2'
]