"""
DeepFake Detection System - Models Package
Created: 2025-06-07
Author: ninjacode911

This module initializes model components and provides high-level model interfaces
with comprehensive error handling and resource management.
"""

import logging
from typing import Dict, Any, Optional, Type
from pathlib import Path
import gc
import weakref

# Import model components
from .base import (
    BaseModel,
    VideoModel,
    AudioModel,
    MultiModalModel,
    ModelConfig,
    ModelMetrics,
    model_registry
)
from .efficientnet import (
    EfficientNetModel,
    EfficientNetWrapper,
    load_efficientnet,
    infer_efficientnet,
    get_gradcam_heatmap
)
from .wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2Wrapper,
    load_wav2vec2, 
    infer_wav2vec2
)
from .factory import ModelFactory, factory
from .logger import get_model_logger
from ..core.exceptions.backend_exceptions import ModelError

# Configure logger
logger = logging.getLogger(__name__)

# Global model cache using weak references
_MODEL_CACHE = weakref.WeakValueDictionary()

class ModelManager:
    """Manages model initialization and lifecycle."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls) -> 'ModelManager':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize model manager."""
        if self._initialized:
            return
            
        try:
            # Set up cache directory
            self._cache_dir = Path(__file__).parent.parent / "data" / "cache" / "models"
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize model registry
            self._model_registry: Dict[str, Type[BaseModel]] = {
                'efficientnet': EfficientNetModel,
                'wav2vec2': Wav2Vec2Model
            }
            
            logger.info("Model manager initialized successfully")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            raise ModelError(
                message="Model manager initialization failed",
                error_code=2000,
                operation="init",
                details={'error': str(e)}
            )

    def get_model(
        self,
        model_type: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseModel:
        """
        Get model instance, using cache if available.
        
        Args:
            model_type: Type of model to get
            config: Optional model configuration
            **kwargs: Additional model arguments
            
        Returns:
            Model instance
        """
        try:
            # Use factory to get model
            return factory.get_model(
                model_type=model_type,
                config=config,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Failed to get model {model_type}: {e}")
            raise ModelError(
                message=f"Failed to get model type '{model_type}'",
                error_code=2001,
                operation="get_model",
                details={
                    'model_type': model_type,
                    'error': str(e)
                }
            )

    def clear_cache(self) -> None:
        """Clear model cache."""
        try:
            _MODEL_CACHE.clear()
            factory.clear_cache()
            gc.collect()
            logger.info("Cleared model caches")
            
        except Exception as e:
            logger.error(f"Failed to clear model cache: {e}")
            raise ModelError(
                message="Failed to clear model cache",
                error_code=2002,
                operation="clear_cache",
                details={'error': str(e)}
            )

    def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            # Clear caches
            self.clear_cache()
            
            # Clean up factory
            factory.cleanup()
            
            # Clean up registry
            self._model_registry.clear()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Cleaned up model resources")
            
        except Exception as e:
            logger.error(f"Model cleanup failed: {e}")
            raise ModelError(
                message="Failed to cleanup model resources",
                error_code=2003,
                operation="cleanup",
                details={'error': str(e)}
            )

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error in model manager destructor: {e}")

# Initialize model manager
model_manager = ModelManager()

# Public exports
__all__ = [
    # Core model components
    'BaseModel',
    'VideoModel',
    'AudioModel', 
    'MultiModalModel',
    'ModelConfig',
    'ModelMetrics',
    
    # Model registry
    'model_registry',
    
    # Model factory
    'ModelFactory',
    'factory',
    
    # Model manager
    'ModelManager',
    'model_manager',
    
    # EfficientNet components
    'EfficientNetModel',
    'EfficientNetWrapper',
    'load_efficientnet',
    'infer_efficientnet',
    'get_gradcam_heatmap',
    
    # Wav2Vec2 components
    'Wav2Vec2Model',
    'Wav2Vec2Wrapper', 
    'load_wav2vec2',
    'infer_wav2vec2'
]