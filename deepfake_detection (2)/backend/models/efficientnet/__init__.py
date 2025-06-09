"""
DeepFake Detection System - EfficientNet Model Package
Created: 2025-06-07
Author: ninjacode911

This module provides EfficientNet model implementations with optimized inference
and resource management.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any

from .model import EfficientNetModel
from .utils import (
    load_efficientnet,
    get_gradcam_heatmap,
    preprocess_image,
    cleanup_gradients
)
from ..base import BaseModel, ModelConfig
from ..logger import get_model_logger
from ...core.exceptions.backend_exceptions import ModelError

# Configure logger
logger = logging.getLogger(__name__)

class EfficientNetWrapper:
    """Wrapper class for EfficientNet model with resource management."""
    
    _instances: Dict[str, 'EfficientNetWrapper'] = {}
    _cache_dir = Path(__file__).parent / "cache"
    
    def __new__(cls, model_name: str, *args, **kwargs) -> 'EfficientNetWrapper':
        """Implement singleton pattern per model name."""
        if model_name not in cls._instances:
            cls._instances[model_name] = super().__new__(cls)
        return cls._instances[model_name]

    def __init__(self, model_name: str, config: Optional[ModelConfig] = None) -> None:
        """
        Initialize EfficientNet wrapper.
        
        Args:
            model_name: Name of EfficientNet model variant
            config: Optional model configuration
        """
        if hasattr(self, '_initialized'):
            return
            
        try:
            self._model_name = model_name
            self._config = config or {}
            self._model = None
            self._cache = {}
            
            # Create cache directory
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Initialized EfficientNet wrapper for {model_name}")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize EfficientNet wrapper: {e}")
            raise ModelError(
                message="EfficientNet wrapper initialization failed",
                error_code=4000,
                operation="init",
                details={'model_name': model_name, 'error': str(e)}
            )

    def load_model(self, weights_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load EfficientNet model weights.
        
        Args:
            weights_path: Optional path to model weights
        """
        try:
            self._model = load_efficientnet(
                model_name=self._model_name,
                weights_path=weights_path,
                config=self._config
            )
            logger.info(f"Loaded EfficientNet model: {self._model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load EfficientNet model: {e}")
            raise ModelError(
                message="Failed to load EfficientNet model",
                error_code=4001, 
                operation="load_model",
                details={
                    'model_name': self._model_name,
                    'weights_path': str(weights_path),
                    'error': str(e)
                }
            )

    def get_heatmap(self, image_path: Union[str, Path]) -> Tuple[Any, Any]:
        """
        Generate GradCAM heatmap for image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of original image and heatmap
        """
        try:
            if not self._model:
                raise ValueError("Model not loaded")
                
            # Check cache
            cache_key = str(image_path)
            if cache_key in self._cache:
                return self._cache[cache_key]
                
            image = preprocess_image(image_path)
            orig_img, heatmap = get_gradcam_heatmap(
                model=self._model,
                image=image,
                layer_name=self._config.get('target_layer', 'top_conv')
            )
            
            # Cache result
            self._cache[cache_key] = (orig_img, heatmap)
            
            return orig_img, heatmap
            
        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}")
            raise ModelError(
                message="Heatmap generation failed",
                error_code=4002,
                operation="get_heatmap", 
                details={'image_path': str(image_path), 'error': str(e)}
            )
        
        finally:
            # Cleanup
            cleanup_gradients()

    def invalidate_cache(self) -> None:
        """Clear cached results."""
        self._cache.clear()
        logger.debug("Cleared EfficientNet cache")

    def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            if self._model:
                del self._model
                self._model = None
            
            self.invalidate_cache()
            
            logger.info("Cleaned up EfficientNet resources")
            
        except Exception as e:
            logger.error(f"Error during EfficientNet cleanup: {e}")
            raise ModelError(
                message="EfficientNet cleanup failed",
                error_code=4003,
                operation="cleanup",
                details={'error': str(e)}
            )

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error in EfficientNet destructor: {e}")

# Initialize default wrapper
default_wrapper = EfficientNetWrapper('efficientnetv2-m')

def infer_efficientnet(*args, **kwargs):
    """Stub for EfficientNet inference."""
    import logging
    logging.getLogger(__name__).info("Stub: infer_efficientnet called")
    return None

__all__ = [
    'EfficientNetWrapper',
    'default_wrapper',
    'load_efficientnet',
    'get_gradcam_heatmap',
    'EfficientNetModel'
]