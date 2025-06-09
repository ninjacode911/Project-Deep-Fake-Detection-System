"""
DeepFake Detection System - Model Manager
Created: 2025-06-07
Author: ninjacode911

This module implements model management for deepfake detection
with proper resource management and error handling.
"""

import logging
import time
import traceback
import threading
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np
import torch
import torch.nn as nn
from contextlib import contextmanager

from .exceptions.backend_exceptions import ModelError, ValidationError, ResourceError
from ..config import config_manager
from ..utils.model_utils import ModelUtils

logger = logging.getLogger(__name__)

class ModelManager:
    """Manager class for deepfake detection models."""

    def __init__(self) -> None:
        """Initialize model manager with configuration."""
        try:
            self._lock = threading.RLock()
            self._memory_limit = config_manager.get("model.memory_limit", 4 * 1024 * 1024 * 1024)  # 4GB
            self._models_dir = Path(config_manager.get("model.directory", "models"))
            self._models_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize model registry
            self._models: Dict[str, Dict[str, Any]] = {}
            self._active_models: Dict[str, nn.Module] = {}
            
            logger.info("Model manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Model manager initialization failed: {e}")
            raise ModelError(
                message="Failed to initialize model manager",
                error_code=5000,
                operation="init",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_model_path(self, model_path: str) -> None:
        """
        Validate model file path.
        
        Args:
            model_path: Path to model file
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check file existence
            if not os.path.exists(model_path):
                raise ValidationError(
                    message="Model file not found",
                    error_code=5001,
                    details={'path': model_path}
                )
                
            # Check file extension
            if not model_path.endswith('.pth'):
                raise ValidationError(
                    message="Invalid model file format",
                    error_code=5002,
                    details={'path': model_path, 'format': os.path.splitext(model_path)[1]}
                )
                
            # Check file size
            file_size = os.path.getsize(model_path)
            max_size = config_manager.get("model.max_file_size", 1 * 1024 * 1024 * 1024)  # 1GB
            if file_size > max_size:
                raise ValidationError(
                    message="Model file too large",
                    error_code=5003,
                    details={
                        'path': model_path,
                        'size': file_size,
                        'max_size': max_size
                    }
                )
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Model path validation failed: {e}")
            raise ValidationError(
                message="Model path validation failed",
                error_code=5004,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_model_config(self, config: Dict[str, Any]) -> None:
        """
        Validate model configuration.
        
        Args:
            config: Model configuration
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            required_fields = {'name', 'version', 'type', 'input_shape', 'output_shape'}
            missing_fields = required_fields - set(config.keys())
            if missing_fields:
                raise ValidationError(
                    message="Missing required model configuration fields",
                    error_code=5005,
                    details={'missing_fields': list(missing_fields)}
                )
                
            # Validate input shape
            if not isinstance(config['input_shape'], (list, tuple)):
                raise ValidationError(
                    message="Invalid input shape format",
                    error_code=5006,
                    details={'input_shape': config['input_shape']}
                )
                
            # Validate output shape
            if not isinstance(config['output_shape'], (list, tuple)):
                raise ValidationError(
                    message="Invalid output shape format",
                    error_code=5007,
                    details={'output_shape': config['output_shape']}
                )
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Model configuration validation failed: {e}")
            raise ValidationError(
                message="Model configuration validation failed",
                error_code=5008,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

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
                    error_code=5009,
                    details={
                        'used': current_memory - start_memory,
                        'limit': self._memory_limit
                    }
                )
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            raise ResourceError(
                message="Memory monitoring failed",
                error_code=5010,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def load_model(
        self,
        model_path: str,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """
        Load model from file.
        
        Args:
            model_path: Path to model file
            config: Model configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            Model ID
        """
        try:
            # Validate inputs
            self._validate_model_path(model_path)
            self._validate_model_config(config)
            
            # Generate model ID
            model_id = f"{config['name']}_{config['version']}"
            
            # Check if model is already loaded
            with self._lock:
                if model_id in self._active_models:
                    return model_id
                    
            # Load model with memory monitoring
            with self._memory_context():
                # Load state dict
                state_dict = torch.load(model_path, map_location='cpu')
                
                if progress_callback:
                    progress_callback(0.3)  # State dict loaded
                    
                # Create model instance
                model = ModelUtils.create_model(config)
                model.load_state_dict(state_dict)
                
                if progress_callback:
                    progress_callback(0.6)  # Model created
                    
                # Move to GPU if available
                if torch.cuda.is_available():
                    model = model.cuda()
                    
                # Set to evaluation mode
                model.eval()
                
                if progress_callback:
                    progress_callback(0.9)  # Model moved to GPU
                    
                # Register model
                with self._lock:
                    self._models[model_id] = {
                        'config': config,
                        'path': model_path,
                        'loaded_at': time.time()
                    }
                    self._active_models[model_id] = model
                    
                if progress_callback:
                    progress_callback(1.0)  # Model registered
                    
                return model_id
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise ModelError(
                message="Failed to load model",
                error_code=5011,
                operation="load_model",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def unload_model(self, model_id: str) -> None:
        """
        Unload model from memory.
        
        Args:
            model_id: Model ID
        """
        try:
            with self._lock:
                if model_id in self._active_models:
                    # Remove model from active models
                    model = self._active_models.pop(model_id)
                    
                    # Clear CUDA cache if model was on GPU
                    if next(model.parameters()).is_cuda:
                        torch.cuda.empty_cache()
                        
                    # Remove from registry
                    self._models.pop(model_id, None)
                    
        except Exception as e:
            logger.error(f"Model unloading failed: {e}")
            raise ModelError(
                message="Failed to unload model",
                error_code=5012,
                operation="unload_model",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def get_model(self, model_id: str) -> Optional[nn.Module]:
        """
        Get model instance.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model instance or None if not found
        """
        try:
            with self._lock:
                return self._active_models.get(model_id)
                
        except Exception as e:
            logger.error(f"Model retrieval failed: {e}")
            raise ModelError(
                message="Failed to get model",
                error_code=5013,
                operation="get_model",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model information.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information or None if not found
        """
        try:
            with self._lock:
                return self._models.get(model_id)
                
        except Exception as e:
            logger.error(f"Model info retrieval failed: {e}")
            raise ModelError(
                message="Failed to get model info",
                error_code=5014,
                operation="get_model_info",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            with self._lock:
                # Unload all models
                for model_id in list(self._active_models.keys()):
                    self.unload_model(model_id)
                    
                # Clear registries
                self._models.clear()
                self._active_models.clear()
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise ModelError(
                message="Failed to cleanup resources",
                error_code=5015,
                operation="cleanup",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def __del__(self) -> None:
        """Cleanup in destructor."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Model manager cleanup in destructor failed: {e}") 