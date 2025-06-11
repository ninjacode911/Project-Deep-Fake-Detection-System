"""
DeepFake Detection System - Model Factory
Created: 2025-06-07
Author: ninjacode911

This module implements a factory pattern for creating and managing deepfake detection models
with proper resource management and error handling.
"""

import logging
import time
import traceback
import threading
import os
from pathlib import Path
from typing import Dict, Any, Optional, Type, Callable
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from ..base.model_base import BaseModel
from ...core.exceptions.backend_exceptions import ModelError
from ...config import config_manager

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class for creating and managing deepfake detection models."""
    
    _instance = None
    _lock = threading.RLock()
    _model_registry: Dict[str, Type[BaseModel]] = {}
    _model_instances: Dict[str, BaseModel] = {}
    _model_weights: Dict[str, str] = {}
    
    def __new__(cls) -> 'ModelFactory':
        """Ensure singleton instance."""
        with cls._lock:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize model factory with configuration."""
        try:
            # Initialize model registry
            self._initialize_registry()
            
            # Load model weights configuration
            self._load_weights_config()
            
            # Initialize resource tracking
            self._active_loading = 0
            self._max_concurrent = config_manager.get("models.max_concurrent_loading", 2)
            
            logger.info("Model factory initialized successfully")
            
        except Exception as e:
            logger.error(f"Model factory initialization failed: {e}")
            raise ModelError(
                message="Failed to initialize model factory",
                error_code=5000,
                operation="init",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _initialize_registry(self) -> None:
        """Initialize model registry with available models."""
        try:
            # Register available models
            from ..vision.efficientnet import EfficientNetModel
            from ..audio.wav2vec2 import Wav2Vec2Model
            
            self._model_registry = {
                'efficientnet': EfficientNetModel,
                'wav2vec2': Wav2Vec2Model
            }
            
        except Exception as e:
            logger.error(f"Model registry initialization failed: {e}")
            raise ModelError(
                message="Failed to initialize model registry",
                error_code=5001,
                operation="initialize_registry",
                details={'error': str(e)}
            )

    def _load_weights_config(self) -> None:
        """Load model weights configuration."""
        try:
            weights_config = config_manager.get("models.weights", {})
            
            # Validate weights configuration
            for model_name, weights_path in weights_config.items():
                if not isinstance(weights_path, str):
                    logger.warning(f"Invalid weights path for {model_name}: path should be a string, got {type(weights_path)}")
                    self._model_weights[model_name] = None # Or a placeholder like "INVALID_PATH_TYPE"
                    continue  # Skip to the next model
                    
                # Validate weights file existence
                if not os.path.exists(weights_path):
                    logger.warning(f"Weights file/directory not found for model {model_name} at path: {weights_path}")
                    self._model_weights[model_name] = None # Store None if path does not exist
                else:
                    self._model_weights[model_name] = weights_path
                
        except Exception as e:
            logger.error(f"Weights configuration loading failed: {e}")
            raise ModelError(
                message="Failed to load weights configuration",
                error_code=5002,
                operation="load_weights_config",
                details={'error': str(e)}
            )

    @staticmethod
    def register_model(name: str, model_class: Type[BaseModel]) -> None:
        """Register a new model class."""
        try:
            if not issubclass(model_class, BaseModel):
                raise ValueError(f"Model class must inherit from BaseModel: {model_class}")
                
            ModelFactory._model_registry[name] = model_class
            logger.info(f"Registered model: {name}")
            
        except Exception as e:
            logger.error(f"Model registration failed: {e}")
            raise ModelError(
                message="Failed to register model",
                error_code=5003,
                operation="register_model",
                details={'model_name': name, 'error': str(e)}
            )

    def create_model(
        self, 
        model_name: str,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> BaseModel:
        """
        Create a model instance with proper initialization.
        
        Args:
            model_name: Name of the model to create
            config: Model configuration
            device: Optional device to load model on
            progress_callback: Optional callback for loading progress
            
        Returns:
            Initialized model instance
        """
        try:
            # Validate model name
            if model_name not in self._model_registry:
                raise ValueError(f"Unknown model: {model_name}")

            # Retrieve weights path early
            weights_path = self._model_weights.get(model_name)

            # Check if weights are missing before proceeding
            if weights_path is None:
                logger.error(f"Cannot create model {model_name}: weights are missing or path was invalid.")
                # Optionally, raise a custom exception here instead of returning None
                # e.g., raise ModelWeightsMissingError(f"Weights missing for model {model_name}")
                return None
                
            # Check if model instance exists
            if model_name in self._model_instances:
                return self._model_instances[model_name]
                
            # Acquire loading slot
            with self._loading_slot():
                # Get model class
                model_class = self._model_registry[model_name]
                
                # weights_path is already retrieved and validated.
                # The original check inside this block for weights_path can be removed or adapted.
                # For clarity, we rely on the already fetched and validated weights_path.
                # If weights_path was None, we would have returned None already.
            
                # Create model instance
                start_time = time.time()
                model = model_class(
                    weights_path=weights_path,
                    config=config,
                    device=device
                )
            
                # Load model with progress tracking
                model.load(progress_callback)
                
                # Store instance
                self._model_instances[model_name] = model
                
                logger.info(
                    f"Model {model_name} created and loaded in "
                    f"{time.time() - start_time:.2f}s"
                )
            
            return model
            
        except Exception as e:
            logger.error(f"Model creation failed: {e}")
            raise ModelError(
                message="Failed to create model",
                error_code=5004,
                operation="create_model",
                details={'model_name': model_name, 'error': str(e)}
            )

    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """Get existing model instance."""
        return self._model_instances.get(model_name)

    def release_model(self, model_name: str) -> None:
        """Release model instance and free resources."""
        try:
            if model_name in self._model_instances:
                model = self._model_instances.pop(model_name)
                model.cleanup()
                logger.info(f"Released model: {model_name}")
            
        except Exception as e:
            logger.error(f"Model release failed: {e}")
            raise ModelError(
                message="Failed to release model",
                error_code=5005,
                operation="release_model",
                details={'model_name': model_name, 'error': str(e)}
            )

    @contextmanager
    def _loading_slot(self):
        """Manage concurrent model loading."""
        try:
            with self._lock:
                while self._active_loading >= self._max_concurrent:
                    time.sleep(0.1)
                self._active_loading += 1
                
            try:
                yield
            finally:
                with self._lock:
                    self._active_loading -= 1
            
        except Exception as e:
            logger.error(f"Model loading slot management failed: {e}")
            raise ModelError(
                message="Failed to manage model loading slot",
                error_code=5006,
                operation="loading_slot",
                details={'error': str(e)}
            )

    def cleanup(self) -> None:
        """Clean up all model instances."""
        try:
            # Release all models
            for model_name in list(self._model_instances.keys()):
                self.release_model(model_name)
            
            # Clear registry
            self._model_registry.clear()
            self._model_weights.clear()
            
            logger.info("Model factory resources cleaned up")
            
        except Exception as e:
            logger.error(f"Model factory cleanup failed: {e}")
            raise ModelError(
                message="Failed to cleanup model factory resources",
                error_code=5007,
                operation="cleanup",
                details={'error': str(e)}
            )

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Model factory cleanup in destructor failed: {e}")