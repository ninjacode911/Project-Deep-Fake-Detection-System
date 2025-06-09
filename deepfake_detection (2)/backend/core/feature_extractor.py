"""
DeepFake Detection System - Feature Extractor
Created: 2025-06-07
Author: ninjacode911

This module implements feature extraction for deepfake detection
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

from .exceptions.backend_exceptions import FeatureError, ValidationError, ResourceError
from ..config import config_manager
from ..utils.model_utils import ModelUtils

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extractor class for deepfake detection features."""

    def __init__(self) -> None:
        """Initialize feature extractor with configuration."""
        try:
            self._lock = threading.RLock()
            self._memory_limit = config_manager.get("feature.memory_limit", 2 * 1024 * 1024 * 1024)  # 2GB
            self._batch_size = config_manager.get("feature.batch_size", 32)
            self._feature_dim = config_manager.get("feature.dimension", 512)
            
            # Initialize feature registry
            self._features: Dict[str, Dict[str, Any]] = {}
            
            logger.info("Feature extractor initialized successfully")
            
        except Exception as e:
            logger.error(f"Feature extractor initialization failed: {e}")
            raise FeatureError(
                message="Failed to initialize feature extractor",
                error_code=6000,
                operation="init",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_input(self, input_data: Union[np.ndarray, torch.Tensor]) -> None:
        """
        Validate input data.
        
        Args:
            input_data: Input data
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if input_data is None:
                raise ValidationError(
                    message="Input data cannot be None",
                    error_code=6001,
                    details={'input_data': input_data}
                )
                
            # Convert to numpy if tensor
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.cpu().numpy()
                
            # Check data type
            if not isinstance(input_data, np.ndarray):
                raise ValidationError(
                    message="Input data must be numpy array or torch tensor",
                    error_code=6002,
                    details={'type': type(input_data)}
                )
                
            # Check shape
            if len(input_data.shape) != 4:  # [batch, channels, height, width]
                raise ValidationError(
                    message="Invalid input shape",
                    error_code=6003,
                    details={'shape': input_data.shape}
                )
                
            # Check values
            if np.isnan(input_data).any():
                raise ValidationError(
                    message="Input data contains NaN values",
                    error_code=6004,
                    details={'shape': input_data.shape}
                )
                
            if np.isinf(input_data).any():
                raise ValidationError(
                    message="Input data contains infinite values",
                    error_code=6005,
                    details={'shape': input_data.shape}
                )
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ValidationError(
                message="Input validation failed",
                error_code=6006,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_model(self, model: nn.Module) -> None:
        """
        Validate model.
        
        Args:
            model: Model instance
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if model is None:
                raise ValidationError(
                    message="Model cannot be None",
                    error_code=6007,
                    details={'model': model}
                )
                
            if not isinstance(model, nn.Module):
                raise ValidationError(
                    message="Invalid model type",
                    error_code=6008,
                    details={'type': type(model)}
                )
                
            # Check if model is in evaluation mode
            if model.training:
                raise ValidationError(
                    message="Model must be in evaluation mode",
                    error_code=6009,
                    details={'training': model.training}
                )
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise ValidationError(
                message="Model validation failed",
                error_code=6010,
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
                    error_code=6011,
                    details={
                        'used': current_memory - start_memory,
                        'limit': self._memory_limit
                    }
                )
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            raise ResourceError(
                message="Memory monitoring failed",
                error_code=6012,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def extract_features(
        self,
        input_data: Union[np.ndarray, torch.Tensor],
        model: nn.Module,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Extract features from input data using model.
        
        Args:
            input_data: Input data
            model: Model instance
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Validate inputs
            self._validate_input(input_data)
            self._validate_model(model)
            
            # Convert to tensor if numpy
            if isinstance(input_data, np.ndarray):
                input_data = torch.from_numpy(input_data).float()
                
            # Move to GPU if available
            if torch.cuda.is_available():
                input_data = input_data.cuda()
                model = model.cuda()
                
            # Process in batches with memory monitoring
            with self._memory_context():
                features = []
                total_batches = (len(input_data) + self._batch_size - 1) // self._batch_size
                
                for i in range(0, len(input_data), self._batch_size):
                    # Get batch
                    batch = input_data[i:i + self._batch_size]
                    
                    # Extract features
                    with torch.no_grad():
                        batch_features = model(batch)
                        
                    # Move to CPU and convert to numpy
                    batch_features = batch_features.cpu().numpy()
                    features.append(batch_features)
                    
                    # Update progress
                    if progress_callback:
                        progress = (i // self._batch_size + 1) / total_batches
                        progress_callback(progress)
                        
                # Concatenate features
                features = np.concatenate(features, axis=0)
                
                # Calculate feature statistics
                feature_stats = self._calculate_feature_stats(features)
                
                # Create feature dictionary
                result = {
                    'features': features,
                    'statistics': feature_stats,
                    'metadata': {
                        'shape': features.shape,
                        'dtype': str(features.dtype),
                        'num_samples': len(features)
                    }
                }
                
                return result
                
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise FeatureError(
                message="Failed to extract features",
                error_code=6013,
                operation="extract_features",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _calculate_feature_stats(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Calculate feature statistics.
        
        Args:
            features: Extracted features
            
        Returns:
            Dictionary of feature statistics
        """
        try:
            # Calculate basic statistics
            mean = np.mean(features, axis=0)
            std = np.std(features, axis=0)
            min_val = np.min(features, axis=0)
            max_val = np.max(features, axis=0)
            
            # Calculate feature quality metrics
            sparsity = np.mean(np.abs(features) < 1e-6)
            variance = np.var(features, axis=0)
            entropy = -np.sum(
                np.mean(features, axis=0) * np.log2(np.mean(features, axis=0) + 1e-10)
            )
            
            return {
                'mean': mean,
                'std': std,
                'min': min_val,
                'max': max_val,
                'sparsity': float(sparsity),
                'variance': variance,
                'entropy': float(entropy)
            }
            
        except Exception as e:
            logger.error(f"Feature statistics calculation failed: {e}")
            raise FeatureError(
                message="Failed to calculate feature statistics",
                error_code=6014,
                operation="calculate_feature_stats",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            with self._lock:
                self._features.clear()
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise FeatureError(
                message="Failed to cleanup resources",
                error_code=6015,
                operation="cleanup",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def __del__(self) -> None:
        """Cleanup in destructor."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Feature extractor cleanup in destructor failed: {e}") 