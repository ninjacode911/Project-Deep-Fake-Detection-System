"""
DeepFake Detection System - EfficientNet Model Implementation
Created: 2025-06-07
Author: ninjacode911

This module implements the EfficientNet model with optimizations
for deepfake detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import gc

from ..base import BaseModel, ModelConfig
from ..logger import get_model_logger
from ...core.exceptions.backend_exceptions import ModelError
from ...types.backend_types import ModelOutput, ImageTensor
from .utils import preprocess_image, cleanup_gradients

# Configure logger
logger = logging.getLogger(__name__)

class EfficientNetModel(BaseModel):
    """EfficientNet model implementation for deepfake detection."""
    
    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize EfficientNet model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__(config)
        
        try:
            self._model_name = config.get('model_name', 'efficientnetv2_m')
            self._num_classes = config.get('num_classes', 2)
            self._dropout_rate = config.get('dropout_rate', 0.2)
            self._pretrained = config.get('pretrained', True)
            self._cache_dir = Path(config.get('cache_dir', './cache'))
            
            # Create cache directory
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            
            self._build_model()
            self._initialize_training()
            logger.info(f"Initialized {self._model_name} model")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise ModelError(
                message="Model initialization failed",
                error_code=4000,
                operation="init",
                details={'error': str(e)}
            )

    def _build_model(self) -> None:
        """Build EfficientNet model architecture."""
        try:
            # Load pretrained model
            self._model = timm.create_model(
                self._model_name,
                pretrained=self._pretrained,
                num_classes=self._num_classes,
                drop_rate=self._dropout_rate
            )
            
            # Move model to device
            self._model = self._model.to(self._device)
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self._model, 'set_grad_checkpointing'):
                self._model.set_grad_checkpointing(enable=True)
                
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise ModelError(
                message="Model building failed",
                error_code=4001,
                operation="build_model",
                details={'error': str(e)}
            )

    def _initialize_training(self) -> None:
        """Initialize training components."""
        try:
            self._criterion = nn.CrossEntropyLoss()
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=self._config.get('learning_rate', 1e-4),
                weight_decay=self._config.get('weight_decay', 1e-2)
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize training: {e}")
            raise ModelError(
                message="Training initialization failed",
                error_code=4002,
                operation="init_training",
                details={'error': str(e)}
            )

    def predict(self, inputs: Union[str, Path, ImageTensor], **kwargs) -> ModelOutput:
        """
        Run prediction on inputs.
        
        Args:
            inputs: Input image path or tensor
            **kwargs: Additional arguments
            
        Returns:
            Model predictions
        """
        try:
            self._model.eval()
            
            # Preprocess input
            if isinstance(inputs, (str, Path)):
                inputs = preprocess_image(inputs)
            
            inputs = inputs.to(self._device)
            
            with torch.no_grad():
                # Record inference time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                outputs = self._model(inputs)
                end_time.record()
                
                torch.cuda.synchronize()
                inference_time = start_time.elapsed_time(end_time)
                
                # Get predictions
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                # Update metrics
                self._metrics['inference_time'] = inference_time
                
                return {
                    'predictions': preds.cpu().numpy(),
                    'probabilities': probs.cpu().numpy(),
                    'inference_time': inference_time
                }
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise ModelError(
                message="Model prediction failed",
                error_code=4003,
                operation="predict",
                details={'error': str(e)}
            )
            
        finally:
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def train_step(self, batch: Tuple[ImageTensor, torch.Tensor]) -> Dict[str, float]:
        """
        Run single training step.
        
        Args:
            batch: Tuple of (inputs, labels)
            
        Returns:
            Training metrics
        """
        try:
            self._model.train()
            
            inputs, labels = batch
            inputs = inputs.to(self._device)
            labels = labels.to(self._device)
            
            # Forward pass
            self._optimizer.zero_grad()
            outputs = self._model(inputs)
            loss = self._criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self._optimizer.step()
            
            return {
                'loss': loss.item()
            }
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            raise ModelError(
                message="Training step failed", 
                error_code=4004,
                operation="train_step",
                details={'error': str(e)}
            )
            
        finally:
            cleanup_gradients()

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        try:
            checkpoint = {
                'model_state_dict': self._model.state_dict(),
                'optimizer_state_dict': self._optimizer.state_dict(),
                'config': self._config,
                'metrics': self._metrics
            }
            
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise ModelError(
                message="Checkpoint saving failed",
                error_code=4005,
                operation="save_checkpoint", 
                details={
                    'path': str(path),
                    'error': str(e)
                }
            )

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        try:
            checkpoint = torch.load(path, map_location=self._device)
            
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self._config.update(checkpoint['config'])
            self._metrics.update(checkpoint['metrics'])
            
            logger.info(f"Loaded checkpoint from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise ModelError(
                message="Checkpoint loading failed",
                error_code=4006,
                operation="load_checkpoint",
                details={
                    'path': str(path),
                    'error': str(e)
                }
            )

    def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            super().cleanup()
            
            # Clear model
            if hasattr(self, '_model'):
                del self._model
            
            # Clear optimizers
            if hasattr(self, '_optimizer'):
                del self._optimizer
                
            # Clear criterion
            if hasattr(self, '_criterion'):
                del self._criterion
                
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Cleaned up model resources")
            
        except Exception as e:
            logger.error(f"Model cleanup failed: {e}")
            raise ModelError(
                message="Model cleanup failed",
                error_code=4007,
                operation="cleanup",
                details={'error': str(e)}
            )