"""
DeepFake Detection System - Model Utilities
Created: 2025-06-07
Author: ninjacode911

This module provides utility functions for model operations with 
optimized performance and resource management.
"""

import json
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from functools import lru_cache
import gc
from contextlib import contextmanager

from ..core.exceptions.backend_exceptions import ConfigError, ModelError
from ..types.backend_types import JsonDict, ModelConfig

# Configure logger
logger = logging.getLogger(__name__)

# Global cache for configs
_CONFIG_CACHE: Dict[str, JsonDict] = {}
_MODEL_CACHE: Dict[str, Any] = {}
_MAX_CACHE_SIZE = 100 * 1024 * 1024  # 100MB

class ModelUtils:
    """Utility class for model operations with resource management."""
    
    @staticmethod
    def cleanup_gpu_memory():
        """Aggressive GPU memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
    @staticmethod
    def get_memory_usage():
        """Monitor memory usage for both CPU and GPU"""
        usage = {
            'cpu_percent': psutil.cpu_percent(),
            'ram_percent': psutil.virtual_memory().percent,
        }
        if torch.cuda.is_available():
            usage['gpu_percent'] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        return usage


    @staticmethod
    @lru_cache(maxsize=32)
    def load_config(config_path: str = "config/model_config.json") -> JsonDict:
        """
        Load and cache configuration from JSON file with fallback defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigError: If configuration loading fails
        """
        try:
            # Check cache first
            if config_path in _CONFIG_CACHE:
                return _CONFIG_CACHE[config_path]
                
            config_file = Path(config_path)
            default_config = {
                "vision_model_m": "pretrained/efficientnetv2_m_finetuned.pth",
                "vision_model_s": "pretrained/efficientnetv2_s_finetuned.pth", 
                "audio_model": "pretrained/Deepfake-audio-detection-V2",
                "weights": {
                    "vision_m": 0.5,
                    "vision_s": 0.0,
                    "audio": 0.5
                },
                "confidence_threshold": 0.5,
                "frame_rate": 30,
                "heatmap_size": [64, 64],
                "cache_size": _MAX_CACHE_SIZE,
                "batch_size": 32,
                "num_workers": 4
            }

            if not config_file.exists():
                logger.warning(f"Config file {config_path} not found. Using defaults.")
                return default_config

            with open(config_file) as f:
                config = json.load(f)
                
            # Validate config
            if not isinstance(config, dict):
                raise ConfigError(
                    message="Invalid config format",
                    error_code=4001,
                    details={'path': config_path}
                )
                
            # Update cache
            _CONFIG_CACHE[config_path] = config
            logger.info(f"Loaded config from {config_path}")
            
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {config_path}: {e}")
            raise ConfigError(
                message="Invalid JSON configuration",
                error_code=4002,
                details={'path': config_path, 'error': str(e)}
            )
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise ConfigError(
                message="Configuration loading failed",
                error_code=4003,
                details={'path': config_path, 'error': str(e)}
            )

    @staticmethod
    def get_device() -> torch.device:
        """
        Get optimal device for model inference with proper error handling.
        
        Returns:
            torch.device: Optimal device for inference
            
        Raises:
            ModelError: If device determination fails
        """
        try:
            if torch.cuda.is_available():
                # Check GPU memory
                gpu_props = torch.cuda.get_device_properties(0)
                if gpu_props.total_memory / (1024**2) >= 2048:  # At least 2GB VRAM
                    device = torch.device("cuda")
                    logger.info(f"Using GPU: {gpu_props.name}")
                else:
                    device = torch.device("cpu")
                    logger.warning("Insufficient GPU memory, falling back to CPU")
            else:
                device = torch.device("cpu")
                logger.info("CUDA not available, using CPU")
                
            return device
            
        except Exception as e:
            logger.error(f"Error determining device: {e}")
            raise ModelError(
                message="Failed to determine compute device",
                error_code=4004,
                details={'error': str(e)}
            )

    @staticmethod
    def combine_predictions(
        visual_probs: List[float],
        audio_probs: List[float],
        weights: Dict[str, float]
    ) -> Tuple[int, List[float]]:
        """
        Combine multi-modal predictions with weighted averaging.
        
        Args:
            visual_probs: Visual model probabilities
            audio_probs: Audio model probabilities 
            weights: Modality weights
            
        Returns:
            Tuple of (prediction class, combined probabilities)
            
        Raises:
            ModelError: If prediction combination fails
        """
        try:
            # Input validation
            if not all(isinstance(p, (list, np.ndarray)) for p in [visual_probs, audio_probs]):
                raise ValueError("Predictions must be lists or numpy arrays")
            
            # Convert to numpy arrays
            visual_probs = np.array(visual_probs, dtype=np.float32)
            audio_probs = np.array(audio_probs, dtype=np.float32)
            
            # Extract and validate weights
            w_visual = weights.get("vision_m", 0.5)
            w_audio = weights.get("audio", 0.5)
            
            total_weight = w_visual + w_audio
            if total_weight == 0:
                raise ValueError("Total weight cannot be zero")
                
            # Normalize weights
            w_visual /= total_weight
            w_audio /= total_weight
            
            # Combine with weighted average
            combined_probs = w_visual * visual_probs + w_audio * audio_probs
            
            # Normalize probabilities
            combined_probs = combined_probs / np.sum(combined_probs)
            
            # Get prediction
            prediction = int(np.argmax(combined_probs))
            
            logger.debug(
                f"Combined predictions: visual={visual_probs.tolist()}, "
                f"audio={audio_probs.tolist()}, weights={weights}, "
                f"result={combined_probs.tolist()}"
            )
            
            return prediction, combined_probs.tolist()
            
        except Exception as e:
            logger.error(f"Failed to combine predictions: {e}")
            raise ModelError(
                message="Prediction combination failed",
                error_code=4005,
                details={
                    'visual_probs': str(visual_probs),
                    'audio_probs': str(audio_probs),
                    'weights': str(weights),
                    'error': str(e)
                }
            )

    @staticmethod
    @contextmanager
    def torch_inference_mode():
        """Context manager for efficient inference."""
        try:
            with torch.inference_mode():
                yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    @staticmethod
    def clear_cache() -> None:
        """Clear all caches and optimize memory."""
        try:
            # Clear caches
            _CONFIG_CACHE.clear()
            _MODEL_CACHE.clear()
            
            # Clear function cache
            ModelUtils.load_config.cache_clear()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Force garbage collection
            gc.collect()
            
            logger.info("Cleared all caches and optimized memory")
            
        except Exception as e:
            logger.error(f"Failed to clear caches: {e}")
            raise ModelError(
                message="Cache clearing failed",
                error_code=4006,
                details={'error': str(e)}
            )

    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """Get current memory usage statistics."""
        try:
            info = {
                'cpu_percent': None,
                'ram_used': None,
                'gpu_used': None
            }
            
            # Get CPU/RAM usage
            try:
                import psutil
                process = psutil.Process()
                info['cpu_percent'] = process.cpu_percent()
                info['ram_used'] = process.memory_info().rss
            except ImportError:
                pass
                
            # Get GPU usage
            if torch.cuda.is_available():
                info['gpu_used'] = torch.cuda.memory_allocated()
                
            return info
            
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {}
    
    @contextmanager
    def model_inference_context():
        try:
            torch.cuda.empty_cache()
            yield
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def optimize_batch_size(available_memory: int) -> int:
        """Dynamically adjust batch size based on available GPU memory"""
        return min(16, available_memory // (224 * 224 * 3 * 4 * 10))  # 10 frames

    def load_model_weights(model: torch.nn.Module, weights_path: Union[str, Path]) -> None:
        """
        Load model weights from a file.
        
        Args:
            model: The PyTorch model to load weights into
            weights_path: Path to the weights file
        """
        try:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
            state_dict = torch.load(weights_path, map_location=get_device())
            model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            logger.error(f"Error loading model weights: {str(e)}")
            raise

def save_model_weights(model: torch.nn.Module, weights_path: Union[str, Path]) -> None:
    """
    Stub for saving model weights.
    
    Args:
        model: The PyTorch model to save
        weights_path: Path to save the weights file
    """
    logger.info(f"Stub: save_model_weights called for {weights_path}")

def optimize_model(model: torch.nn.Module) -> torch.nn.Module:
    """Stub for model optimization."""
    logger.info("Stub: optimize_model called")
    return model

# Initialize utilities
model_utils = ModelUtils()