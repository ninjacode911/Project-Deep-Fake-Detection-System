"""
DeepFake Detection System - Wav2Vec2 Model Utilities
Created: 2025-06-07
Author: ninjacode911

This module provides utility functions for Wav2Vec2 model operations with
optimized performance and error handling.
"""

import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from transformers import AutoModelForAudioClassification
import torchaudio
import gc

from ...core.exceptions.backend_exceptions import ModelError

# Configure logger
logger = logging.getLogger(__name__)

# Cache for processed audio files
_AUDIO_CACHE: Dict[str, Tuple[np.ndarray, int]] = {}
_MAX_CACHE_SIZE = 100

def load_wav2vec2(
    model_path: Union[str, Path],
    device: str = "cuda",
    **kwargs
) -> AutoModelForAudioClassification:
    """
    Load Wav2Vec2 model with optimized settings.
    
    Args:
        model_path: Path to model weights/config
        device: Device to load model on
        **kwargs: Additional model arguments
        
    Returns:
        Loaded model
        
    Raises:
        ModelError: If model loading fails
    """
    try:
        # Validate model path
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
        # Load model with optimized settings
        model = AutoModelForAudioClassification.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            **kwargs
        )
        
        # Move to device and optimize
        model = model.to(device)
        if device == "cuda":
            model = torch.compile(model)  # JIT compilation for GPU
        
        model.eval()
        logger.info(f"Loaded Wav2Vec2 model on {device}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load Wav2Vec2 model: {e}")
        raise ModelError(
            message="Model loading failed",
            error_code=6200,
            operation="load_model",
            details={
                'model_path': str(model_path),
                'device': device,
                'error': str(e)
            }
        )

def load_audio(
    file_path: Union[str, Path],
    target_sr: int = 16000,
    use_cache: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load and preprocess audio file with caching.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        use_cache: Whether to use caching
        
    Returns:
        Tuple of (audio array, sample rate)
    """
    try:
        file_path = str(Path(file_path).resolve())
        
        # Check cache
        if use_cache and file_path in _AUDIO_CACHE:
            logger.debug(f"Using cached audio: {file_path}")
            return _AUDIO_CACHE[file_path]
        
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sr
            )
            waveform = resampler(waveform)
            
        # Convert to numpy array
        audio_array = waveform.numpy().squeeze()
        
        # Update cache
        if use_cache:
            _update_audio_cache(file_path, (audio_array, target_sr))
            
        return audio_array, target_sr
        
    except Exception as e:
        logger.error(f"Failed to load audio file {file_path}: {e}")
        raise ModelError(
            message="Audio loading failed",
            error_code=6201,
            operation="load_audio",
            details={
                'file_path': file_path,
                'error': str(e)
            }
        )

def infer_wav2vec2(
    model: AutoModelForAudioClassification,
    audio_input: Union[str, Path, np.ndarray],
    sample_rate: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run inference with Wav2Vec2 model.
    
    Args:
        model: Loaded model
        audio_input: Audio file path or numpy array
        sample_rate: Sample rate for raw array input
        **kwargs: Additional inference arguments
        
    Returns:
        Dictionary with predictions
    """
    try:
        # Load audio if path provided
        if isinstance(audio_input, (str, Path)):
            audio_input, sample_rate = load_audio(audio_input)
        elif sample_rate is None:
            raise ValueError("Sample rate must be provided for array input")
            
        # Validate input
        if not isinstance(audio_input, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(audio_input)}")
            
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_input).float()
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        # Move to device
        device = next(model.parameters()).device
        audio_tensor = audio_tensor.to(device)
        
        # Run inference
        with torch.inference_mode():
            outputs = model(audio_tensor, **kwargs)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
        result = {
            'predictions': preds.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'labels': model.config.id2label
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise ModelError(
            message="Model inference failed",
            error_code=6202,
            operation="inference",
            details={'error': str(e)}
        )
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def _update_audio_cache(key: str, value: Tuple[np.ndarray, int]) -> None:
    """Update audio cache with size limiting."""
    _AUDIO_CACHE[key] = value
    
    # Remove oldest entries if cache too large
    if len(_AUDIO_CACHE) > _MAX_CACHE_SIZE:
        remove_keys = list(_AUDIO_CACHE.keys())[:-_MAX_CACHE_SIZE]
        for k in remove_keys:
            del _AUDIO_CACHE[k]

def clear_audio_cache() -> None:
    """Clear audio cache."""
    _AUDIO_CACHE.clear()
    gc.collect()