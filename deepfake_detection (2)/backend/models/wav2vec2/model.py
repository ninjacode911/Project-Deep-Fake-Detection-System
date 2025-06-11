"""
Wav2Vec2Model class for audio deepfake detection in the DeepFake Detection System.
Loads Wav2Vec2 model and performs inference on audio waveforms.
"""

import logging
import numpy as np
import torch
import gc
from pathlib import Path
from typing import Dict, Optional, Union, Any
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from ..base import BaseModel
from ...core.exceptions.backend_exceptions import ModelError

logger = logging.getLogger(__name__)

class Wav2Vec2Model(BaseModel):
    """
    Loads and manages the Wav2Vec2 model for deepfake audio detection.
    
    Attributes:
        model (AutoModelForAudioClassification): Loaded Wav2Vec2 model.
        feature_extractor (AutoFeatureExtractor): Feature extractor for preprocessing.
        device (str): Device to run inference on ('cuda' or 'cpu').
        cache (Dict): Inference result cache.
    """
    
    def __init__(self, weights_path: Union[str, Path], device: str = "cuda") -> None:
        """
        Initialize the Wav2Vec2Model.
        
        Args:
            weights_path: Path to the Wav2Vec2 weights directory
            device: Device to run inference on ('cuda' or 'cpu')
            
        Raises:
            ModelError: If initialization fails
        """
        super().__init__({})
        self.weights_loaded = False # Initialize new attribute
        
        try:
            # Set up device
            self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
            self._cache: Dict[str, np.ndarray] = {}
            self._max_cache_size = 1000
            
            # Validate weights path
            if weights_path is None or str(weights_path).strip() == "":
                logger.warning("Wav2Vec2Model: weights_path is not provided or invalid. Model will be non-operational.")
                self.weights_path = None # Ensure weights_path is None if not valid
                # Do not proceed to _load_model_components
            else:
                self.weights_path = Path(weights_path)
                # The check for self.weights_path.exists() will be done in _load_model_components
                self._load_model_components()
                # Success message will be logged in _load_model_components if all goes well
            
        except Exception as e:
            # If _load_model_components raises an error, it will be caught here or by its own try-except.
            # Ensure self.weights_loaded remains False if loading fails.
            self.weights_loaded = False
            logger.error(f"Failed to initialize Wav2Vec2Model: {e}")
            raise ModelError(
                message="Wav2Vec2Model initialization failed",
                error_code=6101,
                operation="init",
                details={'error': str(e)}
            )

    def _load_model_components(self) -> None:
        """Load model and feature extractor."""
        # Ensure weights_loaded is False until successful loading
        self.weights_loaded = False

        if not self.weights_path or not self.weights_path.exists():
            logger.error(f"Wav2Vec2Model: Weights path ({self.weights_path}) does not exist or not provided. Cannot load model components.")
            # self.weights_loaded is already False
            return

        try:
            logger.debug(f"Wav2Vec2Model: Attempting to load model components from {self.weights_path}")
            # Load feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.weights_path,
                local_files_only=True
            )
            
            # Load model
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.weights_path,
                local_files_only=True
            )
            
            # Move model to device and set eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Enable gradient checkpointing if available
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

            self.weights_loaded = True # Set to True on successful load
            logger.info(f"Wav2Vec2Model components loaded successfully from {self.weights_path} on {self.device}")
                
        except Exception as e:
            self.weights_loaded = False # Ensure it's false if an error occurs
            # It's important to log the specific error for debugging
            logger.error(f"Wav2Vec2Model: Failed to load model components from {self.weights_path}: {e}", exc_info=True)
            # Re-raise as ModelError to be caught by __init__ or calling code if needed
            raise ModelError(
                message=f"Failed to load Wav2Vec2Model components from {self.weights_path}",
                error_code=6102, # Keep original error code or define a new one
                operation="load_components",
                details={'error': str(e), 'weights_path': str(self.weights_path)}
            )

    def inference(
        self, 
        audio_waveform: np.ndarray,
        sample_rate: int = 16000,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Run inference on an audio waveform.
        
        Args:
            audio_waveform: Raw audio waveform at 16kHz
            sample_rate: Sample rate of the audio
            use_cache: Whether to use result caching
            
        Returns:
            Probabilities for 5 classes
            
        Raises:
            ModelError: If inference fails
        """
        if not self.weights_loaded:
            logger.error("Wav2Vec2Model: Cannot perform inference, weights are not loaded or model is non-operational.")
            raise ModelError(
                message="Wav2Vec2Model is not operational due to missing or failed loading of weights.",
                error_code=6106, # New error code for this specific state
                operation="inference",
                details={'model_state': 'weights_not_loaded'}
            )

        try:
            # Input validation
            if not isinstance(audio_waveform, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(audio_waveform)}")
            if audio_waveform.ndim != 1:
                raise ValueError(f"Expected 1D array, got shape {audio_waveform.shape}")
                
            # Check cache
            cache_key = self._get_cache_key(audio_waveform)
            if use_cache and cache_key in self._cache:
                logger.debug("Using cached result")
                return self._cache[cache_key]
            
            # Preprocess audio
            inputs = self._preprocess_audio(audio_waveform, sample_rate)
            
            # Run inference
            with torch.inference_mode():
                torch.cuda.empty_cache()  # Clear GPU memory
                logits = self.model(inputs.input_values).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Map probabilities
            mapped_probs = self._map_probabilities(probs[1])
            
            # Update cache
            if use_cache:
                self._update_cache(cache_key, mapped_probs)
                
            return mapped_probs
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise ModelError(
                message="Audio inference failed",
                error_code=6103,
                operation="inference",
                details={'error': str(e)}
            )
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def _preprocess_audio(
        self,
        audio_waveform: np.ndarray,
        sample_rate: int
    ) -> Any:
        """Preprocess audio for model input."""
        try:
            return self.feature_extractor(
                audio_waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=sample_rate * 10,  # 10 second limit
            ).to(self.device)
            
        except Exception as e:
            raise ModelError(
                message="Audio preprocessing failed",
                error_code=6104,
                operation="preprocess",
                details={'error': str(e)}
            )

    def _map_probabilities(self, fake_prob: float) -> np.ndarray:
        """Map binary probability to 5 class probabilities."""
        if fake_prob >= 0.9:
            return np.array([0.8, 0.15, 0.03, 0.01, 0.01])  # Fake
        elif fake_prob >= 0.7:
            return np.array([0.15, 0.6, 0.15, 0.05, 0.05])  # Likely Fake
        elif fake_prob >= 0.3:
            return np.array([0.05, 0.15, 0.6, 0.15, 0.05])  # Neutral
        elif fake_prob >= 0.1:
            return np.array([0.05, 0.05, 0.15, 0.6, 0.15])  # Likely Real
        else:
            return np.array([0.01, 0.01, 0.03, 0.15, 0.8])  # Real

    def _get_cache_key(self, audio_waveform: np.ndarray) -> str:
        """Generate cache key from audio data."""
        return str(hash(audio_waveform.tobytes()))

    def _update_cache(self, key: str, value: np.ndarray) -> None:
        """Update cache with new result."""
        self._cache[key] = value
        
        # Implement cache size limiting
        if len(self._cache) > self._max_cache_size:
            # Remove oldest entries
            remove_keys = list(self._cache.keys())[:-self._max_cache_size]
            for k in remove_keys:
                del self._cache[k]

    def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            # Clear cache
            self._cache.clear()
            
            # Delete model
            if hasattr(self, 'model'):
                del self.model
                
            # Delete feature extractor
            if hasattr(self, 'feature_extractor'):
                del self.feature_extractor
                
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Wav2Vec2Model resources cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise ModelError(
                message="Model cleanup failed",
                error_code=6105,
                operation="cleanup",
                details={'error': str(e)}
            )

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Cleanup in destructor failed: {e}")