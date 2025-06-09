"""
DeepFake Detection System - Audio Handler
Created: 2025-06-07
Author: ninjacode911

This module implements audio processing and feature extraction for deepfake detection
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
import soundfile as sf
import librosa
from contextlib import contextmanager

from .exceptions.backend_exceptions import AudioError, ValidationError, ResourceError
from ..config import config_manager
from ..utils.model_utils import ModelUtils

logger = logging.getLogger(__name__)

class AudioHandler:
    """Handler class for audio processing and feature extraction."""

    def __init__(self) -> None:
        """Initialize audio handler with configuration."""
        try:
            self._lock = threading.RLock()
            self._memory_limit = config_manager.get("audio.memory_limit", 2 * 1024 * 1024 * 1024)  # 2GB
            self._supported_formats = {'.wav', '.mp3', '.flac', '.ogg'}
            self._sampling_rate = config_manager.get("audio.sampling_rate", 16000)
            self._max_duration = config_manager.get("audio.max_duration", 10)  # seconds
            
            logger.info("Audio handler initialized successfully")
            
        except Exception as e:
            logger.error(f"Audio handler initialization failed: {e}")
            raise AudioError(
                message="Failed to initialize audio handler",
                error_code=3000,
                operation="init",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_audio(self, audio_path: str) -> None:
        """
        Validate audio file.

        Args:
            audio_path: Path to audio file

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check file existence
            if not os.path.exists(audio_path):
                raise ValidationError(
                    message="Audio file not found",
                    error_code=3001,
                    details={'path': audio_path}
                )
                
            # Check file extension
            ext = os.path.splitext(audio_path)[1].lower()
            if ext not in self._supported_formats:
                raise ValidationError(
                    message="Unsupported audio format",
                    error_code=3002,
                    details={'path': audio_path, 'format': ext}
                )
                
            # Check file size
            file_size = os.path.getsize(audio_path)
            max_size = config_manager.get("audio.max_file_size", 100 * 1024 * 1024)  # 100MB
            if file_size > max_size:
                raise ValidationError(
                    message="Audio file too large",
                    error_code=3003,
                    details={
                        'path': audio_path,
                        'size': file_size,
                        'max_size': max_size
                    }
                )
                
            # Check if file is corrupted
            try:
                with sf.SoundFile(audio_path) as f:
                    duration = len(f) / f.samplerate
                    if duration <= 0:
                        raise ValidationError(
                            message="Invalid audio duration",
                            error_code=3004,
                            details={'path': audio_path, 'duration': duration}
                        )
            except Exception as e:
                raise ValidationError(
                    message="Corrupted audio file",
                    error_code=3005,
                    details={'path': audio_path, 'error': str(e)}
                )
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Audio validation failed: {e}")
            raise ValidationError(
                message="Audio validation failed",
                error_code=3006,
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
                    error_code=3007,
                    details={
                        'used': current_memory - start_memory,
                        'limit': self._memory_limit
                    }
                )
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            raise ResourceError(
                message="Memory monitoring failed",
                error_code=3008,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def process_audio(
        self,
        audio_path: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Process audio file and extract features.

        Args:
            audio_path: Path to audio file
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary of extracted features
        """
        try:
            # Validate audio
            self._validate_audio(audio_path)
            
            # Load audio with memory monitoring
            with self._memory_context():
                # Load audio file
                audio_data, sr = librosa.load(
                audio_path,
                    sr=self._sampling_rate,
                    duration=self._max_duration
                )
                
                if progress_callback:
                    progress_callback(0.3)  # Loading complete
                
                # Extract features
                mfcc = librosa.feature.mfcc(
                    y=audio_data,
                    sr=sr,
                    n_mfcc=13
                )
                
                if progress_callback:
                    progress_callback(0.6)  # MFCC extraction complete
                
                # Calculate spectral features
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=audio_data,
                    sr=sr
                )
                spectral_bandwidth = librosa.feature.spectral_bandwidth(
                    y=audio_data,
                    sr=sr
                )
                
                if progress_callback:
                    progress_callback(0.8)  # Spectral features complete
                
                # Calculate quality metrics
                noise_score = self.get_noise_score({'audio_data': audio_data})
                clarity_score = self.get_clarity_score({'audio_data': audio_data})
                
                if progress_callback:
                    progress_callback(1.0)  # Processing complete
                
                # Create feature dictionary
                features = {
                    'audio_data': audio_data,
                    'mfcc': mfcc,
                    'spectral_centroid': spectral_centroid,
                    'spectral_bandwidth': spectral_bandwidth,
                    'quality_metrics': {
                        'noise_score': noise_score,
                        'clarity_score': clarity_score
                    },
                    'metadata': {
                        'sampling_rate': sr,
                        'duration': len(audio_data) / sr,
                        'num_samples': len(audio_data)
                    }
                }
                
                return features

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise AudioError(
                message="Failed to process audio",
                error_code=3009,
                operation="process_audio",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def get_noise_score(self, features: Dict[str, Any]) -> float:
        """
        Calculate audio noise score.

        Args:
            features: Audio features

        Returns:
            Noise score (0-1, higher means more noise)
        """
        try:
            # Extract audio data
            audio_data = features.get('audio_data')
            if audio_data is None:
                return 1.0  # Assume worst case
                
            # Calculate signal-to-noise ratio
            signal_power = np.mean(np.square(audio_data))
            noise_power = np.mean(np.square(audio_data - np.mean(audio_data)))
            
            if noise_power == 0:
                return 0.0  # Perfect signal
                
            snr = 10 * np.log10(signal_power / noise_power)
            
            # Convert SNR to noise score (0-1)
            # Assuming SNR range from -20dB to 40dB
            noise_score = 1.0 - min(max((snr + 20) / 60, 0), 1)
            
            return float(noise_score)

        except Exception as e:
            logger.error(f"Noise score calculation failed: {e}")
            return 1.0  # Assume worst case

    def get_clarity_score(self, features: Dict[str, Any]) -> float:
        """
        Calculate audio clarity score.

        Args:
            features: Audio features

        Returns:
            Clarity score (0-1, higher means clearer)
        """
        try:
            # Extract audio data
            audio_data = features.get('audio_data')
            if audio_data is None:
                return 0.0  # Assume worst case
                
            # Calculate spectral flatness
            spectrum = np.abs(np.fft.rfft(audio_data))
            spectral_flatness = np.exp(np.mean(np.log(spectrum + 1e-10))) / np.mean(spectrum)
            
            # Calculate zero-crossing rate
            zero_crossings = np.sum(np.diff(np.signbit(audio_data)))
            zcr = zero_crossings / (len(audio_data) - 1)
            
            # Combine metrics
            clarity_score = 0.7 * (1 - spectral_flatness) + 0.3 * (1 - zcr)
            
            return float(clarity_score)

        except Exception as e:
            logger.error(f"Clarity score calculation failed: {e}")
            return 0.0  # Assume worst case

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise AudioError(
                message="Failed to cleanup resources",
                error_code=3010,
                operation="cleanup",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def __del__(self) -> None:
        """Cleanup in destructor."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Audio handler cleanup in destructor failed: {e}")
