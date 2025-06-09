"""
DeepFake Detection System - Core Detector
Created: 2025-06-07
Author: ninjacode911

This module implements the core deepfake detection logic with model management,
caching, and resource optimization for both video and audio deepfake detection.
"""

import logging
import time
import traceback
import threading
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import mimetypes
import magic

from .exceptions.backend_exceptions import (
    ModelError, CacheError, VideoError, AudioError,
    ValidationError, ResourceError
)
from .cache.cache_manager import CacheManager
from .audio_handler import AudioHandler
from .video_handler import VideoHandler
from ..models.factory import ModelFactory
from ..types.backend_types import DetectionResult, ModelConfig, ProcessingStats
from ..config import config_manager
from ..utils.model_utils import ModelUtils

logger = logging.getLogger(__name__)

class Detector:
    """Core detector class for integrated video and audio deepfake detection."""

    def __init__(self) -> None:
        """Initialize detector with models and handlers."""
        try:
            # Initialize cache and handlers
            self._cache_manager = CacheManager(
                Path(__file__).parent.parent / "data" / "cache" / "detection"
            )
            self._video_handler = VideoHandler()
            self._audio_handler = AudioHandler()
            self._model_factory = ModelFactory()
            self._lock = threading.RLock()

            # Configure parallel processing
            max_workers = min(
                config_manager.get("detection.max_workers", 4),
                ModelUtils.get_optimal_thread_count()
            )
            self._executor = ThreadPoolExecutor(max_workers=max_workers)

            # Initialize model containers
            self._models: Dict[str, Any] = {}
            self._model_stats: Dict[str, ProcessingStats] = {}
            
            # Set device with proper error handling
            self._device = self._get_optimal_device()
            
            # Initialize models
            self._initialize_models()
            
            logger.info(f"Detector initialized successfully on {self._device}")
            
        except Exception as e:
            logger.error(f"Detector initialization failed: {e}")
            raise ModelError(
                message="Failed to initialize detector",
                error_code=1000,
                operation="init",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def get_health_status(self) -> Dict[str, Any]:
        return {
            'models_loaded': list(self._models.keys()),
            'device': str(self._device),
            'cache_size': self._cache_manager.size,
            'memory_usage': ModelUtils.get_memory_usage(),
            'model_stats': self.get_model_stats()
        }
    
    def _get_optimal_device(self) -> torch.device:
        """Determine optimal device with memory checks."""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                required_memory = config_manager.get("models.min_gpu_memory", 2 * 1024 * 1024 * 1024)  # 2GB
                if gpu_memory >= required_memory:
                    return torch.device("cuda")
            return torch.device("cpu")
        except Exception as e:
            logger.warning(f"Device detection failed, defaulting to CPU: {e}")
            return torch.device("cpu")

    def _initialize_models(self) -> None:
        """Initialize video and audio detection models."""
        try:
            model_configs = config_manager.get("models", {})
            
            # Required models check
            required_models = {"vision", "audio"}
            available_models = set(model_configs.keys())
            if not required_models.issubset(available_models):
                missing = required_models - available_models
                raise ModelError(
                    message="Missing required models",
                    error_code=1001,
                    details={'missing_models': list(missing)}
                )

            # Initialize models with resource management
            with ModelUtils.model_inference_context():
                for model_name, config in model_configs.items():
                    self._models[model_name] = self._model_factory.create_model(
                        model_name,
                        config,
                        device=self._device
                    )
                    self._model_stats[model_name] = ProcessingStats()

            logger.info(f"Initialized {len(self._models)} models successfully")
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise ModelError(
                message="Failed to initialize detection models",
                error_code=1100,
                operation="initialize_models",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_media_path(self, media_path: str) -> None:
        """
        Validate media file path and format.
        
        Args:
            media_path: Path to media file
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check file existence
            if not os.path.exists(media_path):
                raise ValidationError(
                    message="Media file not found",
                    error_code=1004,
                    details={'path': media_path}
                )
                
            # Check file size
            file_size = os.path.getsize(media_path)
            max_size = config_manager.get("detection.max_file_size", 1024 * 1024 * 1024)  # 1GB
            if file_size > max_size:
                raise ValidationError(
                    message="File size exceeds maximum limit",
                    error_code=1005,
                    details={
                        'path': media_path,
                        'size': file_size,
                        'max_size': max_size
                    }
                )
                
            # Check file type
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(media_path)
            
            if not file_type.startswith(('video/', 'audio/')):
                raise ValidationError(
                    message="Unsupported file type",
                    error_code=1006,
                    details={
                        'path': media_path,
                        'type': file_type
                    }
                )
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Media validation failed: {e}")
            raise ValidationError(
                message="Media validation failed",
                error_code=1007,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_options(self, options: Dict[str, Any]) -> None:
        """
        Validate detection options.
        
        Args:
            options: Detection options
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if not isinstance(options, dict):
                raise ValidationError(
                    message="Invalid options type",
                    error_code=1008,
                    details={'type': type(options)}
                )
                
            # Validate threshold
            threshold = options.get('threshold')
            if threshold is not None:
                if not isinstance(threshold, (int, float)):
                    raise ValidationError(
                        message="Invalid threshold type",
                        error_code=1009,
                        details={'type': type(threshold)}
                    )
                if not 0 <= threshold <= 1:
                    raise ValidationError(
                        message="Threshold out of range",
                        error_code=1010,
                        details={'value': threshold}
                    )
                    
            # Validate batch size
            batch_size = options.get('batch_size')
            if batch_size is not None:
                if not isinstance(batch_size, int):
                    raise ValidationError(
                        message="Invalid batch size type",
                        error_code=1011,
                        details={'type': type(batch_size)}
                    )
                if batch_size <= 0:
                    raise ValidationError(
                        message="Invalid batch size",
                        error_code=1012,
                        details={'value': batch_size}
                    )
                    
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Options validation failed: {e}")
            raise ValidationError(
                message="Options validation failed",
                error_code=1013,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _update_progress(
        self,
        stage: str,
        progress: float,
        callback: Optional[Callable[[int], None]]
    ) -> None:
        """
        Update progress with stage weights.
        
        Args:
            stage: Current processing stage
            progress: Stage progress (0-1)
            callback: Progress callback function
        """
        if callback is None:
            return
            
        try:
            # Define stage weights
            stage_weights = {
                'video_extract': 0.3,
                'audio_extract': 0.2,
                'detection': 0.5
            }
            
            # Calculate overall progress
            overall_progress = int(sum(
                stage_weights.get(s, 0) * (progress if s == stage else 0)
                for s in stage_weights
            ) * 100)
            
            # Call progress callback
            callback(overall_progress)
            
        except Exception as e:
            logger.error(f"Progress update failed: {e}")
            # Don't raise error to avoid interrupting processing

    def detect(
        self, 
        media_path: str, 
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> DetectionResult:
        """
        Detect deepfakes using both video and audio analysis.

        Args:
            media_path: Path to media file
            options: Detection options
            progress_callback: Optional callback for progress updates

        Returns:
            DetectionResult with combined video and audio analysis
        """
        try:
            # Validate inputs
            self._validate_media_path(media_path)
            if options:
                self._validate_options(options)

            start_time = time.time()
            cache_key = self._generate_cache_key(media_path, options)

            # Check cache with validation
            cached_result = self._get_validated_cache(cache_key, media_path)
            if cached_result is not None:
                return cached_result

            # Extract features with progress tracking
            video_features = self._video_handler.extract_features(
                media_path,
                lambda p: self._update_progress('video_extract', p, progress_callback)
            )
            audio_features = self._audio_handler.process_audio(
                media_path,
                lambda p: self._update_progress('audio_extract', p, progress_callback)
            )
            
            # Run detection with resource optimization
            with ModelUtils.model_inference_context():
                future_results = []
                for model_name, model in self._models.items():
                    future = self._executor.submit(
                        self._run_model_detection,
                        model_name,
                        model,
                        video_features if 'vision' in model_name else None,
                        audio_features if 'audio' in model_name else None
                    )
                    future_results.append(future)

                # Collect results with progress tracking
                model_results = {}
                for i, future in enumerate(future_results):
                    model_name, result = future.result()
                    model_results[model_name] = result
                    progress = (i + 1) / len(future_results)
                    self._update_progress('detection', progress, progress_callback)

            # Calculate final result with weighted combination
            final_score = self._aggregate_results(model_results)
            processing_time = time.time() - start_time
            
            # Create comprehensive result
            result = DetectionResult(
                is_fake=final_score > config_manager.get("detection.threshold", 0.5),
                confidence=final_score,
                model_scores=model_results,
                processing_time=processing_time,
                frame_scores=video_features.get('frame_scores'),
                audio_scores=audio_features.get('segment_scores'),
                metadata={
                    'device': str(self._device),
                    'models_used': list(self._models.keys()),
                    'timestamp': time.time()
                }
            )

            # Cache with validation
            self._cache_manager.put(cache_key, result.to_dict())
            
            logger.info(f"Detection completed in {processing_time:.2f}s with score {final_score:.3f}")
            return result

        except Exception as e:
            logger.error(f"Detection failed for {media_path}: {e}")
            raise ModelError(
                message="Detection failed",
                error_code=1101,
                operation="detect",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _get_validated_cache(self, cache_key: str, media_path: str) -> Optional[DetectionResult]:
        """Get and validate cached result."""
        try:
            cached = self._cache_manager.get(cache_key)
            if not cached:
                return None
            
            if not os.path.exists(media_path):
                return None
            
            mtime = os.path.getmtime(media_path)
            if mtime > cached.get('timestamp', 0):
                return None
            
            return DetectionResult(**cached)
            
        except Exception as e:
            logger.warning(f"Cache validation failed: {e}")
            return None

    def _run_model_detection(
        self,
        model_name: str,
        model: Any,
        video_features: Optional[np.ndarray] = None,
        audio_features: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Run detection with a single model and collect quality metrics.
        
        Args:
            model_name: Name of the model
            model: Model instance
            video_features: Optional video features
            audio_features: Optional audio features
            
        Returns:
            Tuple of (model_name, result_dict)
        """
        try:
            # Initialize result dictionary
            result = {
                'score': 0.0,
                'confidence': 0.0,
                'quality_metrics': {}
            }
            
            # Run model inference
            if 'vision' in model_name and video_features is not None:
                # Process video features
                with torch.no_grad():
                    predictions = model.predict(video_features)
                    
                # Calculate quality metrics for vision
                face_detection = self._video_handler.get_face_detection_quality(video_features)
                blur_score = self._video_handler.get_blur_score(video_features)
                
                result.update({
                    'score': float(predictions.mean()),
                    'confidence': float(predictions.std()),
                    'quality_metrics': {
                        'face_detection': face_detection,
                        'blur_score': blur_score
                    }
                })
                
            elif 'audio' in model_name and audio_features is not None:
                # Process audio features
                with torch.no_grad():
                    predictions = model.predict(audio_features)
                    
                # Calculate quality metrics for audio
                noise_score = self._audio_handler.get_noise_score(audio_features)
                clarity_score = self._audio_handler.get_clarity_score(audio_features)
                
                result.update({
                    'score': float(predictions.mean()),
                    'confidence': float(predictions.std()),
                    'quality_metrics': {
                        'noise_score': noise_score,
                        'clarity_score': clarity_score
                    }
                })

            # Update model statistics
            self._model_stats[model_name].update(
                inference_time=time.time() - self._model_stats[model_name].last_inference_time,
                memory_usage=model.get_memory_usage()
            )
            
            return model_name, result
            
        except Exception as e:
            logger.error(f"Model detection failed for {model_name}: {e}")
            raise ModelError(
                message=f"Detection failed for {model_name}",
                error_code=1103,
                operation="run_model_detection",
                details={'model_name': model_name, 'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _aggregate_results(self, model_results: Dict[str, Dict[str, Any]]) -> float:
        """
        Aggregate model results using hybrid approach with confidence and content quality.
        
        Args:
            model_results: Dictionary of model results
            
        Returns:
            Final detection score
        """
        try:
            # Get hybrid configuration
            hybrid_config = config_manager.get("detection.hybrid", {})
            base_weights = hybrid_config.get("base_weights", {"vision": 0.5, "audio": 0.5})
            content_quality = hybrid_config.get("content_quality", {})
            validation = hybrid_config.get("validation", {})
            
            # Initialize weights and scores
            weights = {}
            scores = {}
            confidences = {}
            
            # Process each model result
            for model_name, result in model_results.items():
                score = result['score']
                confidence = result['confidence']
                quality_metrics = result['quality_metrics']
                
                # Store scores and confidences
                scores[model_name] = score
                confidences[model_name] = confidence
                
                # Calculate base weight
                base_weight = base_weights.get(model_name, 0.5)
                
                # Adjust weight based on confidence
                confidence_weight = confidence / sum(confidences.values())
                
                # Adjust weight based on content quality
                quality_weight = 1.0
                if model_name == 'vision':
                    face_detection = quality_metrics.get('face_detection', 0.0)
                    blur_score = quality_metrics.get('blur_score', 0.0)
                    
                    if face_detection > content_quality['vision']['face_detection_threshold']:
                        quality_weight *= content_quality['vision']['quality_weight_multiplier']
                    if blur_score < content_quality['vision']['blur_threshold']:
                        quality_weight *= content_quality['vision']['quality_weight_multiplier']
                        
                elif model_name == 'audio':
                    noise_score = quality_metrics.get('noise_score', 0.0)
                    clarity_score = quality_metrics.get('clarity_score', 0.0)
                    
                    if noise_score < content_quality['audio']['noise_threshold']:
                        quality_weight *= content_quality['audio']['quality_weight_multiplier']
                    if clarity_score > content_quality['audio']['clarity_threshold']:
                        quality_weight *= content_quality['audio']['quality_weight_multiplier']
                
                # Calculate final weight
                weights[model_name] = base_weight * confidence_weight * quality_weight
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            # Check for high confidence cases
            high_confidence_threshold = validation.get('high_confidence_threshold', 0.8)
            for model_name, confidence in confidences.items():
                if confidence > high_confidence_threshold:
                    # If one model has high confidence, use its score
                    return scores[model_name]
            
            # Check for model disagreement
            score_disagreement_threshold = validation.get('score_disagreement_threshold', 0.3)
            if abs(scores.get('vision', 0) - scores.get('audio', 0)) > score_disagreement_threshold:
                # If models disagree significantly, use the more confident one
                max_confidence_model = max(confidences.items(), key=lambda x: x[1])[0]
                return scores[max_confidence_model]
            
            # Calculate weighted average
            final_score = sum(scores[model] * weights[model] for model in scores)
            
            # Validate final score
            min_confidence = validation.get('min_confidence_threshold', 0.5)
            if max(confidences.values()) < min_confidence:
                logger.warning("Low confidence in all models")
            
            return final_score
            
        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")
            raise ModelError(
                message="Failed to aggregate model results",
                error_code=1102,
                operation="aggregate_results",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _generate_cache_key(self, media_path: str, options: Optional[Dict[str, Any]]) -> str:
        """Generate cache key from file path and options."""
        mtime = os.path.getmtime(media_path)
        options_str = str(sorted(options.items())) if options else ""
        return f"detection_{media_path}_{mtime}_{options_str}"

    def get_model_stats(self) -> Dict[str, ProcessingStats]:
        """Get model performance statistics."""
        return self._model_stats

    def cleanup(self) -> None:
        """Clean up detector resources."""
        try:
            # Clean up models
            for model in self._models.values():
                model.cleanup()
            self._models.clear()

            # Clean up handlers
            self._video_handler.cleanup()
            self._audio_handler.cleanup()
            
            # Clean up cache
            self._cache_manager.cleanup()
            
            # Shutdown thread pool
            self._executor.shutdown(wait=True)
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Detector resources cleaned up")
            
        except Exception as e:
            logger.error(f"Detector cleanup failed: {e}")
            raise ModelError(
                message="Failed to cleanup detector resources",
                error_code=1104,
                operation="cleanup",
                details={'error': str(e)}
            )

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Detector cleanup in destructor failed: {e}")