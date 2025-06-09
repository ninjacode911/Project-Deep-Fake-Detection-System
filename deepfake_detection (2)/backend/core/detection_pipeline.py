"""
DeepFake Detection System - Detection Pipeline
Created: 2025-06-07
Author: ninjacode911

This module implements the main detection pipeline for deepfake detection
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
from datetime import datetime
from contextlib import contextmanager

from .exceptions.backend_exceptions import PipelineError, ValidationError, ResourceError
from .model_manager import ModelManager
from .feature_extractor import FeatureExtractor
from .result_aggregator import ResultAggregator
from .report_generator import ReportGenerator
from .media_processor import MediaProcessor
from ..config import config_manager

logger = logging.getLogger(__name__)

class DetectionPipeline:
    """Pipeline class for deepfake detection."""

    def __init__(self) -> None:
        """Initialize detection pipeline with components."""
        try:
            self._lock = threading.RLock()
            self._memory_limit = config_manager.get("pipeline.memory_limit", 4 * 1024 * 1024 * 1024)  # 4GB
            self._batch_size = config_manager.get("pipeline.batch_size", 32)
            self._min_confidence = config_manager.get("pipeline.min_confidence", 0.5)
            
            # Initialize components
            self._model_manager = ModelManager()
            self._feature_extractor = FeatureExtractor()
            self._result_aggregator = ResultAggregator()
            self._report_generator = ReportGenerator()
            self._media_processor = MediaProcessor()
            
            # Initialize pipeline registry
            self._pipelines: Dict[str, Dict[str, Any]] = {}
            
            logger.info("Detection pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Detection pipeline initialization failed: {e}")
            raise PipelineError(
                message="Failed to initialize detection pipeline",
                error_code=10000,
                operation="init",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_input(self, media_path: str, model_path: str) -> None:
        """
        Validate pipeline input.
        
        Args:
            media_path: Path to media file
            model_path: Path to model file
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Validate media path
            if not os.path.exists(media_path):
                raise ValidationError(
                    message="Media file does not exist",
                    error_code=10001,
                    details={'path': media_path}
                )
                
            # Validate model path
            if not os.path.exists(model_path):
                raise ValidationError(
                    message="Model file does not exist",
                    error_code=10002,
                    details={'path': model_path}
                )
                
            # Validate model format
            model_ext = os.path.splitext(model_path)[1].lower()
            if model_ext not in ['.pth', '.pt', '.onnx']:
                raise ValidationError(
                    message="Unsupported model format",
                    error_code=10003,
                    details={
                        'path': model_path,
                        'format': model_ext
                    }
                )
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            raise ValidationError(
                message="Input validation failed",
                error_code=10004,
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
                    error_code=10005,
                    details={
                        'used': current_memory - start_memory,
                        'limit': self._memory_limit
                    }
                )
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            raise ResourceError(
                message="Memory monitoring failed",
                error_code=10006,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def process_media(
        self,
        media_path: str,
        model_path: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Process media file through detection pipeline.
        
        Args:
            media_path: Path to media file
            model_path: Path to model file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Detection results
        """
        try:
            # Validate input
            self._validate_input(media_path, model_path)
            
            # Process media with memory monitoring
            with self._memory_context():
                # Load model
                model = self._model_manager.load_model(model_path)
                
                if progress_callback:
                    progress_callback(0.1)  # Model loaded
                    
                # Extract metadata
                metadata = self._media_processor.extract_metadata(media_path)
                
                if progress_callback:
                    progress_callback(0.2)  # Metadata extracted
                    
                # Process frames
                results = []
                batch_frames = []
                batch_metadata = []
                
                for frame, frame_metadata in self._media_processor.process_media(
                    media_path,
                    lambda p: progress_callback(0.2 + 0.4 * p) if progress_callback else None
                ):
                    # Add to batch
                    batch_frames.append(frame)
                    batch_metadata.append(frame_metadata)
                    
                    # Process batch if full
                    if len(batch_frames) >= self._batch_size:
                        # Extract features
                        features = self._feature_extractor.extract_features(
                            batch_frames,
                            model
                        )
                        
                        # Get predictions
                        predictions = model(features)
                        
                        # Add results
                        for pred, meta in zip(predictions, batch_metadata):
                            if pred['confidence'] >= self._min_confidence:
                                results.append({
                                    'prediction': pred['prediction'],
                                    'confidence': pred['confidence'],
                                    'timestamp': meta['timestamp'],
                                    'frame_number': meta['frame_number']
                                })
                                
                        # Clear batch
                        batch_frames.clear()
                        batch_metadata.clear()
                        
                # Process remaining frames
                if batch_frames:
                    features = self._feature_extractor.extract_features(
                        batch_frames,
                        model
                    )
                    predictions = model(features)
                    
                    for pred, meta in zip(predictions, batch_metadata):
                        if pred['confidence'] >= self._min_confidence:
                            results.append({
                                'prediction': pred['prediction'],
                                'confidence': pred['confidence'],
                                'timestamp': meta['timestamp'],
                                'frame_number': meta['frame_number']
                            })
                            
                if progress_callback:
                    progress_callback(0.6)  # Frames processed
                    
                # Aggregate results
                aggregated = self._result_aggregator.aggregate_results(
                    results,
                    lambda p: progress_callback(0.6 + 0.2 * p) if progress_callback else None
                )
                
                if progress_callback:
                    progress_callback(0.8)  # Results aggregated
                    
                # Generate report
                report = self._report_generator.generate_report(
                    aggregated,
                    media_path,
                    lambda p: progress_callback(0.8 + 0.2 * p) if progress_callback else None
                )
                
                if progress_callback:
                    progress_callback(1.0)  # Report generated
                    
                return report
                
        except Exception as e:
            logger.error(f"Media processing failed: {e}")
            raise PipelineError(
                message="Failed to process media",
                error_code=10007,
                operation="process_media",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            with self._lock:
                self._pipelines.clear()
                
            # Cleanup components
            self._model_manager.cleanup()
            self._feature_extractor.cleanup()
            self._result_aggregator.cleanup()
            self._report_generator.cleanup()
            self._media_processor.cleanup()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise PipelineError(
                message="Failed to cleanup resources",
                error_code=10008,
                operation="cleanup",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def __del__(self) -> None:
        """Cleanup in destructor."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Detection pipeline cleanup in destructor failed: {e}") 