"""
DeepFake Detection System - Video Handler
Created: 2025-06-07
Author: ninjacode911

This module implements video processing and feature extraction for deepfake detection
with proper resource management and error handling.
"""

import logging
import time
import traceback
import threading
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import cv2
import numpy as np
import torch
from contextlib import contextmanager

from .exceptions.backend_exceptions import VideoError, ValidationError, ResourceError
from ..config import config_manager
from ..utils.model_utils import ModelUtils

logger = logging.getLogger(__name__)

class VideoHandler:
    """Handler class for video processing and feature extraction."""

    def __init__(self) -> None:
        """Initialize video handler with configuration."""
        try:
            self._lock = threading.RLock()
            self._face_detector = None
            self._memory_limit = config_manager.get("video.memory_limit", 4 * 1024 * 1024 * 1024)  # 4GB
            self._supported_formats = {'.mp4', '.avi', '.mov', '.mkv'}
            
            logger.info("Video handler initialized successfully")
            
        except Exception as e:
            logger.error(f"Video handler initialization failed: {e}")
            raise VideoError(
                message="Failed to initialize video handler",
                error_code=2000,
                operation="init",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_video(self, video_path: str) -> None:
        """
        Validate video file.

        Args:
            video_path: Path to video file

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check file existence
            if not os.path.exists(video_path):
                raise ValidationError(
                    message="Video file not found",
                    error_code=2001,
                    details={'path': video_path}
                )
                
            # Check file extension
            ext = os.path.splitext(video_path)[1].lower()
            if ext not in self._supported_formats:
                raise ValidationError(
                    message="Unsupported video format",
                    error_code=2002,
                    details={'path': video_path, 'format': ext}
                )
                
            # Check file size
            file_size = os.path.getsize(video_path)
            max_size = config_manager.get("video.max_file_size", 1024 * 1024 * 1024)  # 1GB
            if file_size > max_size:
                raise ValidationError(
                    message="Video file too large",
                    error_code=2003,
                    details={
                        'path': video_path,
                        'size': file_size,
                        'max_size': max_size
                    }
                )
                
            # Check if file is corrupted
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValidationError(
                    message="Corrupted video file",
                    error_code=2004,
                    details={'path': video_path}
                )
            cap.release()
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Video validation failed: {e}")
            raise ValidationError(
                message="Video validation failed",
                error_code=2005,
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
                    error_code=2006,
                    details={
                        'used': current_memory - start_memory,
                        'limit': self._memory_limit
                    }
                )
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            raise ResourceError(
                message="Memory monitoring failed",
                error_code=2007,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def extract_features(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Extract features from video for deepfake detection.
        
        Args:
            video_path: Path to video file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary of extracted features
        """
        try:
            # Validate video
            self._validate_video(video_path)
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise VideoError(
                    message="Failed to open video",
                    error_code=2008,
                    details={'path': video_path}
                )
                
            try:
                # Get video properties
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                
                # Calculate frame sampling
                target_frames = config_manager.get("video.target_frames", 30)
                frame_interval = max(1, total_frames // target_frames)
                
                # Initialize feature containers
                frames = []
                frame_scores = []
                current_frame = 0
                
                # Process frames with memory monitoring
                with self._memory_context():
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                            
                        if current_frame % frame_interval == 0:
                            # Process frame
                            processed_frame = self._preprocess_frame(frame)
                            frames.append(processed_frame)
                            
                            # Calculate frame quality score
                            quality_score = self._calculate_frame_quality(frame)
                            frame_scores.append(quality_score)
                            
                            # Update progress
                            if progress_callback:
                                progress = (current_frame + 1) / total_frames
                                progress_callback(progress)
                                
                        current_frame += 1
                        
                # Convert to numpy arrays
                frames = np.stack(frames) if frames else np.array([])
                frame_scores = np.array(frame_scores) if frame_scores else np.array([])
                
                # Create feature dictionary
                features = {
                    'frames': frames,
                    'frame_scores': frame_scores,
                    'metadata': {
                        'total_frames': total_frames,
                        'fps': fps,
                        'duration': duration,
                        'sampled_frames': len(frames)
                    }
                }
                
                return features
                
            finally:
                cap.release()

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise VideoError(
                message="Failed to extract features",
                error_code=2009,
                operation="extract_features",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess video frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        try:
            # Resize frame
            target_size = config_manager.get("video.target_size", (224, 224))
            resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            
            # Convert to RGB
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            elif resized.shape[2] == 4:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
            elif resized.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized

        except Exception as e:
            logger.error(f"Frame preprocessing failed: {e}")
            raise VideoError(
                message="Failed to preprocess frame",
                error_code=2010,
                operation="preprocess_frame",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _calculate_frame_quality(self, frame: np.ndarray) -> float:
        """
        Calculate frame quality score.
        
        Args:
            frame: Input frame
            
        Returns:
            Quality score (0-1)
        """
        try:
            # Calculate blur score
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = 1.0 - min(np.var(laplacian) / 1000.0, 1.0)
            
            # Calculate brightness score
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Calculate contrast score
            contrast = np.std(gray) / 128.0
            contrast_score = min(contrast, 1.0)
            
            # Combine scores
            quality_score = (
                0.4 * (1.0 - blur_score) +
                0.3 * brightness_score +
                0.3 * contrast_score
            )
            
            return float(quality_score)

        except Exception as e:
            logger.error(f"Frame quality calculation failed: {e}")
            return 0.0  # Return worst quality on error

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self._face_detector is not None:
                self._face_detector = None
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise VideoError(
                message="Failed to cleanup resources",
                error_code=2011,
                operation="cleanup",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def __del__(self) -> None:
        """Cleanup in destructor."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Video handler cleanup in destructor failed: {e}")