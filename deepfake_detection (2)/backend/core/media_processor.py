"""
DeepFake Detection System - Media Processor
Created: 2025-06-07
Author: ninjacode911

This module implements media processing for deepfake detection
with proper resource management and error handling.
"""

import logging
import time
import traceback
import threading
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable, Generator
import numpy as np
import cv2
import torch
from datetime import datetime
from contextlib import contextmanager

from .exceptions.backend_exceptions import MediaError, ValidationError, ResourceError
from ..config import config_manager

logger = logging.getLogger(__name__)

class MediaProcessor:
    """Processor class for media files."""

    def __init__(self) -> None:
        """Initialize media processor with configuration."""
        try:
            self._lock = threading.RLock()
            self._memory_limit = config_manager.get("media.memory_limit", 2 * 1024 * 1024 * 1024)  # 2GB
            self._max_frame_size = config_manager.get("media.max_frame_size", (1920, 1080))
            self._supported_formats = config_manager.get("media.supported_formats", [".mp4", ".avi", ".mov"])
            
            # Initialize processor registry
            self._processors: Dict[str, Dict[str, Any]] = {}
            
            logger.info("Media processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Media processor initialization failed: {e}")
            raise MediaError(
                message="Failed to initialize media processor",
                error_code=9000,
                operation="init",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_media_path(self, media_path: str) -> None:
        """
        Validate media file path.
        
        Args:
            media_path: Path to media file
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check if path exists
            if not os.path.exists(media_path):
                raise ValidationError(
                    message="Media file does not exist",
                    error_code=9001,
                    details={'path': media_path}
                )
                
            # Check file format
            file_ext = os.path.splitext(media_path)[1].lower()
            if file_ext not in self._supported_formats:
                raise ValidationError(
                    message="Unsupported media format",
                    error_code=9002,
                    details={
                        'path': media_path,
                        'format': file_ext,
                        'supported': self._supported_formats
                    }
                )
                
            # Check file size
            file_size = os.path.getsize(media_path)
            max_size = config_manager.get("media.max_file_size", 10 * 1024 * 1024 * 1024)  # 10GB
            if file_size > max_size:
                raise ValidationError(
                    message="Media file too large",
                    error_code=9003,
                    details={
                        'path': media_path,
                        'size': file_size,
                        'max_size': max_size
                    }
                )
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Media path validation failed: {e}")
            raise ValidationError(
                message="Media path validation failed",
                error_code=9004,
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
                    error_code=9005,
                    details={
                        'used': current_memory - start_memory,
                        'limit': self._memory_limit
                    }
                )
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            raise ResourceError(
                message="Memory monitoring failed",
                error_code=9006,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def process_media(
        self,
        media_path: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """
        Process media file and yield frames with metadata.
        
        Args:
            media_path: Path to media file
            progress_callback: Optional callback for progress updates
            
        Yields:
            Tuple of (frame, metadata)
        """
        try:
            # Validate media path
            self._validate_media_path(media_path)
            
            # Open video capture
            cap = cv2.VideoCapture(media_path)
            if not cap.isOpened():
                raise MediaError(
                    message="Failed to open media file",
                    error_code=9007,
                    details={'path': media_path}
                )
                
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Process frames with memory monitoring
            with self._memory_context():
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # Resize frame if needed
                    if (width, height) > self._max_frame_size:
                        scale = min(
                            self._max_frame_size[0] / width,
                            self._max_frame_size[1] / height
                        )
                        new_size = (int(width * scale), int(height * scale))
                        frame = cv2.resize(frame, new_size)
                        
                    # Create frame metadata
                    metadata = {
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'resolution': frame.shape[:2],
                        'fps': fps,
                        'duration': duration,
                        'total_frames': total_frames
                    }
                    
                    # Update progress
                    if progress_callback:
                        progress = frame_count / total_frames
                        progress_callback(progress)
                        
                    yield frame, metadata
                    frame_count += 1
                    
            # Release resources
            cap.release()
            
        except Exception as e:
            logger.error(f"Media processing failed: {e}")
            raise MediaError(
                message="Failed to process media",
                error_code=9008,
                operation="process_media",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def extract_metadata(
        self,
        media_path: str
    ) -> Dict[str, Any]:
        """
        Extract metadata from media file.
        
        Args:
            media_path: Path to media file
            
        Returns:
            Dictionary of metadata
        """
        try:
            # Validate media path
            self._validate_media_path(media_path)
            
            # Open video capture
            cap = cv2.VideoCapture(media_path)
            if not cap.isOpened():
                raise MediaError(
                    message="Failed to open media file",
                    error_code=9009,
                    details={'path': media_path}
                )
                
            # Extract metadata
            metadata = {
                'path': media_path,
                'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': float(cap.get(cv2.CAP_PROP_FPS)),
                'duration': float(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)),
                'resolution': (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ),
                'format': os.path.splitext(media_path)[1].lower(),
                'size': os.path.getsize(media_path),
                'created': datetime.fromtimestamp(
                    os.path.getctime(media_path)
                ).isoformat(),
                'modified': datetime.fromtimestamp(
                    os.path.getmtime(media_path)
                ).isoformat()
            }
            
            # Release resources
            cap.release()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            raise MediaError(
                message="Failed to extract metadata",
                error_code=9010,
                operation="extract_metadata",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            with self._lock:
                self._processors.clear()
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise MediaError(
                message="Failed to cleanup resources",
                error_code=9011,
                operation="cleanup",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def __del__(self) -> None:
        """Cleanup in destructor."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Media processor cleanup in destructor failed: {e}") 