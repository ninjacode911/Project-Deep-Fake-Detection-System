"""
DeepFake Detection System - Core Components
Created: 2025-06-07
Author: ninjacode911

This module initializes and exports core components of the backend system.
It provides centralized access to detection, media handling, and threading utilities.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Import core components
from .detector import Detector
from .video_handler import VideoHandler
from .audio_handler import AudioHandler
from .thread import AnalysisThread
from .cache.cache_manager import CacheManager
from .exceptions.backend_exceptions import (
    BackendError,
    ModelError,
    VideoError,
    AudioError
)

# Configure logger
logger = logging.getLogger(__name__)

class CoreManager:
    """Manages core component initialization and lifecycle."""
    
    _instance: Optional['CoreManager'] = None
    _initialized = False

    def __new__(cls) -> 'CoreManager':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize core components."""
        if self._initialized:
            return
            
        try:
            # Initialize cache system
            cache_dir = Path(__file__).parent.parent / "data" / "cache"
            self.cache = CacheManager(cache_dir)
            
            # Initialize handlers
            self.video_handler = VideoHandler()
            self.audio_handler = AudioHandler()
            
            # Initialize detector
            self.detector = Detector()
            
            logger.info("Core components initialized successfully")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Core initialization failed: {e}")
            raise BackendError(
                message="Failed to initialize core components",
                error_code=1000,
                component="Core",
                operation="init",
                details={'error': str(e)}
            )

    def cleanup(self) -> None:
        """Clean up core resources."""
        try:
            if hasattr(self, 'cache'):
                self.cache.cleanup()
            if hasattr(self, 'video_handler'):
                self.video_handler.cleanup()
            if hasattr(self, 'audio_handler'):
                self.audio_handler.cleanup()
            if hasattr(self, 'detector'):
                self.detector.cleanup()
                
            logger.info("Core components cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Core cleanup failed: {e}")
            raise BackendError(
                message="Failed to cleanup core components",
                error_code=1001,
                component="Core",
                operation="cleanup",
                details={'error': str(e)}
            )

# Initialize core manager
core_manager = CoreManager()

# Register cleanup
import atexit
atexit.register(core_manager.cleanup)

# Version information
__version__ = "1.0.0"
__author__ = "ninjacode911"

# Export public interfaces
__all__ = [
    'Detector',
    'VideoHandler',
    'AudioHandler',
    'AnalysisThread',
    'CacheManager',
    'BackendError',
    'ModelError',
    'VideoError',
    'AudioError',
    'core_manager'
]