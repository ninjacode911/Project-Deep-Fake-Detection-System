"""
DeepFake Detection System - Exception Management
Created: 2025-06-07
Author: ninjacode911

This module initializes custom exceptions for the backend components.
It provides a hierarchical exception system for better error handling and reporting.
"""

from typing import Optional
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

class DeepFakeDetectionError(Exception):
    """Base exception class for all backend errors."""
    
    def __init__(self, message: str, error_code: Optional[int] = None) -> None:
        self.message = message
        self.error_code = error_code
        self.log_error()
        super().__init__(self.message)

    def log_error(self) -> None:
        """Log the error with appropriate severity."""
        logger.error(f"Error {self.error_code}: {self.message}")

class ModelError(DeepFakeDetectionError):
    """Exceptions related to model operations."""
    pass

class VideoError(DeepFakeDetectionError):
    """Exceptions related to video processing."""
    pass

class AudioError(DeepFakeDetectionError):
    """Exceptions related to audio processing."""
    pass

class CacheError(DeepFakeDetectionError):
    """Exceptions related to caching operations."""
    pass

class ConfigError(DeepFakeDetectionError):
    """Exceptions related to configuration."""
    pass

class DatabaseError(DeepFakeDetectionError):
    """Exceptions related to database operations."""
    pass

class ThreadError(DeepFakeDetectionError):
    """Exceptions related to thread operations."""
    pass

# Error codes for different categories
ERROR_CODES = {
    'MODEL': {
        'LOAD_ERROR': 1001,
        'INFERENCE_ERROR': 1002,
        'WEIGHT_ERROR': 1003
    },
    'VIDEO': {
        'EXTRACTION_ERROR': 2001,
        'PROCESSING_ERROR': 2002,
        'FORMAT_ERROR': 2003
    },
    'AUDIO': {
        'EXTRACTION_ERROR': 3001,
        'PROCESSING_ERROR': 3002,
        'FORMAT_ERROR': 3003
    },
    'CACHE': {
        'INIT_ERROR': 4001,
        'STORAGE_ERROR': 4002,
        'CLEANUP_ERROR': 4003
    },
    'CONFIG': {
        'LOAD_ERROR': 5001,
        'VALIDATION_ERROR': 5002,
        'UPDATE_ERROR': 5003
    },
    'DATABASE': {
        'CONNECTION_ERROR': 6001,
        'QUERY_ERROR': 6002,
        'INTEGRITY_ERROR': 6003
    },
    'THREAD': {
        'START_ERROR': 7001,
        'RUNTIME_ERROR': 7002,
        'CLEANUP_ERROR': 7003
    }
}

__all__ = [
    'DeepFakeDetectionError',
    'ModelError',
    'VideoError',
    'AudioError',
    'CacheError',
    'ConfigError',
    'DatabaseError',
    'ThreadError',
    'ERROR_CODES'
]