"""
DeepFake Detection System - Type Definitions
Created: 2025-06-07
Author: ninjacode911

This module exports all type definitions and enums used throughout the backend.
Provides proper type checking and validation functionality.
"""

from typing import Dict, List, Union, Optional, Any, TypeVar, Callable, Type
from enum import Enum, auto
from dataclasses import dataclass
from pathlib import Path

from .backend_types import (
    DetectionResult,
    ModelConfig,
    VideoFrame,
    AudioSegment,
    ProcessingStats,
    CacheConfig,
    DatabaseConfig,
    ValidationResult,
    ErrorDetails
)

# Type aliases for common types
PathLike = Union[str, Path]
JsonDict = Dict[str, Any]
ModelOutput = Union[float, List[float]]
FrameData = Union[VideoFrame, AudioSegment]

# Type variable for generic functions
T = TypeVar('T')
CallbackType = Callable[[T], None]

class MediaType(Enum):
    """Supported media types."""
    VIDEO = auto()
    AUDIO = auto()
    IMAGE = auto()

class ProcessingStatus(Enum):
    """Processing status indicators."""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

class ErrorLevel(Enum):
    """Error severity levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

class ModelType(Enum):
    """Supported model types."""
    EFFICIENT_NET = "efficientnet"
    WAV2VEC = "wav2vec2"
    ENSEMBLE = "ensemble"

@dataclass
class SystemConfig:
    """System configuration container."""
    cache_config: CacheConfig
    db_config: DatabaseConfig
    model_configs: Dict[str, ModelConfig]
    
# Export all defined types
__all__ = [
    # Core types
    'DetectionResult',
    'ModelConfig', 
    'VideoFrame',
    'AudioSegment',
    'ProcessingStats',
    'CacheConfig',
    'DatabaseConfig',
    'ValidationResult',
    'ErrorDetails',
    'SystemConfig',
    
    # Type aliases
    'PathLike',
    'JsonDict',
    'ModelOutput',
    'FrameData',
    'CallbackType',
    
    # Enums
    'MediaType',
    'ProcessingStatus', 
    'ErrorLevel',
    'ModelType',
    
    # Type vars
    'T'
]

def validate_type(value: Any, expected_type: Type[T]) -> ValidationResult:
    """
    Validate value against expected type.
    
    Args:
        value: Value to validate
        expected_type: Expected type
        
    Returns:
        ValidationResult with validation status and error details
    """
    try:
        if not isinstance(value, expected_type):
            return ValidationResult(
                valid=False,
                error=ErrorDetails(
                    message=f"Invalid type: expected {expected_type.__name__}, got {type(value).__name__}",
                    error_code=4001
                )
            )
        return ValidationResult(valid=True)
        
    except Exception as e:
        return ValidationResult(
            valid=False,
            error=ErrorDetails(
                message=f"Validation error: {str(e)}",
                error_code=4002
            )
        )