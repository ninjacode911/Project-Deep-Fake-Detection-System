"""
DeepFake Detection System - Backend Type Definitions
Created: 2025-06-07
Author: ninjacode911

This module defines core types and data structures used throughout the backend.
"""
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, TypeVar, Tuple, TypedDict, Literal
from enum import Enum, auto
from pathlib import Path
from datetime import datetime
import logging
import time
import traceback

from ..core.exceptions.backend_exceptions import ValidationError

logger = logging.getLogger(__name__)

# Type aliases
ImageTensor = Union[torch.Tensor, np.ndarray]
AudioTensor = Union[torch.Tensor, np.ndarray]
ModelWeights = Dict[str, torch.Tensor]
JsonDict = Dict[str, Any]

@dataclass(frozen=True)
class ErrorDetails:
    """Error details container."""
    message: str
    error_code: int
    timestamp: datetime = field(default_factory=datetime.now)
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelMetrics:
    """Metrics for model performance tracking."""
    model_name: str
    inference_time: float
    memory_usage: float 
    confidence_score: float
    frame_count: int
    error_rate: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class ValidationResult:
    """Validation result container."""
    valid: bool
    error: Optional[ErrorDetails] = None
    warnings: List[str] = field(default_factory=list)

@dataclass
class ModelConfig:
    """Model configuration container."""
    model_type: str
    weights_path: Path
    device: str = "cuda"
    batch_size: int = 32
    num_workers: int = 4
    cache_size: int = 100
    optimize_memory: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProcessingStats:
    """Processing statistics container."""
    inference_time: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    batch_size: int = 1
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class VideoFrame:
    """Video frame container."""
    data: ImageTensor
    frame_number: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def cleanup(self) -> None:
        """Clean up frame resources."""
        del self.data
        torch.cuda.empty_cache()

@dataclass
class AudioSegment:
    """Audio segment container."""
    data: AudioTensor
    sample_rate: int
    start_time: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def cleanup(self) -> None:
        """Clean up audio resources."""
        del self.data
        torch.cuda.empty_cache()

@dataclass
class DetectionResult:
    """Add frontend-friendly error reporting"""
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': 'success',
            'is_fake': self.is_fake,
            'confidence': self.confidence,
            'model_scores': self.model_scores,
            'processing_time': self.processing_time,
            'frame_scores': self.frame_scores,
            'audio_scores': self.audio_scores,
            'metadata': self.metadata,
            'error': None
        }

@dataclass
class CacheConfig:
    """Cache configuration container."""
    max_size: int = 1024 * 1024 * 1024  # 1GB
    ttl: int = 3600  # 1 hour
    cleanup_interval: int = 300  # 5 minutes
    persistent: bool = False
    compression: bool = True
    path: Optional[Path] = None

@dataclass
class DatabaseConfig:
    """Database configuration container."""
    url: str
    max_connections: int = 10
    timeout: int = 30
    retry_limit: int = 3
    cache_enabled: bool = True
    debug: bool = False

class ProcessingMode(Enum):
    """Processing mode options."""
    SYNC = auto()
    ASYNC = auto()
    BATCH = auto()

class ModelPrecision(Enum):
    """Model precision options."""
    FP32 = "float32"
    FP16 = "float16"
    INT8 = "int8"

class CacheStrategy(Enum):
    """Cache strategy options."""
    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    FIFO = "first_in_first_out"

@dataclass
class ResourceMonitor:
    """Resource usage monitor."""
    gpu_memory_used: int = 0
    cpu_memory_used: int = 0
    disk_usage: int = 0
    process_time: float = 0.0
    
    def log_usage(self) -> None:
        """Log current resource usage."""
        if torch.cuda.is_available():
            self.gpu_memory_used = torch.cuda.memory_allocated()
        # Add CPU/disk monitoring

class ErrorResult:
    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': 'error',
            'error': {
                'code': self.error_code,
                'message': self.message,
                'details': self.details
            }
        }

@dataclass
class OptimizationConfig:
    """Model optimization configuration."""
    precision: ModelPrecision = ModelPrecision.FP32
    dynamic_shapes: bool = True
    gradient_checkpointing: bool = False
    channels_last: bool = True
    compile_model: bool = True
    torch_script: bool = False

def validate_config(config: Any) -> ValidationResult:
    """Validate configuration object."""
    try:
        if not isinstance(config, (ModelConfig, CacheConfig, DatabaseConfig)):
            return ValidationResult(
                valid=False,
                error=ErrorDetails(
                    message="Invalid config type",
                    error_code=5001
                )
            )
        return ValidationResult(valid=True)
    except Exception as e:
        return ValidationResult(
            valid=False,
            error=ErrorDetails(
                message=f"Validation error: {str(e)}",
                error_code=5002
            )
        )

# Type variable for generic type hints
T = TypeVar('T')

class MediaType(str, Enum):
    """Supported media types."""
    VIDEO = "video"
    AUDIO = "audio"

class ModelType(str, Enum):
    """Supported model types."""
    EFFICIENTNET = "efficientnet"
    WAV2VEC2 = "wav2vec2"

class DetectionResult(TypedDict):
    """Detection result type."""
    prediction: float  # 0-1 score
    confidence: float  # 0-1 confidence
    processing_time: float  # seconds
    timestamp: datetime
    metadata: Dict[str, Any]

class ModelMetrics(TypedDict):
    """Model performance metrics type."""
    inference_time: float  # seconds
    memory_usage: float  # MB
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class DetectionOptions:
    """Detection options with validation."""
    model_type: ModelType
    media_type: MediaType
    batch_size: int = 32
    use_cache: bool = True
    progress_callback: Optional[callable] = None
    
    def __post_init__(self):
        """Validate options after initialization."""
        try:
            # Validate model type
            if not isinstance(self.model_type, ModelType):
                raise ValueError(f"Invalid model type: {self.model_type}")
                
            # Validate media type
            if not isinstance(self.media_type, MediaType):
                raise ValueError(f"Invalid media type: {self.media_type}")
                
            # Validate batch size
            if not isinstance(self.batch_size, int) or self.batch_size < 1:
                raise ValueError(f"Invalid batch size: {self.batch_size}")
                
            # Validate progress callback
            if self.progress_callback is not None:
                if not callable(self.progress_callback):
                    raise ValueError("Progress callback must be callable")
                    
        except Exception as e:
            logger.error(f"Detection options validation failed: {e}")
            raise ValidationError(
                message="Invalid detection options",
                error_code=8000,
                operation="validate_options",
                details={'error': str(e)}
            )

@dataclass
class ModelConfig:
    """Model configuration with validation."""
    model_type: ModelType
    weights_path: str
    input_size: Tuple[int, int]
    batch_size: int = 32
    device: str = "cuda"
    use_mixed_precision: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        try:
            # Validate model type
            if not isinstance(self.model_type, ModelType):
                raise ValueError(f"Invalid model type: {self.model_type}")
                
            # Validate weights path
            if not isinstance(self.weights_path, str):
                raise ValueError(f"Invalid weights path: {self.weights_path}")
                
            # Validate input size
            if not isinstance(self.input_size, tuple) or len(self.input_size) != 2:
                raise ValueError(f"Invalid input size: {self.input_size}")
                
            if not all(isinstance(x, int) and x > 0 for x in self.input_size):
                raise ValueError(f"Input size dimensions must be positive integers: {self.input_size}")
                
            # Validate batch size
            if not isinstance(self.batch_size, int) or self.batch_size < 1:
                raise ValueError(f"Invalid batch size: {self.batch_size}")
                
            # Validate device
            if not isinstance(self.device, str):
                raise ValueError(f"Invalid device: {self.device}")
                
            if self.device not in ["cuda", "cpu"]:
                raise ValueError(f"Unsupported device: {self.device}")
                
        except Exception as e:
            logger.error(f"Model configuration validation failed: {e}")
            raise ValidationError(
                message="Invalid model configuration",
                error_code=8001,
                operation="validate_config",
                details={'error': str(e)}
            )

@dataclass
class CacheConfig:
    """Cache configuration with validation."""
    max_size: int = 1000  # MB
    ttl: int = 3600  # seconds
    cleanup_interval: int = 300  # seconds
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        try:
            # Validate max size
            if not isinstance(self.max_size, int) or self.max_size < 1:
                raise ValueError(f"Invalid max size: {self.max_size}")
                
            # Validate TTL
            if not isinstance(self.ttl, int) or self.ttl < 0:
                raise ValueError(f"Invalid TTL: {self.ttl}")
                
            # Validate cleanup interval
            if not isinstance(self.cleanup_interval, int) or self.cleanup_interval < 1:
                raise ValueError(f"Invalid cleanup interval: {self.cleanup_interval}")
                
        except Exception as e:
            logger.error(f"Cache configuration validation failed: {e}")
            raise ValidationError(
                message="Invalid cache configuration",
                error_code=8002,
                operation="validate_config",
                details={'error': str(e)}
            )

@dataclass
class DatabaseConfig:
    """Database configuration with validation."""
    path: str = "data/detection.db"
    backup_dir: str = "data/backups"
    max_backups: int = 5
    max_size_mb: int = 100
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        try:
            # Validate path
            if not isinstance(self.path, str):
                raise ValueError(f"Invalid database path: {self.path}")
                
            # Validate backup directory
            if not isinstance(self.backup_dir, str):
                raise ValueError(f"Invalid backup directory: {self.backup_dir}")
                
            # Validate max backups
            if not isinstance(self.max_backups, int) or self.max_backups < 1:
                raise ValueError(f"Invalid max backups: {self.max_backups}")
                
            # Validate max size
            if not isinstance(self.max_size_mb, int) or self.max_size_mb < 1:
                raise ValueError(f"Invalid max size: {self.max_size_mb}")
                
        except Exception as e:
            logger.error(f"Database configuration validation failed: {e}")
            raise ValidationError(
                message="Invalid database configuration",
                error_code=8003,
                operation="validate_config",
                details={'error': str(e)}
            )

def validate_detection_result(result: DetectionResult) -> None:
    """
    Validate detection result.

    Args:
        result: Detection result to validate
    """
    try:
        # Validate prediction
        if not isinstance(result['prediction'], (int, float)):
            raise ValueError(f"Invalid prediction type: {type(result['prediction'])}")
            
        if not 0 <= result['prediction'] <= 1:
            raise ValueError(f"Invalid prediction value: {result['prediction']}")
            
        # Validate confidence
        if not isinstance(result['confidence'], (int, float)):
            raise ValueError(f"Invalid confidence type: {type(result['confidence'])}")
            
        if not 0 <= result['confidence'] <= 1:
            raise ValueError(f"Invalid confidence value: {result['confidence']}")
            
        # Validate processing time
        if not isinstance(result['processing_time'], (int, float)):
            raise ValueError(f"Invalid processing time type: {type(result['processing_time'])}")
            
        if result['processing_time'] < 0:
            raise ValueError(f"Invalid processing time value: {result['processing_time']}")
            
        # Validate timestamp
        if not isinstance(result['timestamp'], datetime):
            raise ValueError(f"Invalid timestamp type: {type(result['timestamp'])}")
            
        # Validate metadata
        if not isinstance(result['metadata'], dict):
            raise ValueError(f"Invalid metadata type: {type(result['metadata'])}")
            
    except Exception as e:
        logger.error(f"Detection result validation failed: {e}")
        raise ValidationError(
            message="Invalid detection result",
            error_code=8004,
            operation="validate_result",
            details={'error': str(e)}
        )

def validate_model_metrics(metrics: ModelMetrics) -> None:
    """
    Validate model metrics.

    Args:
        metrics: Model metrics to validate
    """
    try:
        # Validate inference time
        if not isinstance(metrics['inference_time'], (int, float)):
            raise ValueError(f"Invalid inference time type: {type(metrics['inference_time'])}")
            
        if metrics['inference_time'] < 0:
            raise ValueError(f"Invalid inference time value: {metrics['inference_time']}")
            
        # Validate memory usage
        if not isinstance(metrics['memory_usage'], (int, float)):
            raise ValueError(f"Invalid memory usage type: {type(metrics['memory_usage'])}")
            
        if metrics['memory_usage'] < 0:
            raise ValueError(f"Invalid memory usage value: {metrics['memory_usage']}")
            
        # Validate timestamp
        if not isinstance(metrics['timestamp'], datetime):
            raise ValueError(f"Invalid timestamp type: {type(metrics['timestamp'])}")
            
        # Validate metadata
        if not isinstance(metrics['metadata'], dict):
            raise ValueError(f"Invalid metadata type: {type(metrics['metadata'])}")
            
    except Exception as e:
        logger.error(f"Model metrics validation failed: {e}")
        raise ValidationError(
            message="Invalid model metrics",
            error_code=8005,
            operation="validate_metrics",
            details={'error': str(e)}
        )

class ModelOutput(dict):
    """Stub for model output."""
    pass

class ImageTensor:
    """Stub for image tensor."""
    pass

__all__ = [
    # Data containers
    'ErrorDetails',
    'ValidationResult',
    'ModelConfig',
    'ProcessingStats',
    'VideoFrame',
    'AudioSegment',
    'DetectionResult',
    'CacheConfig',
    'DatabaseConfig',
    'ResourceMonitor',
    'OptimizationConfig',
    
    # Enums
    'ProcessingMode',
    'ModelPrecision',
    'CacheStrategy',
    
    # Type aliases
    'ImageTensor',
    'AudioTensor',
    'ModelWeights',
    'JsonDict',
    
    # Functions
    'validate_config',
    
    # Type vars
    'T'
]