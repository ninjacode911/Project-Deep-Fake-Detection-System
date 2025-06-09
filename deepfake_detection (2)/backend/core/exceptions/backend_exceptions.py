"""
DeepFake Detection System - Backend Exceptions
Created: 2025-06-07
Author: ninjacode911

This module defines custom exceptions for the backend with proper error handling
and resource management.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class BackendError(Exception):
    """Base exception for backend errors."""
    def __init__(self, message: str, error_code: int = 0, component: str = "", operation: str = "", details: dict = None):
        self.message = message
        self.error_code = error_code
        self.component = component
        self.operation = operation
        self.details = details or {}
        super().__init__(self.message)

class ResourceError(BackendError):
    """Exception raised for resource-related errors."""
    def __init__(self, message: str, resource_type: str = "", details: dict = None):
        super().__init__(
            message=message,
            error_code=1001,
            component="Resource",
            operation="management",
            details=details or {}
        )
        self.resource_type = resource_type

class ModelError(BackendError):
    """Exception raised for model-related errors."""
    def __init__(self, message: str, model_name: str = "", details: dict = None):
        super().__init__(
            message=message,
            error_code=1002,
            component="Model",
            operation="inference",
            details=details or {}
        )
        self.model_name = model_name

class VideoError(BackendError):
    """Exception for video processing errors."""
    
    def __init__(
        self,
        message: str,
        error_code: int,
        operation: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize video error."""
        super().__init__(message, error_code, operation, details)
        
        # Validate error code range
        if not 6000 <= error_code < 7000:
            raise ValueError(f"Invalid video error code: {error_code}")

class AudioError(BackendError):
    """Exception for audio processing errors."""
    
    def __init__(
        self,
        message: str,
        error_code: int,
        operation: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize audio error."""
        super().__init__(message, error_code, operation, details)
        
        # Validate error code range
        if not 7000 <= error_code < 8000:
            raise ValueError(f"Invalid audio error code: {error_code}")

class CacheError(BackendError):
    """Exception for cache-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: int,
        operation: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize cache error."""
        super().__init__(message, error_code, operation, details)
        
        # Validate error code range
        if not 8000 <= error_code < 9000:
            raise ValueError(f"Invalid cache error code: {error_code}")

class DatabaseError(BackendError):
    """Exception raised for database-related errors."""
    def __init__(self, message: str, operation: str = "", details: dict = None):
        super().__init__(
            message=message,
            error_code=1003,
            component="Database",
            operation=operation,
            details=details or {}
        )

class ThreadError(BackendError):
    """Exception for thread-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: int,
        operation: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize thread error."""
        super().__init__(message, error_code, operation, details)
        
        # Validate error code range
        if not 10000 <= error_code < 11000:
            raise ValueError(f"Invalid thread error code: {error_code}")

class ConfigError(BackendError):
    """Exception for configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: int,
        operation: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize configuration error."""
        super().__init__(message, error_code, operation, details)
        
        # Validate error code range
        if not 11000 <= error_code < 12000:
            raise ValueError(f"Invalid config error code: {error_code}")

class ValidationError(BackendError):
    """Exception for validation-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: int,
        operation: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize validation error."""
        super().__init__(message, error_code, operation, details)

        # Validate error code range
        if not 12000 <= error_code < 13000:
            raise ValueError(f"Invalid validation error code: {error_code}")

def validate_error_code(error_code: int, component: str) -> None:
    """
    Validate error code range for component.

    Args:
        error_code: Error code to validate
        component: Component name
    """
    try:
        # Define valid ranges
        ranges = {
            'model': (5000, 6000),
            'video': (6000, 7000),
            'audio': (7000, 8000),
            'cache': (8000, 9000),
            'database': (9000, 10000),
            'thread': (10000, 11000),
            'config': (11000, 12000),
            'validation': (12000, 13000)
}

        # Get range for component
        if component not in ranges:
            raise ValueError(f"Unknown component: {component}")
            
        start, end = ranges[component]
        
        # Validate range
        if not start <= error_code < end:
            raise ValueError(
                f"Invalid error code {error_code} for {component}. "
                f"Must be in range [{start}, {end})"
            )
            
    except Exception as e:
        logger.error(f"Error code validation failed: {e}")
        raise ValidationError(
            message="Invalid error code",
            error_code=12000,
            operation="validate_error_code",
            details={'error': str(e), 'component': component, 'error_code': error_code}
        )

def get_error_context(error: BackendError) -> Dict[str, Any]:
    """
    Get error context for logging and debugging.

    Args:
        error: Backend error

    Returns:
        Error context dictionary
    """
    try:
        return {
            'message': error.message,
            'error_code': error.error_code,
            'operation': error.operation,
            'details': error.details,
            'timestamp': error.timestamp.isoformat(),
            'type': error.__class__.__name__,
            'traceback': traceback.format_exc()
        }
        
    except Exception as e:
        logger.error(f"Failed to get error context: {e}")
        return {
            'message': str(error),
            'error': str(e)
        }

def handle_error(error: Exception) -> BackendError:
    """
    Convert exception to appropriate backend error.

    Args:
        error: Exception to handle

    Returns:
        Backend error
    """
    try:
        if isinstance(error, BackendError):
            return error
            
        # Map common exceptions to backend errors
        if isinstance(error, ValueError):
            return ValidationError(
                message=str(error),
                error_code=12001,
                operation="handle_error",
                details={'error': str(error)}
            )
            
        if isinstance(error, FileNotFoundError):
            return ConfigError(
                message=str(error),
                error_code=11001,
                operation="handle_error",
                details={'error': str(error)}
            )
            
        if isinstance(error, PermissionError):
            return ConfigError(
                message=str(error),
                error_code=11002,
                operation="handle_error",
                details={'error': str(error)}
            )
            
        # Default to generic backend error
        return BackendError(
            message=str(error),
            error_code=5000,
            operation="handle_error",
            details={'error': str(error)}
        )
        
    except Exception as e:
        logger.error(f"Error handling failed: {e}")
        return BackendError(
            message="Failed to handle error",
            error_code=5001,
            operation="handle_error",
            details={'error': str(e), 'original_error': str(error)}
        )