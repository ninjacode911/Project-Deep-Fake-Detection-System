"""
DeepFake Detection System - Backend Package
Created: 2025-06-07
Author: ninjacode911

This module initializes and exports core backend components with comprehensive
error handling, resource management and performance optimization.
"""
import torch 
import logging
import atexit
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import threading
# Configure base logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import core components
try:
    from .core.exceptions.backend_exceptions import BackendError
    from .core import (
        Detector, 
        VideoHandler,
        AudioHandler,
        core_manager
    )
    from .database import Database, DatabaseManager, db_manager
    from .models import ModelManager, model_manager
    from .utils import ResourceManager, CacheManager
    from .config import ConfigManager, config_manager
    from .types import ModelConfig, DetectionResult
except ImportError as e:
    logger.critical(f"Failed to import core components: {e}")
    raise ImportError(f"Failed to import core components: {e}")

class Backend:
    """Main backend interface with resource management."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls) -> 'Backend':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize backend components."""
        if self._initialized:
            return
            
        try:
            # Initialize managers
            self.config = config_manager
            self.db = db_manager
            self.models = model_manager
            self.core = core_manager
            
            # Initialize resource tracking
            self._resources: Dict[str, Any] = {}
            self._initialized = True
            
            logger.info("Backend initialized successfully")
            
        except Exception as e:
            logger.critical(f"Backend initialization failed: {e}")
            raise BackendError(
                message="Backend initialization failed",
                error_code=1000,
                component="Backend",
                operation="init",
                details={'error': str(e)}
            )

    _cleanup_lock = threading.Lock()
    
    def cleanup(self) -> None:
        """Clean up all backend resources."""
        try:
            # Clean up components in reverse initialization order
            if hasattr(self, 'core'):
                self.core.cleanup()
            if hasattr(self, 'models'):
                self.models.cleanup()
            if hasattr(self, 'db'):
                self.db.cleanup()
            if hasattr(self, 'config'):
                self.config.cleanup()
                
            # Clear resources
            self._resources.clear()
            
            logger.info("Backend resources cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Backend cleanup failed: {e}")
            raise BackendError(
                message="Backend cleanup failed",
                error_code=1001,
                component="Backend",
                operation="cleanup",
                details={'error': str(e)}
            )

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error in backend destructor: {e}")

# Initialize backend singleton
backend = Backend()

# Register cleanup handler
atexit.register(backend.cleanup)

# Version information
__version__ = "1.0.0"
__author__ = "ninjacode911"

# Export public interfaces
__all__ = [
    # Main interface
    'Backend',
    'backend',
    
    # Core components
    'Detector',
    'VideoHandler', 
    'AudioHandler',
    'core_manager',
    
    # Database
    'Database',
    'DatabaseManager',
    'db_manager',
    
    # Models
    'ModelManager',
    'model_manager',
    'ModelConfig',
    
    # Types
    'DetectionResult',
    
    # Errors
    'BackendError',
    
    # Version info
    '__version__',
    '__author__'
]