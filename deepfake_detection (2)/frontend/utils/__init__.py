#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepFake Detection System - Frontend Utilities
Created on: 2025-04-28 07:50:42 UTC
Last Modified: 2025-06-08 13:00:00 UTC
Author: ninjacode911
"""
import sys
import os
import logging
from pathlib import Path
import tempfile
import gc

# Configure package logging
logger = logging.getLogger(__name__)

# Import utility modules
from .logger import setup_logger, LogLevel, LogFormatter
from .error_handler import ErrorHandler, ErrorLevel, handle_error

# Package exports
__all__ = [
    'setup_logger',
    'LogLevel',
    'LogFormatter',
    'ErrorHandler', 
    'ErrorLevel',
    'handle_error'
]

def initialize_logging():
    """Initialize frontend logging."""
    try:
        logger = logging.getLogger('frontend')
        logger.setLevel(logging.INFO)
        
        # File handler
        log_path = Path('logs/frontend/frontend.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Remove existing handlers to prevent duplicates
        logger.handlers.clear()
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info("Frontend logging initialized")
        
    except Exception as e:
        print(f"Failed to initialize logging: {e}", file=sys.stderr)
        sys.exit(1)

# Initialize logging
initialize_logging()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Global cleanup function
def cleanup_resources():
    """Cleanup application resources."""
    try:
        # Clear any temporary files
        temp_dir = Path(tempfile.gettempdir()) / "deepfake_detection"
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
        
        # Force garbage collection
        gc.collect()
        
        # Get logger instance
        logger = logging.getLogger('frontend')
        logger.info("Frontend resources cleaned up")
    except Exception as e:
        # Use print as fallback if logger is not available
        print(f"Cleanup failed: {e}", file=sys.stderr)

# Register cleanup
import atexit
atexit.register(cleanup_resources)