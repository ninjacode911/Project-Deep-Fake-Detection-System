"""
DeepFake Detection System - Test Package Initialization
Created: 2025-06-07
Author: ninjacode911

This module initializes testing environment with proper configuration,
logging, fixtures, and utilities for comprehensive testing.
"""

import os
import sys
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure test logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(tempfile.gettempdir(), 'test_deepfake.log'))
    ]
)

logger = logging.getLogger('deepfake.tests')

# Suppress specific warnings during tests
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Test configuration
TEST_CONFIG: Dict[str, Any] = {
    'TEMP_DIR': tempfile.mkdtemp(prefix='deepfake_test_'),
    'CACHE_DIR': tempfile.mkdtemp(prefix='deepfake_test_cache_'),
    'MODEL_WEIGHTS_DIR': os.path.join(PROJECT_ROOT, 'models', 'weights'),
    'TEST_DATA_DIR': os.path.join(PROJECT_ROOT, 'tests', 'data'),
    'MAX_CACHE_SIZE': 100 * 1024 * 1024,  # 100MB
    'DEBUG': True
}

def cleanup_test_env() -> None:
    """Clean up temporary test directories and resources."""
    try:
        # Clean up temp directories
        if os.path.exists(TEST_CONFIG['TEMP_DIR']):
            for root, dirs, files in os.walk(TEST_CONFIG['TEMP_DIR'], topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(TEST_CONFIG['TEMP_DIR'])
            
        if os.path.exists(TEST_CONFIG['CACHE_DIR']):
            for root, dirs, files in os.walk(TEST_CONFIG['CACHE_DIR'], topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(TEST_CONFIG['CACHE_DIR'])
            
    except Exception as e:
        logger.error(f"Error cleaning up test environment: {str(e)}")
        raise

def setup_test_env() -> None:
    """Initialize test environment with required directories and resources."""
    try:
        # Create necessary directories
        os.makedirs(TEST_CONFIG['TEMP_DIR'], exist_ok=True)
        os.makedirs(TEST_CONFIG['CACHE_DIR'], exist_ok=True)
        os.makedirs(TEST_CONFIG['TEST_DATA_DIR'], exist_ok=True)
        
        # Initialize test resources
        logger.info("Test environment initialized successfully")
        
    except Exception as e:
        logger.error(f"Error setting up test environment: {str(e)}")
        raise

class TestError(Exception):
    """Custom exception for test-related errors."""
    def __init__(self, message: str, error_code: int = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

# Initialize test environment on import
setup_test_env()

# Register cleanup on exit
import atexit
atexit.register(cleanup_test_env)

__all__ = [
    'TEST_CONFIG',
    'setup_test_env',
    'cleanup_test_env',
    'TestError',
    'logger'
]