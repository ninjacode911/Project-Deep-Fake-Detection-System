"""
DeepFake Detection System - Test Package
Created: 2025-06-07
Author: ninjacode911
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
import shutil
import cv2

import threading
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure test logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/test.log')
    ]
)

# Test configuration
TEST_CONFIG = {
    'TEST_DATA_DIR': Path(PROJECT_ROOT) / 'test_data',
    'TEMP_DIR': Path(tempfile.mkdtemp(prefix='deepfake_test_')),
    'SAMPLE_VIDEO': Path(PROJECT_ROOT) / 'test_data' / 'sample.mp4',
    'CACHE_DIR': Path(tempfile.mkdtemp(prefix='deepfake_test_cache_')),
    'MAX_CACHE_SIZE': 100 * 1024 * 1024  # 100MB
}

def verify_test_data():
    """Verify test data exists and is valid."""
    try:
        # Create test directories
        TEST_CONFIG['TEST_DATA_DIR'].mkdir(exist_ok=True, parents=True)
        
        sample_video = TEST_CONFIG['SAMPLE_VIDEO']
        if not sample_video.exists():
            source = Path(PROJECT_ROOT) / 'test_video.mp4'
            if source.exists():
                shutil.copy(source, sample_video)
            else:
                raise FileNotFoundError(
                    f"Sample video not found at {sample_video} or {source}"
                )
        
        # Basic video validation
        cap = cv2.VideoCapture(str(sample_video))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {sample_video}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            raise ValueError(f"Invalid video file: {sample_video}")
            
        cap.release()
        logging.info(f"Test video verified: {sample_video} ({frame_count} frames)")
        
    except Exception as e:
        logging.error(f"Test data verification failed: {e}")
        raise

def cleanup():
    """Cleanup temporary test resources."""
    try:
        for key in ['TEMP_DIR', 'CACHE_DIR']:
            if TEST_CONFIG[key].exists():
                shutil.rmtree(TEST_CONFIG[key])
    except Exception as e:
        logging.error(f"Cleanup failed: {e}")

# Register cleanup
import atexit
atexit.register(cleanup)

# Create required directories
TEST_CONFIG['TEST_DATA_DIR'].mkdir(exist_ok=True, parents=True)

# Initialize test environment
verify_test_data()