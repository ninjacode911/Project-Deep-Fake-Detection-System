"""
DeepFake Detection System - Test Configuration and Fixtures
Created: 2025-06-07
Author: ninjacode911

This module provides pytest fixtures and configuration for testing with
proper resource management and cleanup.
"""

import pytest
import torch
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from contextlib import contextmanager

from ..models.efficientnet import EfficientNetModel
from ..models.wav2vec2 import Wav2Vec2Model
from ..core.detector import DeepFakeDetector
from ..database import DatabaseManager
from . import TEST_CONFIG, logger

@pytest.fixture(scope="session")
def test_device() -> str:
    """Provide test device with fallback to CPU."""
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as e:
        logger.warning(f"Error checking CUDA availability: {e}, falling back to CPU")
        return "cpu"

@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Provide temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp(prefix="deepfake_test_"))
    try:
        yield temp_path
    finally:
        try:
            shutil.rmtree(temp_path)
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {e}")

@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return TEST_CONFIG

@pytest.fixture(scope="function")
def vision_model(test_device: str, temp_dir: Path) -> Generator[EfficientNetModel, None, None]:
    """Provide initialized vision model for testing."""
    model = None
    try:
        model = EfficientNetModel(
            config={
                'model_name': 'efficientnetv2_m',
                'num_classes': 2,
                'device': test_device,
                'cache_dir': str(temp_dir)
            }
        )
        yield model
    finally:
        if model is not None:
            try:
                model.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up vision model: {e}")

@pytest.fixture(scope="function")
def audio_model(test_device: str, temp_dir: Path) -> Generator[Wav2Vec2Model, None, None]:
    """Provide initialized audio model for testing."""
    model = None
    try:
        model = Wav2Vec2Model(
            weights_path=TEST_CONFIG['MODEL_WEIGHTS_DIR'] / "wav2vec2",
            device=test_device
        )
        yield model
    finally:
        if model is not None:
            try:
                model.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up audio model: {e}")

@pytest.fixture(scope="function")
def detector(
    vision_model: EfficientNetModel,
    audio_model: Wav2Vec2Model,
    test_config: Dict[str, Any]
) -> Generator[DeepFakeDetector, None, None]:
    """Provide initialized detector for testing."""
    detector_instance = None
    try:
        detector_instance = DeepFakeDetector(
            vision_model=vision_model,
            audio_model=audio_model,
            config=test_config
        )
        yield detector_instance
    finally:
        if detector_instance is not None:
            try:
                detector_instance.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up detector: {e}")

@pytest.fixture(scope="session")
def db_manager(temp_dir: Path) -> Generator[DatabaseManager, None, None]:
    """Provide database manager for testing."""
    manager = None
    try:
        db_path = temp_dir / "test.db"
        manager = DatabaseManager(
            db_path=str(db_path),
            max_connections=5
        )
        yield manager
    finally:
        if manager is not None:
            try:
                manager.cleanup()
                if db_path.exists():
                    db_path.unlink()
            except Exception as e:
                logger.error(f"Error cleaning up database: {e}")

@contextmanager
def catch_warnings_as_errors():
    """Context manager to treat warnings as errors during tests."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield

@pytest.fixture(autouse=True)
def cleanup_cuda_memory():
    """Automatically cleanup CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            logger.error(f"Error cleaning up CUDA memory: {e}")

def pytest_configure(config):
    """Configure pytest environment."""
    config.addinivalue_line(
        "markers",
        "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA is not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )