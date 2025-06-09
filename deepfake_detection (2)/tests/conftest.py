"""
Test configuration and fixtures for pytest.
"""
import pytest
import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import QApplication

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for GUI tests."""
    app = QApplication.instance() or QApplication(sys.argv)
    yield app
    app.quit()

@pytest.fixture(scope="session")
def test_data_dir():
    """Get test data directory."""
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def sample_video_path(test_data_dir):
    """Get path to sample test video."""
    video_path = test_data_dir / "sample.mp4"
    if not video_path.exists():
        pytest.skip("Test video not found")
    return str(video_path)

@pytest.fixture(scope="session")
def detector():
    """Create detector instance for testing."""
    from backend.core.detector import Detector
    detector = Detector()
    yield detector
    detector.cleanup() 