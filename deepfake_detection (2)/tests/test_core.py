import pytest
import torch
from pathlib import Path
from backend.core.detector import Detector
from backend.database.database import Database
from backend.utils.model_utils import ModelUtils

@pytest.fixture
def detector():
    """Fixture to create detector instance."""
    return Detector()

@pytest.fixture
def sample_video(tmp_path):
    """Fixture to create a temporary test video."""
    video_path = tmp_path / "test.mp4"
    # You can add code here to create a sample video file
    return str(video_path)

def test_detector_initialization(detector):
    """Test detector initialization."""
    assert detector is not None
    assert hasattr(detector, '_video_handler')
    assert hasattr(detector, '_audio_handler')

def test_detection_workflow(detector, sample_video):
    """Test complete detection workflow."""
    try:
        result = detector.detect(
            sample_video,
            progress_callback=lambda x: print(f"Progress: {x}%")
        )
        
        # Check result structure
        assert hasattr(result, 'is_fake')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'model_scores')
        assert 0 <= result.confidence <= 1
        
    except Exception as e:
        pytest.fail(f"Detection failed: {e}")

def test_model_loading():
    """Test model loading and resource management."""
    try:
        detector = Detector()
        assert detector._models is not None
        
        # Check GPU memory cleanup
        if torch.cuda.is_available():
            initial_mem = torch.cuda.memory_allocated()
            detector.cleanup()
            final_mem = torch.cuda.memory_allocated()
            assert final_mem <= initial_mem
            
    except Exception as e:
        pytest.fail(f"Model loading failed: {e}")