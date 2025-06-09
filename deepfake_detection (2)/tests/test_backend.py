import unittest
import torch
import numpy as np
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.core.detector import Detector
from backend.models.efficientnet import EfficientNetModel
from backend.models.wav2vec2 import Wav2Vec2Model
from backend.utils.model_utils import ModelUtils

class TestBackend(unittest.TestCase):
    def setUp(self):
        """Initialize test environment."""
        self.detector = Detector()
        self.test_video = "test_data/sample.mp4"

    def tearDown(self):
        """Cleanup resources."""
        if hasattr(self, 'detector'):
            self.detector.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_model_loading(self):
        """Test model initialization."""
        self.assertIsNotNone(self.detector)
        self.assertTrue(hasattr(self.detector, '_video_handler'))
        self.assertTrue(hasattr(self.detector, '_audio_handler'))

    def test_video_processing(self):
        """Test video preprocessing."""
        if not Path(self.test_video).exists():
            self.skipTest("Test video not found")
            
        result = self.detector.detect(self.test_video)
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'is_fake'))
        self.assertIsInstance(result.confidence, float)
        
    def test_memory_management(self):
        """Test memory handling."""
        if torch.cuda.is_available():
            initial_mem = torch.cuda.memory_allocated()
            _ = self.detector.detect(self.test_video)
            self.detector.cleanup()
            final_mem = torch.cuda.memory_allocated()
            self.assertLessEqual(final_mem, initial_mem * 1.1)

if __name__ == "__main__":
    unittest.main()