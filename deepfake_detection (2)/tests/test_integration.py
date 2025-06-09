import unittest
import sys
import torch
from pathlib import Path
from PyQt6.QtWidgets import QApplication

from backend.core.detector import Detector
from frontend.widgets.main_window import DeepFakeDetectionSystem
from frontend.api.thread import AnalysisThread

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize test environment."""
        cls.app = QApplication.instance() or QApplication(sys.argv)
        cls.test_video = "test_data/sample.mp4"
        
    def setUp(self):
        """Setup test components."""
        self.window = DeepFakeDetectionSystem()
        self.detector = Detector()
        
    def tearDown(self):
        """Cleanup resources."""
        self.window.close()
        if hasattr(self, 'detector'):
            self.detector.cleanup()
        
    def test_analysis_workflow(self):
        """Test complete analysis workflow."""
        if not Path(self.test_video).exists():
            self.skipTest("Test video not found")
            
        # Start analysis
        thread = AnalysisThread(self.detector, self.test_video)
        thread.start()
        thread.wait()  # Wait for completion
        
        # Verify results
        self.assertTrue(hasattr(thread, 'result'))
        self.assertIsNotNone(thread.result)
        
    def test_ui_updates(self):
        """Test UI updates during analysis."""
        progress_values = []
        
        def track_progress(value):
            progress_values.append(value)
            
        if not Path(self.test_video).exists():
            self.skipTest("Test video not found")
            
        thread = AnalysisThread(self.detector, self.test_video)
        thread.progress_updated.connect(track_progress)
        thread.start()
        thread.wait()
        
        # Verify progress updates
        self.assertTrue(len(progress_values) > 0)
        self.assertEqual(progress_values[-1], 100)

if __name__ == "__main__":
    unittest.main()