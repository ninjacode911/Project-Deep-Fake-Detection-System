import unittest
import sys
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt

from frontend.widgets.main_window import DeepFakeDetectionSystem
from frontend.widgets.file_manager import FileManager
from frontend.widgets.video_preview import VideoPreview

class TestFrontend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create QApplication instance."""
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        """Initialize test components."""
        self.window = DeepFakeDetectionSystem()
        
    def tearDown(self):
        """Cleanup resources."""
        self.window.close()
        
    def test_window_initialization(self):
        """Test main window setup."""
        self.assertIsNotNone(self.window)
        self.assertEqual(self.window.windowTitle(), "DeepFake Detection System")
        
    def test_file_manager(self):
        """Test file manager widget."""
        file_manager = self.window.findChild(FileManager)
        self.assertIsNotNone(file_manager)
        
    def test_video_preview(self):
        """Test video preview widget."""
        preview = self.window.findChild(VideoPreview)
        self.assertIsNotNone(preview)
        
    def test_drag_drop(self):
        """Test drag and drop functionality."""
        file_manager = self.window.findChild(FileManager)
        self.assertTrue(file_manager.acceptDrops())

if __name__ == "__main__":
    unittest.main()