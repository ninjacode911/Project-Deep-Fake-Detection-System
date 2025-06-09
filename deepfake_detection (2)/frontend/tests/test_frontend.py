import pytest
import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

# Add the frontend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from widgets.main_window import MainWindow
from api.client import APIClient
from resources.theme_manager import ThemeManager
from utils.logger import setup_logger
from utils.error_handler import ErrorHandler

@pytest.fixture
def app():
    """Create a QApplication instance."""
    return QApplication(sys.argv)

@pytest.fixture
def main_window(app):
    """Create and return the main window instance."""
    window = MainWindow()
    window.show()
    return window

@pytest.fixture
def api_client():
    """Create and return an API client instance."""
    return APIClient()

@pytest.fixture
def theme_manager():
    """Create and return a theme manager instance."""
    return ThemeManager()

def test_application_startup(app, main_window):
    """Test that the application starts up correctly."""
    assert main_window.isVisible()
    assert main_window.windowTitle() == "Deepfake Detection"
    
    # Check if all main widgets are present
    assert main_window.video_preview is not None
    assert main_window.timeline is not None
    assert main_window.controls is not None
    assert main_window.classification is not None
    assert main_window.heatmap is not None
    assert main_window.detected_issues is not None

def test_theme_switching(app, main_window, theme_manager):
    """Test theme switching functionality."""
    # Test dark theme
    theme_manager.set_theme("dark")
    assert theme_manager.current_theme == "dark"
    
    # Test light theme
    theme_manager.set_theme("light")
    assert theme_manager.current_theme == "light"

def test_video_loading(app, main_window):
    """Test video loading functionality."""
    # Simulate loading a test video
    test_video_path = "test_data/sample.mp4"
    if os.path.exists(test_video_path):
        main_window.load_video(test_video_path)
        assert main_window.video_preview.is_playing is False
        assert main_window.timeline.duration > 0

def test_api_connection(app, api_client):
    """Test API client connection."""
    # Test API connection
    try:
        api_client.check_connection()
        assert True
    except Exception as e:
        pytest.fail(f"API connection failed: {str(e)}")

def test_error_handling(app, main_window):
    """Test error handling functionality."""
    error_handler = ErrorHandler()
    
    # Test error logging
    test_error = "Test error message"
    error_handler.handle_error(test_error)
    
    # Check if error was logged
    log_file = "logs/error.log"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log_content = f.read()
            assert test_error in log_content

def test_resource_loading(app, main_window):
    """Test resource loading functionality."""
    # Check if resources are loaded
    assert os.path.exists("resources/resources_rc.py")
    assert os.path.exists("resources/icons/chevron-down.svg")
    assert os.path.exists("resources/styles/dark_theme.qss")

def test_widget_interactions(app, main_window):
    """Test widget interactions."""
    # Test video controls
    QTest.mouseClick(main_window.controls.play_button, Qt.MouseButton.LeftButton)
    assert main_window.video_preview.is_playing is True
    
    QTest.mouseClick(main_window.controls.pause_button, Qt.MouseButton.LeftButton)
    assert main_window.video_preview.is_playing is False
    
    # Test timeline interaction
    QTest.mouseClick(main_window.timeline, Qt.MouseButton.LeftButton)
    assert main_window.video_preview.current_frame > 0

def test_file_management(app, main_window):
    """Test file management functionality."""
    # Test recent files
    test_file = "test_data/sample.mp4"
    if os.path.exists(test_file):
        main_window.file_manager.add_recent_file(test_file)
        recent_files = main_window.file_manager.get_recent_files()
        assert test_file in recent_files

def test_cleanup(app, main_window):
    """Test cleanup functionality."""
    # Test application cleanup
    main_window.close()
    assert not main_window.isVisible()

if __name__ == "__main__":
    pytest.main([__file__]) 