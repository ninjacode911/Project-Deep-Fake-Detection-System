#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepFake Detection System - Frontend Package Initialization
Created on: 2025-04-30 17:30:00 UTC
Last Modified: 2025-06-07 14:30:00 UTC
Author: ninjacode911
"""

import sys
import logging
import atexit
import gc
from typing import List, Type, Dict
from pathlib import Path
from PyQt6.QtWidgets import QWidget

__version__ = "1.0.0"
__author__ = "ninjacode911"

logger = logging.getLogger(__name__)

# Track initialized components for cleanup
_initialized_components: Dict[str, bool] = {}
_widget_instances: List[QWidget] = []

def _cleanup_components() -> None:
    """Cleanup all initialized components on exit."""
    try:
        # Cleanup widgets in reverse initialization order
        for widget in reversed(_widget_instances):
            if hasattr(widget, 'cleanup'):
                widget.cleanup()
        _widget_instances.clear()
        
        # Clear caches
        gc.collect()
        logger.info("Frontend components cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

# Register cleanup handler
atexit.register(_cleanup_components)

# Initialize core components
try:
    # Import utilities first
    from .utils import setup_logger, ErrorHandler, LogLevel
    from .config import config_manager
    from .resources import theme_manager
    _initialized_components['utilities'] = True
    
    # Import widgets in dependency order
    from .widgets.file_manager import FileManager
    from .widgets.video_preview import VideoPreviewWidget
    from .widgets.video_controls import VideoControlsWidget
    from .widgets.timeline import TimelineWidget
    from .widgets.classification import ClassificationWidget
    from .widgets.manipulation import ManipulationWidget
    from .widgets.heatmap import HeatmapWidget
    from .widgets.detected_issues import DetectedIssuesWidget
    from .widgets.recent_files import RecentFilesWidget
    from .widgets.main_window import DeepFakeDetectionSystem
    _initialized_components['widgets'] = True

    # Import API components
    from .api import APIClient, AnalysisThread
    _initialized_components['api'] = True
    
    # Define component groups with proper typing
    CORE_WIDGETS: List[Type[QWidget]] = [
        DeepFakeDetectionSystem,
        VideoPreviewWidget,
        TimelineWidget
    ]

    ANALYSIS_WIDGETS: List[Type[QWidget]] = [
        ClassificationWidget,
        ManipulationWidget, 
        HeatmapWidget,
        DetectedIssuesWidget
    ]

    CONTROL_WIDGETS: List[Type[QWidget]] = [
        FileManager,
        VideoControlsWidget,
        RecentFilesWidget
    ]

    # Initialize centralized error handling
    error_handler = ErrorHandler()
    _initialized_components['error_handler'] = True
    
    # Configure logging with backup
    try:
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        backup_dir = log_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        setup_logger(
            filename=log_dir / "frontend.log",
            log_level=LogLevel.INFO,
            max_size=5_242_880,  # 5MB
            backup_count=3,
            backup_dir=backup_dir,
            log_to_console=True
        )
        _initialized_components['logging'] = True
        
    except Exception as e:
        print(f"Warning: Logging setup failed: {e}", file=sys.stderr)
        # Continue without logging
    
    # Validate initialization
    if not all(_initialized_components.values()):
        missing = [k for k, v in _initialized_components.items() if not v]
        raise ImportError(f"Components failed to initialize: {missing}")
        
    logger.info(f"Frontend package v{__version__} initialized successfully")

except ImportError as e:
    logger.critical(f"Failed to import required components: {e}")
    raise ImportError(f"Failed to import required components: {e}")
except Exception as e:
    logger.critical(f"Frontend initialization failed: {e}")
    raise RuntimeError(f"Frontend initialization failed: {e}")

# Public exports
__all__ = [
    # Main application
    'DeepFakeDetectionSystem',
    
    # Core utilities
    'config_manager',
    'theme_manager',
    'error_handler',
    
    # Widget groups
    'CORE_WIDGETS',
    'ANALYSIS_WIDGETS', 
    'CONTROL_WIDGETS',
    
    # Individual widgets
    'FileManager',
    'VideoPreviewWidget',
    'VideoControlsWidget',
    'TimelineWidget',
    'ClassificationWidget',
    'ManipulationWidget',
    'HeatmapWidget',
    'DetectedIssuesWidget',
    'RecentFilesWidget',
    
    # API components
    'APIClient',
    'AnalysisThread',
    
    # Version info
    '__version__',
    '__author__'
]

def cleanup():
    """Manual cleanup function if needed before exit."""
    _cleanup_components()