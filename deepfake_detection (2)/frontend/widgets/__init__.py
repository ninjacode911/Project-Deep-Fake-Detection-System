#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepFake Detection System - Frontend Widgets Package
Created on: 2025-04-28 07:50:42 UTC
Last Modified: 2025-06-07 14:30:00 UTC
Author: ninjacode911

This module manages widget initialization, caching, and cleanup for the DeepFake Detection System.
"""

import logging
from typing import List, Type, Dict, Any
from pathlib import Path
from PyQt6.QtWidgets import QWidget

# Core widgets
from .video_preview import VideoPreviewWidget
from .video_controls import VideoControlsWidget
from .timeline import TimelineWidget

# Analysis widgets
from .classification import ClassificationWidget
from .manipulation import ManipulationWidget
from .heatmap import HeatmapWidget
from .detected_issues import DetectedIssuesWidget

# File handling widgets
from .file_manager import FileManager
from .recent_files import RecentFilesWidget

logger = logging.getLogger(__name__)

# Widget groups for organized management
CORE_WIDGETS: List[Type[QWidget]] = [
    VideoPreviewWidget,  
    VideoControlsWidget,
    TimelineWidget
]

ANALYSIS_WIDGETS: List[Type[QWidget]] = [
    ClassificationWidget,
    ManipulationWidget,
    HeatmapWidget,
    DetectedIssuesWidget
]

FILE_WIDGETS: List[Type[QWidget]] = [
    FileManager,
    RecentFilesWidget
]

class WidgetManager:
    """Centralized widget management system."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not WidgetManager._initialized:
            self._active_widgets: Dict[str, QWidget] = {}
            self._cache_status: Dict[str, bool] = {}
            WidgetManager._initialized = True
            logger.info("Widget manager initialized")
            
    def register_widget(self, widget: QWidget) -> None:
        """Register a widget for management."""
        try:
            widget_name = widget.__class__.__name__
            self._active_widgets[widget_name] = widget
            self._cache_status[widget_name] = False
            logger.debug(f"Registered widget: {widget_name}")
        except Exception as e:
            logger.error(f"Failed to register widget: {e}")
            
    def cleanup_all(self) -> None:
        """Clean up all managed widgets."""
        try:
            for widget_name, widget in self._active_widgets.items():
                try:
                    if hasattr(widget, 'cleanup'):
                        widget.cleanup()
                    logger.debug(f"Cleaned up widget: {widget_name}")
                except Exception as e:
                    logger.error(f"Failed to cleanup {widget_name}: {e}")
            self._active_widgets.clear()
            self._cache_status.clear()
        except Exception as e:
            logger.error(f"Failed to perform complete cleanup: {e}")
            
    def invalidate_caches(self) -> None:
        """Invalidate caches of all widgets."""
        try:
            for widget_name, widget in self._active_widgets.items():
                try:
                    if hasattr(widget, 'clear_cache'):
                        widget.clear_cache()
                        self._cache_status[widget_name] = False
                    logger.debug(f"Invalidated cache for: {widget_name}")
                except Exception as e:
                    logger.error(f"Failed to invalidate cache for {widget_name}: {e}")
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            
    def sync_widget_states(self) -> None:
        """Synchronize states across all widgets."""
        try:
            for widget in self._active_widgets.values():
                if hasattr(widget, 'reset_state'):
                    widget.reset_state()
            logger.debug("Widget states synchronized")
        except Exception as e:
            logger.error(f"State synchronization failed: {e}")

# Initialize widget manager
widget_manager = WidgetManager()

__all__ = [
    # Widget Groups
    'CORE_WIDGETS',
    'ANALYSIS_WIDGETS',
    'FILE_WIDGETS',
    
    # Core Widgets
    'VideoPreviewWidget',
    'VideoControlsWidget',
    'TimelineWidget',
    
    # Analysis Widgets
    'ClassificationWidget',
    'ManipulationWidget',
    'HeatmapWidget',
    'DetectedIssuesWidget',
    
    # File Handling Widgets
    'FileManager',
    'RecentFilesWidget',
    
    # Widget Management
    'widget_manager',
    'WidgetManager'
]