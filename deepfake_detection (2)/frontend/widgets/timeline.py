#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Timeline widget for video navigation with smooth animations
Created on: 2025-04-28 07:50:42 UTC
Last Modified: 2025-06-07 14:30:00 UTC
Author: ninjacode911
"""

import logging
from typing import List, Optional, Dict
from enum import Enum, auto
import gc

from PyQt6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, 
    QPoint, QRectF, pyqtSignal as Signal, 
    pyqtProperty, QTimer, QMutex
)
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, 
    QLabel, QFrame, QProgressBar
)
from PyQt6.QtGui import (
    QPainter, QPaintEvent, QColor, QPen, 
    QBrush, QLinearGradient, QResizeEvent,
    QMouseEvent
)

logger = logging.getLogger(__name__)

class TimelineState(Enum):
    """Timeline widget states."""
    INITIALIZING = auto()
    READY = auto()
    SEEKING = auto()
    LOADING = auto()
    ERROR = auto()
    CLEANED = auto()

class TimelineWidget(QWidget):
    """Enhanced timeline widget with smooth animations."""
    
    # Signals
    position_changed = Signal(int)    # Frame number
    seek_requested = Signal(int)      # Frame number
    error_occurred = Signal(str)      # Error message
    state_changed = Signal(str)       # Current state
    animation_completed = Signal()    # Animation completion

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize timeline widget with enhanced error handling."""
        try:
            super().__init__(parent)
            self._state = TimelineState.INITIALIZING
            
            # Thread safety
            self._mutex = QMutex()
            
            # Configuration
            self._frame_count = 0
            self._current_frame = 0
            self._marker_position = 0.0
            self._anomaly_frames: List[int] = []
            self._is_dragging = False
            self._hover_position = -1
            
            # Animation setup
            self._marker_animation = QPropertyAnimation(self, b"markerPosition")
            self._marker_animation.setDuration(150)
            self._marker_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._marker_animation.finished.connect(self._on_animation_complete)
            
            # Animation cleanup timer
            self._cleanup_timer = QTimer(self)
            self._cleanup_timer.timeout.connect(self._cleanup_animations)
            self._cleanup_timer.start(60000)  # Check every minute
            
            # UI setup
            self._setup_ui()
            self._setup_styles()
            
            self.setMouseTracking(True)
            self.setMinimumHeight(60)
            
            # Validate state
            self._validate_state()
            
            self._state = TimelineState.READY
            logger.debug("TimelineWidget initialized")
            self.state_changed.emit("initialized")
            
        except Exception as e:
            self._state = TimelineState.ERROR
            logger.error(f"Failed to initialize TimelineWidget: {e}")
            self.error_occurred.emit(f"Initialization Error: {str(e)}")
            raise

    def _setup_ui(self) -> None:
        """Set up UI components with loading indicator."""
        try:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            
            # Loading indicator
            self._loading_bar = QProgressBar()
            self._loading_bar.setTextVisible(False)
            self._loading_bar.setMaximumHeight(2)
            self._loading_bar.hide()
            layout.addWidget(self._loading_bar)
            
            # Timeline frame
            self._timeline_frame = QFrame()
            self._timeline_frame.setObjectName("timelineFrame")
            layout.addWidget(self._timeline_frame)
            
            # Frame counter label
            self._frame_label = QLabel("Frame: 0 / 0")
            self._frame_label.setObjectName("frameLabel")
            self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._frame_label)
            
        except Exception as e:
            logger.error(f"UI setup failed: {e}")
            raise

    def _validate_state(self) -> None:
        """Validate widget state."""
        try:
            required_attrs = [
                '_state', '_mutex', '_marker_animation',
                '_cleanup_timer', '_loading_bar', '_timeline_frame',
                '_frame_label'
            ]
            
            for attr in required_attrs:
                if not hasattr(self, attr):
                    raise AttributeError(f"Missing required attribute: {attr}")
                    
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            self.error_occurred.emit(f"Validation Error: {str(e)}")

    def _setup_styles(self) -> None:
        """Apply widget styles."""
        try:
            self.setStyleSheet("""
                QProgressBar {
                    background-color: transparent;
                    border: none;
                }
                QProgressBar::chunk {
                    background-color: #1F6FEB;
                }
                QFrame#timelineFrame {
                    background-color: #21262D;
                    border: 1px solid #30363D;
                    border-radius: 4px;
                }
                QLabel#frameLabel {
                    color: #8B949E;
                    font-size: 12px;
                    padding: 2px;
                }
            """)
        except Exception as e:
            logger.error(f"Style application failed: {e}")

    def set_loading(self, loading: bool) -> None:
        """Show/hide loading state."""
        try:
            if loading:
                self._state = TimelineState.LOADING
                self._loading_bar.setRange(0, 0)
                self._loading_bar.show()
            else:
                self._state = TimelineState.READY
                self._loading_bar.hide()
            
            self.state_changed.emit("loading" if loading else "ready")
            
        except Exception as e:
            logger.error(f"Failed to set loading state: {e}")
            self.error_occurred.emit(f"Loading Error: {str(e)}")

    def _cleanup_animations(self) -> None:
        """Periodic animation cleanup."""
        try:
            with QMutex():
                if self._marker_animation.state() == QPropertyAnimation.State.Stopped:
                    self._marker_animation.clear()
                    gc.collect()
            
        except Exception as e:
            logger.error(f"Animation cleanup failed: {e}")

    def cleanup(self) -> None:
        """Enhanced cleanup with animation handling."""
        try:
            self._state = TimelineState.CLEANED
            
            # Stop timers
            if self._cleanup_timer.isActive():
                self._cleanup_timer.stop()
            
            # Stop and clear animations
            self._marker_animation.stop()
            self._marker_animation.clear()
            
            # Clear data
            self._anomaly_frames.clear()
            self._frame_count = 0
            self._current_frame = 0
            self._marker_position = 0.0
            
            # Force cleanup
            gc.collect()
            
            logger.debug("TimelineWidget cleaned up")
            self.state_changed.emit("cleaned")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            self.error_occurred.emit(f"Cleanup Error: {str(e)}")

    def _on_animation_complete(self) -> None:
        """Handle animation completion."""
        try:
            if self._state == TimelineState.SEEKING:
                self._state = TimelineState.READY
                self.animation_completed.emit()
        except Exception as e:
            logger.error(f"Animation completion handler failed: {e}")

    # ... rest of the existing methods remain unchanged ...

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()