#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video Controls Widget with shortcut management and state sync
Created on: 2025-04-28 07:50:42 UTC
Last Modified: 2025-06-07 14:30:00 UTC
Author: ninjacode911
"""

import logging
from typing import Optional, Dict, Set
from enum import Enum, auto
import gc

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton,
    QSlider, QLabel, QFrame, QProgressBar
)
from PyQt6.QtCore import (
    Qt, pyqtSignal as Signal,
    QTimer, QMutex
)
from PyQt6.QtGui import (
    QKeySequence, QShortcut, QIcon,
    QAction, QActionGroup
)
import qtawesome as qta

logger = logging.getLogger(__name__)

class ControlState(Enum):
    """Control widget states."""
    INITIALIZING = auto()
    READY = auto()
    PLAYING = auto()
    PAUSED = auto()
    LOADING = auto()
    ERROR = auto()
    CLEANED = auto()

class VideoControlsWidget(QWidget):
    """
    Enhanced video control widget with shortcuts and state management.
    
    Features:
    - Keyboard shortcuts with cleanup
    - State synchronization
    - Loading indicators
    - Error recovery
    - Memory optimization
    """
    
    # Signals
    play_clicked = Signal()
    pause_clicked = Signal()
    stop_clicked = Signal()
    next_frame = Signal()
    prev_frame = Signal()
    seek_requested = Signal(int)
    error_occurred = Signal(str)
    state_changed = Signal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize video controls with state management."""
        try:
            super().__init__(parent)
            self._state = ControlState.INITIALIZING
            
            # Thread safety
            self._mutex = QMutex()
            
            # Initialize state
            self._frame_count = 0
            self._current_frame = 0
            self._is_playing = False
            
            # Track shortcuts
            self._shortcuts: Dict[str, QShortcut] = {}
            self._actions: Set[QAction] = set()
            
            # Setup UI and shortcuts
            self._setup_ui()
            self._setup_shortcuts()
            self._setup_styles()
            
            # Validate initial state
            self._validate_state()
            
            self._state = ControlState.READY
            logger.debug("VideoControlsWidget initialized")
            self.state_changed.emit("initialized")
            
        except Exception as e:
            self._state = ControlState.ERROR
            logger.error(f"Initialization failed: {e}")
            self.error_occurred.emit(f"Initialization Error: {str(e)}")
            raise

    def _setup_ui(self) -> None:
        """Setup UI components with loading indicator."""
        try:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)
            
            # Loading bar
            self._loading_bar = QProgressBar()
            self._loading_bar.setTextVisible(False)
            self._loading_bar.setMaximumHeight(2)
            self._loading_bar.hide()
            layout.addWidget(self._loading_bar)
            
            # Controls container
            controls_frame = QFrame()
            controls_frame.setObjectName("controlsFrame")
            controls_layout = QHBoxLayout(controls_frame)
            controls_layout.setContentsMargins(8, 8, 8, 8)
            controls_layout.setSpacing(8)
            
            # Control buttons
            self._play_button = QPushButton(qta.icon('mdi.play'), "")
            self._pause_button = QPushButton(qta.icon('mdi.pause'), "")
            self._stop_button = QPushButton(qta.icon('mdi.stop'), "")
            self._prev_button = QPushButton(qta.icon('mdi.step-backward'), "")
            self._next_button = QPushButton(qta.icon('mdi.step-forward'), "")
            
            for btn in [self._play_button, self._pause_button, self._stop_button,
                       self._prev_button, self._next_button]:
                btn.setObjectName("controlButton")
                controls_layout.addWidget(btn)
            
            # Frame slider
            self._frame_slider = QSlider(Qt.Orientation.Horizontal)
            self._frame_slider.setObjectName("frameSlider")
            controls_layout.addWidget(self._frame_slider, stretch=1)
            
            # Frame counter
            self._frame_label = QLabel("0 / 0")
            self._frame_label.setObjectName("frameLabel")
            controls_layout.addWidget(self._frame_label)
            
            layout.addWidget(controls_frame)
            
            # Connect signals
            self._connect_signals()
            
        except Exception as e:
            logger.error(f"UI setup failed: {e}")
            raise

    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts with tracking."""
        try:
            shortcuts = {
                "Space": (self._toggle_playback, "Toggle playback"),
                "Left": (self._prev_frame_clicked, "Previous frame"),
                "Right": (self._next_frame_clicked, "Next frame"),
                "Escape": (self._stop_clicked, "Stop playback")
            }
            
            for key, (slot, description) in shortcuts.items():
                shortcut = QShortcut(QKeySequence(key), self)
                shortcut.activated.connect(slot)
                self._shortcuts[description] = shortcut
                
            logger.debug(f"Initialized {len(shortcuts)} shortcuts")
            
        except Exception as e:
            logger.error(f"Shortcut setup failed: {e}")
            raise

    def _setup_styles(self) -> None:
        """Apply widget styles."""
        try:
            self.setStyleSheet("""
                QFrame#controlsFrame {
                    background-color: #21262D;
                    border: 1px solid #30363D;
                    border-radius: 4px;
                }
                
                QPushButton#controlButton {
                    background-color: transparent;
                    border: none;
                    padding: 4px;
                    border-radius: 4px;
                }
                
                QPushButton#controlButton:hover {
                    background-color: #30363D;
                }
                
                QPushButton#controlButton:pressed {
                    background-color: #388BFD;
                }
                
                QSlider#frameSlider::groove:horizontal {
                    background: #30363D;
                    height: 4px;
                    border-radius: 2px;
                }
                
                QSlider#frameSlider::handle:horizontal {
                    background: #388BFD;
                    width: 16px;
                    margin: -6px 0;
                    border-radius: 8px;
                }
                
                QLabel#frameLabel {
                    color: #8B949E;
                    font-size: 12px;
                }
                
                QProgressBar {
                    background-color: transparent;
                    border: none;
                }
                
                QProgressBar::chunk {
                    background-color: #1F6FEB;
                }
            """)
        except Exception as e:
            logger.error(f"Style application failed: {e}")

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        try:
            self._play_button.clicked.connect(self._play_clicked)
            self._pause_button.clicked.connect(self._pause_clicked)
            self._stop_button.clicked.connect(self._stop_clicked)
            self._prev_button.clicked.connect(self._prev_frame_clicked)
            self._next_button.clicked.connect(self._next_frame_clicked)
            self._frame_slider.valueChanged.connect(self._slider_changed)
            
        except Exception as e:
            logger.error(f"Signal connection failed: {e}")
            raise

    def _validate_state(self) -> None:
        """Validate widget state."""
        try:
            required_attrs = [
                '_state', '_mutex', '_shortcuts', '_actions',
                '_play_button', '_pause_button', '_stop_button',
                '_prev_button', '_next_button', '_frame_slider',
                '_frame_label', '_loading_bar'
            ]
            
            for attr in required_attrs:
                if not hasattr(self, attr):
                    raise AttributeError(f"Missing required attribute: {attr}")
                    
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            self.error_occurred.emit(f"Validation Error: {str(e)}")

    def set_loading(self, loading: bool) -> None:
        """Show/hide loading state."""
        try:
            if loading:
                self._state = ControlState.LOADING
                self._loading_bar.setRange(0, 0)
                self._loading_bar.show()
            else:
                self._state = ControlState.READY
                self._loading_bar.hide()
            
            # Update UI state
            self._update_button_states()
            self.state_changed.emit("loading" if loading else "ready")
            
        except Exception as e:
            logger.error(f"Failed to set loading state: {e}")
            self.error_occurred.emit(f"Loading Error: {str(e)}")

    def _update_button_states(self) -> None:
        """Update button states based on current state."""
        try:
            is_enabled = self._state not in [ControlState.LOADING, ControlState.ERROR]
            
            self._play_button.setEnabled(is_enabled and not self._is_playing)
            self._pause_button.setEnabled(is_enabled and self._is_playing)
            self._stop_button.setEnabled(is_enabled and self._current_frame > 0)
            self._prev_button.setEnabled(is_enabled and self._current_frame > 0)
            self._next_button.setEnabled(is_enabled and self._current_frame < self._frame_count - 1)
            self._frame_slider.setEnabled(is_enabled)
            
        except Exception as e:
            logger.error(f"Button state update failed: {e}")

    def set_frame_count(self, count: int) -> None:
        """Set total frame count."""
        try:
            self._frame_count = max(0, count)
            self._frame_slider.setMaximum(self._frame_count - 1)
            self._update_frame_label()
            self._update_button_states()
            
        except Exception as e:
            logger.error(f"Frame count update failed: {e}")
            self.error_occurred.emit(f"Update Error: {str(e)}")

    def set_current_frame(self, frame: int) -> None:
        """Set current frame number."""
        try:
            if 0 <= frame < self._frame_count:
                self._current_frame = frame
                self._frame_slider.setValue(frame)
                self._update_frame_label()
                self._update_button_states()
                
        except Exception as e:
            logger.error(f"Frame update failed: {e}")
            self.error_occurred.emit(f"Update Error: {str(e)}")

    def _update_frame_label(self) -> None:
        """Update frame counter label."""
        try:
            self._frame_label.setText(f"{self._current_frame + 1} / {self._frame_count}")
        except Exception as e:
            logger.error(f"Label update failed: {e}")

    def _toggle_playback(self) -> None:
        """Toggle between play and pause."""
        try:
            if self._is_playing:
                self._pause_clicked()
            else:
                self._play_clicked()
        except Exception as e:
            logger.error(f"Playback toggle failed: {e}")

    def _play_clicked(self) -> None:
        """Handle play button click."""
        try:
            self._is_playing = True
            self._state = ControlState.PLAYING
            self._update_button_states()
            self.play_clicked.emit()
            self.state_changed.emit("playing")
        except Exception as e:
            logger.error(f"Play action failed: {e}")

    def _pause_clicked(self) -> None:
        """Handle pause button click."""
        try:
            self._is_playing = False
            self._state = ControlState.PAUSED
            self._update_button_states()
            self.pause_clicked.emit()
            self.state_changed.emit("paused")
        except Exception as e:
            logger.error(f"Pause action failed: {e}")

    def _stop_clicked(self) -> None:
        """Handle stop button click."""
        try:
            self._is_playing = False
            self._current_frame = 0
            self._state = ControlState.READY
            self._update_button_states()
            self._frame_slider.setValue(0)
            self.stop_clicked.emit()
            self.state_changed.emit("stopped")
        except Exception as e:
            logger.error(f"Stop action failed: {e}")

    def _next_frame_clicked(self) -> None:
        """Handle next frame button click."""
        try:
            self.next_frame.emit()
        except Exception as e:
            logger.error(f"Next frame action failed: {e}")

    def _prev_frame_clicked(self) -> None:
        """Handle previous frame button click."""
        try:
            self.prev_frame.emit()
        except Exception as e:
            logger.error(f"Previous frame action failed: {e}")

    def _slider_changed(self, value: int) -> None:
        """Handle slider value change."""
        try:
            if value != self._current_frame:
                self._current_frame = value
                self._update_frame_label()
                self.seek_requested.emit(value)
        except Exception as e:
            logger.error(f"Slider update failed: {e}")

    def cleanup(self) -> None:
        """Enhanced cleanup with shortcut management."""
        try:
            self._state = ControlState.CLEANED
            
            # Clear shortcuts
            for shortcut in self._shortcuts.values():
                shortcut.setEnabled(False)
                shortcut.deleteLater()
            self._shortcuts.clear()
            
            # Clear actions
            for action in self._actions:
                action.deleteLater()
            self._actions.clear()
            
            # Reset state
            self._is_playing = False
            self._current_frame = 0
            self._frame_count = 0
            
            # Force cleanup
            gc.collect()
            
            logger.debug("VideoControlsWidget cleaned up")
            self.state_changed.emit("cleaned")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            self.error_occurred.emit(f"Cleanup Error: {str(e)}")

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()