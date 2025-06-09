#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Video Preview Widget with optimized playback and caching
Created on: 2025-04-28 07:50:42 UTC
Last Modified: 2025-06-07 14:30:00 UTC
Author: ninjacode911
"""

import cv2
import logging
import gc
import numpy as np
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Dict, Tuple
from collections import OrderedDict

from PyQt6.QtCore import (Qt, QTimer, QThread, QMutex, 
                         pyqtSignal as Signal, pyqtSlot as Slot)
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QFrame,
                            QProgressBar)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPaintEvent, QResizeEvent

logger = logging.getLogger(__name__)

class PreviewState(Enum):
    """Widget states."""
    INITIALIZING = auto()
    READY = auto()
    LOADING = auto()
    PLAYING = auto()
    PAUSED = auto()
    ERROR = auto()
    CLEANED = auto()

class FrameCache:
    """Enhanced LRU cache for video frames with optimization."""
    
    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._mutex = QMutex()
        self._memory_usage = 0
        self._max_memory = 512 * 1024 * 1024  # 512MB limit
        
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get frame from cache with thread safety."""
        with QMutex():
            if frame_number in self._cache:
                frame = self._cache.pop(frame_number)
                self._cache[frame_number] = frame
                return frame
        return None

    def add_frame(self, frame_number: int, frame: np.ndarray) -> None:
        """Add frame to cache with memory management."""
        try:
            with QMutex():
                # Check memory limit
                frame_size = frame.nbytes
                if frame_number not in self._cache:
                    while (self._memory_usage + frame_size > self._max_memory or 
                           len(self._cache) >= self.max_size):
                        if not self._cache:
                            break
                        _, old_frame = self._cache.popitem(last=False)
                        self._memory_usage -= old_frame.nbytes
                
                    self._cache[frame_number] = frame.copy()
                    self._memory_usage += frame_size
                    
        except Exception as e:
            logger.error(f"Frame cache error: {e}")

    def clear(self) -> None:
        """Clear cache and free memory."""
        try:
            with QMutex():
                self._cache.clear()
                self._memory_usage = 0
                gc.collect()
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

class FrameLoader(QThread):
    """Enhanced asynchronous frame loader with prefetching."""
    
    frame_loaded = Signal(int, np.ndarray)
    error_occurred = Signal(str)
    
    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self._stop_flag = False
        self._current_frame = 0
        self._cap = None
        self._mutex = QMutex()
        self._prefetch_size = 5
        
    def run(self):
        """Load frames with prefetching."""
        try:
            self._cap = cv2.VideoCapture(self.video_path)
            while not self._stop_flag:
                with QMutex():
                    if not self._cap.isOpened():
                        break
                        
                    # Load current frame
                    ret, frame = self._cap.read()
                    if not ret:
                        break
                        
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frame_loaded.emit(self._current_frame, frame)
                    
                    # Prefetch next frames
                    for i in range(1, self._prefetch_size + 1):
                        if self._stop_flag:
                            break
                        next_frame = self._current_frame + i
                        ret, frame = self._cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frame_loaded.emit(next_frame, frame)
                    
                    self._current_frame += 1
                    
        except Exception as e:
            self.error_occurred.emit(f"Frame loading error: {e}")
        finally:
            if self._cap:
                self._cap.release()

    def stop(self):
        """Stop frame loading and cleanup."""
        try:
            self._stop_flag = True
            self.wait()
            if self._cap:
                self._cap.release()
                self._cap = None
        except Exception as e:
            logger.error(f"Loader stop error: {e}")

class VideoPreviewWidget(QWidget):
    """Enhanced video preview widget with optimized caching."""
    
    frame_changed = Signal(int)
    playback_finished = Signal()
    error_occurred = Signal(str)
    state_changed = Signal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        try:
            super().__init__(parent)
            self._state = PreviewState.INITIALIZING
            
            # Initialize state
            self._current_frame = 0
            self._frame_count = 0
            self._fps = 30
            self._is_playing = False
            self._frame_cache = FrameCache()
            self._frame_loader = None
            self._playback_timer = QTimer(self)
            
            # UI setup
            self._setup_ui()
            self._setup_connections()
            self._validate_state()
            
            self._state = PreviewState.READY
            logger.debug("VideoPreviewWidget initialized")
            self.state_changed.emit("initialized")
            
        except Exception as e:
            self._state = PreviewState.ERROR
            logger.error(f"Initialization failed: {e}")
            self.error_occurred.emit(str(e))

    def _validate_state(self) -> bool:
        """Validate widget state."""
        try:
            required_attrs = [
                '_frame_cache', '_frame_loader', '_playback_timer',
                '_display', '_frame', '_loading_bar'
            ]
            
            for attr in required_attrs:
                if not hasattr(self, attr):
                    raise AttributeError(f"Missing required attribute: {attr}")
                    
            return True
            
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            self.error_occurred.emit(str(e))
            return False

    def _setup_ui(self):
        """Initialize UI components with loading indicator."""
        try:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            
            # Loading indicator
            self._loading_bar = QProgressBar()
            self._loading_bar.setTextVisible(False)
            self._loading_bar.setMaximumHeight(2)
            self._loading_bar.hide()
            layout.addWidget(self._loading_bar)
            
            # Video frame
            self._frame = QFrame()
            self._frame.setStyleSheet("""
                QFrame {
                    background-color: #161B22;
                    border: 1px solid #30363D;
                    border-radius: 8px;
                }
            """)
            layout.addWidget(self._frame)
            
            # Display label
            self._display = QLabel()
            self._display.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self._display)
            
        except Exception as e:
            logger.error(f"UI setup failed: {e}")
            raise

    def _setup_connections(self):
        """Setup signal connections."""
        self._playback_timer.timeout.connect(self._next_frame)

    def set_loading(self, loading: bool) -> None:
        """Show/hide loading state."""
        try:
            if loading:
                self._state = PreviewState.LOADING
                self._loading_bar.setRange(0, 0)
                self._loading_bar.show()
            else:
                self._state = PreviewState.READY
                self._loading_bar.hide()
            
            self.state_changed.emit("loading" if loading else "ready")
            
        except Exception as e:
            logger.error(f"Failed to set loading state: {e}")

    def load_video(self, video_path: str) -> bool:
        """Load video with loading indicator."""
        self.set_loading(True)
        try:
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            # Stop existing loader
            if self._frame_loader:
                self._frame_loader.stop()
                
            # Clear cache
            self._frame_cache.clear()
                
            # Initialize new loader
            self._frame_loader = FrameLoader(video_path, self)
            self._frame_loader.frame_loaded.connect(self._on_frame_loaded)
            self._frame_loader.error_occurred.connect(self.error_occurred)
            self._frame_loader.start()
            
            # Get video properties
            cap = cv2.VideoCapture(video_path)
            self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            self._current_frame = 0
            self._playback_timer.setInterval(1000 / self._fps)
            
            logger.info(f"Video loaded: {video_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load video: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
        finally:
            self.set_loading(False)

    @Slot(int, np.ndarray)
    def _on_frame_loaded(self, frame_number: int, frame: np.ndarray):
        """Handle loaded frame."""
        try:
            self._frame_cache.add_frame(frame_number, frame)
            if frame_number == self._current_frame:
                self._display_frame(frame)
                self.frame_changed.emit(frame_number)
        except Exception as e:
            logger.error(f"Frame loading error: {e}")

    def _display_frame(self, frame: np.ndarray):
        """Display frame with proper scaling."""
        try:
            h, w = frame.shape[:2]
            frame_w = self.width()
            frame_h = int(frame_w * h / w)
            
            # Convert to QImage
            bytes_per_line = 3 * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Scale and display
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                frame_w, frame_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self._display.setPixmap(scaled_pixmap)
            
        except Exception as e:
            logger.error(f"Frame display error: {e}")

    def play(self):
        """Start video playback."""
        try:
            if not self._is_playing:
                self._is_playing = True
                self._state = PreviewState.PLAYING
                self._playback_timer.start()
                self.state_changed.emit("playing")
        except Exception as e:
            logger.error(f"Play error: {e}")

    def pause(self):
        """Pause video playback."""
        try:
            if self._is_playing:
                self._is_playing = False
                self._state = PreviewState.PAUSED
                self._playback_timer.stop()
                self.state_changed.emit("paused")
        except Exception as e:
            logger.error(f"Pause error: {e}")

    def _next_frame(self):
        """Display next frame with error recovery."""
        try:
            if self._current_frame >= self._frame_count - 1:
                self.pause()
                self.playback_finished.emit()
                return
                
            self._current_frame += 1
            frame = self._frame_cache.get_frame(self._current_frame)
            
            if frame is not None:
                self._display_frame(frame)
                self.frame_changed.emit(self._current_frame)
                
        except Exception as e:
            logger.error(f"Frame advance error: {e}")
            self.error_occurred.emit(str(e))
            self.pause()  # Safety pause on error

    def seek(self, frame_number: int):
        """Seek to specific frame."""
        try:
            if 0 <= frame_number < self._frame_count:
                self._current_frame = frame_number
                frame = self._frame_cache.get_frame(frame_number)
                if frame is not None:
                    self._display_frame(frame)
                    self.frame_changed.emit(frame_number)
                    
        except Exception as e:
            logger.error(f"Seek error: {e}")
            self.error_occurred.emit(str(e))

    def cleanup(self):
        """Enhanced cleanup with proper resource management."""
        try:
            self._state = PreviewState.CLEANED
            
            # Stop playback and loading
            if self._is_playing:
                self.pause()
            
            # Stop frame loader
            if self._frame_loader:
                self._frame_loader.stop()
            
            # Stop timer
            if self._playback_timer.isActive():
                self._playback_timer.stop()
            
            # Clear frame cache
            self._frame_cache.clear()
            
            # Force cleanup
            gc.collect()
            
            logger.debug("VideoPreviewWidget cleaned up")
            self.state_changed.emit("cleaned")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            self.error_occurred.emit(str(e))

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

    def resizeEvent(self, event: QResizeEvent):
        """Handle widget resize."""
        try:
            super().resizeEvent(event)
            if self._current_frame < self._frame_count:
                frame = self._frame_cache.get_frame(self._current_frame)
                if frame is not None:
                    self._display_frame(frame)
        except Exception as e:
            logger.error(f"Resize error: {e}")