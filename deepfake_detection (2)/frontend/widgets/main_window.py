#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepFake Detection System - Main Window 
Created on: 2025-04-28 07:50:42 UTC
Last Modified: 2025-06-05 14:30:00 UTC
Author: ninjacode911

This module provides the main window for the DeepFake Detection System, integrating all widgets
and handling video analysis, playback, and result visualization.
"""

import logging
import sys
import gc
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Union, Tuple, Any
import resources_rc
import cv2
import numpy as np
import psutil
import qtawesome as qta
from PyQt6.QtCore import QObject, Qt, QSize, QTimer, pyqtSignal as Signal
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QProgressBar, QComboBox, QFrame,
    QSizePolicy, QApplication, QMessageBox, QFileDialog
)
from PyQt6.QtGui import QFont, QIcon
from typing import Optional, List, Dict, Union, Tuple, Any
from frontend.widgets.file_manager import FileManager
from frontend.resources.theme_manager import theme_manager
from frontend.widgets.recent_files import RecentFilesWidget
from frontend.widgets.video_controls import VideoControlsWidget
from frontend.widgets.timeline import TimelineWidget
from frontend.widgets.video_preview import VideoPreviewWidget
from frontend.widgets.heatmap import HeatmapWidget
from frontend.widgets.detected_issues import DetectedIssuesWidget
from frontend.widgets.manipulation import ManipulationWidget
from frontend.widgets.classification import ClassificationWidget
from backend.core.detector import Detector
from backend.database.database import Database
from frontend.config import config_manager

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._detection_thread = None
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._cleanup_resources)
        self._cleanup_timer.start(300000)  # 5 min cleanup

    def _cleanup_resources(self):
        if hasattr(self, 'detector'):
            self.detector.cleanup()

class MemoryManager:
        """Memory management utilities for optimizing application performance."""
        def __init__(self, threshold_mb: int = 1000):
            self.threshold_mb = threshold_mb
            self.last_cleanup = time.time()
            self.cleanup_interval = 30  # seconds
        self._monitor_timer = QTimer()
        self._monitor_timer.timeout.connect(self.check_memory)
        self._monitor_timer.start(5000)  # Check every 5 seconds
        logger.debug("Memory manager initialized")

        def check_memory(self) -> bool:
        """Check if memory usage exceeds threshold."""
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
            if memory_mb > self.threshold_mb:
                logger.warning(f"Memory usage high: {memory_mb:.1f}MB")
                return True
            return False
            except Exception as e:
                logger.error(f"Memory check failed: {e}")
                return False

        def optimize_memory(self, force: bool = False):
        """Optimize memory usage."""
            try:
                current_time = time.time()
                if force or (current_time - self.last_cleanup > self.cleanup_interval 
                            and self.check_memory()):
                # Clear video frames
                if hasattr(self, 'video_player'):
                    self.video_player.prefetcher.clear_cache()
                
                # Clear OpenCV windows
                    cv2.destroyAllWindows()
                
                # Force garbage collection
                gc.collect()
                
                    self.last_cleanup = current_time
                    logger.debug("Memory optimization performed")
            except Exception as e:
                logger.error(f"Memory optimization failed: {e}")

    def cleanup(self):
        """Cleanup resources."""
        try:
            self._monitor_timer.stop()
            self.optimize_memory(force=True)
        except Exception as e:
            logger.error(f"Memory manager cleanup failed: {e}")

class FramePrefetcher(QObject):
        """Thread-safe frame prefetching and caching mechanism."""
        def __init__(self, video_player, cache_size: int = 5):
            super().__init__()
            self.video_player = video_player
            self.cache_size = cache_size
            self._frame_cache = {}
            self._prefetch_thread = None
            self._stop_prefetch = False
            self._lock = threading.Lock()
            logger.debug("Frame prefetcher initialized")

        def start_prefetching(self, current_frame: int):
            try:
                if self._prefetch_thread and self._prefetch_thread.is_alive():
                    self._stop_prefetch = True
                    self._prefetch_thread.join()
                
                self._stop_prefetch = False
                self._prefetch_thread = threading.Thread(
                    target=self._prefetch_frames,
                    args=(current_frame,),
                    daemon=True
                )
                self._prefetch_thread.start()
            except Exception as e:
                logger.error(f"Failed to start prefetching: {e}")

        def _prefetch_frames(self, start_frame: int) -> None:
            """Prefetch frames with proper locking."""
            try:
                for i in range(start_frame + 1, min(start_frame + self.cache_size, self.total_frames)):
                    with self._lock:
                        if self._stop_prefetch:
                            return
                        if i not in self._frame_cache:
                            if self.cap and self.cap.isOpened():
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                                ret, frame = self.cap.read()
                                if ret:
                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    self._frame_cache[i] = frame
            except Exception as e:
                logger.error(f"Prefetch error: {e}")

        def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
            with self._lock:
                return self._frame_cache.get(frame_number)

        def clear_cache(self):
            with self._lock:
                self._frame_cache.clear()
                self._stop_prefetch = True



class VideoPlayer(QObject):
    """Enhanced video player class to handle frame-based playback and synchronization."""
    frame_changed = Signal(int)  # Signal to notify frame changes
    playback_state_changed = Signal(bool)  # Signal to notify play/pause state
    error_occurred = Signal(str)  # Signal to emit errors

    def __init__(self):
        """Initialize the VideoPlayer with enhanced features."""
        try:
            super().__init__()
            self._initialize_variables()
            self._setup_timer()
            self.prefetcher = FramePrefetcher(self)
            self.memory_manager = MemoryManager()
            self._frame_lock = threading.Lock()
            self._is_loading = False
            logger.debug("VideoPlayer initialized with prefetching and memory management")
        except Exception as e:
            error_msg = f"Failed to initialize VideoPlayer: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def _initialize_variables(self):
        """Initialize instance variables."""
        self.current_time = 0.0
        self.total_time = 0.0
        self.current_frame = 0
        self.total_frames = 1  # Default to 1 to prevent seeking errors
        self.is_playing = False
        self.cap = None  # OpenCV VideoCapture object
        self.fps = 30.0  # Default FPS
        self.mock_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        self.mock_frame[:] = [50, 50, 50]  # Dark gray
        self._last_frame = None
        self._frame_buffer = {}
        self._buffer_size = 10
        logger.debug("VideoPlayer variables initialized")

    def _setup_timer(self):
        """Setup playback timer."""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        logger.debug("VideoPlayer timer initialized")

    def load_video(self, file_path: str):
        """Load video frames from file."""
        if self._is_loading:
            logger.warning("Video loading already in progress")
            return

        try:
            self._is_loading = True
            self.release()
            
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Video file not found: {file_path}")
            
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video file: {file_path}")
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.total_frames <= 0:
                self.total_frames = 450  # Mock total frames (15 seconds at 30 FPS)
                logger.warning("Video has no frames, using mock total_frames=450")
            
            self.total_time = self.total_frames / self.fps if self.fps > 0 else 0.0
            self.current_frame = 0
            self.current_time = 0.0
            
            # Initialize frame buffer
            self._frame_buffer.clear()
            self._load_initial_frames()
            
            logger.info(f"Loaded video with {self.total_frames} frames from {file_path} at {self.fps} FPS")
            self.frame_changed.emit(self.current_frame)
            
        except Exception as e:
            error_msg = f"Failed to load video: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
        finally:
            self._is_loading = False

    def _load_initial_frames(self):
        """Load initial frames into buffer."""
        try:
            if not self.cap or not self.cap.isOpened():
                return
                
            for i in range(min(self._buffer_size, self.total_frames)):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self._frame_buffer[i] = frame
        except Exception as e:
            logger.error(f"Failed to load initial frames: {e}")

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get frame with caching."""
        try:
            with self._frame_lock:
                # Check buffer first
                if frame_number in self._frame_buffer:
                    return self._frame_buffer[frame_number]
                
                # Check prefetcher
                prefetched = self.prefetcher.get_frame(frame_number)
                if prefetched is not None:
                    self._frame_buffer[frame_number] = prefetched
                    return prefetched
                
                # Load frame directly
                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = self.cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self._frame_buffer[frame_number] = frame
                        return frame
                
                return None
        except Exception as e:
            logger.error(f"Failed to get frame {frame_number}: {e}")
            return None

    def _manage_buffer(self):
        """Manage frame buffer size."""
        try:
            with self._frame_lock:
                if len(self._frame_buffer) > self._buffer_size:
                    # Remove frames furthest from current frame
                    frames = sorted(self._frame_buffer.keys())
                    to_remove = len(frames) - self._buffer_size
                    for i in range(to_remove):
                        self._frame_buffer.pop(frames[i])
        except Exception as e:
            logger.error(f"Buffer management failed: {e}")

    def play(self):
        """Start playback."""
        try:
            if not self.is_playing and self.total_frames > 0:
                self.is_playing = True
                self.timer.start(int(1000 / self.fps))
                self.playback_state_changed.emit(True)
                logger.debug("Video playback started")
        except Exception as e:
            error_msg = f"Failed to start playback: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def pause(self):
        """Pause playback."""
        try:
            if self.is_playing:
                self.is_playing = False
                self.timer.stop()
                self.playback_state_changed.emit(False)
                logger.debug("Video paused")
        except Exception as e:
            error_msg = f"Failed to pause playback: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def forward(self):
        """Skip forward 5 seconds (or equivalent frames)."""
        try:
            if self.total_frames > 0:
                skip_frames = int(self.fps * 5)
                self.current_frame = min(self.current_frame + skip_frames, self.total_frames - 1)
                self.current_time = self.current_frame / self.fps if self.fps > 0 else 0.0
                self.frame_changed.emit(self.current_frame)
                logger.debug(f"Video forwarded to frame {self.current_frame} ({self.current_time:.2f}s)")
        except Exception as e:
            error_msg = f"Failed to fast-forward: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def backward(self):
        """Skip backward 5 seconds (or equivalent frames)."""
        try:
            if self.total_frames > 0:
                skip_frames = int(self.fps * 5)
                self.current_frame = max(self.current_frame - skip_frames, 0)
                self.current_time = self.current_frame / self.fps if self.fps > 0 else 0.0
                self.frame_changed.emit(self.current_frame)
                logger.debug(f"Video rewound to frame {self.current_frame} ({self.current_time:.2f}s)")
        except Exception as e:
            error_msg = f"Failed to rewind: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def seek(self, percentage: float):
        """Seek to a specific position (percentage)."""
        try:
            if not 0 <= percentage <= 100:
                raise ValueError(f"Percentage must be in [0, 100], got {percentage}")
            if self.total_frames > 0:
                self.current_frame = int((percentage / 100) * (self.total_frames - 1))
                self.current_time = self.current_frame / self.fps if self.fps > 0 else 0.0
                self.frame_changed.emit(self.current_frame)
                logger.debug(f"Video sought to frame {self.current_frame} ({self.current_time:.2f}s) via percentage {percentage}")
        except Exception as e:
            error_msg = f"Failed to seek video: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def seek_to_frame(self, frame: int):
        """Seek to a specific frame."""
        try:
            if self.total_frames <= 0:
                logger.warning("Total frames is %d, cannot seek", self.total_frames)
                return
            if not 0 <= frame < self.total_frames:
                raise ValueError(f"Frame {frame} out of range [0, {self.total_frames-1}]")
            self.current_frame = frame
            self.current_time = self.current_frame / self.fps if self.fps > 0 else 0.0
            self.frame_changed.emit(self.current_frame)
            logger.debug(f"Video sought to frame {self.current_frame} ({self.current_time:.2f}s)")
        except Exception as e:
            error_msg = f"Failed to seek to frame: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def update_progress(self):
        """Update playback progress."""
        try:
            if self.is_playing and self.total_frames > 0:
                self.current_frame += 1
                self.current_time = self.current_frame / self.fps if self.fps > 0 else 0.0
                if self.current_frame >= self.total_frames - 1:
                    self.pause()
                    self.current_frame = self.total_frames - 1
                    self.current_time = self.total_time
                self.frame_changed.emit(self.current_frame)
                logger.debug(f"Progress updated to frame {self.current_frame} ({self.current_time:.2f}s)")
        except Exception as e:
            error_msg = f"Failed to update playback progress: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def set_duration(self, duration: float):
        """Set the video duration (in seconds)."""
        try:
            if duration <= 0:
                raise ValueError("Duration must be positive")
            self.total_time = duration
            self.total_frames = max(1, int(duration * self.fps))
            logger.debug(f"Video duration set to {duration:.2f}s ({self.total_frames} frames)")
        except Exception as e:
            error_msg = f"Failed to set duration: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def release(self):
        """Release video resources."""
        try:
            self.pause()
            if self.cap:
                self.cap.release()
            self.cap = None
            self._frame_buffer.clear()
                self.prefetcher.clear_cache()
            logger.debug("Video resources released")
        except Exception as e:
            logger.error(f"Failed to release video resources: {e}")

    def cleanup(self) -> None:
        """Cleanup all resources."""
        try:
            self.release()
            self.memory_manager.cleanup()
            logger.debug("VideoPlayer cleanup completed")
        except Exception as e:
            logger.error(f"VideoPlayer cleanup failed: {e}")

class DeepFakeDetectionSystem(QMainWindow):
    """Main window for the DeepFake Detection System."""
    
    analysis_progress = Signal(int)  # Signal to update analysis progress
    
    def _optimize_memory(self) -> None:
        """Optimize memory usage."""
        try:
            process = psutil.Process()
            if process.memory_info().rss > 1024 * 1024 * 1000:  # 1GB
                self._emergency_cleanup()
                gc.collect()
                logger.info("Memory optimization performed")
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")

    def _emergency_cleanup(self) -> None:
        """Emergency resource cleanup."""
        try:
            self.video_player.clear_cache()
            self.heatmap_widget.clear_cache()
            self._cache.clear()
            gc.collect()
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
                    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepFake Detection System")
        self.setMinimumSize(1200, 800)
        
        # Initialize components
        self.file_manager = FileManager()
        self.current_file_path = None
        self.video_player = VideoPlayer()
        self.detector = Detector()
        self.db = Database()
        self.total_frames = 1
        self.anomalies = []
        self.fps = 30.0
        self.class_mapping = ["Fake", "Likely Fake", "Neutral", "Likely Real", "Real"]
        
        # Setup UI
        self._init_ui()
        
        # Connect signals
        self._connect_signals()
        
        # Apply styles
        self._apply_styles()
        
        # Set default UI state
        self._set_default_ui_state()
        
        logger.info("Main window initialized successfully")

    def _init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Header
        self.header = self._create_header()
        main_layout.addWidget(self.header)
        
        # Main content
        main_content = QWidget()
        main_layout.addWidget(main_content)
        content_layout = QVBoxLayout(main_content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        
        # Columns
        columns_splitter = QSplitter(Qt.Orientation.Horizontal)
        columns_splitter.setHandleWidth(1)
        columns_splitter.setStyleSheet("QSplitter::handle { background: #30363d; }")
        
        # Left sidebar
        self.sidebar = self._create_left_sidebar()
        columns_splitter.addWidget(self.sidebar)
        
        # Middle column
        self.middle = self._create_middle_column()
        columns_splitter.addWidget(self.middle)
        
        # Right column
        self.right = self._create_right_column()
        columns_splitter.addWidget(self.right)
        
        # Set column proportions (20%, 40%, 40%)
        columns_splitter.setSizes([240, 480, 480])
        content_layout.addWidget(columns_splitter)

        # Bottom section
        self.bottom = self._create_bottom_section()
        content_layout.addWidget(self.bottom)
        
        # Set stretch factors
        content_layout.setStretch(0, 2)
        content_layout.setStretch(1, 1)
    
    def _create_header(self):
        """Create the header widget."""
        self.header_frame = QFrame()
        self.header_frame.setFixedHeight(64)
        layout = QHBoxLayout(self.header_frame)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(16)
        
        # Title with icon
        title_frame = QFrame()
        title_layout = QHBoxLayout(title_frame)
        title_layout.setSpacing(8)
        
        icon = qta.icon('fa5s.video', color='#1F6FEB')
        title_icon = QLabel()
        title_icon.setPixmap(icon.pixmap(32, 32))
        title_layout.addWidget(title_icon)
        
        self.title = QLabel("DeepFake Detection System")
        self.title.setFont(QFont("Open Sans", 14, QFont.Weight.Bold))
        title_layout.addWidget(self.title)
        
        layout.addWidget(title_frame)
        
        layout.addStretch()
        
        # User info
        self.user_frame = QFrame()
        user_layout = QHBoxLayout(self.user_frame)
        user_layout.setContentsMargins(12, 4, 12, 4)
        user_layout.setSpacing(4)
        
        user_icon = qta.icon('mdi.account-circle-outline', color='#8B949E')
        user_label = QLabel()
        user_label.setPixmap(user_icon.pixmap(16, 16))
        user_layout.addWidget(user_label)
        
        user_text = QLabel("Logged in as: ninjacode911")
        user_text.setFont(QFont("Open Sans", 10))
        user_layout.addWidget(user_text)
        
        layout.addWidget(self.user_frame)
        
        return self.header_frame
    
    def _create_left_sidebar(self):
        """Create the left sidebar widget."""
        self.sidebar_frame = QFrame()
        layout = QVBoxLayout(self.sidebar_frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # Import section
        self.import_frame = QFrame()
        import_layout = QVBoxLayout(self.import_frame)
        import_layout.setSpacing(12)
        
        self.import_title = QLabel("IMPORT")
        self.import_title.setFont(QFont("Open Sans", 10, QFont.Weight.Bold))
        import_layout.addWidget(self.import_title)
        
        self.upload_btn = QPushButton("Upload")
        self.upload_btn.setIcon(qta.icon('mdi.upload', color='white'))
        self.upload_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        import_layout.addWidget(self.upload_btn)
        
        self.record_btn = QPushButton("Record")
        self.record_btn.setIcon(qta.icon('mdi.video', color='white'))
        self.record_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.record_btn.setEnabled(False)
        import_layout.addWidget(self.record_btn)
        
        layout.addWidget(self.import_frame)
        
        # Recent files section
        self.recent_frame = QFrame()
        recent_layout = QVBoxLayout(self.recent_frame)
        recent_layout.setSpacing(12)
        
        # Add "Recent Files" title
        self.recent_title = QLabel("RECENT FILES")
        self.recent_title.setFont(QFont("Open Sans", 10, QFont.Weight.Bold))
        recent_layout.addWidget(self.recent_title)

        self.recent_files_widget = RecentFilesWidget()
        self.recent_files_widget.setFixedHeight(400)
        recent_layout.addWidget(self.recent_files_widget)

        layout.addWidget(self.recent_frame)
        layout.addStretch()
        
        return self.sidebar_frame
    
    def _create_middle_column(self):
        """Create the middle column widget for analysis results."""
        self.middle_frame = QFrame()
        layout = QVBoxLayout(self.middle_frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        
        # Title
        self.title_frame = QFrame()
        title_layout = QHBoxLayout(self.title_frame)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)
        
        self.analysis_title = QLabel("ANALYSIS RESULTS")
        self.analysis_title.setFont(QFont("Open Sans", 10, QFont.Weight.Bold))
        title_layout.addWidget(self.analysis_title)
        
        analytics_icon = qta.icon('mdi.chart-bar', color='#8B949E')
        title_icon = QLabel()
        title_icon.setPixmap(analytics_icon.pixmap(16, 16))
        title_layout.addWidget(title_icon)
        
        layout.addWidget(self.title_frame)
        
        # Classification
        self.class_frame = QFrame()
        class_layout = QVBoxLayout(self.class_frame)
        class_layout.setContentsMargins(0, 0, 0, 0)
        class_layout.setSpacing(4)
        
        self.classification_widget = ClassificationWidget()
        class_layout.addWidget(self.classification_widget)
        
        layout.addWidget(self.class_frame)
        
        # Manipulation probability
        self.manip_frame = QFrame()
        self.manip_frame.setFixedHeight(50)  # Halve the space
        manip_layout = QVBoxLayout(self.manip_frame)
        manip_layout.setContentsMargins(0, 0, 0, 0)
        manip_layout.setSpacing(4)
        
        self.manip_label = QLabel("Manipulation Probability:")
        manip_layout.addWidget(self.manip_label)
        
        self.manipulation_widget = ManipulationWidget()
        manip_layout.addWidget(self.manipulation_widget)
        
        layout.addWidget(self.manip_frame)
        
        # Heatmap visualization
        self.heatmap_frame = QFrame()
        heatmap_layout = QVBoxLayout(self.heatmap_frame)
        heatmap_layout.setContentsMargins(0, 0, 0, 0)
        heatmap_layout.setSpacing(4)

        heatmap_title_layout = QHBoxLayout()
        heatmap_title_layout.addStretch()

        self.intensity_btn = QPushButton("Adjust Intensity")
        self.intensity_btn.setIcon(qta.icon('mdi.tune', color='white'))
        self.intensity_btn.clicked.connect(self.adjust_heatmap_intensity)
        heatmap_title_layout.addWidget(self.intensity_btn)

        self.heatmap_combo = QComboBox()
        self.heatmap_combo.addItems(["Grad-CAM", "EigenCAM"])
        self.heatmap_combo.currentIndexChanged.connect(self.update_heatmap_type)
        heatmap_title_layout.addWidget(self.heatmap_combo)

        heatmap_layout.addLayout(heatmap_title_layout)

        self.heatmap_widget = HeatmapWidget()
        self.heatmap_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.heatmap_widget.setMinimumSize(200, 300)  # Increase height to ensure visibility
        heatmap_layout.addWidget(self.heatmap_widget)

        layout.addWidget(self.heatmap_frame)

        # Detected issues
        self.issues_frame = QFrame()
        issues_layout = QVBoxLayout(self.issues_frame)
        issues_layout.setContentsMargins(0, 0, 0, 0)
        issues_layout.setSpacing(4)

        self.issues_widget = DetectedIssuesWidget()
        issues_layout.addWidget(self.issues_widget)

        layout.addWidget(self.issues_frame)
        
        # Removed layout.addStretch() to prevent overlap
        
        return self.middle_frame
    
    def _create_right_column(self):
        """Create the right column widget for video preview."""
        self.right_frame = QFrame()
        layout = QVBoxLayout(self.right_frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Title
        self.video_title_frame = QFrame()
        title_layout = QHBoxLayout(self.video_title_frame)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)
        
        self.video_title = QLabel("VIDEO PREVIEW")
        self.video_title.setFont(QFont("Open Sans", 10, QFont.Weight.Bold))
        title_layout.addWidget(self.video_title)
        
        movie_icon = qta.icon('mdi.movie', color='#8B949E')
        title_icon = QLabel()
        title_icon.setPixmap(movie_icon.pixmap(16, 16))
        title_layout.addWidget(title_icon)
        
        layout.addWidget(self.video_title_frame)
        
        # Video preview
        self.video_preview = VideoPreviewWidget()
        # Removed setFixedHeight to allow expansion
        layout.addWidget(self.video_preview)
        
        # Playback controls
        self.controls_frame = VideoControlsWidget()
        layout.addWidget(self.controls_frame)
        
        # Playback progress
        self.progress_frame = QFrame()
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setSpacing(4)
        
        self.progress_title = QLabel("Playback Progress")
        self.progress_title.setFont(QFont("Open Sans", 10))
        progress_layout.addWidget(self.progress_title)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addWidget(self.progress_frame)
        
        # Anomalies and actions
        self.anomalies_frame = QFrame()
        self.anomalies_frame.setFixedHeight(50)  # Reduce space
        anomalies_layout = QHBoxLayout(self.anomalies_frame)
        anomalies_layout.setSpacing(8)
        
        self.anomalies_frame_left = QFrame()
        anomalies_layout_left = QHBoxLayout(self.anomalies_frame_left)
        anomalies_layout_left.setContentsMargins(0, 0, 0, 0)
        anomalies_layout_left.setSpacing(4)
        
        warn_icon = qta.icon('mdi.alert', color='#8B949E')
        warn_label = QLabel()
        warn_label.setPixmap(warn_icon.pixmap(16, 16))
        anomalies_layout_left.addWidget(warn_label)

        self.anomalies_label = QLabel("Detected anomalies at: --")
        anomalies_layout_left.addWidget(self.anomalies_label)
        anomalies_layout.addWidget(self.anomalies_frame_left)

        anomalies_layout.addStretch()

        self.export_btn = QPushButton("Export")
        self.export_btn.setIcon(qta.icon('mdi.download', color='white'))
        self.export_btn.clicked.connect(self.export_results)
        anomalies_layout.addWidget(self.export_btn)

        self.share_btn = QPushButton("Share")
        self.share_btn.setIcon(qta.icon('mdi.share', color='white'))
        self.share_btn.clicked.connect(self.share_results)
        anomalies_layout.addWidget(self.share_btn)
        
        layout.addWidget(self.anomalies_frame)
        
        # Removed layout.addStretch() to allow video preview to expand
        
        return self.right_frame

    def _create_bottom_section(self):
        """Create the bottom section widget for frame-by-frame analysis."""
        self.bottom_frame = QFrame()
        self.bottom_frame.setFixedHeight(200)
        layout = QVBoxLayout(self.bottom_frame)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        # Title and controls
        self.bottom_title_frame = QFrame()
        title_layout = QHBoxLayout(self.bottom_title_frame)
        title_layout.setSpacing(16)
        
        title_subframe = QFrame()
        title_subframe_layout = QHBoxLayout(title_subframe)
        title_subframe_layout.setContentsMargins(0, 0, 0, 0)
        title_subframe_layout.setSpacing(4)
        
        self.frame_title = QLabel("FRAME-BY-FRAME ANALYSIS")
        self.frame_title.setFont(QFont("Open Sans", 10, QFont.Weight.Bold))
        title_subframe_layout.addWidget(self.frame_title)
        
        movie_icon = qta.icon('mdi.movie', color='#8B949E')
        title_icon = QLabel()
        title_icon.setPixmap(movie_icon.pixmap(16, 16))
        title_subframe_layout.addWidget(title_icon)
        
        title_layout.addWidget(title_subframe)
        
        title_layout.addStretch()
        
        self.stats_frame = QFrame()
        stats_layout = QHBoxLayout(self.stats_frame)
        stats_layout.setSpacing(16)
        
        self.frame_label_frame = QFrame()
        frame_layout = QHBoxLayout(self.frame_label_frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(8)
        self.frame_text = QLabel("Frame:")
        frame_layout.addWidget(self.frame_text)
        self.frame_label = QLabel("0/0")
        frame_layout.addWidget(self.frame_label)
        stats_layout.addWidget(self.frame_label_frame)
        
        self.lip_frame = QFrame()
        lip_layout = QHBoxLayout(self.lip_frame)
        lip_layout.setContentsMargins(0, 0, 0, 0)
        lip_layout.setSpacing(8)
        self.lip_text = QLabel("Lip-Sync Error:")
        lip_layout.addWidget(self.lip_text)
        self.lip_label = QLabel("0%")
        lip_layout.addWidget(self.lip_label)
        stats_layout.addWidget(self.lip_frame)

        # Navigation buttons
        self.nav_frame = QFrame()
        nav_layout = QHBoxLayout(self.nav_frame)
        nav_layout.setSpacing(8)

        self.skip_prev_btn = QPushButton()
        self.skip_prev_btn.setIcon(qta.icon('mdi.skip-previous', color='#E6EDF3'))
        self.skip_prev_btn.clicked.connect(self.skip_previous)
        nav_layout.addWidget(self.skip_prev_btn)

        self.prev_btn = QPushButton()
        self.prev_btn.setIcon(qta.icon('mdi.chevron-left', color='#E6EDF3'))
        self.prev_btn.clicked.connect(self.previous_frame)
        nav_layout.addWidget(self.prev_btn)

        self.bottom_play_btn = QPushButton()
        self.bottom_play_btn.setIcon(qta.icon('mdi.play', color='white'))
        self.bottom_play_btn.clicked.connect(self.toggle_playback)
        nav_layout.addWidget(self.bottom_play_btn)

        self.next_btn = QPushButton()
        self.next_btn.setIcon(qta.icon('mdi.chevron-right', color='#E6EDF3'))
        self.next_btn.clicked.connect(self.next_frame)
        nav_layout.addWidget(self.next_btn)

        self.skip_next_btn = QPushButton()
        self.skip_next_btn.setIcon(qta.icon('mdi.skip-next', color='#E6EDF3'))
        self.skip_next_btn.clicked.connect(self.skip_next)
        nav_layout.addWidget(self.skip_next_btn)

        stats_layout.addWidget(self.nav_frame)
        
        title_layout.addWidget(self.stats_frame)
        
        layout.addWidget(self.bottom_title_frame)
        
        # Timeline
        self.timeline = TimelineWidget()
        layout.addWidget(self.timeline)
        
        return self.bottom_frame
    
    def _connect_signals(self):
        """Connect signals to slots with enhanced error handling and recovery."""
        try:
            # File selection signals
            self._safe_connect(
                self.file_manager.file_selected,
                self.start_analysis,
                "file_manager.file_selected -> start_analysis"
            )
            self._safe_connect(
                self.file_manager.error_occurred,
                self.show_error,
                "file_manager.error_occurred -> show_error"
            )
            self._safe_connect(
                self.file_manager.recent_files_updated,
                self.recent_files_widget.update_files,
                "file_manager.recent_files_updated -> recent_files_widget.update_files"
            )
            self._safe_connect(
                self.recent_files_widget.file_selected,
                self.start_analysis,
                "recent_files_widget.file_selected -> start_analysis"
            )
            self._safe_connect(
                self.recent_files_widget.error_occurred,
                self.show_error,
                "recent_files_widget.error_occurred -> show_error"
            )

            # Video player signals
            self._safe_connect(
                self.video_player.frame_changed,
                self.on_frame_selected,
                "video_player.frame_changed -> on_frame_selected"
            )
            self._safe_connect(
                self.video_player.playback_state_changed,
                self._update_playback_state,
                "video_player.playback_state_changed -> _update_playback_state"
            )
            self._safe_connect(
                self.video_player.error_occurred,
                self.show_error,
                "video_player.error_occurred -> show_error"
            )

            # Video controls signals
            self._safe_connect(
                self.controls_frame.play_pause_clicked,
                self.toggle_playback,
                "controls_frame.play_pause_clicked -> toggle_playback"
            )
            self._safe_connect(
                self.controls_frame.rewind_clicked,
                self.video_player.backward,
                "controls_frame.rewind_clicked -> video_player.backward"
            )
            self._safe_connect(
                self.controls_frame.fast_forward_clicked,
                self.video_player.forward,
                "controls_frame.fast_forward_clicked -> video_player.forward"
            )
            self._safe_connect(
                self.controls_frame.seek_position,
                self.video_player.seek,
                "controls_frame.seek_position -> video_player.seek"
            )

            # Video preview signals
            self._safe_connect(
                self.video_preview.frame_changed,
                self.on_frame_selected,
                "video_preview.frame_changed -> on_frame_selected"
            )
            self._safe_connect(
                self.video_preview.error_occurred,
                self.show_error,
                "video_preview.error_occurred -> show_error"
            )

            # Timeline signals
            self._safe_connect(
                self.timeline.frame_selected,
                self.on_frame_selected,
                "timeline.frame_selected -> on_frame_selected"
            )
            self._safe_connect(
                self.timeline.error_occurred,
                self.show_error,
                "timeline.error_occurred -> show_error"
            )

            # Detected issues signals
            self._safe_connect(
                self.issues_widget.frame_selected,
                self.on_frame_selected,
                "issues_widget.frame_selected -> on_frame_selected"
            )
            self._safe_connect(
                self.issues_widget.error_occurred,
                self.show_error,
                "issues_widget.error_occurred -> show_error"
            )

            # Heatmap signals
            self._safe_connect(
                self.heatmap_widget.error_occurred,
                self.show_error,
                "heatmap_widget.error_occurred -> show_error"
            )

            # Manipulation widget signals
            self._safe_connect(
                self.manipulation_widget.error_occurred,
                self.show_error,
                "manipulation_widget.error_occurred -> show_error"
            )

            # Analysis progress signals
            self._safe_connect(
                self.analysis_progress,
                self._update_analysis_progress,
                "analysis_progress -> _update_analysis_progress"
            )

            # Upload button signals
            self._safe_connect(
                self.upload_btn.clicked,
                self.file_manager.show_file_dialog,
                "upload_btn.clicked -> file_manager.show_file_dialog"
            )

            logger.info("All signals connected successfully")
            
        except Exception as e:
            error_msg = f"Failed to connect signals: {str(e)}"
            logger.error(error_msg)
            self.handle_error(
                "Signal Connection Error",
                error_msg,
                self._recover_connections
            )

    def _recover_connections(self):
        """Attempt to recover signal connections."""
        try:
            # Disconnect all signals first
            self._disconnect_all_signals()
            # Retry connecting all signals
            self._connect_signals()
            logger.info("Signal connections recovered successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to recover signal connections: {e}")

    def _disconnect_all_signals(self):
        """Safely disconnect all signals."""
        try:
            # Disconnect file manager signals
            self.file_manager.file_selected.disconnect()
            self.file_manager.error_occurred.disconnect()
            self.file_manager.recent_files_updated.disconnect()
            
            # Disconnect other signals...
            # Add disconnection for each signal
            
            logger.debug("All signals disconnected")
        except Exception as e:
            logger.warning(f"Error during signal disconnection: {e}")
    
    def toggle_playback(self):
        """Toggle video playback state."""
        try:
            logger.debug("Toggle playback button clicked")
            if self.video_player.is_playing:
                self.video_player.pause()
            else:
                self.video_player.play()
            logger.debug("Playback toggled")
        except Exception as e:
            error_msg = f"Failed to toggle playback: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def _update_playback_state(self, is_playing: bool):
        """Update UI elements based on playback state."""
        try:
            icon = qta.icon('mdi.pause' if is_playing else 'mdi.play', color='white')
            self.bottom_play_btn.setIcon(icon)
            self.controls_frame.update_play_pause_icon(icon)
            logger.debug(f"Playback state updated: {'Playing' if is_playing else 'Paused'}")
        except Exception as e:
            error_msg = f"Failed to update playback state UI: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def start_analysis(self, file_path: str):
        """Start video analysis."""
        try:
            self.current_file_path = file_path
            self._set_default_ui_state()
            
            # Load video for playback
            self.video_player.load_video(file_path)
            self.fps = self.video_player.fps
            self.total_frames = self.video_player.total_frames
            self.video_preview.load_video(file_path)
            self.video_preview.setTotalFrames(self.total_frames)
            self.timeline.setTotalFrames(self.total_frames)
            self.controls_frame.update_time(0, self.video_player.total_time)
            self.frame_label.setText(f"0/{self.total_frames-1 if self.total_frames > 0 else 0}")
            cached_result = self.db.get_analysis(file_path)
            
            if cached_result:
                logger.info(f"Using cached analysis for video: {file_path}")
                self.handle_analysis_result(cached_result)
                return
            
            # Perform analysis
            logger.info(f"Starting analysis for video: {file_path}")
            result = self.detector.detect(file_path, progress_callback=self._emit_analysis_progress)
            self.db.save_analysis(file_path, result)
            self.handle_analysis_result(result)
            logger.info(f"Analysis completed for video: {file_path}")
        except Exception as e:
            error_msg = f"Failed to start analysis: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def _emit_analysis_progress(self, percentage: int):
        """Emit analysis progress signal."""
        self.analysis_progress.emit(percentage)
    
    def _update_analysis_progress(self, percentage: int):
        """Update progress bar during analysis."""
        try:
            self.progress_bar.setValue(percentage)
            logger.debug(f"Analysis progress updated: {percentage}%")
        except Exception as e:
            error_msg = f"Failed to update analysis progress: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def handle_analysis_result(self, result: dict):
        """Handle analysis result from the backend."""
        try:
            prediction = result.get('prediction', 0)
            probs = result.get('probs', [0.1, 0.2, 0.5, 0.15, 0.05])
            confidence = max(probs) if probs else 0.0
            heatmap_data = result.get('heatmap', None)
            self.anomalies = result.get('anomalies', [])
            
            # Update classification
            classification = self.class_mapping[prediction] if 0 <= prediction < len(self.class_mapping) else "Neutral"
            self.classification_widget.update_classification(classification, confidence)
            self.video_preview.update_classification(classification)
            
            # Update manipulation probabilities
            self.manipulation_widget.update_probabilities(probs)
            
            # Update detected issues
            self.issues_widget.update_issues(self.anomalies)
            
            # Update heatmap
            self.heatmap_widget.update_data(heatmap_data)
            
            # Update recent files
            if self.current_file_path:
                self.recent_files_widget.update_file_analysis(self.current_file_path, result)
            
            # Update timeline and video preview with anomalies
            anomaly_frames = []
            for anomaly in self.anomalies:
                timestamp = anomaly.get('timestamp', 0.0)
                frame = int(timestamp * self.fps)
                if 0 <= frame < self.total_frames:
                    anomaly_frames.append(frame)
            self.timeline.update_anomalies(anomaly_frames)
            self.video_preview.update_anomalies(anomaly_frames)
            
            # Update anomalies label
            if anomaly_frames:
                self.anomalies_label.setText(f"Detected anomalies at frames: {', '.join(map(str, anomaly_frames))}")
            else:
                self.anomalies_label.setText("No anomalies detected")
            
            logger.info("Analysis results processed successfully")
        except Exception as e:
            error_msg = f"Failed to process analysis result: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def update_progress(self, percentage: float):
        """Update progress bar and labels based on percentage."""
        try:
            self.progress_bar.setValue(int(percentage))
            self.controls_frame.setCurrentPosition(int(percentage))
            self.controls_frame.update_time(
                self.video_player.current_time,
                self.video_player.total_time
            )
            logger.debug(f"Updated progress UI to {percentage:.2f}%")
        except Exception as e:
            error_msg = f"Failed to update progress UI: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)

    def on_frame_selected(self, frame: int):
        """Handle frame selection from timeline, video preview, or detected issues."""
        try:
            logger.debug(f"on_frame_selected called with frame: {frame}")
            if self.total_frames <= 0:
                logger.warning("Total frames is %d, cannot handle frame selection", self.total_frames)
                return
            
            self.video_player.seek_to_frame(frame)
            self.timeline.seek_to(frame)
            self.video_preview.seek_to_frame(frame)
            
            # Update frame display
            frame_data = self.video_player.get_frame(frame)
            if frame_data is not None:
                self.video_preview.update_frame(frame_data, frame)
            
            # Update UI elements
            percentage = (frame / (self.total_frames - 1)) * 100 if self.total_frames > 1 else 0.0
            self.update_progress(percentage)
            self.frame_label.setText(f"{frame}/{self.total_frames-1 if self.total_frames > 0 else 0}")
            
            # Update lip-sync error
            lip_error = 0.0
            for anomaly in self.anomalies:
                anomaly_frame = int(anomaly.get('timestamp', 0.0) * self.fps)
                if anomaly_frame == frame and anomaly.get('issue') == "Lip-sync mismatch":
                    lip_error = anomaly.get('confidence', 0.0) * 100
                    break
            self.lip_label.setText(f"{lip_error:.1f}%")
            
            logger.debug(f"Frame selected: {frame}")
        except Exception as e:
            error_msg = f"Failed to handle frame selection: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def previous_frame(self):
        """Move to the previous frame."""
        try:
            logger.debug("Previous frame button clicked")
            if self.video_player.current_frame > 0:
                self.on_frame_selected(self.video_player.current_frame - 1)
            logger.debug("Moved to previous frame")
        except Exception as e:
            error_msg = f"Failed to move to previous frame: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def next_frame(self):
        """Move to the next frame."""
        try:
            logger.debug("Next frame button clicked")
            if self.video_player.current_frame < self.total_frames - 1:
                self.on_frame_selected(self.video_player.current_frame + 1)
            logger.debug("Moved to next frame")
        except Exception as e:
            error_msg = f"Failed to move to next frame: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def skip_previous(self):
        """Skip to the start."""
        try:
            logger.debug("Skip previous button clicked")
            self.on_frame_selected(0)
            logger.debug("Skipped to start")
        except Exception as e:
            error_msg = f"Failed to skip to start: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def skip_next(self):
        """Skip to the end."""
        try:
            logger.debug("Skip next button clicked")
            if self.total_frames > 0:
                self.on_frame_selected(self.total_frames - 1)
            logger.debug("Skipped to end")
        except Exception as e:
            error_msg = f"Failed to skip to end: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def adjust_heatmap_intensity(self):
        """Adjust heatmap intensity (placeholder)."""
        try:
            QMessageBox.information(self, "Adjust Heatmap Intensity", "Intensity adjustment controlled via slider in HeatmapWidget.")
            logger.debug("Heatmap intensity adjustment triggered")
        except Exception as e:
            error_msg = f"Failed to adjust heatmap intensity: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def update_heatmap_type(self):
        """Update heatmap visualization type."""
        try:
            heatmap_type = self.heatmap_combo.currentText()
            self.heatmap_widget.update_type(heatmap_type)
            logger.debug(f"Heatmap type updated to {heatmap_type}")
        except Exception as e:
            error_msg = f"Failed to update heatmap type: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def export_results(self):
        """Export analysis results to a file."""
        try:
            if not self.current_file_path:
                raise ValueError("No analysis results to export")
            export_path, _ = QFileDialog.getSaveFileName(self, "Export Results", "", "JSON Files (*.json)")
            if export_path:
                import json
                result = {
                    "file_path": self.current_file_path,
                    "classification": self.classification_widget.result_label.text(),
                    "anomalies": self.anomalies,
                    "probabilities": self.manipulation_widget.probs,
                    "heatmap_type": self.heatmap_combo.currentText()
                }
                with open(export_path, 'w') as f:
                    json.dump(result, f, indent=4)
                QMessageBox.information(self, "Export Successful", f"Results exported to {export_path}")
            logger.info(f"Exported results to {export_path}")
        except Exception as e:
            error_msg = f"Failed to export results: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def share_results(self):
        """Share analysis results (placeholder)."""
        try:
            QMessageBox.information(self, "Share Results", "Share results functionality not implemented yet.")
            logger.debug("Share results triggered")
        except Exception as e:
            error_msg = f"Failed to share results: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    
    def show_error(self, message: str):
        """Display error message."""
        try:
            QMessageBox.critical(self, "Error", message)
            logger.error(f"Error displayed: {message}")
        except Exception as e:
            logger.error(f"Failed to display error message: {str(e)}")

    def _set_default_ui_state(self):
        """Set the default UI state before any video is loaded."""
        try:
            self.classification_widget.update_classification("Not Analyzed", 0.0)
            self.manipulation_widget.update_probabilities([0.2] * 5)
            self.issues_widget.update_issues([])
            self.heatmap_widget.update_data(None)
            self.frame_label.setText("0/0")
            self.lip_label.setText("0.0%")
            self.total_frames = 1
            self.fps = 30.0
            self.anomalies = []
            self.anomalies_label.setText("No anomalies detected")
            self.progress_bar.setValue(0)
            self.controls_frame.setSliderPosition(0)
            self.controls_frame.update_time(0, 0)
            self.timeline.setTotalFrames(1)
            self.video_preview.setTotalFrames(1)
            self.timeline.update_anomalies([])
            self.video_preview.update_anomalies([])
            self.video_preview.update_classification("Not Analyzed")
            self.video_player.release()
            logger.info("Default UI state set successfully")
        except Exception as e:
            error_msg = f"Failed to set default UI state: {str(e)}"
            logger.error(error_msg)
            self.show_error(error_msg)
    def handle_error(self, error_type: str, error_msg: str, recovery_action: Optional[callable] = None):
        """Centralized error handling system."""
        try:
            logger.error(f"{error_type}: {error_msg}")
            
            # Create detailed error message
            detailed_msg = f"{error_type}\n\n{error_msg}"
            if recovery_action:
                detailed_msg += "\n\nClick OK to attempt recovery."
            
            # Show error dialog
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText(detailed_msg)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
            
            if msg_box.exec() == QMessageBox.StandardButton.Ok and recovery_action:
                try:
                    recovery_action()
                    logger.info(f"Recovery action completed for {error_type}")
                except Exception as e:
                    logger.error(f"Recovery action failed: {e}")
                    self.show_error(f"Recovery failed: {e}")
                    
        except Exception as e:
            logger.critical(f"Error handler failed: {e}")
    def _recover_video_playback(self):
        """Attempt to recover video playback."""
        try:
            if self.video_player:
                self.video_player.release()
            if self.current_file_path:
                self.video_player.load_video(self.current_file_path)
        except Exception as e:
            raise RuntimeError(f"Playback recovery failed: {e}")

    def _recover_analysis(self):
        """Attempt to recover analysis state."""
        try:
            self._set_default_ui_state()
            if self.current_file_path:
                self.start_analysis(self.current_file_path)
        except Exception as e:
            raise RuntimeError(f"Analysis recovery failed: {e}")

    def _optimize_memory(self):
        """Enhanced memory optimization."""
        try:
            # Clear caches
            if hasattr(self, 'video_player'):
                self.video_player.prefetcher.clear_cache()
            
            # Release unused resources
            gc.collect()
            
            # Monitor memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            
            if memory_info.rss > 1024 * 1024 * 1000:  # 1000 MB
                logger.warning("High memory usage detected")
                self._emergency_cleanup()
                
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")

    def _emergency_cleanup(self):
        """Emergency resource cleanup."""
        try:
            # Release video resources
            if self.video_player:
                self.video_player.release()
            
            # Clear widget caches
            self.video_preview.cleanup()
            self.heatmap_widget.cleanup()
            self.timeline.cleanup()
            
            # Force garbage collection
            gc.collect()
            gc.collect()  # Second pass
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")

    def _track_analysis_progress(self, percentage: int):
        """Enhanced progress tracking with error recovery."""
        try:
            self._update_analysis_progress(percentage)
            
            # Check for timeout
            if hasattr(self, '_last_progress_time'):
                elapsed = time.time() - self._last_progress_time
                if elapsed > 30:  # 30 seconds timeout
                    raise TimeoutError("Analysis timeout")
            
            self._last_progress_time = time.time()
            
        except Exception as e:
            self.handle_error(
                "Analysis Progress Error",
                str(e),
                self._recover_analysis
            )

    def _safe_connect(self, signal, slot, connection_name: str):
        """Safe signal connection with error handling."""
        try:
            signal.connect(slot)
            logger.debug(f"Connected {connection_name}")
        except Exception as e:
            self.handle_error(
                "Signal Connection Error",
                f"Failed to connect {connection_name}: {e}",
                lambda: self._retry_connection(signal, slot, connection_name)
            )

    def _retry_connection(self, signal, slot, connection_name: str):
        """Retry failed signal connection."""
        try:
            # Disconnect if already connected
            try:
                signal.disconnect(slot)
            except:
                pass
            
            # Retry connection
            signal.connect(slot)
            logger.info(f"Successfully reconnected {connection_name}")
            
        except Exception as e:
            raise RuntimeError(f"Connection retry failed: {e}")

    def _apply_styles(self):
        """Apply global styles."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0D1117;
                color: #E6EDF3;
                font-family: 'Open Sans', sans-serif;
            }
            QFrame#header {
                background-color: #161B22;
                border-bottom: 1px solid #30363d;
            }
            QFrame#sidebar {
                background-color: #161B22;
                border-right: 1px solid #30363d;
            }
            QFrame#middle {
                background-color: #161B22;
                border: 1px solid #30363d;
                border-radius: 6px;
            }
            QFrame#right {
                background-color: #161B22;
                border: 1px solid #30363d;
                border-radius: 6px;
            }
            QFrame#bottom {
                background-color: #161B22;
                border-top: 1px solid #30363d;
            }
            QLabel {
                color: #E6EDF3;
            }
            QLabel#title, QLabel#section_title {
                color: #8B949E;
            }
            QLabel#class_value {
                color: #F85149;
            }
            QLabel#class_label, QLabel#manip_label {
                color: #8B949E;
            }
            QLabel#confidence_label {
                background-color: rgba(248, 81, 73, 0.2);
                color: #F85149;
                padding: 2px 8px;
                border-radius: 9999px;
            }
            QLabel#frame_label {
                background-color: rgba(31, 111, 235, 0.2);
                color: #1F6FEB;
                padding: 2px 8px;
                border-radius: 6px;
            }
            QLabel#lip_label {
                background-color: rgba(248, 81, 73, 0.2);
                color: #F85149;
                padding: 2px 8px;
                border-radius: 6px;
            }
            QPushButton {
                background-color: #21262d;
                color: #E6EDF3;
                border: none;
                padding: 6px 12px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #30363d;
            }
            QPushButton#upload_btn {
                background-color: #1F6FEB;
                color: white;
            }
            QPushButton#upload_btn:hover {
                background-color: #1a5fd0;
            }
            QPushButton#play_btn {
                background-color: #1F6FEB;
            }
            QPushButton#play_btn:hover {
                background-color: #1a5fd0;
            }
            QProgressBar {
                background-color: #0D1117;
                border-radius: 4px;
                height: 8px;
            }
            QProgressBar::chunk {
                background-color: #1F6FEB;
                border-radius: 4px;
            }
            QProgressBar#class_progress::chunk {
                background-color: #F85149;
            }
            QComboBox {
                background-color: #21262d;
                color: #E6EDF3;
                padding: 4px;
                border-radius: 6px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #21262d;
                color: #E6EDF3;
                selection-background-color: #30363d;
            }
            QLabel#video_preview, QLabel#heatmap_placeholder {
                background-color: #0D1117;
                border-radius: 6px;
            }
            QLabel#timeline {
                background-color: #0D1117;
                border-radius: 6px;
            }
        """)
        
        # Apply object names for styling
        self.header_frame.setObjectName("header")
        self.sidebar_frame.setObjectName("sidebar")
        self.middle_frame.setObjectName("middle")
        self.right_frame.setObjectName("right")
        self.bottom_frame.setObjectName("bottom")
        self.title.setObjectName("title")
        self.import_title.setObjectName("section_title")
        self.recent_title.setObjectName("section_title")  # Added for Recent Files title
        self.analysis_title.setObjectName("section_title")
        self.manip_label.setObjectName("manip_label")
        self.upload_btn.setObjectName("upload_btn")
        self.bottom_play_btn.setObjectName("play_btn")
        self.frame_text.setObjectName("frame_text")
        self.frame_label.setObjectName("frame_label")
        self.lip_text.setObjectName("lip_text")
        self.lip_label.setObjectName("lip_label")
        self.frame_title.setObjectName("section_title")
        self.video_title.setObjectName("section_title")
        self.video_preview.setObjectName("video_preview")
        self.heatmap_widget.setObjectName("heatmap_placeholder")
        self.timeline.setObjectName("timeline")
            
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            self.timeline.clear_anomalies()
            self.heatmap_widget.update_data(None)
            self.manipulation_widget.update_probabilities([0.2] * 5)
            self.issues_widget.update_issues([])
            self.video_player.release()
            logger.info("Application closed successfully")
            event.accept()
        except Exception as e:
            logger.error(f"Failed to close application: {str(e)}")
            self.show_error(f"Failed to close application: {str(e)}")
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DeepFakeDetectionSystem()
    window.show()
    sys.exit(app.exec())