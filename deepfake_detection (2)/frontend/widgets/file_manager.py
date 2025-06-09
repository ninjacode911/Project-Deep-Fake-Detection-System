#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced File Manager Widget with drag-drop support and retry logic
Created on: 2025-04-28 07:50:42 UTC 
Last Modified: 2025-06-07 14:30:00 UTC
Author: ninjacode911
"""

import os
import logging
import time
from typing import List, Set, Optional, Dict
from pathlib import Path
from functools import wraps

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QLabel, QFrame, QHBoxLayout, QProgressBar, QMessageBox
)
from PyQt6.QtCore import (
    Qt, pyqtSignal as Signal, QMimeData,
    QUrl, QFileInfo, QTimer
)
from PyQt6.QtGui import (
    QDragEnterEvent, QDropEvent, QColor
)
import qtawesome as qta

from frontend.config import config_manager

logger = logging.getLogger(__name__)

def retry_on_error(retries: int = 3, delay: float = 1.0):
    """Decorator for retry logic."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_error = None
            for attempt in range(retries):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < retries - 1:
                        time.sleep(delay)
            
            logger.error(f"All {retries} attempts failed: {last_error}")
            self.error_occurred.emit(str(last_error))
            return None
        return wrapper
    return decorator

class DropZone(QFrame):
    """Enhanced drop zone with visual feedback."""
    
    files_dropped = Signal(list)  # List of file paths
    error_occurred = Signal(str)  # Error message
    
    def __init__(self, parent: Optional[QWidget] = None):
        try:
            super().__init__(parent)
            self._setup_ui()
            self._is_dragging = False
            self._retry_count: Dict[str, int] = {}
            self.setAcceptDrops(True)
            
        except Exception as e:
            logger.error(f"Failed to initialize DropZone: {e}")
            raise

    # ... (rest of DropZone implementation remains the same)

class FileManager(QWidget):
    """Enhanced file manager widget with drag-drop support and retry logic."""
    
    file_selected = Signal(str)  # File path
    error_occurred = Signal(str)  # Error message
    retry_attempted = Signal(str)  # Operation being retried
    state_changed = Signal(str)  # Current state
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the file manager with retry handling."""
        try:
            super().__init__(parent)
            self._recent_files: Set[str] = set()
            self._retry_count: Dict[str, int] = {}
            self._pending_retries: Dict[str, QTimer] = {}
            self._loading = False
            self._setup_ui()
            self._setup_retry_handlers()
            logger.debug("FileManager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize FileManager: {e}")
            raise

    def _setup_retry_handlers(self) -> None:
        """Setup retry mechanism."""
        try:
            self._retry_timer = QTimer(self)
            self._retry_timer.timeout.connect(self._process_pending_retries)
            self._retry_timer.start(5000)  # Check every 5 seconds
            
        except Exception as e:
            logger.error(f"Failed to setup retry handlers: {e}")
            raise

    @retry_on_error(retries=3)
    def browse_files(self) -> None:
        """Open file browser dialog with retry logic."""
        try:
            if self._loading:
                return
                
            self._loading = True
            self.state_changed.emit("browsing")
            
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
            file_dialog.setNameFilter(
                "Video Files (*.mp4 *.avi *.mov *.mkv *.webm)")
            
            if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
                files = file_dialog.selectedFiles()
                if files:
                    self._handle_dropped_files(files)
                    
        except Exception as e:
            error_msg = f"File browse error: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self._schedule_retry("browse_files")
            
        finally:
            self._loading = False
            self.state_changed.emit("ready")

    def _schedule_retry(self, operation: str) -> None:
        """Schedule operation retry."""
        try:
            if operation not in self._retry_count:
                self._retry_count[operation] = 0
                
            if self._retry_count[operation] < 3:
                self._retry_count[operation] += 1
                delay = self._retry_count[operation] * 2000  # Exponential backoff
                
                timer = QTimer(self)
                timer.setSingleShot(True)
                timer.timeout.connect(lambda: self._retry_operation(operation))
                timer.start(delay)
                
                self._pending_retries[operation] = timer
                self.retry_attempted.emit(operation)
                logger.debug(f"Scheduled retry for {operation}")
                
        except Exception as e:
            logger.error(f"Failed to schedule retry: {e}")

    def _retry_operation(self, operation: str) -> None:
        """Retry failed operation."""
        try:
            if operation == "browse_files":
                self.browse_files()
            elif operation.startswith("handle_file:"):
                _, file_path = operation.split(":", 1)
                self._handle_dropped_files([file_path])
                
            if operation in self._pending_retries:
                self._pending_retries[operation].stop()
                del self._pending_retries[operation]
                
        except Exception as e:
            logger.error(f"Retry failed for {operation}: {e}")

    def _process_pending_retries(self) -> None:
        """Process any pending retry operations."""
        try:
            current_time = time.time()
            for operation, timer in list(self._pending_retries.items()):
                if not timer.isActive():
                    del self._pending_retries[operation]
                    del self._retry_count[operation]
                    
        except Exception as e:
            logger.error(f"Failed to process retries: {e}")

    def cleanup(self) -> None:
        """Enhanced cleanup with retry handler cleanup."""
        try:
            # Stop retry timer
            if hasattr(self, '_retry_timer'):
                self._retry_timer.stop()
            
            # Clear pending retries
            for timer in self._pending_retries.values():
                timer.stop()
            self._pending_retries.clear()
            self._retry_count.clear()
            
            # Clear recent files
            self.clear_recent_files()
            
            # Cleanup signals
            try:
                self.file_selected.disconnect()
                self.error_occurred.disconnect()
                self.retry_attempted.disconnect()
                self.state_changed.disconnect()
            except:
                pass
            
            logger.debug("FileManager cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def _validate_state(self) -> bool:
        """Validate widget state."""
        try:
            required_attributes = [
                '_recent_files',
                '_retry_count',
                '_pending_retries',
                '_loading',
                '_retry_timer',
                'drop_zone',
                'browse_button'
            ]
            
            return all(hasattr(self, attr) for attr in required_attributes)
            
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            return False