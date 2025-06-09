"""
DeepFake Detection System - Thread Management
Created: 2025-06-07
Author: ninjacode911

This module implements thread management for asynchronous deepfake detection
with proper resource handling and error management.
"""

import logging
import time
import gc
import threading
from typing import Optional, Dict, Any, Callable
from queue import Queue
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from .exceptions.backend_exceptions import ThreadError
from .detector import Detector
from ..config import config_manager

logger = logging.getLogger(__name__)

class AnalysisThread(QThread):
    """Thread for asynchronous deepfake detection analysis."""
    
    # Signals
    progress_updated = pyqtSignal(int)  # Progress percentage
    result_ready = pyqtSignal(dict)     # Analysis results
    error_occurred = pyqtSignal(str)    # Error message
    memory_warning = pyqtSignal(float)  # Memory usage percentage
    
    def __init__(
        self,
        media_path: str,
        options: Optional[Dict[str, Any]] = None,
        parent=None
    ) -> None:
        """
        Initialize analysis thread.

        Args:
            media_path: Path to media file
            options: Analysis options
            parent: Parent QObject
        """
        super().__init__(parent)
        self.media_path = Path(media_path)
        self.options = options or {}
        self._detector: Optional[Detector] = None
        self._stop_flag = threading.Event()
        self._pause_flag = threading.Event()
        self._result_queue = Queue()
        self._error_queue = Queue()
        
        # Resource management
        self._resource_lock = threading.Lock()
        self._max_memory_percent = config_manager.get(
            "performance.max_memory_percent", 75.0
        )
        
        self._initialize()

    def _initialize(self) -> None:
        """Initialize thread resources."""
        try:
            with self._resource_lock:
                self._detector = Detector()
                logger.debug("Analysis thread initialized")
        except Exception as e:
            logger.error(f"Thread initialization failed: {e}")
            raise ThreadError(
                message="Thread initialization failed",
                error_code=7001,
                operation="initialize",
                details={'error': str(e)}
            )

    def run(self) -> None:
        """Execute analysis in background thread."""
        try:
            if not self.media_path.exists():
                raise FileNotFoundError(f"Media file not found: {self.media_path}")

            # Monitor resources
            self._start_resource_monitoring()
            
            # Analysis progress callback
            def progress_callback(percent: int) -> None:
                if self._stop_flag.is_set():
                    raise InterruptedError("Analysis cancelled")
                if self._pause_flag.is_set():
                    self._pause_flag.wait()
                self.progress_updated.emit(percent)

            # Run analysis
            with self._resource_lock:
                result = self._detector.detect(
                    str(self.media_path),
                    options=self.options,
                    progress_callback=progress_callback
                )

            if self._stop_flag.is_set():
                return

            self._result_queue.put(result)
            self.result_ready.emit(result)
            logger.info(f"Analysis completed for {self.media_path}")

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._error_queue.put(error_msg)
            self.error_occurred.emit(error_msg)
            
        finally:
            self._cleanup()

    def _start_resource_monitoring(self) -> None:
        """Start monitoring system resources."""
        def monitor_resources():
            import psutil
            while not self._stop_flag.is_set():
                try:
                    memory_percent = psutil.Process().memory_percent()
                    if memory_percent > self._max_memory_percent:
                        self.memory_warning.emit(memory_percent)
                        gc.collect()
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"Resource monitoring error: {e}")

        self._monitor_thread = threading.Thread(
            target=monitor_resources,
            daemon=True
        )
        self._monitor_thread.start()

    def pause(self) -> None:
        """Pause analysis."""
        self._pause_flag.set()
        logger.debug("Analysis paused")

    def resume(self) -> None:
        """Resume analysis."""
        self._pause_flag.clear()
        logger.debug("Analysis resumed")

    def stop(self) -> None:
        """Stop analysis."""
        self._stop_flag.set()
        self._pause_flag.clear()
        logger.debug("Analysis stop requested")

    def _cleanup(self) -> None:
        """Clean up thread resources."""
        try:
            with self._resource_lock:
                if self._detector is not None:
                    self._detector.cleanup()
                    self._detector = None
                
                # Clear queues
                while not self._result_queue.empty():
                    self._result_queue.get()
                while not self._error_queue.empty():
                    self._error_queue.get()
                
                # Force garbage collection
                gc.collect()
                
            logger.debug("Thread resources cleaned up")
            
        except Exception as e:
            logger.error(f"Thread cleanup failed: {e}")
            raise ThreadError(
                message="Thread cleanup failed",
                error_code=7003,
                operation="cleanup",
                details={'error': str(e)}
            )

    def get_last_error(self) -> Optional[str]:
        """Get last error message from queue."""
        try:
            return self._error_queue.get_nowait()
        except:
            return None

    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get analysis result from queue."""
        try:
            return self._result_queue.get_nowait()
        except:
            return None

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        try:
            self.stop()
            self._cleanup()
        except Exception as e:
            logger.error(f"Thread deletion cleanup failed: {e}")