import logging
from typing import Optional, Dict, Any
from pathlib import Path
import time

from PyQt6.QtCore import QThread, pyqtSignal as Signal, QTimer

from .client import APIClient

logger = logging.getLogger(__name__)

class DetectionThread(QThread):
    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def run(self):
        try:
            result = self.detector.detect(
                self.media_path,
                progress_callback=self.progress_updated.emit
            )
            self.result_ready.emit(result.to_dict())
        except Exception as e:
            self.error_occurred.emit(str(e))

class AnalysisThread(QThread):
    """Thread for running video analysis in background."""
    
    # Signals
    analysis_complete = Signal(dict)  # Emits analysis results
    analysis_error = Signal(str)      # Emits error messages
    progress_updated = Signal(int)    # Emits progress percentage

    def __init__(self, file_path: str, parent=None, timeout: int = 300):
        """Initialize analysis thread."""
        super().__init__(parent)
        self.file_path = file_path
        self.client = APIClient()
        self._is_cancelled = False
        self._timeout = timeout
        self._start_time = None
        self._timeout_timer = QTimer()
        self._timeout_timer.timeout.connect(self._check_timeout)
        logger.debug(f"Analysis thread initialized for {file_path}")

    def run(self):
        """Execute analysis in background thread."""
        try:
            if not Path(self.file_path).exists():
                raise FileNotFoundError(f"Video file not found: {self.file_path}")

            # Start timeout timer
            self._start_time = time.time()
            self._timeout_timer.start(1000)  # Check every second

            # Progress callback
            def update_progress(percentage: int):
                if self._is_cancelled:
                    raise InterruptedError("Analysis cancelled")
                self.progress_updated.emit(percentage)

            # Run analysis
            result = self.client.analyze_video(
                self.file_path,
                progress_callback=update_progress
            )

            if self._is_cancelled:
                logger.info("Analysis cancelled")
                return

            self.analysis_complete.emit(result)
            logger.info("Analysis completed successfully")

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            logger.error(error_msg)
            self.analysis_error.emit(error_msg)
        finally:
            self._cleanup()

    def _check_timeout(self):
        """Check if analysis has exceeded timeout."""
        if self._start_time and time.time() - self._start_time > self._timeout:
            self._is_cancelled = True
            self.analysis_error.emit("Analysis timed out")
            self._cleanup()

    def cancel(self):
        """Cancel ongoing analysis."""
        self._is_cancelled = True
        logger.info("Analysis cancellation requested")
        self._cleanup()

    def _cleanup(self):
        """Cleanup resources."""
        try:
            self._timeout_timer.stop()
            if hasattr(self, 'client'):
                self.client.cleanup()
        except Exception as e:
            logger.error(f"Thread cleanup failed: {e}")