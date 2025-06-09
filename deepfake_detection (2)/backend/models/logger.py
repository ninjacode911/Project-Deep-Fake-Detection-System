"""
DeepFake Detection System - Model Logger
Created: 2025-06-07
Author: ninjacode911

Enhanced logging system for model operations with performance optimization,
error tracking, and resource management.
"""

import logging
import sys
import time
import json
import queue
import threading
import traceback
from typing import Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from concurrent.futures import ThreadPoolExecutor

# Configure constants
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5
LOG_QUEUE_SIZE = 1000

class ModelLoggerError(Exception):
    """Custom exception for logger errors."""
    pass

class ModelLogger:
    """
    Enhanced logger for model operations with performance optimization and error tracking.
    
    Features:
    - Asynchronous logging using queue
    - Performance metrics tracking
    - Memory usage monitoring
    - Error aggregation
    - Resource cleanup
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'ModelLogger':
        """Ensure singleton pattern with thread-safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize logger with advanced configuration."""
        if not hasattr(self, '_initialized'):
            try:
                self._initialized = True
                self._setup_logger()
                self._setup_metrics()
                self._start_background_tasks()
            except Exception as e:
                raise ModelLoggerError(f"Logger initialization failed: {str(e)}")

    def _setup_logger(self) -> None:
        """Set up the logging system with queue-based handlers."""
        try:
            # Create logs directory
            self.log_dir = Path("logs/models")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create logger
            self.logger = logging.getLogger("model_logger")
            self.logger.setLevel(logging.DEBUG)
            
            # Set up queue
            self._log_queue = queue.Queue(maxsize=LOG_QUEUE_SIZE)
            
            # Create handlers
            self._setup_handlers()
            
            # Create queue listener
            self._listener = QueueListener(
                self._log_queue,
                *self.logger.handlers,
                respect_handler_level=True
            )
            self._listener.start()
            
        except Exception as e:
            raise ModelLoggerError(f"Logger setup failed: {str(e)}")

    def _setup_handlers(self) -> None:
        """Configure logging handlers with rotation and formatting."""
        try:
            # File handler
            log_file = self.log_dir / f"model_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = RotatingFileHandler(
                filename=log_file,
                maxBytes=MAX_LOG_SIZE,
                backupCount=BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
        except Exception as e:
            raise ModelLoggerError(f"Handler setup failed: {str(e)}")

    def _setup_metrics(self) -> None:
        """Initialize performance metrics tracking."""
        self._metrics = {
            'error_count': 0,
            'warning_count': 0,
            'start_time': time.time(),
            'operations': {}
        }
        self._metrics_lock = threading.Lock()

    def _start_background_tasks(self) -> None:
        """Initialize background tasks for maintenance."""
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._executor.submit(self._periodic_cleanup)
        self._executor.submit(self._periodic_metrics_export)

    def log(self, level: int, message: str, **kwargs: Any) -> None:
        """
        Log a message with the specified level and additional context.
        
        Args:
            level: Logging level
            message: Log message
            **kwargs: Additional context
        """
        try:
            extra = self._prepare_log_context(**kwargs)
            record = logging.LogRecord(
                name=self.logger.name,
                level=level,
                pathname=__file__,
                lineno=0,
                msg=message,
                args=(),
                exc_info=None,
                extra=extra
            )
            self._log_queue.put_nowait(record)
            self._update_metrics(level)
            
        except queue.Full:
            sys.stderr.write(f"Warning: Log queue full, message dropped: {message}\n")
        except Exception as e:
            sys.stderr.write(f"Error logging message: {str(e)}\n")

    def _prepare_log_context(self, **kwargs: Any) -> Dict[str, Any]:
        """Prepare context information for logging."""
        context = {
            'timestamp': datetime.now().isoformat(),
            'thread_id': threading.get_ident()
        }
        context.update(kwargs)
        return {'context': context}

    def _update_metrics(self, level: int) -> None:
        """Update logging metrics with thread safety."""
        with self._metrics_lock:
            if level >= logging.ERROR:
                self._metrics['error_count'] += 1
            elif level >= logging.WARNING:
                self._metrics['warning_count'] += 1

    def _periodic_cleanup(self) -> None:
        """Periodic cleanup of old log files."""
        while True:
            try:
                self._cleanup_old_logs()
                time.sleep(86400)  # Daily cleanup
            except Exception as e:
                sys.stderr.write(f"Cleanup error: {str(e)}\n")

    def _cleanup_old_logs(self) -> None:
        """Remove old log files beyond retention period."""
        try:
            retention_days = 30
            current_time = time.time()
            
            for log_file in self.log_dir.glob("*.log*"):
                if (current_time - log_file.stat().st_mtime) > (retention_days * 86400):
                    log_file.unlink()
                    
        except Exception as e:
            sys.stderr.write(f"Log cleanup failed: {str(e)}\n")

    def _periodic_metrics_export(self) -> None:
        """Export metrics periodically."""
        while True:
            try:
                self._export_metrics()
                time.sleep(3600)  # Hourly export
            except Exception as e:
                sys.stderr.write(f"Metrics export error: {str(e)}\n")

    def _export_metrics(self) -> None:
        """Export current metrics to JSON file."""
        try:
            metrics_file = self.log_dir / "metrics.json"
            with metrics_file.open('w') as f:
                json.dump(self._metrics, f, indent=2)
        except Exception as e:
            sys.stderr.write(f"Metrics export failed: {str(e)}\n")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current logging metrics."""
        with self._metrics_lock:
            return self._metrics.copy()

    def cleanup(self) -> None:
        """Clean up resources properly."""
        try:
            self._listener.stop()
            self._executor.shutdown(wait=True)
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
        except Exception as e:
            sys.stderr.write(f"Logger cleanup failed: {str(e)}\n")

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        self.cleanup()

# Create global logger instance
model_logger = ModelLogger()

# Convenience methods
def debug(message: str, **kwargs: Any) -> None:
    model_logger.log(logging.DEBUG, message, **kwargs)

def info(message: str, **kwargs: Any) -> None:
    model_logger.log(logging.INFO, message, **kwargs)

def warning(message: str, **kwargs: Any) -> None:
    model_logger.log(logging.WARNING, message, **kwargs)

def error(message: str, **kwargs: Any) -> None:
    model_logger.log(logging.ERROR, message, **kwargs)

def critical(message: str, **kwargs: Any) -> None:
    model_logger.log(logging.CRITICAL, message, **kwargs)

def get_model_logger() -> logging.Logger:
    """
    Stub for getting a model logger.
    
    Returns:
        A logger instance for model logging.
    """
    return logging.getLogger(__name__)