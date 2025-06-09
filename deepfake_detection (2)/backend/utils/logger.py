"""
DeepFake Detection System - Logger Module
Created: 2025-06-07
Author: ninjacode911

Enhanced logging system with performance optimization and resource management.
"""

import os
import sys
import time 
import json
import queue
import logging
import threading
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from concurrent.futures import ThreadPoolExecutor

# Constants
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5
LOG_QUEUE_SIZE = 1000
METRICS_EXPORT_INTERVAL = 3600  # 1 hour
LOG_RETENTION_DAYS = 30

class LogLevel(Enum):
    """Available logging levels with descriptions."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING 
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class LoggerError(Exception):
    """Custom exception for logger errors."""
    pass

class LogFormatter(logging.Formatter):
    """Enhanced log formatter with color support and context."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[1;31m' # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and context."""
        if hasattr(record, 'context'):
            record.msg = f"{record.msg} | Context: {record.context}"
            
        if sys.stdout.isatty():  # Check if output is terminal
            level_name = record.levelname
            if level_name in self.COLORS:
                record.levelname = f"{self.COLORS[level_name]}{level_name}{self.RESET}"
        
        return super().format(record)

class Logger:
    """Enhanced logger with performance optimization and resource management."""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> 'Logger':
        """Ensure singleton pattern with thread safety."""
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
                raise LoggerError(f"Logger initialization failed: {str(e)}")

    def _setup_logger(self) -> None:
        """Set up logging system with queue-based handlers."""
        try:
            # Create logs directory
            self.log_dir = Path("logs")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create logger
            self.logger = logging.getLogger("deepfake_detector")
            self.logger.setLevel(logging.DEBUG)
            
            # Set up queue
            self._log_queue = queue.Queue(maxsize=LOG_QUEUE_SIZE)
            
            # Configure handlers
            self._setup_handlers()
            
            # Create queue listener
            self._listener = QueueListener(
                self._log_queue,
                *self.logger.handlers,
                respect_handler_level=True
            )
            self._listener.start()
            
        except Exception as e:
            raise LoggerError(f"Logger setup failed: {str(e)}")

    def _setup_handlers(self) -> None:
        """Configure logging handlers with rotation and formatting."""
        try:
            # File handler
            log_file = self.log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
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
            formatter = LogFormatter(LOG_FORMAT, DATE_FORMAT)
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # Add handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
        except Exception as e:
            raise LoggerError(f"Handler setup failed: {str(e)}")

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
        """Log a message with specified level and context."""
        try:
            extra = self._prepare_context(**kwargs)
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

    def _prepare_context(self, **kwargs: Any) -> Dict[str, Any]:
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
            current_time = time.time()
            for log_file in self.log_dir.glob("*.log*"):
                if (current_time - log_file.stat().st_mtime) > (LOG_RETENTION_DAYS * 86400):
                    log_file.unlink()
        except Exception as e:
            sys.stderr.write(f"Log cleanup failed: {str(e)}\n")

    def _periodic_metrics_export(self) -> None:
        """Export metrics periodically."""
        while True:
            try:
                self._export_metrics()
                time.sleep(METRICS_EXPORT_INTERVAL)
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
logger = Logger()

# Convenience logging methods
def debug(message: str, **kwargs: Any) -> None:
    logger.log(LogLevel.DEBUG.value, message, **kwargs)

def info(message: str, **kwargs: Any) -> None:
    logger.log(LogLevel.INFO.value, message, **kwargs)

def warning(message: str, **kwargs: Any) -> None:
    logger.log(LogLevel.WARNING.value, message, **kwargs)

def error(message: str, **kwargs: Any) -> None:
    logger.log(LogLevel.ERROR.value, message, **kwargs)

def critical(message: str, **kwargs: Any) -> None:
    logger.log(LogLevel.CRITICAL.value, message, **kwargs)

def setup_logger(
    name: str,
    level: Union[int, str] = LogLevel.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional path to log file
        console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = LogFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str = "backend", level: int = LogLevel.INFO):
    return setup_logger(name, level)

def configure_logging():
    """Configure logging for the application."""
    setup_logger("backend", LogLevel.INFO)