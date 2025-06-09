#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging configuration for the frontend components.
Author: ninjacode911
Created: 2025-06-05
"""

import os
import sys
import time
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Union

class LogLevel(Enum):
    """Available logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO 
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class LogFormatter(logging.Formatter):
    """Custom log formatter with color support."""
    
    COLORS = {
        'DEBUG': '\033[0;36m',  # Cyan
        'INFO': '\033[0;32m',   # Green
        'WARNING': '\033[0;33m', # Yellow
        'ERROR': '\033[0;31m',   # Red
        'CRITICAL': '\033[1;31m' # Bold Red
    }
    RESET = '\033[0m'

    def format(self, record):
        # Add color if outputting to terminal
        if sys.stdout.isatty():
            record.levelname = (f"{self.COLORS.get(record.levelname, '')}"
                              f"{record.levelname}{self.RESET}")
        return super().format(record)

def setup_logger(filename: str, max_size: int = 5_242_880, backup_count: int = 3):
    """Setup rotating file logger."""
    handler = RotatingFileHandler(
        filename=filename,
        maxBytes=max_size,
        backupCount=backup_count
    )
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def setup_logger(
    filename: Union[str, Path],
    log_level: LogLevel = LogLevel.INFO,
    max_size: int = 5_242_880,  # 5MB
    backup_count: int = 3,
    log_to_console: bool = True
) -> None:
    """
    Configure application logging with rotation and optional console output.
    
    Args:
        filename: Log file path
        log_level: Logging level to use
        max_size: Maximum size of each log file in bytes
        backup_count: Number of backup files to keep
        log_to_console: Whether to also log to console
    """
    try:
        # Create logs directory if needed
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = LogFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level.value)

        # Remove existing handlers
        root_logger.handlers.clear()

        # File handler
        file_handler = RotatingFileHandler(
            filename=filename,
            maxBytes=max_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Console handler (optional)
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            root_logger.addHandler(console_handler)

        # Log initialization
        logger = logging.getLogger(__name__)
        logger.info(
            "Logging system initialized at %s on %s",
            datetime.now().strftime("%I:%M %p"),
            datetime.now().strftime("%A, %B %d, %Y")
        )
        
    except Exception as e:
        print(f"Failed to initialize logging: {e}", file=sys.stderr)
        sys.exit(1)