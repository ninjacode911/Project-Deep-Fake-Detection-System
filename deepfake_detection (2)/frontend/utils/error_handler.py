#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Error handling utilities for the frontend components.
"""

import logging
from enum import Enum
from typing import Optional, Callable, Any , Deque , Dict
from PyQt6.QtWidgets import QMessageBox
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorLevel(Enum):
    """Error severity levels."""
    INFO = "Information"
    WARNING = "Warning" 
    ERROR = "Error"
    CRITICAL = "Critical Error"

class ErrorHandler:
    """Centralized error handling for the application."""
    
    def __init__(self, max_errors: int = 100):
        self._error_queue = deque(maxlen=max_errors)
        self.error_callbacks: dict[str, Callable] = {}
        
    def register_callback(self, error_type: str, callback: Callable) -> None:
        """Register error handling callback."""
        self.error_callbacks[error_type] = callback
        
    def handle_error(self, error: Exception, level: ErrorLevel, 
                    context: str = "", show_dialog: bool = True) -> None:
        """Handle an error with appropriate logging and user feedback."""
        try:
            error_msg = f"{context}: {str(error)}" if context else str(error)
            
            # Log error
            if level == ErrorLevel.CRITICAL:
                logger.critical(error_msg, exc_info=True)
            elif level == ErrorLevel.ERROR:
                logger.error(error_msg, exc_info=True)
            elif level == ErrorLevel.WARNING:
                logger.warning(error_msg)
            else:
                logger.info(error_msg)
                
            # Show dialog if requested
            if show_dialog:
                QMessageBox.critical(None, level.value, error_msg)
                
            # Call registered error callbacks
            error_type = type(error).__name__
            if error_type in self.error_callbacks:
                self.error_callbacks[error_type](error)
                
        except Exception as e:
            logger.critical(f"Error handler failed: {e}", exc_info=True)
            
    def cleanup(self) -> None:
        """Clean up resources."""
        self._error_queue.clear()
        self.error_callbacks.clear()

# Global instance
error_handler = ErrorHandler()

def handle_error(error: Exception, level: ErrorLevel = ErrorLevel.ERROR,
                context: str = "", show_dialog: bool = True) -> None:
    """Convenience function to handle errors using global handler."""
    error_handler.handle_error(error, level, context, show_dialog)