#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Theme manager for the DeepFake Detection System.
Handles loading and applying themes/styles.
Created on: 2025-06-05 14:30:00 UTC
Author: ninjacode911
"""

import logging
from pathlib import Path
from typing import Optional, Dict
from PyQt6.QtWidgets import QApplication
from functools import lru_cache

logger = logging.getLogger(__name__)

class ThemeManager:
    """Manages application theming and styles."""
    
    def __init__(self):
        """Initialize theme manager with paths and default theme."""
        self.style_dir = Path(__file__).parent / "styles"
        self.icons_dir = Path(__file__).parent / "icons"
        self.current_theme = "dark"
        self._theme_cache: Dict[str, str] = {}
        self._current_theme: Optional[str] = None
        logger.debug("ThemeManager initialized")

    @lru_cache(maxsize=4)
    def load_theme(self, theme_name: str) -> str:
        """Load and cache theme."""
        try:
            if theme_name in self._cache:
                return self._cache[theme_name]
                
            theme_path = self._get_theme_path(theme_name)
            theme_data = self._load_theme_file(theme_path)
            
            if not self._validate_theme(theme_data):
                raise ValueError(f"Invalid theme: {theme_name}")
                
            self._cache[theme_name] = theme_data
            return theme_data
            
        except Exception as e:
            logger.error(f"Theme loading failed: {e}")
            return self.load_theme("dark")  # Fallback


    def apply_theme(self, app: QApplication, theme_name: str = "dark") -> bool:
        """Apply theme to application."""
        try:
            stylesheet = self.load_theme(theme_name)
            if stylesheet:
                app.setStyleSheet(stylesheet)
                self.current_theme = theme_name
                logger.info(f"Applied theme: {theme_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to apply theme: {e}")
            return False

    def validate_theme(self, theme_data: str) -> bool:
        """Validate theme data contains required selectors."""
        required_selectors = {
            "QMainWindow", "QWidget", "QPushButton", 
            "QLabel", "QSlider", "QComboBox"
        }
        return all(selector in theme_data for selector in required_selectors)

    def cleanup(self) -> None:
        """Clean up resources."""
        self._cache.clear()
        self.load_theme.cache_clear()

# Global instance
theme_manager = ThemeManager()