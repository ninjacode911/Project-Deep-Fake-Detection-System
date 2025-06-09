#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recent Files Widget with enhanced caching and performance
Created on: 2025-04-28 07:50:42 UTC
Last Modified: 2025-06-07 14:30:00 UTC
Author: ninjacode911
"""

import logging
import gc
from typing import Optional, List, Dict, Union, Tuple, Any, OrderedDict
from pathlib import Path
from functools import lru_cache
from datetime import datetime
from enum import Enum, auto

from PyQt6.QtWidgets import (
    QListWidget, QWidget, QMenu, QMessageBox, 
    QListWidgetItem, QFrame, QVBoxLayout, 
    QPushButton, QLabel, QProgressBar
)
from PyQt6.QtCore import (
    pyqtSignal as Signal, Qt, QPoint, 
    QTimer, QThread, QMutex
)
from PyQt6.QtGui import (
    QAction, QIcon, QColor, QPainter, 
    QFont, QPalette, QBrush
)
import qtawesome as qta

from frontend.config import config_manager
from backend.database import Database

logger = logging.getLogger(__name__)

class WidgetState(Enum):
    """Widget states for validation."""
    INITIALIZING = auto()
    READY = auto()
    UPDATING = auto()
    ERROR = auto()
    CLEANED = auto()

class CacheManager:
    """Manager for caching file history and analysis results."""
    
    def __init__(self, max_entries: int = 50):
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._max_entries = max_entries
        self._mutex = QMutex()
        self._background_colors: Dict[int, QColor] = {}
        self._last_validation = 0.0

    def get(self, key: str) -> Optional[Dict]:
        """Get cached entry."""
        with QMutex():
            return self._cache.get(key)

    def set(self, key: str, value: Dict) -> None:
        """Add entry to cache."""
        with QMutex():
            if len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def remove(self, key: str) -> None:
        """Remove entry from cache."""
        with QMutex():
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        with QMutex():
            self._cache.clear()
            self._background_colors.clear()
            self._last_validation = 0.0

    def validate(self) -> bool:
        """Validate cache integrity."""
        try:
            current_time = datetime.now().timestamp()
            if current_time - self._last_validation < 300:  # 5 minutes
                return True
                
            with QMutex():
                # Remove invalid entries
                invalid_keys = []
                for key, value in self._cache.items():
                    if key.startswith("prediction_"):
                        file_path = key.split("_", 1)[1]
                        if not Path(file_path).exists():
                            invalid_keys.append(key)
                
                for key in invalid_keys:
                    self._cache.pop(key)
                
                self._last_validation = current_time
                return True
                
        except Exception as e:
            logger.error(f"Cache validation failed: {e}")
            return False

    @lru_cache(maxsize=10)
    def get_background_color(self, prediction: int) -> QColor:
        """Get cached background color for prediction class."""
        return self._background_colors.get(
            prediction, 
            QColor("#21262d")
        )

    def set_background_color(self, prediction: int, color: QColor) -> None:
        """Cache background color for prediction class."""
        self._background_colors[prediction] = color


class RecentFilesWidget(QListWidget):
    """
    Enhanced widget for displaying recent files with caching.
    
    Features:
    - LRU caching for file history
    - Background color caching
    - Lazy loading of entries
    - Automatic cleanup of invalid entries
    - Context menu operations
    - Error handling and logging
    - State validation
    - Loading indicators
    """
    
    file_selected = Signal(str)  # Emits file path
    error_occurred = Signal(str)  # Emits error messages
    list_updated = Signal(int)   # Emits count of valid entries
    state_changed = Signal(str)  # Emits current state
    
    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the widget."""
        try:
            super().__init__(parent)
            self._state = WidgetState.INITIALIZING
            self._setup_widget()
            self._initialize_data()
            self._setup_cache()
            self._setup_loading_indicator()
            self._apply_styles()
            
            # Setup periodic updates
            self._update_timer = QTimer()
            self._update_timer.timeout.connect(self.update_files)
            self._update_timer.start(300000)  # Update every 5 minutes
            
            # Setup cache validation timer
            self._cache_validator = QTimer()
            self._cache_validator.timeout.connect(self._validate_cache)
            self._cache_validator.start(60000)  # Validate every minute
            
            self.update_files()
            self._state = WidgetState.READY
            self.state_changed.emit("initialized")
            logger.debug("RecentFilesWidget initialized")
            
        except Exception as e:
            self._state = WidgetState.ERROR
            logger.error(f"Initialization failed: {e}")
            self.error_occurred.emit(f"Widget initialization failed: {e}")

    def _setup_widget(self) -> None:
        """Configure widget properties."""
        try:
            self.setFixedHeight(400)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setWordWrap(True)
            self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.customContextMenuRequested.connect(self._show_context_menu)
            self.itemClicked.connect(self._on_item_clicked)
            
        except Exception as e:
            logger.error(f"Widget setup failed: {e}")
            raise

    def _setup_loading_indicator(self) -> None:
        """Setup loading progress bar."""
        try:
            self._loading_bar = QProgressBar(self)
            self._loading_bar.setTextVisible(False)
            self._loading_bar.setMaximumHeight(2)
            self._loading_bar.hide()
            
            # Add to layout
            if self.layout() is None:
                layout = QVBoxLayout(self)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self._loading_bar)
                
        except Exception as e:
            logger.error(f"Loading indicator setup failed: {e}")

    def _set_loading(self, loading: bool) -> None:
        """Show/hide loading state."""
        try:
            if loading:
                self._state = WidgetState.UPDATING
                self._loading_bar.setRange(0, 0)  # Indeterminate
                self._loading_bar.show()
            else:
                self._state = WidgetState.READY
                self._loading_bar.hide()
                
            self.setEnabled(not loading)
            self.state_changed.emit("loading" if loading else "ready")
            
        except Exception as e:
            logger.error(f"Failed to set loading state: {e}")

    def _validate_cache(self) -> None:
        """Validate cache integrity."""
        try:
            if not self._cache.validate():
                logger.warning("Cache validation failed, clearing cache")
                self._cache.clear()
                self.update_files()
        except Exception as e:
            logger.error(f"Cache validation failed: {e}")

    def _validate_state(self) -> bool:
        """Validate widget state."""
        try:
            required_attrs = [
                '_cache', 'db', '_state', '_loading_bar',
                '_update_timer', '_cache_validator'
            ]
            
            for attr in required_attrs:
                if not hasattr(self, attr):
                    raise AttributeError(f"Missing required attribute: {attr}")
            
            return True
            
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            self.error_occurred.emit(f"Validation Error: {str(e)}")
            return False

    def _initialize_data(self) -> None:
        """Initialize data sources and mappings."""
        try:
            self.db = Database()
            self.class_mapping = ["Fake", "Likely Fake", "Neutral", "Likely Real", "Real"]
            self.class_colors = ["#F85149", "#FF8C61", "#8B949E", "#54AEFF", "#2EA043"]
            self._is_updating = False
            
        except Exception as e:
            logger.error(f"Data initialization failed: {e}")
            raise

    def _setup_cache(self) -> None:
        """Initialize cache manager."""
        try:
            self._cache = CacheManager()
            # Pre-cache background colors
            for i, color in enumerate(self.class_colors):
                self._cache.set_background_color(i, QColor(color).darker(150))
                
        except Exception as e:
            logger.error(f"Cache setup failed: {e}")
            raise

    def _apply_styles(self) -> None:
        """Apply widget styles."""
        try:
            self.setStyleSheet("""
                QListWidget {
                    background-color: #161B22;
                    border: none;
                    outline: none;
                    padding: 4px;
                }
                QListWidget::item {
                    background-color: #21262D;
                    border: 1px solid #30363D;
                    border-radius: 4px;
                    padding: 8px;
                    margin: 2px 4px;
                }
                QListWidget::item:hover {
                    border-color: #388BFD;
                }
                QListWidget::item:selected {
                    background-color: #1F6FEB;
                    border-color: #388BFD;
                }
                QScrollBar:vertical {
                    background-color: #161B22;
                    width: 8px;
                    margin: 0px;
                }
                QScrollBar::handle:vertical {
                    background-color: #30363D;
                    border-radius: 4px;
                }
                QScrollBar::handle:vertical:hover {
                    background-color: #388BFD;
                }
                QScrollBar::add-line:vertical,
                QScrollBar::sub-line:vertical {
                    height: 0px;
                }
                QProgressBar {
                    background-color: #161B22;
                    border: none;
                }
                QProgressBar::chunk {
                    background-color: #1F6FEB;
                }
            """)
            logger.debug("Styles applied")
            
        except Exception as e:
            logger.error(f"Style application failed: {e}")

    @lru_cache(maxsize=1)
    def _get_classification(self, prediction: int) -> str:
        """Get cached classification text."""
        if 0 <= prediction < len(self.class_mapping):
            return self.class_mapping[prediction]
        return "Unknown"

    def _create_list_item(self, file_path: str, classification: str, 
                         prediction: int) -> QListWidgetItem:
        """Create a formatted list item."""
        try:
            item = QListWidgetItem(f"{Path(file_path).name} - {classification}")
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            item.setData(Qt.ItemDataRole.UserRole + 1, prediction)
            
            font = QFont("Open Sans", 12)
            item.setFont(font)
            item.setForeground(Qt.GlobalColor.white)
            
            # Use cached background color
            color = self._cache.get_background_color(prediction)
            item.setBackground(color)
            
            item.setToolTip(f"Path: {file_path}\nClassification: {classification}")
            return item
            
        except Exception as e:
            logger.error(f"Item creation failed: {e}")
            raise

    def update_files(self) -> None:
        """Update file list with caching."""
        if self._is_updating:
            return

        self._is_updating = True
        self._set_loading(True)
        
        try:
            self.clear()
            history = self._get_file_history()
            
            for entry in history:
                self._add_history_item(entry)
            
            count = self.count()
            self.list_updated.emit(count)
            logger.info(f"Updated list with {count} entries")
            
        except Exception as e:
            error_msg = f"Update failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            
        finally:
            self._is_updating = False
            self._set_loading(False)

    def _get_file_history(self) -> List[Dict]:
        """Get file history with caching."""
        try:
            # Try to get from cache first
            cache_key = "file_history"
            cached = self._cache.get(cache_key)
            if cached:
                return cached

            # Get from database
            history = self.db.get_history()
            self._cache.set(cache_key, history)
            return history
            
        except Exception as e:
            logger.warning(f"History fetch failed: {e}, using config")
            return [{"file_path": fp, "result": None} 
                   for fp in config_manager.get_recent_files()]

    def _add_history_item(self, entry: Dict) -> None:
        """Add history entry with caching."""
        try:
            file_path = entry['file_path']
            result = entry.get('result', None)
            
            # Try to get cached prediction
            cache_key = f"prediction_{file_path}"
            prediction = self._cache.get(cache_key)
            
            if prediction is None:
                prediction = self._get_prediction(result)
                self._cache.set(cache_key, prediction)
            
            classification = self._get_classification(prediction)
            item = self._create_list_item(file_path, classification, prediction)
            self.addItem(item)
            
        except Exception as e:
            logger.error(f"Failed to add item: {e}")

    def _get_prediction(self, result: Optional[Dict]) -> int:
        """Get prediction from result."""
        if result and isinstance(result.get('prediction'), int):
            return result['prediction']
        return 0

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle item selection."""
        try:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            self.file_selected.emit(file_path)
            logger.debug(f"Selected: {file_path}")
            
        except Exception as e:
            error_msg = f"Selection failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def _show_context_menu(self, position: QPoint) -> None:
        """Show context menu."""
        try:
            item = self.itemAt(position)
            if not item:
                return

            menu = QMenu(self)
            analyze_action = menu.addAction(
                qta.icon('mdi.play', color='white'),
                "Analyze Again"
            )
            remove_action = menu.addAction(
                qta.icon('mdi.delete', color='#F85149'),
                "Remove from History"
            )

            action = menu.exec(self.mapToGlobal(position))
            if action == analyze_action:
                self._on_item_clicked(item)
            elif action == remove_action:
                self._remove_item(item)
                
        except Exception as e:
            logger.error(f"Context menu error: {e}")

    def _remove_item(self, item: QListWidgetItem) -> None:
        """Remove item and clear cache."""
        try:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            self.db.remove_analysis(file_path)
            config_manager.remove_recent_file(file_path)
            
            # Clear cached entries
            self._cache.remove(f"prediction_{file_path}")
            self._cache.remove("file_history")
            
            self.update_files()
            logger.info(f"Removed: {file_path}")
            
        except Exception as e:
            error_msg = f"Remove failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def clear_history(self) -> None:
        """Clear history and cache."""
        self._set_loading(True)
        try:
            self.db.clear_history()
            config_manager.clear_recent_files()
            self._cache.clear()
            self.clear()
            self.list_updated.emit(0)
            logger.info("History cleared")
            
        except Exception as e:
            error_msg = f"Clear failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
        finally:
            self._set_loading(False)

    def update_file_analysis(self, file_path: str, result: dict) -> None:
        """Update analysis with cache refresh."""
        self._set_loading(True)
        try:
            self.db.save_analysis(file_path, result)
            
            # Clear cached entries
            self._cache.remove(f"prediction_{file_path}")
            self._cache.remove("file_history")
            
            self.update_files()
            logger.info(f"Updated analysis: {file_path}")
            
        except Exception as e:
            error_msg = f"Update failed: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
        finally:
            self._set_loading(False)

    def cleanup(self) -> None:
        """Enhanced cleanup with proper resource management."""
        try:
            self._state = WidgetState.CLEANED
            
            # Stop timers
            if hasattr(self, '_update_timer'):
                self._update_timer.stop()
            if hasattr(self, '_cache_validator'):
                self._cache_validator.stop()
            
            # Clear caches
            if hasattr(self, '_cache'):
                self._cache.clear()
                self._get_classification.cache_clear()
            
            # Clear UI
            self.clear()
            
            # Clear database connection
            if hasattr(self, 'db'):
                self.db = None
            
            # Force cleanup
            gc.collect()
            
            logger.debug("Cleanup complete")
            self.state_changed.emit("cleaned")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()