import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
import gc

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QFrame, 
    QLabel, QPushButton, QHBoxLayout, QStackedWidget,
    QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal as Signal, QTimer, QSize
from PyQt6.QtGui import QColor, QPalette, QFont, QIcon
import qtawesome as qta

logger = logging.getLogger(__name__)

class FilterState(Enum):
    """Filter states for issue display."""
    ALL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()

@dataclass
class Issue:
    """Data class for detected issues."""
    id: int
    type: str
    severity: str
    frame: int
    confidence: float
    description: str
    timestamp: datetime

class IssueCard(QFrame):
    """Card widget for displaying individual issues."""
    
    clicked = Signal(int)  # Issue ID
    
    def __init__(self, issue: Issue, parent=None):
        super().__init__(parent)
        self.issue = issue
        self._setup_ui()
        self._apply_styles()
        
    def _setup_ui(self):
        """Initialize card UI elements."""
        try:
            layout = QVBoxLayout(self)
            layout.setSpacing(8)
            layout.setContentsMargins(12, 12, 12, 12)

            # Header with type and severity
            header = QHBoxLayout()
            
            type_label = QLabel(self.issue.type)
            type_label.setObjectName("issueType")
            header.addWidget(type_label)
            
            severity_label = QLabel(self.issue.severity)
            severity_label.setObjectName(f"severity{self.issue.severity}")
            header.addWidget(severity_label)
            
            header.addStretch()
            
            # Confidence indicator
            conf_label = QLabel(f"{self.issue.confidence:.1%}")
            conf_label.setObjectName("confidence")
            header.addWidget(conf_label)
            
            layout.addLayout(header)

            # Description
            desc = QLabel(self.issue.description)
            desc.setObjectName("description")
            desc.setWordWrap(True)
            layout.addWidget(desc)

            # Footer with frame and timestamp
            footer = QHBoxLayout()
            
            frame_label = QLabel(f"Frame: {self.issue.frame}")
            frame_label.setObjectName("frameNum")
            footer.addWidget(frame_label)
            
            footer.addStretch()
            
            time_label = QLabel(self.issue.timestamp.strftime("%H:%M:%S"))
            time_label.setObjectName("timestamp")
            footer.addWidget(time_label)
            
            layout.addLayout(footer)

        except Exception as e:
            logger.error(f"Failed to setup issue card UI: {e}")
            raise

    def _apply_styles(self):
        """Apply custom styles to the card."""
        self.setObjectName("issueCard")
        self.setStyleSheet("""
            QFrame#issueCard {
                background-color: #21262D;
                border: 1px solid #30363D;
                border-radius: 6px;
            }
            QFrame#issueCard:hover {
                border-color: #388BFD;
            }
            QLabel#issueType {
                color: #E6EDF3;
                font-weight: bold;
                font-size: 14px;
            }
            QLabel#severityHigh {
                color: #F85149;
                font-weight: bold;
                padding: 2px 8px;
                background: rgba(248, 81, 73, 0.1);
                border-radius: 10px;
            }
            QLabel#severityMedium {
                color: #F8E957;
                font-weight: bold;
                padding: 2px 8px;
                background: rgba(248, 233, 87, 0.1);
                border-radius: 10px;
            }
            QLabel#severityLow {
                color: #2EA043;
                font-weight: bold;
                padding: 2px 8px;
                background: rgba(46, 160, 67, 0.1);
                border-radius: 10px;
            }
            QLabel#confidence {
                color: #388BFD;
                font-weight: bold;
            }
            QLabel#description {
                color: #8B949E;
                font-size: 13px;
                line-height: 1.4;
            }
            QLabel#frameNum, QLabel#timestamp {
                color: #6E7681;
                font-size: 12px;
            }
        """)

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        super().mousePressEvent(event)
        self.clicked.emit(self.issue.id)

class DetectedIssuesWidget(QWidget):
    """Widget for displaying detected deepfake issues with enhanced UI."""
    
    issue_selected = Signal(int)  # Issue ID
    error_occurred = Signal(str)
    state_changed = Signal(str)  # State description
    
    def __init__(self, parent: Optional[QWidget] = None):
        try:
            super().__init__(parent)
            self._issues: Dict[int, Issue] = {}
            self._issue_cache: Dict[int, IssueCard] = {}
            self._current_filter = FilterState.ALL
            self._loading = False
            self._visible_issues: Set[int] = set()
            
            # Initialize UI
            self._setup_ui()
            self._apply_styles()
            self._validate_state()
            
            # Setup cleanup timer
            self._cleanup_timer = QTimer(self)
            self._cleanup_timer.timeout.connect(self._perform_cache_cleanup)
            self._cleanup_timer.start(300000)  # 5 minutes
            
            logger.debug("DetectedIssuesWidget initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize DetectedIssuesWidget: {e}")
            raise

    def _validate_state(self) -> None:
        """Validate widget state."""
        try:
            required_attributes = [
                '_issues', '_issue_cache', '_current_filter',
                '_container', '_container_layout', '_empty_label',
                '_filter_buttons', '_loading_indicator'
            ]
            
            for attr in required_attributes:
                if not hasattr(self, attr):
                    raise AttributeError(f"Missing required attribute: {attr}")
                    
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            self.error_occurred.emit(str(e))

    def _setup_ui(self):
        """Set up the user interface."""
        try:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)

            # Header with title and filters
            header = QHBoxLayout()
            
            title = QLabel("Detected Issues")
            title.setObjectName("sectionTitle")
            header.addWidget(title)
            
            # Filter buttons
            self._filter_buttons = {}
            for filter_type in FilterState:
                btn = QPushButton(filter_type.name.title())
                btn.setObjectName(f"filter{filter_type.name.lower()}")
                btn.setCheckable(True)
                btn.clicked.connect(lambda x, t=filter_type: self._apply_filter(t))
                header.addWidget(btn)
                self._filter_buttons[filter_type] = btn
            
            self._filter_buttons[FilterState.ALL].setChecked(True)
            header.addStretch()
            
            layout.addLayout(header)

            # Loading indicator
            self._loading_indicator = QProgressBar()
            self._loading_indicator.setTextVisible(False)
            self._loading_indicator.setObjectName("loadingIndicator")
            self._loading_indicator.hide()
            layout.addWidget(self._loading_indicator)

            # Scroll area for issues
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setObjectName("issueScroll")
            
            self._container = QWidget()
            self._container_layout = QVBoxLayout(self._container)
            self._container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            scroll.setWidget(self._container)
            
            layout.addWidget(scroll)

            # Empty state
            self._empty_label = QLabel("No issues detected")
            self._empty_label.setObjectName("emptyState")
            self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._container_layout.addWidget(self._empty_label)
            
        except Exception as e:
            logger.error(f"Failed to setup UI: {e}")
            raise

    def set_loading(self, loading: bool) -> None:
        """Set loading state with UI feedback."""
        try:
            self._loading = loading
            self._loading_indicator.setVisible(loading)
            self._empty_label.setText("Loading issues..." if loading else "No issues detected")
            self._empty_label.setVisible(loading or not self._issues)
            
            # Disable interaction during loading
            for btn in self._filter_buttons.values():
                btn.setEnabled(not loading)
            
            self.state_changed.emit("loading" if loading else "ready")
            
        except Exception as e:
            logger.error(f"Failed to set loading state: {e}")
            self.error_occurred.emit(str(e))

    def reset_state(self) -> None:
        """Reset widget to initial state."""
        try:
            # Clear data structures
            self._issues.clear()
            self._issue_cache.clear()
            self._visible_issues.clear()
            
            # Reset UI
            self.clear()
            self._current_filter = FilterState.ALL
            self._filter_buttons[FilterState.ALL].setChecked(True)
            
            # Force garbage collection
            gc.collect()
            
            self.state_changed.emit("reset")
            logger.debug("Widget state reset")
            
        except Exception as e:
            logger.error(f"State reset failed: {e}")
            self.error_occurred.emit(str(e))

    def _perform_cache_cleanup(self) -> None:
        """Perform periodic cache cleanup."""
        try:
            # Remove cached cards for non-visible issues
            to_remove = []
            for issue_id in self._issue_cache:
                if issue_id not in self._visible_issues:
                    to_remove.append(issue_id)
            
            for issue_id in to_remove:
                del self._issue_cache[issue_id]
            
            gc.collect()
            logger.debug(f"Cleaned up {len(to_remove)} cached issues")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

    def add_issue(self, issue: Issue) -> None:
        """Add a new issue to the widget."""
        try:
            self._issues[issue.id] = issue
            
            # Create or reuse card
            if issue.id in self._issue_cache:
                card = self._issue_cache[issue.id]
            else:
                card = IssueCard(issue)
                card.clicked.connect(self._on_issue_selected)
                self._issue_cache[issue.id] = card
            
            # Hide empty state if needed
            self._empty_label.setVisible(False)
            
            # Add card based on current filter
            if self._should_show_issue(issue):
                self._container_layout.addWidget(card)
                self._visible_issues.add(issue.id)
                
            logger.debug(f"Added issue {issue.id}")
            
        except Exception as e:
            error_msg = f"Failed to add issue: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def _should_show_issue(self, issue: Issue) -> bool:
        """Check if issue should be shown based on current filter."""
        try:
            if self._current_filter == FilterState.ALL:
                return True
            return issue.severity.upper() == self._current_filter.name
        except Exception as e:
            logger.error(f"Filter check failed: {e}")
            return False

    def _apply_filter(self, filter_type: FilterState) -> None:
        """Apply severity filter with optimized updates."""
        try:
            self._current_filter = filter_type
            self._visible_issues.clear()
            
            # Update button states
            for state, btn in self._filter_buttons.items():
                btn.setChecked(state == filter_type)
            
            # Optimize layout updates
            self._container.setUpdatesEnabled(False)
            try:
                self._update_visible_issues()
            finally:
                self._container.setUpdatesEnabled(True)
            
            self.state_changed.emit(f"filter_{filter_type.name.lower()}")
            logger.debug(f"Applied filter: {filter_type.name}")
            
        except Exception as e:
            error_msg = f"Failed to apply filter: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def _update_visible_issues(self) -> None:
        """Update visible issues with optimized layout handling."""
        try:
            # Clear current layout
            while self._container_layout.count():
                item = self._container_layout.takeAt(0)
                if item.widget():
                    item.widget().setVisible(False)
            
            # Add matching issues
            matching_issues = False
            for issue_id, issue in self._issues.items():
                if self._should_show_issue(issue):
                    matching_issues = True
                    self._visible_issues.add(issue_id)
                    
                    if issue_id in self._issue_cache:
                        card = self._issue_cache[issue_id]
                        card.setVisible(True)
                    else:
                        card = IssueCard(issue)
                        card.clicked.connect(self._on_issue_selected)
                        self._issue_cache[issue_id] = card
                    
                    self._container_layout.addWidget(card)
            
            # Update empty state
            self._empty_label.setVisible(not matching_issues)
            
        except Exception as e:
            logger.error(f"Failed to update visible issues: {e}")
            self.error_occurred.emit(str(e))

    def _on_issue_selected(self, issue_id: int) -> None:
        """Handle issue selection."""
        try:
            self.issue_selected.emit(issue_id)
            logger.debug(f"Issue selected: {issue_id}")
        except Exception as e:
            logger.error(f"Failed to handle issue selection: {e}")
            self.error_occurred.emit(str(e))

    def clear(self) -> None:
        """Clear all issues."""
        try:
            self._issues.clear()
            self._visible_issues.clear()
            
            while self._container_layout.count():
                item = self._container_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            
            self._empty_label.setVisible(True)
            logger.debug("Cleared all issues")
            
        except Exception as e:
            error_msg = f"Failed to clear issues: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def cleanup(self) -> None:
        """Enhanced cleanup with cache invalidation."""
        try:
            # Clear all issues
            self.clear()
            
            # Clear caches
            self._issue_cache.clear()
            self._visible_issues.clear()
            
            # Stop cleanup timer
            if self._cleanup_timer.isActive():
                self._cleanup_timer.stop()
            
            # Force garbage collection
            gc.collect()
            
            logger.debug("DetectedIssuesWidget cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")