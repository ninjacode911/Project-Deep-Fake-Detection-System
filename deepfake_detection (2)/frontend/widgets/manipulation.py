import logging
from typing import Optional, List, Dict, Union, Tuple, Any
import gc
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QFrame, QMenu,
                            QProgressBar)
from PyQt6.QtCore import (pyqtSignal as Signal, Qt, QRect, QPropertyAnimation,
                         QEasingCurve, QPoint, pyqtProperty as Property,
                         QTimer)
from PyQt6.QtGui import (QPainter, QColor, QFont, QMouseEvent, QPaintEvent,
                        QAction, QContextMenuEvent, QPainterPath, QLinearGradient)
import qtawesome as qta

logger = logging.getLogger(__name__)

class AnimatedBar:
    """Helper class for smooth bar animations with spring effect."""
    
    def __init__(self, start_value: float = 0.2):
        self._value = start_value
        self._target = start_value
        self._velocity = 0.0
        self._spring_constant = 0.3
        self._damping = 0.7
        self._animation = None

    def get_value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        self._value = value

    def update_spring(self) -> bool:
        """Update spring animation. Returns True if still animating."""
        force = (self._target - self._value) * self._spring_constant
        self._velocity = (self._velocity + force) * self._damping
        self._value += self._velocity
        return abs(self._velocity) > 0.0001 or abs(self._target - self._value) > 0.0001

    def reset(self) -> None:
        """Reset animation state."""
        self._value = 0.2
        self._target = 0.2
        self._velocity = 0.0

    value = Property(float, get_value, set_value)

class ManipulationWidget(QWidget):
    """
    Enhanced widget for displaying manipulation probabilities with advanced animations.
    
    Features:
    - Smooth spring-based animations
    - Interactive tooltips and context menu
    - Error handling and validation
    - Automatic probability normalization
    - Gradient overlays and glow effects
    - Export capabilities
    """
    
    error_occurred = Signal(str)
    probability_updated = Signal(int, float)
    animation_completed = Signal()
    state_changed = Signal(str)
    
    def __init__(self, parent: Optional[QWidget] = None):
        try:
            super().__init__(parent)
            
            # Initialize state
            self._setup_data()
            self._init_ui()
            self._apply_styles()
            self._setup_animations()
            self._validate_state()
            
            logger.info("ManipulationWidget initialized")
            self.state_changed.emit("initialized")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.error_occurred.emit(str(e))

    def _validate_state(self) -> None:
        """Validate widget state."""
        try:
            required_attrs = [
                '_bars', '_labels', '_colors', '_animations',
                '_animation_timer', '_spring_animation_active',
                '_loading_indicator'
            ]
            
            for attr in required_attrs:
                if not hasattr(self, attr):
                    raise AttributeError(f"Missing required attribute: {attr}")
                    
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            self.error_occurred.emit(f"Validation Error: {str(e)}")

    def _setup_data(self) -> None:
        """Initialize data structures."""
        self._bars = [AnimatedBar() for _ in range(5)]
        self._labels = ["Fake", "Likely Fake", "Neutral", "Likely Real", "Real"]
        self._colors = {
            "normal": ["#F85149", "#FF8C61", "#8B949E", "#54AEFF", "#2EA043"],
            "hover": ["#FF6661", "#FFA07A", "#A0A8B3", "#70BFFF", "#4ABF58"],
            "gradient": {
                "start": ["#FF362E", "#FF7042", "#6E7681", "#388BFD", "#238636"],
                "end": ["#F85149", "#FF8C61", "#8B949E", "#54AEFF", "#2EA043"]
            }
        }
        self._hovered_bar: Optional[int] = None
        self._animations: List[QPropertyAnimation] = []
        self._animation_timer = None
        self._spring_animation_active = False

    def _init_ui(self) -> None:
        """Initialize the user interface."""
        try:
            layout = QVBoxLayout()
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(8)

            # Title section with icon
            title_frame = QFrame()
            title_layout = QHBoxLayout(title_frame)
            title_layout.setContentsMargins(0, 0, 0, 0)

            icon_label = QLabel()
            icon = qta.icon('mdi.chart-bar', color='#8B949E')
            icon_label.setPixmap(icon.pixmap(16, 16))
            title_layout.addWidget(icon_label)

            title = QLabel("MANIPULATION PROBABILITIES")
            title.setObjectName("sectionTitle")
            title_layout.addWidget(title)
            title_layout.addStretch()

            layout.addWidget(title_frame)

            # Loading indicator
            self._loading_indicator = QProgressBar()
            self._loading_indicator.setTextVisible(False)
            self._loading_indicator.setMaximumHeight(2)
            self._loading_indicator.hide()
            layout.addWidget(self._loading_indicator)

            self.setMinimumHeight(200)
            self.setLayout(layout)

        except Exception as e:
            logger.error(f"Failed to initialize UI: {e}")
            self.error_occurred.emit(f"UI Error: {str(e)}")

    def _setup_animations(self) -> None:
        """Setup animation timer with performance optimization."""
        try:
            self._animation_timer = QTimer(self)
            self._animation_timer.setInterval(16)  # 60 FPS
            self._animation_timer.timeout.connect(self._update_spring_animations)
            
        except Exception as e:
            logger.error(f"Animation setup failed: {e}")
            self.error_occurred.emit(f"Animation Error: {str(e)}")

    def _update_spring_animations(self) -> None:
        """Update spring-based animations with sync."""
        try:
            still_animating = False
            for i, bar in enumerate(self._bars):
                if bar.update_spring():
                    still_animating = True
                    self.probability_updated.emit(i, bar.value)
            
            if still_animating:
                self.update()
            else:
                self._animation_timer.stop()
                self._spring_animation_active = False
                self.animation_completed.emit()
                
        except Exception as e:
            logger.error(f"Animation update failed: {e}")
            self._stop_animations()

    def _stop_animations(self) -> None:
        """Safely stop all animations."""
        try:
            if self._animation_timer and self._animation_timer.isActive():
                self._animation_timer.stop()
            self._spring_animation_active = False
            self.state_changed.emit("animations_stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop animations: {e}")

    def set_loading(self, loading: bool) -> None:
        """Show/hide loading state."""
        try:
            self._loading_indicator.setVisible(loading)
            if loading:
                self._loading_indicator.setRange(0, 0)
            else:
                self._loading_indicator.setRange(0, 100)
            self.state_changed.emit("loading" if loading else "ready")
            
        except Exception as e:
            logger.error(f"Failed to set loading state: {e}")

    def reset_state(self) -> None:
        """Reset widget to initial state."""
        try:
            # Stop animations
            self._stop_animations()
            
            # Reset bars
            for bar in self._bars:
                bar.reset()
            
            # Clear state
            self._hovered_bar = None
            self._spring_animation_active = False
            
            # Update UI
            self.update()
            self.state_changed.emit("reset")
            
        except Exception as e:
            logger.error(f"Reset failed: {e}")
            self.error_occurred.emit(f"Reset Error: {str(e)}")

    def cleanup(self) -> None:
        """Enhanced cleanup with proper resource management."""
        try:
            # Stop animations
            self._stop_animations()
            
            # Clear animations
            self._animations.clear()
            
            # Reset bars
            for bar in self._bars:
                bar.reset()
            
            # Clear state
            self._hovered_bar = None
            self._spring_animation_active = False
            
            # Force cleanup
            gc.collect()
            
            logger.debug("ManipulationWidget cleaned up")
            self.state_changed.emit("cleaned_up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            self.error_occurred.emit(f"Cleanup Error: {str(e)}")

    # ... (rest of the existing methods remain unchanged)

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()