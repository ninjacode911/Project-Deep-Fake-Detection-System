#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ClassificationWidget for displaying deepfake detection results.
Created on: 2025-04-28 07:50:42 UTC
Last Modified: 2025-06-07 14:30:00 UTC
Author: ninjacode911
"""

import logging
from typing import Optional, Dict, Any
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, 
                           QHBoxLayout, QFrame, QSizePolicy)
from PyQt6.QtCore import (Qt, QTimer, QPropertyAnimation, 
                         QEasingCurve, pyqtSignal, pyqtProperty)
from PyQt6.QtGui import (QFont, QColor, QPainter, QPaintEvent, 
                        QLinearGradient, QResizeEvent)
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)

class ClassificationWidget(QWidget):
    """Displays classification results with animated confidence visualization."""
    
    # Signals
    state_changed = pyqtSignal(str)  # Emits current state
    animation_completed = pyqtSignal()
    
    # Class constants
    ANIMATION_DURATION = 500  # ms
    UPDATE_INTERVAL = 16      # ms (~60 FPS)
    CONFIDENCE_STEPS = 100    # Animation smoothness
    
    # Classification colors
    STATUS_COLORS = {
        "Fake": "#F85149",
        "Likely Fake": "#FF8C61", 
        "Neutral": "#8B949E",
        "Likely Real": "#54AEFF",
        "Real": "#2EA043",
        "Not Analyzed": "#6E7681"
    }

    def __init__(self, parent: Optional[QWidget] = None):
        try:
            super().__init__(parent)
            self._state = ClassificationState()
            self._animation_state = AnimationState.IDLE
            self._cache: Dict[str, Any] = {}
            
            # Optimize animation timer
            self._animation_timer = QTimer(self)
            self._animation_timer.setTimerType(Qt.TimerType.PreciseTimer)
            
            # Initialize UI with state validation
            self._setup_ui()
            self._validate_initial_state()
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self._handle_error("Initialization failed", e)

    def reset_state(self) -> None:
        """Reset widget to initial state."""
        try:
            # Stop any running animations
            self._stop_animations()
            
            # Reset state
            self._state = ClassificationState()
            self._animation_state = AnimationState.IDLE
            
            # Clear cache
            self._cache.clear()
            
            # Reset UI
            self._update_ui_from_state()
            self.state_changed.emit("reset")
            
        except Exception as e:
            logger.error(f"State reset failed: {e}")
            self._handle_error("Reset failed", e)

    def _stop_animations(self) -> None:
        """Safely stop all animations."""
        try:
            if self._animation_timer.isActive():
                self._animation_timer.stop()
            self._animation_state = AnimationState.IDLE
            self._state.is_animating = False
            
        except Exception as e:
            logger.error(f"Animation stop failed: {e}")

    def _validate_initial_state(self) -> None:
        """Validate initial widget state."""
        try:
            required_attributes = [
                '_state', '_animation_state', '_cache',
                '_animation_timer', 'result_label', 
                'confidence_label'
            ]
            
            for attr in required_attributes:
                if not hasattr(self, attr):
                    raise AttributeError(f"Missing required attribute: {attr}")
                    
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            self._handle_error("Validation failed", e)

    def _update_ui_from_state(self) -> None:
        """Update UI elements from current state."""
        try:
            self.result_label.setText(self._state.class_name)
            self.confidence_label.setText(f"{self._state.confidence:.1%}")
            
            # Update colors
            self._update_colors()
            
            # Trigger repaint if needed
            if self._state.is_animating:
                self.update()
                
        except Exception as e:
            logger.error(f"UI update failed: {e}")
            self._handle_error("UI update failed", e)

    def _handle_error(self, context: str, error: Exception) -> None:
        """Centralized error handling."""
        try:
            self._state.is_error = True
            self._animation_state = AnimationState.ERROR
            
            error_msg = f"{context}: {str(error)}"
            logger.error(error_msg)
            
            self.error_occurred.emit(error_msg)
            self._show_error_state()
            
        except Exception as e:
            logger.critical(f"Error handler failed: {e}")

    def _setup_ui(self) -> None:
        """Set up the user interface components."""
        try:
            # Main layout
            layout = QVBoxLayout(self)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

            # Result frame
            result_frame = QFrame()
            result_frame.setObjectName("resultFrame")
            result_frame.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Preferred
            )
            
            result_layout = QHBoxLayout(result_frame)
            result_layout.setSpacing(12)

            # Classification label
            self.result_label = QLabel(self._class_name)
            self.result_label.setFont(QFont("Open Sans", 16, QFont.Weight.Bold))
            self.result_label.setObjectName("resultLabel")
            result_layout.addWidget(self.result_label)

            # Confidence label
            self.confidence_label = QLabel("0%")
            self.confidence_label.setObjectName("confidenceLabel")
            result_layout.addWidget(self.confidence_label)
            
            layout.addWidget(result_frame)

            # Set minimum size
            self.setMinimumHeight(120)
            
        except Exception as e:
            logger.error(f"Failed to setup UI: {e}")
            raise

    # Add new state management classes
    @dataclass
    class ClassificationState:
        """Classification widget state."""
        class_name: str = "Not Analyzed"
        confidence: float = 0.0
        is_animating: bool = False
        is_error: bool = False
        last_update: float = 0.0

    class AnimationState(Enum):
        """Animation states."""
        IDLE = auto()
        RUNNING = auto()
        ERROR = auto()
        PAUSED = auto()

    def _apply_styles(self) -> None:
        """Apply widget styles."""
        try:
            self.setStyleSheet("""
                QFrame#resultFrame {
                    background-color: #161B22;
                    border: 1px solid #30363D;
                    border-radius: 8px;
                    padding: 16px;
                }
                
                QLabel#resultLabel {
                    color: #E6EDF3;
                    font-weight: bold;
                }
                
                QLabel#confidenceLabel {
                    background-color: rgba(31, 111, 235, 0.2);
                    color: #1F6FEB;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-weight: bold;
                }
            """)
        except Exception as e:
            logger.error(f"Failed to apply styles: {e}")
            raise

    def update_classification(self, class_name: str, confidence: float) -> None:
        """Update classification results."""
        if not self._is_initialized:
            logger.warning("Widget not initialized")
            return

        try:
            # Validate inputs
            if not isinstance(confidence, (int, float)):
                raise ValueError(f"Confidence must be numeric, got {type(confidence)}")
            if not 0 <= confidence <= 1:
                raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
            if not isinstance(class_name, str):
                raise ValueError(f"Class name must be string, got {type(class_name)}")

            # Update state
            self._class_name = class_name
            self._target_confidence = confidence
            self._animation_progress = 0.0

            # Update UI
            self.result_label.setText(class_name)
            self.result_label.setStyleSheet(
                f"color: {self.STATUS_COLORS.get(class_name, self.STATUS_COLORS['Neutral'])};"
            )
            
            # Start animation
            if not self._animation_timer.isActive():
                self._animation_timer.start()

            # Update confidence display
            self.confidence_label.setText(f"{confidence:.1%}")
            
            # Emit update signal
            self.classification_updated.emit(class_name, confidence)
            logger.debug(f"Classification updated: {class_name} ({confidence:.2%})")

        except Exception as e:
            error_msg = f"Failed to update classification: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            self._show_error_state()

    def _update_animation(self) -> None:
        """Update confidence animation state."""
        try:
            if self._animation_progress >= 1.0:
                self._animation_timer.stop()
                return

            # Update animation progress
            self._animation_progress += self.UPDATE_INTERVAL / self.ANIMATION_DURATION
            self._animation_progress = min(1.0, self._animation_progress)

            # Apply easing
            eased_progress = QEasingCurve.OutCubic(self._animation_progress)
            
            # Update current confidence
            self._current_confidence = (
                (1 - eased_progress) * self._current_confidence + 
                eased_progress * self._target_confidence
            )

            # Trigger repaint
            self.update()

        except Exception as e:
            logger.error(f"Animation update failed: {e}")
            self._animation_timer.stop()

    def paintEvent(self, event: QPaintEvent) -> None:
        """Custom paint event for confidence visualization."""
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # Define gradient based on confidence
            gradient = QLinearGradient(0, 0, self.width(), 0)
            confidence_color = self._get_confidence_color(self._current_confidence)
            
            gradient.setColorAt(0, confidence_color)
            gradient.setColorAt(self._current_confidence, confidence_color)
            gradient.setColorAt(self._current_confidence + 0.001, QColor("#21262D"))
            gradient.setColorAt(1, QColor("#21262D"))

            # Draw confidence bar
            bar_height = 4
            y_pos = self.height() - bar_height - 16
            
            painter.fillRect(
                16,                    # x
                y_pos,                 # y
                self.width() - 32,     # width
                bar_height,            # height
                gradient
            )

        except Exception as e:
            logger.error(f"Paint event failed: {e}")

    def _get_confidence_color(self, confidence: float) -> QColor:
        """Get color based on confidence level."""
        if confidence < 0.3:
            return QColor(self.STATUS_COLORS["Fake"])
        elif confidence < 0.7:
            return QColor(self.STATUS_COLORS["Likely Fake"])
        return QColor(self.STATUS_COLORS["Real"])

    def _show_error_state(self) -> None:
        """Display error state."""
        try:
            self.result_label.setText("Error")
            self.result_label.setStyleSheet(f"color: {self.STATUS_COLORS['Fake']};")
            self.confidence_label.setText("N/A")
            self._current_confidence = 0.0
            self._target_confidence = 0.0
            self.update()
        except Exception as e:
            logger.error(f"Failed to show error state: {e}")

    def cleanup(self) -> None:
        """Enhanced cleanup with cache invalidation."""
        try:
            # Stop animations
            self._stop_animations()
            
            # Clear cache
            self._cache.clear()
            
            # Reset state
            self._state = ClassificationState()
            self._animation_state = AnimationState.IDLE
            
            # Clean up signals
            try:
                self.classification_updated.disconnect()
                self.error_occurred.disconnect()
                self.state_changed.disconnect()
                self.animation_completed.disconnect()
            except:
                pass
            
            logger.debug("ClassificationWidget cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle widget resize."""
        super().resizeEvent(event)
        self.update()  # Ensure confidence bar is redrawn correctly