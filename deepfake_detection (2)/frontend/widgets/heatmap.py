import logging
from typing import Optional, List, Dict, Union, Tuple, Any
import numpy as np
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QSlider, QComboBox, 
                            QLabel, QHBoxLayout, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal as Signal, QPoint, QSize, QRect, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QMouseEvent
import qtawesome as qta

logger = logging.getLogger(__name__)

class HeatmapColorMap:
    """Enhanced colormap configurations for heatmap visualization."""
    
    PRESETS = {
        "Viridis": [
            (0.0, (68, 1, 84)),    
            (0.25, (58, 82, 139)), 
            (0.5, (32, 144, 140)), 
            (0.75, (94, 201, 98)), 
            (1.0, (253, 231, 37))  
        ],
        "Inferno": [
            (0.0, (0, 0, 4)),     
            (0.25, (85, 16, 106)), 
            (0.5, (184, 55, 121)), 
            (0.75, (246, 148, 58)),
            (1.0, (252, 255, 164)) 
        ],
        "DeepFake": [
            (0.0, (13, 17, 23)),    
            (0.25, (31, 111, 235)), 
            (0.5, (47, 129, 247)),  
            (0.75, (248, 81, 73)),  
            (1.0, (255, 255, 255))  
        ]
    }

    @staticmethod
    def get_color(value: float, preset: str = "DeepFake") -> QColor:
        """Get interpolated color for a value between 0 and 1."""
        try:
            if preset not in HeatmapColorMap.PRESETS:
                preset = "DeepFake"
                
            stops = HeatmapColorMap.PRESETS[preset]
            
            # Find the color stops to interpolate between
            for i in range(len(stops) - 1):
                v1, c1 = stops[i]
                v2, c2 = stops[i + 1]
                
                if v1 <= value <= v2:
                    # Linear interpolation
                    t = (value - v1) / (v2 - v1)
                    r = int(c1[0] * (1 - t) + c2[0] * t)
                    g = int(c1[1] * (1 - t) + c2[1] * t)
                    b = int(c1[2] * (1 - t) + c2[2] * t)
                    return QColor(r, g, b)
                    
            # Fall back to extremes
            if value <= 0:
                return QColor(*stops[0][1])
            return QColor(*stops[-1][1])
            
        except Exception as e:
            logger.error(f"Color interpolation failed: {e}")
            return QColor(0, 0, 0)


class HeatmapWidget(QWidget):
    """
    Widget for displaying heatmap visualizations of deepfake analysis.
    
    Features:
    - Adjustable intensity with slider (0.5x to 2.0x)
    - Switchable visualization types (Grad-CAM/EigenCAM)
    - Interactive region highlighting
    - Smooth color gradient visualization
    - Mock data generation for testing
    - Efficient image caching
    - Multiple color presets
    
    Signals:
        error_occurred (str): Emitted when an error occurs
        region_selected (tuple): Emitted when user clicks a region (x, y, intensity)
        state_changed (str): Emitted when widget state changes
        cache_cleared (None): Emitted when cache is cleared
    """
    
    error_occurred = Signal(str)
    region_selected = Signal(tuple)  # (x, y, intensity_value)
    state_changed = Signal(str)
    cache_cleared = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the HeatmapWidget."""
        try:
            super().__init__(parent)
            
            # Data state
            self._heatmap_data: Optional[np.ndarray] = None
            self._intensity: float = 1.0
            self._heatmap_type: str = "Grad-CAM"
            self._highlighted_region: Optional[Tuple[int, int]] = None
            self._color_preset = "DeepFake"
            
            # Caching and memory management
            self._cache: Dict[Tuple[float, str], QPixmap] = {}
            self._cache_size = 0
            self._cache_timer = QTimer(self)
            self._cache_timer.timeout.connect(self._cleanup_cache)
            self._cache_timer.start(300000)  # Cleanup every 5 minutes
            
            # Loading state
            self._is_loading = False
            
            # Initialize UI
            self._setup_ui()
            self._apply_styles()
            self._init_mock_data()
            
            # Set size policies
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding, 
                QSizePolicy.Policy.Expanding
            )
            
            logger.info("HeatmapWidget initialized")
            self.state_changed.emit("initialized")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.error_occurred.emit(f"Initialization Error: {str(e)}")

    def _apply_styles(self) -> None:
        """Apply custom styles to the widget."""
        try:
            self.setStyleSheet("""
                QLabel#sectionTitle {
                    color: #E6EDF3;
                    font-size: 12px;
                    font-weight: bold;
                }
                QLabel#heatmapDisplay {
                    background-color: #161B22;
                    border: 1px solid #30363D;
                    border-radius: 4px;
                }
                QLabel#controlLabel {
                    color: #8B949E;
                    font-size: 11px;
                }
                QComboBox {
                    background-color: #21262D;
                    border: 1px solid #30363D;
                    border-radius: 4px;
                    color: #C9D1D9;
                    padding: 4px;
                }
                QComboBox:hover {
                    border-color: #388BFD;
                }
                QSlider::groove:horizontal {
                    background-color: #21262D;
                    height: 4px;
                    border-radius: 2px;
                }
                QSlider::handle:horizontal {
                    background-color: #388BFD;
                    width: 12px;
                    margin: -4px 0;
                    border-radius: 6px;
                }
                QSlider::handle:horizontal:hover {
                    background-color: #1F6FEB;
                }
            """)
        except Exception as e:
            logger.error(f"Style application failed: {str(e)}")
            self.error_occurred.emit(f"Style Error: {str(e)}")

    def _setup_ui(self) -> None:
        """Set up the user interface components."""
        try:
            layout = QVBoxLayout()
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(8)

            # Title with icon
            title_layout = QHBoxLayout()
            icon_label = QLabel()
            icon = qta.icon('mdi.fire', color='#8B949E')
            icon_label.setPixmap(icon.pixmap(16, 16))
            title_layout.addWidget(icon_label)
            
            title = QLabel("HEATMAP VISUALIZATION")
            title.setObjectName("sectionTitle")
            title_layout.addWidget(title)
            title_layout.addStretch()
            layout.addLayout(title_layout)

            # Loading indicator
            self._loading_bar = QProgressBar()
            self._loading_bar.setTextVisible(False)
            self._loading_bar.setFixedHeight(2)
            self._loading_bar.hide()
            layout.addWidget(self._loading_bar)

            # Heatmap display
            self.heatmap_label = QLabel()
            self.heatmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.heatmap_label.setFixedSize(256, 176)
            self.heatmap_label.setObjectName("heatmapDisplay")
            self.heatmap_label.mousePressEvent = self._on_mouse_press
            layout.addWidget(self.heatmap_label)

            # Controls layout
            controls_layout = QHBoxLayout()
            controls_layout.setSpacing(16)

            # Color preset selection
            color_layout = QHBoxLayout()
            color_label = QLabel("Color:")
            color_label.setObjectName("controlLabel")
            self.color_combo = QComboBox()
            self.color_combo.addItems(list(HeatmapColorMap.PRESETS.keys()))
            self.color_combo.currentTextChanged.connect(self._update_color_preset)
            color_layout.addWidget(color_label)
            color_layout.addWidget(self.color_combo)
            controls_layout.addLayout(color_layout, stretch=1)

            # Intensity control
            intensity_layout = QHBoxLayout()
            intensity_label = QLabel("Intensity:")
            intensity_label.setObjectName("controlLabel")
            self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
            self.intensity_slider.setRange(50, 200)
            self.intensity_slider.setValue(100)
            self.intensity_slider.valueChanged.connect(self._update_intensity)
            intensity_layout.addWidget(intensity_label)
            intensity_layout.addWidget(self.intensity_slider)
            controls_layout.addLayout(intensity_layout, stretch=2)

            # Type selection
            type_layout = QHBoxLayout()
            type_label = QLabel("Type:")
            type_label.setObjectName("controlLabel")
            self.type_combo = QComboBox()
            self.type_combo.addItems(["Grad-CAM", "EigenCAM"])
            self.type_combo.currentTextChanged.connect(self._update_type)
            type_layout.addWidget(type_label)
            type_layout.addWidget(self.type_combo)
            controls_layout.addLayout(type_layout, stretch=1)

            layout.addLayout(controls_layout)
            self.setLayout(layout)

        except Exception as e:
            logger.error(f"Failed to setup UI: {str(e)}")
            self.error_occurred.emit(f"UI Setup Error: {str(e)}")

    def _init_mock_data(self) -> None:
        """Initialize mock heatmap data with Gaussian blobs."""
        try:
            x, y = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
            
            # Create different patterns for each type
            if self._heatmap_type == "Grad-CAM":
                # Central focus with some surrounding activity
                center = np.exp(-(x**2 + y**2) / 0.3)
                surrounding = np.exp(-((x-0.5)**2 + (y-0.5)**2) / 1.0) * 0.3
                data = np.clip(center + surrounding, 0, 1)
            else:
                # Multiple attention regions for EigenCAM
                blob1 = np.exp(-((x-0.3)**2 + (y-0.3)**2) / 0.2)
                blob2 = np.exp(-((x+0.3)**2 + (y+0.3)**2) / 0.3) * 0.7
                data = np.clip(blob1 + blob2, 0, 1)

            self._heatmap_data = data
            self._update_heatmap_display()
            logger.debug(f"Mock data initialized for {self._heatmap_type}")

        except Exception as e:
            logger.error(f"Failed to initialize mock data: {str(e)}")
            self.error_occurred.emit(f"Mock Data Error: {str(e)}")

    def _update_heatmap_display(self) -> None:
        """Update the heatmap visualization."""
        try:
            if self._heatmap_data is None:
                self.heatmap_label.setText("No data available")
                return

            # Check cache first
            cache_key = (self._intensity, self._color_preset)
            if cache_key in self._cache:
                pixmap = self._cache[cache_key]
            else:
                # Create new pixmap
                height, width = self._heatmap_data.shape
                image = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Apply intensity adjustment
                adjusted = np.clip(self._heatmap_data * self._intensity, 0, 1)
                
                # Apply color mapping
                for i in range(height):
                    for j in range(width):
                        color = HeatmapColorMap.get_color(adjusted[i, j], self._color_preset)
                        image[i, j] = [color.blue(), color.green(), color.red()]
                
                # Convert to QImage and then QPixmap
                q_img = QImage(image.data, width, height, 3 * width, 
                            QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                # Add highlight if needed
                if self._highlighted_region is not None:
                    self._draw_highlight(pixmap)
                
                # Cache the result
                self._cache[cache_key] = pixmap
                self._cache_size += pixmap.byteCount()

            # Scale and display
            scaled_pixmap = pixmap.scaled(
                self.heatmap_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.heatmap_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            logger.error(f"Display update failed: {str(e)}")
            self.error_occurred.emit(f"Display Error: {str(e)}")

    def _cleanup_cache(self) -> None:
        """Periodic cache cleanup."""
        try:
            if self._cache_size > 100 * 1024 * 1024:  # 100MB limit
                self._cache.clear()
                self._cache_size = 0
                self.cache_cleared.emit()
                logger.debug("Cache cleared")
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Stop timers
            self._cache_timer.stop()
            
            # Clear data
            self._heatmap_data = None
            self._highlighted_region = None
            
            # Clear cache
            self._cache.clear()
            self._cache_size = 0
            
            logger.debug("HeatmapWidget cleaned up")
            self.state_changed.emit("cleaned")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            self.error_occurred.emit(f"Cleanup Error: {str(e)}")

    def set_loading(self, loading: bool) -> None:
        """Show/hide loading indicator."""
        try:
            self._is_loading = loading
            self._loading_bar.setVisible(loading)
            if loading:
                self._loading_bar.setRange(0, 0)  # Indeterminate
            else:
                self._loading_bar.setRange(0, 100)
            self.state_changed.emit("loading" if loading else "ready")
        except Exception as e:
            logger.error(f"Loading state change failed: {e}")

    def _draw_highlight(self, pixmap: QPixmap) -> None:
        """Draw highlight box around selected region."""
        try:
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Create outer glow effect
            x, y = self._highlighted_region
            for i in range(3):
                painter.setPen(QColor(255, 255, 255, 255 - i * 50))
                rect = QRect(x - 5 - i, y - 5 - i, 10 + i * 2, 10 + i * 2)
                painter.drawRect(rect)
            
            painter.end()

        except Exception as e:
            logger.error(f"Highlight drawing failed: {str(e)}")

    def _update_intensity(self, value: int) -> None:
        """Update visualization intensity."""
        try:
            self._intensity = value / 100.0
            self._update_heatmap_display()
            logger.debug(f"Intensity updated to {self._intensity:.2f}x")
        except Exception as e:
            logger.error(f"Intensity update failed: {str(e)}")
            self.error_occurred.emit(f"Intensity Error: {str(e)}")

    def _update_type(self, heatmap_type: str) -> None:
        """Update heatmap visualization type."""
        try:
            self._heatmap_type = heatmap_type
            self._init_mock_data()
            logger.debug(f"Type updated to {heatmap_type}")
        except Exception as e:
            logger.error(f"Type update failed: {str(e)}")
            self.error_occurred.emit(f"Type Error: {str(e)}")

    def _update_color_preset(self, preset: str) -> None:
        """Update colormap preset."""
        try:
            if preset in HeatmapColorMap.PRESETS:
                self._color_preset = preset
                self._update_heatmap_display()
                logger.debug(f"Color preset updated to {preset}")
        except Exception as e:
            logger.error(f"Color preset update failed: {str(e)}")
            self.error_occurred.emit(f"Color Preset Error: {str(e)}")

    def _on_mouse_press(self, event: QMouseEvent) -> None:
        """Handle mouse click events."""
        try:
            pos = event.position()
            if not isinstance(pos, QPoint):
                pos = pos.toPoint()

            # Map click to heatmap coordinates
            size = self.heatmap_label.size()
            x = int(64 * pos.x() / size.width())
            y = int(64 * pos.y() / size.height())
            
            # Validate coordinates
            if 0 <= x < 64 and 0 <= y < 64:
                self._highlighted_region = (x, y)
                intensity = float(self._heatmap_data[y, x])
                self.region_selected.emit((x, y, intensity))
                self._update_heatmap_display()
                logger.debug(f"Region selected at ({x}, {y})")

        except Exception as e:
            logger.error(f"Mouse event failed: {str(e)}")
            self.error_occurred.emit(f"Input Error: {str(e)}")

    def update_data(self, data: Union[np.ndarray, list, None]) -> None:
        """Update heatmap with new data."""
        try:
            if data is None:
                logger.debug("No data provided, using mock data")
                self._init_mock_data()
                return

            # Convert and validate data
            arr = np.array(data, dtype=np.float32)
            if arr.shape != (64, 64):
                raise ValueError(f"Invalid shape: {arr.shape}, expected (64, 64)")

            self._heatmap_data = np.clip(arr, 0, 1)
            self._highlighted_region = None
            self._update_heatmap_display()
            logger.info("Heatmap data updated successfully")
            self.state_changed.emit("data_updated")

        except Exception as e:
            logger.error(f"Data update failed: {str(e)}")
            self.error_occurred.emit(f"Data Error: {str(e)}")

    def reset(self) -> None:
        """Reset widget to initial state."""
        try:
            self._highlighted_region = None
            self.intensity_slider.setValue(100)
            self.type_combo.setCurrentText("Grad-CAM")
            self._init_mock_data()
            logger.debug("Widget reset to initial state")
            self.state_changed.emit("reset")
        except Exception as e:
            logger.error(f"Reset failed: {str(e)}")
            self.error_occurred.emit(f"Reset Error: {str(e)}")