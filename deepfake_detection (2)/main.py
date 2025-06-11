#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point for the DeepFake Detection System.
Created on: 2025-04-28 07:50:42 UTC
Last Modified: 2025-06-07 14:30:00 UTC
Author: ninjacode911
"""

import sys
import logging
import traceback
import gc
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import signal
import psutil
import qtawesome as qta
from PyQt6.QtWidgets import QApplication, QMessageBox, QSplashScreen, QProgressBar
from PyQt6.QtGui import QIcon, QPixmap, QColor
from PyQt6.QtCore import Qt, QTimer, QSize

# Frontend imports
from frontend.widgets.main_window import DeepFakeDetectionSystem
from frontend.config import config_manager
from frontend.resources import theme_manager
from frontend.utils.error_handler import ErrorHandler
from frontend.utils.logger import setup_logger, LogLevel

# Backend imports
from backend.database.database import Database
from backend.core.detector import Detector

logger = logging.getLogger(__name__)

def initialize_logging(debug_mode: bool = False) -> None:
    """Set up logging with rotation and backup."""
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        backup_dir = log_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        log_level = LogLevel.DEBUG if debug_mode else LogLevel.INFO
        
        setup_logger(
            filename=log_file,
            log_level=log_level,
            max_size=5_242_880,  # 5MB
            backup_count=3,
            backup_dir=backup_dir,
            log_to_console=True
        )
        
        logger.info("Application logging initialized at %s IST on %s", 
                    datetime.now().strftime("%I:%M %p"),
                    datetime.now().strftime("%A, %B %d, %Y"))
                    
    except Exception as e:
        print(f"Failed to initialize logging: {e}", file=sys.stderr)
        sys.exit(1)

def check_system_requirements() -> Dict[str, bool]:
    """Enhanced system requirements check."""
    try:
        requirements = {
            'memory': False,
            'cpu': False,
            'disk': False,
            'gpu': False
        }
        
        # Check available memory (minimum 4GB)
        memory = psutil.virtual_memory()
        requirements['memory'] = memory.available >= 4 * 1024 * 1024 * 1024
        
        # Check CPU cores (minimum 2 cores)
        requirements['cpu'] = psutil.cpu_count() >= 2
        
        # Check disk space (minimum 1GB)
        disk = psutil.disk_usage('/')
        requirements['disk'] = disk.free >= 1 * 1024 * 1024 * 1024
        
        # Log system info
        logger.info("System Check Results:")
        logger.info(f"Memory Available: {memory.available / (1024**3):.2f} GB")
        logger.info(f"CPU Cores: {psutil.cpu_count()}")
        logger.info(f"Free Disk Space: {disk.free / (1024**3):.2f} GB")
        
        return requirements
        
    except Exception as e:
        logger.error(f"System check failed: {e}")
        return {'memory': False, 'cpu': False, 'disk': False, 'gpu': False}

def show_system_warnings(requirements: Dict[str, bool]) -> None:
    """Show warnings for unmet system requirements."""
    warnings = []
    
    if not requirements['memory']:
        warnings.append("Low memory available (4GB recommended)")
    if not requirements['cpu']:
        warnings.append("Insufficient CPU cores (2 cores recommended)")
    if not requirements['disk']:
        warnings.append("Low disk space (1GB required)")
        
    if warnings:
        warning_text = "System Requirements Warning:\n\n" + "\n".join(warnings)
        QMessageBox.warning(None, "System Check", warning_text)

def initialize_database() -> Optional[Database]:
    """Initialize database with retry logic."""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            db_path = Path("deepfake_detection.db")
            db = Database(db_path=str(db_path))
            logger.info(f"Database initialized successfully at {db_path}")
            return db
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database initialization attempt {attempt + 1} failed: {e}")
                QTimer.singleShot(retry_delay * 1000, lambda: None)
            else:
                logger.error(f"Database initialization failed after {max_retries} attempts: {e}")
                raise

def create_splash_screen() -> QSplashScreen:
    """Create enhanced splash screen with progress indicator."""
    try:
        # Create styled splash screen
        splash_pixmap = QPixmap(500, 300)
        splash_pixmap.fill(QColor("#0D1117"))
        
        splash = QSplashScreen(splash_pixmap)
        
        # Add progress bar
        progress = QProgressBar(splash)
        progress.setGeometry(50, 260, 400, 4)
        progress.setStyleSheet("""
            QProgressBar {
                background-color: #21262D;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #1F6FEB;
                border-radius: 2px;
            }
        """)
        progress.setRange(0, 0)  # Indeterminate progress
        
        # Add title and message
        splash.showMessage(
            "DeepFake Detection System\n\nInitializing components...",
            Qt.AlignmentFlag.AlignCenter,
            QColor("#E6EDF3")
        )
        
        return splash, progress
        
    except Exception as e:
        logger.error(f"Failed to create splash screen: {e}")
        raise

def cleanup_resources() -> None:
    """Cleanup resources before shutdown"""
    try:
        logger.info("Starting graceful shutdown...")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Clear model caches
        from backend.utils.model_utils import ModelUtils
        ModelUtils.clear_cache()
        
        # Close database connections
        from backend.database.database import Database
        Database.close_connections()
        
        # Clear temporary files
        temp_dir = Path("data/temp")
        if temp_dir.exists():
            for file in temp_dir.glob("*"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {file}: {e}")
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Cleanup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def signal_handler(signum: int, frame: Optional[Any]) -> None:
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}")
    cleanup_resources()
    sys.exit(0)

def main() -> None:
    """Enhanced main application entry point."""
    error_handler = None
    
    try:
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Initialize logging
        initialize_logging(debug_mode='--debug' in sys.argv)
        
        # Create error handler
        error_handler = ErrorHandler()
        
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("DeepFake Detection System")
        app.setApplicationVersion("1.0.0")
        app.aboutToQuit.connect(cleanup_resources)
        
        # Show splash screen
        splash, progress = create_splash_screen()
        splash.show()
        app.processEvents()
        
        # Check system requirements
        requirements = check_system_requirements()
        show_system_warnings(requirements)
        
        # Initialize components
        steps = [
            ("Initializing database...", initialize_database),
            ("Loading configuration...", config_manager.load),
            ("Applying theme...", lambda: theme_manager.apply_theme(app, "dark")),
            ("Initializing detector...", lambda: Detector())
        ]
        
        db_instance: Optional[Database] = None
        detector_instance: Optional[Detector] = None

        for message, step_func in steps:
            splash.showMessage(f"\n\n{message}",
                             Qt.AlignmentFlag.AlignCenter,
                             QColor("#E6EDF3"))
            app.processEvents()
            
            try:
                if message == "Initializing database...":
                    db_instance = step_func()
                    if db_instance is None:
                        raise RuntimeError("Database initialization returned None.")
                    logger.info("Database instance captured.")
                elif message == "Initializing detector...":
                    detector_instance = step_func()
                    if detector_instance is None:
                        # This case should ideally not happen if Detector() constructor succeeds
                        # or if ModelFactory.create_model now correctly returns None and Detector handles it.
                        # However, the Detector constructor itself could fail before returning an instance.
                        raise RuntimeError("Detector initialization returned None.")
                    logger.info("Detector instance captured.")
                else:
                    # For config_manager.load and theme_manager.apply_theme
                    step_func()
            except Exception as e:
                logger.error(f"Failed at step: '{message}': {e}", exc_info=True)
                # Show a critical error message to the user immediately
                QMessageBox.critical(None, "Startup Error", f"A critical error occurred during: {message}.\n{str(e)}\n\nApplication will exit.")
                sys.exit(1) # Exit immediately on critical failure during startup steps

        # Ensure critical instances are created
        if db_instance is None:
            logger.error("Database instance was not created during startup steps!")
            QMessageBox.critical(None, "Startup Error", "Database failed to initialize. Application will exit.")
            sys.exit(1)

        if detector_instance is None:
            logger.error("Detector instance was not created during startup steps!")
            QMessageBox.critical(None, "Startup Error", "Detector failed to initialize. Application will exit.")
            sys.exit(1)

        # Create main window and pass instances
        window = DeepFakeDetectionSystem(detector=detector_instance, database=db_instance)
        window.setWindowTitle("DeepFake Detection System")
        window.setGeometry(100, 100, 1280, 720)
        
        # Set window icon
        icon = qta.icon('fa5s.video', color='#1F6FEB')
        window.setWindowIcon(icon.pixmap(QSize(32, 32)))
        
        # Register cleanup
        app.aboutToQuit.connect(cleanup_resources)
        
        # Show window
        QTimer.singleShot(1500, splash.close)
        QTimer.singleShot(1500, window.show)
        
        logger.info("Application startup completed successfully")
        
        # Start event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error("Critical error during startup", exc_info=True)
        
        if 'app' not in locals():
            app = QApplication(sys.argv)
            
        error_msg = (
            f"Failed to start application:\n\n{str(e)}\n\n"
            f"Details:\n{traceback.format_exc()}\n\n"
            "The application will now exit."
        )
        
        if error_handler:
            error_handler.handle_error(Exception("Startup failed"), error_msg)
        else:
            QMessageBox.critical(None, "Critical Error", error_msg)
            
        sys.exit(1)

if __name__ == "__main__":
    main()