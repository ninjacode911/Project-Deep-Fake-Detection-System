"""
DeepFake Detection System - Database Manager
Created: 2025-06-07
Author: ninjacode911

This module implements a thread-safe database manager for storing and retrieving
detection results and metadata with proper error handling and resource management.
"""

import logging
import time
import traceback
import threading
import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime
import shutil

from ..core.exceptions.backend_exceptions import DatabaseError
from ..config import config_manager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Thread-safe database manager for detection results."""

    _instance = None
    _lock = threading.RLock()

    def __new__(cls) -> 'DatabaseManager':
        """Ensure singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self) -> None:
        """Initialize database manager."""
        try:
            # Get database configuration
            self.db_path = Path(config_manager.get("database.path", "data/detection.db"))
            self.backup_dir = Path(config_manager.get("database.backup_dir", "data/backups"))
            self.max_backups = config_manager.get("database.max_backups", 5)
            self.max_size_mb = config_manager.get("database.max_size_mb", 100)
            
            # Create directories
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize connection
            self._conn: Optional[sqlite3.Connection] = None
            self._cursor: Optional[sqlite3.Cursor] = None
            
            # Initialize database
        self._initialize_database()
            
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Database manager initialization failed: {e}")
            raise DatabaseError(
                message="Failed to initialize database manager",
                error_code=7000,
                operation="init",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _initialize_database(self) -> None:
        """Initialize database schema."""
        try:
            # Connect to database
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._cursor = self._conn.cursor()
                
                # Create tables
            self._cursor.executescript("""
                CREATE TABLE IF NOT EXISTS detection_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_path TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    prediction REAL NOT NULL,
                        confidence REAL NOT NULL,
                    processing_time REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    metadata TEXT NOT NULL,
                    UNIQUE(media_path, timestamp)
                    );
                    
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    inference_time REAL NOT NULL,
                    memory_usage REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    metadata TEXT NOT NULL
                );
                
                CREATE INDEX IF NOT EXISTS idx_detection_media_path 
                ON detection_results(media_path);
                    
                CREATE INDEX IF NOT EXISTS idx_detection_timestamp 
                ON detection_results(timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_metrics_model_name 
                ON model_metrics(model_name);
                
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON model_metrics(timestamp);
                """)
                
            self._conn.commit()
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise DatabaseError(
                message="Failed to initialize database schema",
                error_code=7001,
                operation="initialize_database",
                details={'error': str(e)}
            )

    def store_detection_result(
        self,
        media_path: str,
        media_type: str,
        prediction: float,
        confidence: float,
        processing_time: float,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Store detection result in database.

        Args:
            media_path: Path to media file
            media_type: Type of media (video/audio)
            prediction: Detection prediction
            confidence: Prediction confidence
            processing_time: Processing time in seconds
            metadata: Additional metadata
        """
        try:
            with self._lock:
                # Validate inputs
                if not os.path.exists(media_path):
                    raise FileNotFoundError(f"Media file not found: {media_path}")
                    
                if media_type not in ['video', 'audio']:
                    raise ValueError(f"Invalid media type: {media_type}")
                    
                if not 0 <= prediction <= 1:
                    raise ValueError(f"Invalid prediction value: {prediction}")
                    
                if not 0 <= confidence <= 1:
                    raise ValueError(f"Invalid confidence value: {confidence}")
                    
                if processing_time < 0:
                    raise ValueError(f"Invalid processing time: {processing_time}")
                    
                # Check database size
                self._check_database_size()
                
                # Insert result
                self._cursor.execute("""
                    INSERT INTO detection_results (
                        media_path, media_type, prediction, confidence,
                        processing_time, timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    media_path,
                    media_type,
                    prediction,
                    confidence,
                    processing_time,
                    datetime.now().isoformat(),
                    json.dumps(metadata)
                ))
                
                self._conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to store detection result: {e}")
            raise DatabaseError(
                message="Failed to store detection result",
                error_code=7002,
                operation="store_detection_result",
                details={'error': str(e)}
            )

    def store_model_metrics(
        self,
        model_name: str,
        inference_time: float,
        memory_usage: float,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Store model performance metrics.

        Args:
            model_name: Name of the model
            inference_time: Inference time in seconds
            memory_usage: Memory usage in MB
            metadata: Additional metrics
        """
        try:
            with self._lock:
                # Validate inputs
                if not model_name:
                    raise ValueError("Model name cannot be empty")
                    
                if inference_time < 0:
                    raise ValueError(f"Invalid inference time: {inference_time}")
                    
                if memory_usage < 0:
                    raise ValueError(f"Invalid memory usage: {memory_usage}")
                
                # Insert metrics
                self._cursor.execute("""
                    INSERT INTO model_metrics (
                        model_name, inference_time, memory_usage,
                        timestamp, metadata
                    ) VALUES (?, ?, ?, ?, ?)
                """, (
                    model_name,
                    inference_time,
                    memory_usage,
                    datetime.now().isoformat(),
                    json.dumps(metadata)
                    ))
                
                self._conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store model metrics: {e}")
            raise DatabaseError(
                message="Failed to store model metrics",
                error_code=7003,
                operation="store_model_metrics",
                details={'error': str(e)}
            )

    def get_detection_history(
        self,
        media_path: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get detection history with optional filters.

        Args:
            media_path: Optional media path filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of results

        Returns:
            List of detection results
        """
        try:
            with self._lock:
                # Build query
                query = "SELECT * FROM detection_results WHERE 1=1"
                params = []
                
                if media_path:
                    query += " AND media_path = ?"
                    params.append(media_path)
                    
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                    
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                    
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                # Execute query
                self._cursor.execute(query, params)
                results = self._cursor.fetchall()
                
                # Convert to dictionaries
                columns = [col[0] for col in self._cursor.description]
                return [
                    {
                        col: json.loads(val) if col == 'metadata' else val
                        for col, val in zip(columns, row)
                    }
                    for row in results
                ]
            
        except Exception as e:
            logger.error(f"Failed to get detection history: {e}")
            raise DatabaseError(
                message="Failed to get detection history",
                error_code=7004,
                operation="get_detection_history",
                details={'error': str(e)}
            )

    def get_model_metrics(
        self,
        model_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get model metrics with optional filters.

        Args:
            model_name: Optional model name filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of results

        Returns:
            List of model metrics
        """
        try:
            with self._lock:
                # Build query
                query = "SELECT * FROM model_metrics WHERE 1=1"
                params = []
                
                if model_name:
                    query += " AND model_name = ?"
                    params.append(model_name)
                    
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                    
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                    
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                # Execute query
                self._cursor.execute(query, params)
                results = self._cursor.fetchall()
                
                # Convert to dictionaries
                columns = [col[0] for col in self._cursor.description]
                return [
                    {
                        col: json.loads(val) if col == 'metadata' else val
                        for col, val in zip(columns, row)
                    }
                    for row in results
                ]
                
        except Exception as e:
            logger.error(f"Failed to get model metrics: {e}")
            raise DatabaseError(
                message="Failed to get model metrics",
                error_code=7005,
                operation="get_model_metrics",
                details={'error': str(e)}
            )

    def _check_database_size(self) -> None:
        """Check and handle database size limits."""
        try:
            # Get current size in MB
            size_mb = self.db_path.stat().st_size / (1024 * 1024)
            
            if size_mb > self.max_size_mb:
                # Create backup
                self._create_backup()
                
                # Clear old data
                self._clear_old_data()
                
        except Exception as e:
            logger.error(f"Database size check failed: {e}")
            raise DatabaseError(
                message="Failed to check database size",
                error_code=7006,
                operation="check_database_size",
                details={'error': str(e)}
            )

    def _create_backup(self) -> None:
        """Create database backup."""
        try:
            # Generate backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"detection_{timestamp}.db"
            
            # Copy database file
            shutil.copy2(self.db_path, backup_path)
            
            # Clean up old backups
            self._cleanup_old_backups()
            
            logger.info(f"Created database backup: {backup_path}")
            
        except Exception as e:
            logger.error(f"Database backup creation failed: {e}")
            raise DatabaseError(
                message="Failed to create database backup",
                error_code=7007,
                operation="create_backup",
                details={'error': str(e)}
            )

    def _cleanup_old_backups(self) -> None:
        """Clean up old database backups."""
        try:
            # Get all backup files
            backups = sorted(
                self.backup_dir.glob("detection_*.db"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            # Remove excess backups
            for backup in backups[self.max_backups:]:
                backup.unlink()
                logger.info(f"Removed old backup: {backup}")
                
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
            raise DatabaseError(
                message="Failed to cleanup old backups",
                error_code=7008,
                operation="cleanup_old_backups",
                details={'error': str(e)}
            )

    def _clear_old_data(self) -> None:
        """Clear old data from database."""
        try:
            # Keep last 1000 results
            self._cursor.execute("""
                DELETE FROM detection_results
                WHERE id NOT IN (
                    SELECT id FROM detection_results
                    ORDER BY timestamp DESC
                    LIMIT 1000
                )
            """)
            
            # Keep last 1000 metrics
            self._cursor.execute("""
                DELETE FROM model_metrics
                WHERE id NOT IN (
                    SELECT id FROM model_metrics
                    ORDER BY timestamp DESC
                    LIMIT 1000
                )
            """)
            
            self._conn.commit()
            
            # Vacuum database
            self._conn.execute("VACUUM")
                
            logger.info("Cleared old data from database")
            
        except Exception as e:
            logger.error(f"Failed to clear old data: {e}")
            raise DatabaseError(
                message="Failed to clear old data",
                error_code=7009,
                operation="clear_old_data",
                details={'error': str(e)}
            )

    def cleanup(self) -> None:
        """Clean up database resources."""
        try:
            with self._lock:
                if self._cursor is not None:
                    self._cursor.close()
                    self._cursor = None
                    
                if self._conn is not None:
                    self._conn.close()
                    self._conn = None
                        
            logger.info("Database resources cleaned up")
            
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
            raise DatabaseError(
                message="Failed to cleanup database resources",
                error_code=7010,
                operation="cleanup",
                details={'error': str(e)}
            )

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Database cleanup in destructor failed: {e}")