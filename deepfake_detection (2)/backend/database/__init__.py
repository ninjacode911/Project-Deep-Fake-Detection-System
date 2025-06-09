"""
DeepFake Detection System - Database Management
Created: 2025-06-07
Author: ninjacode911

This module initializes database components and provides exports.
It includes error handling, connection management, and configuration.
"""

from typing import Dict, Any, Optional, Type
import logging
from pathlib import Path

from .database import Database
from ..core.exceptions.backend_exceptions import DatabaseError
from ..config import config_manager

# Configure module logger
logger = logging.getLogger(__name__)

# Database configuration defaults
DEFAULT_CONFIG = {
    'db_path': 'data/history.db',
    'max_connections': 5,
    'timeout': 30,
    'cache_size': 2000,
    'journal_mode': 'WAL'  # Write-Ahead Logging for better concurrency
}

class DatabaseManager:
    """Manages database connections and lifecycle."""
    
    _instance: Optional['DatabaseManager'] = None
    _initialized: bool = False
    _connections: Dict[str, Database] = {}
    
    def __new__(cls) -> 'DatabaseManager':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize database manager."""
        if self._initialized:
            return
            
        try:
            # Load configuration
            self.config = {
                **DEFAULT_CONFIG,
                **config_manager.get('database', {})
            }
            
            # Create data directory
            db_path = Path(self.config['db_path'])
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize main database connection
            self._connections['main'] = Database(
                db_path=str(db_path),
                timeout=self.config['timeout'],
                cache_size=self.config['cache_size']
            )
            
            logger.info("Database manager initialized successfully")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise DatabaseError(
                message="Failed to initialize database manager",
                error_code=5000,
                operation="init",
                details={'error': str(e)}
            )

    def get_connection(self, name: str = 'main') -> Database:
        """Get a database connection by name."""
        if name not in self._connections:
            raise DatabaseError(
                message=f"Database connection '{name}' not found",
                error_code=5001,
                operation="get_connection",
                details={'connection_name': name}
            )
        return self._connections[name]

    def cleanup(self) -> None:
        """Clean up database connections."""
        try:
            for name, conn in self._connections.items():
                try:
                    conn.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up connection '{name}': {e}")
            
            self._connections.clear()
            logger.info("Database connections cleaned up")
            
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
            raise DatabaseError(
                message="Failed to cleanup database connections",
                error_code=5002,
                operation="cleanup",
                details={'error': str(e)}
            )

    def __del__(self) -> None:
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Database manager cleanup in destructor failed: {e}")

# Initialize database manager instance
db_manager = DatabaseManager()

# Export database components
__all__ = [
    'Database',
    'DatabaseManager', 
    'db_manager',
    'DatabaseError'
]