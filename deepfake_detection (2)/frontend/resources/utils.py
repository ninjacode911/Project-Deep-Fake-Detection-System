"""Utility functions for resource management."""
from pathlib import Path
from typing import Optional, Union, BinaryIO
import logging
from PyQt6.QtCore import QFile, QIODevice
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def resource_file(path: Union[str, Path]) -> BinaryIO:
    """Safely manage resource file handling."""
    file = QFile(str(path))
    try:
        if not file.open(QIODevice.OpenModeFlag.ReadOnly):
            raise IOError(f"Cannot open resource: {path}")
        yield file
    finally:
        if file and file.isOpen():
            file.close()

def load_resource(path: str) -> Optional[str]:
    """Load resource file content with proper cleanup."""
    try:
        if not Path(path).exists():
            raise FileNotFoundError(f"Resource not found: {path}")
            
        with resource_file(path) as file:
            data = file.readAll()
            if data.isEmpty():
                raise ValueError(f"Empty resource file: {path}")
            return data.data().decode()
            
    except Exception as e:
        logger.error(f"Failed to load resource {path}: {e}")
        return None