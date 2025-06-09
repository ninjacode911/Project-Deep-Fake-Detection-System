import logging
import requests
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from requests.adapters import HTTPAdapter, Retry

from backend.core.detector import Detector
from backend.database.database import Database

logger = logging.getLogger(__name__)

class APIClient:
    """Client for interacting with the backend detector and database."""
    
    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        self._base_url = base_url
        self._timeout = timeout
        self._session = requests.Session()
        self._retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        self._session.mount('http://', HTTPAdapter(max_retries=self._retry_strategy))
        self._session.mount('https://', HTTPAdapter(max_retries=self._retry_strategy))
        
        # Initialize backend components
        try:
        self.db = Database()
        self.detector = Detector()
            logger.info("Backend components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize backend components: {e}")
            raise RuntimeError("Failed to initialize backend components")

    def analyze_video(self, file_path: str, 
                     progress_callback: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
        """Analyze video file using either local or remote backend."""
        try:
            if self._base_url:
                return self._remote_analysis(file_path, progress_callback)
            else:
                return self._local_analysis(file_path, progress_callback)
        except ConnectionError as e:
            logger.error(f"Connection failed: {e}")
            raise ConnectionError("Failed to connect to analysis server")
        except TimeoutError as e:
            logger.error(f"Analysis timeout: {e}")
            raise TimeoutError("Analysis timed out - please try again")
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise RuntimeError(f"Analysis failed: {str(e)}")

    def _local_analysis(self, file_path: str, 
                       progress_callback: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
        """Perform analysis using local backend."""
        try:
            result = self.detector.detect(file_path, progress_callback)
            self.db.save_analysis(file_path, result)
            return result
        except Exception as e:
            logger.error(f"Local analysis failed: {e}")
            raise RuntimeError(f"Local analysis failed: {e}")

    def _remote_analysis(self, file_path: str,
                        progress_callback: Optional[Callable[[int], None]] = None) -> Dict[str, Any]:
        """Perform analysis using remote backend."""
        try:
            self._validate_connection()
            
            # Prepare file for upload
            with open(file_path, 'rb') as f:
                files = {'video': f}
                response = self._session.post(
                    f"{self._base_url}/analyze",
                    files=files,
                    timeout=self._timeout
                )
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Remote analysis failed: {e}")
            raise ConnectionError(f"Remote analysis failed: {e}")

    def get_cached_result(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get cached analysis result if available."""
        try:
            return self.db.get_analysis(file_path)
        except Exception as e:
            logger.error(f"Failed to get cached result: {e}")
            return None

    def _validate_connection(self) -> None:
        """Validate API connection before analysis."""
        if not self._base_url:
            return
            
        try:
            response = self._session.get(
                f"{self._base_url}/health", 
                timeout=self._timeout
            )
            response.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"API connection failed: {e}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self._session.close()
            if hasattr(self, 'db'):
            self.db.close()
            if hasattr(self, 'detector'):
                self.detector.cleanup()
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")