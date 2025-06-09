import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self._config_path = Path(config_path) if config_path else Path(__file__).parent / "file_manager.json"
        self._config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with error handling."""
        try:
            if not self._config_path.exists():
                return self._create_default_config()
            
            with open(self._config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._create_default_config()

    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        config = {
            "default_upload_directory": str(Path.home() / "Videos"),
            "recent_files": [],
            "window_state": {
                "geometry": None,
                "maximized": False
            },
            "preferences": {
                "theme": "dark",
                "language": "en",
                "auto_analyze": True
            }
        }
        try:
            with open(self._config_path, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save default config: {e}")
        return config

    def _validate_config(self) -> None:
        """Validate configuration structure."""
        required_fields = {
            "default_upload_directory": str,
            "recent_files": list,
            "window_state": dict,
            "preferences": dict
        }
        
        for field, field_type in required_fields.items():
            if field not in self._config:
                logger.warning(f"Missing config field: {field}")
                self._config[field] = self._create_default_config()[field]
            if not isinstance(self._config[field], field_type):
                logger.warning(f"Invalid type for {field}")
                self._config[field] = self._create_default_config()[field]

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        try:
            self._config[key] = value
            with open(self._config_path, 'w') as f:
                json.dump(self._config, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

config_manager = ConfigManager()