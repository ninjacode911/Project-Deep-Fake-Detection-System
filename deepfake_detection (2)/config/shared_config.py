import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SharedConfig:
    """Shared configuration accessible by both frontend and backend"""
    
    _instance = None
    _config: Dict[str, Any] = {}
    _last_modified: float = 0
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize with default config if not already initialized"""
        if not self._config:
            self._default_config = {
                "models": {
                    "vision": {
                        "type": "efficientnet",
                        "weights_path": "pretrained/efficientnetv2_m_finetuned.pth",
                        "batch_size": 16,
                        "input_size": [224, 224]
                    },
                    "audio": {
                        "type": "wav2vec2",
                        "weights_path": "pretrained/Deepfake-audio-detection-V2",
                        "sample_rate": 16000
                    }
                },
                "detection": {
                    "weights": {
                        "vision": 0.5,
                        "audio": 0.5
                    },
                    "thresholds": {
                        "fake": 0.9,
                        "likely_fake": 0.7
                    }
                },
                "performance": {
                    "cache_size_mb": 512,
                    "cleanup_interval": 300
                }
            }
            self._config = self._default_config.copy()

    def load_config(self, config_path: str = "config/config.json") -> None:
        """Load configuration from JSON file"""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file {config_path} not found, using defaults")
                return

            with open(config_file, 'r') as f:
                loaded_config = json.load(f)
                
            # Update config while preserving defaults for missing keys
            self._update_nested_dict(self._config, loaded_config)
            self._last_modified = datetime.now().timestamp()
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load shared config: {e}")
            raise

    def _update_nested_dict(self, d1: Dict, d2: Dict) -> None:
        """Update nested dictionary while preserving structure"""
        for k, v in d2.items():
            if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                self._update_nested_dict(d1[k], v)
            else:
                d1[k] = v

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with dot notation support"""
        try:
            keys = key.split('.')
            value = self._config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def validate(self) -> bool:
        """Validate configuration structure"""
        try:
            required_keys = {'models', 'detection', 'performance'}
            if not all(k in self._config for k in required_keys):
                return False
            return True
        except Exception:
            return False

# Global instance
shared_config = SharedConfig()