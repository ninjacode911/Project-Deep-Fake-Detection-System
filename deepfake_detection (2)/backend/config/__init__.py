#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepFake Detection System - Backend Configuration Management
Created on: 2025-06-07 14:30:00 UTC
Author: ninjacode911

This module provides thread-safe configuration management with caching,
validation, and error handling for the backend components.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class ConfigManager:
    """Thread-safe configuration manager with caching and validation."""
    
    _instance = None
    _lock = threading.Lock()
    _config_cache: Dict[str, Any] = {}
    _last_modified: Optional[float] = None
    _cache_valid = False

    def __new__(cls) -> 'ConfigManager':
        """Ensure singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self) -> None:
        """Initialize configuration manager."""
        self._base_path = Path(__file__).parent
        self._config_path = self._base_path / "model_config.json"
        self._cache_lifetime = 300  # 5 minutes cache lifetime
        
        # Default configuration
        self._default_config = {
            "vision_model_m": "pretrained/efficientnetv2_m_finetuned.pth",
            "vision_model_s": "pretrained/efficientnetv2_s_finetuned.pth",
            "audio_model": "pretrained/Deepfake-audio-detection-V2",
            "weights": {
                "vision_m": 0.5,
                "vision_s": 0.0,
                "audio": 0.5
            },
            "confidence_threshold": 0.5,
            "frame_rate": 30,
            "heatmap_size": [64, 64],
            "cache": {
                "enabled": True,
                "max_size_mb": 1024,
                "cleanup_interval": 300
            },
            "performance": {
                "batch_size": 32,
                "num_workers": 4,
                "pin_memory": True,
                "prefetch_factor": 2
            }
        }

    def load_config(self) -> Dict[str, Any]:
        """Load configuration with caching and validation."""
        try:
            with self._lock:
                # Check if cache is valid
                if self._is_cache_valid():
                    logger.debug("Using cached configuration")
                    return self._config_cache

                # Read and parse config file
                if not self._config_path.exists():
                    logger.warning(f"Config file not found at {self._config_path}, creating default")
                    self._create_default_config()
                    return self._default_config

                config = self._read_config()
                self._validate_config(config)
                self._update_cache(config)
                
                logger.info("Configuration loaded successfully")
                return config

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config: {e}")
            raise ConfigError(f"Invalid configuration format: {e}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise ConfigError(f"Configuration error: {e}")

    def _read_config(self) -> Dict[str, Any]:
        """Read configuration file with error handling."""
        try:
            with open(self._config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read config file: {e}")
            raise ConfigError(f"Cannot read configuration: {e}")

    def _create_default_config(self) -> None:
        """Create default configuration file."""
        try:
            with open(self._config_path, 'w') as f:
                json.dump(self._default_config, f, indent=4)
            logger.info("Created default configuration file")
        except Exception as e:
            logger.error(f"Failed to create default config: {e}")
            raise ConfigError(f"Cannot create default configuration: {e}")

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure and values."""
        try:
            # Check required sections
            required_sections = {"vision_model_m", "vision_model_s", "audio_model", 
                               "weights", "confidence_threshold", "frame_rate", "heatmap_size"}
            missing = required_sections - set(config.keys())
            if missing:
                raise ConfigError(f"Missing required sections: {missing}")

            # Validate weights
            weights = config.get("weights", {})
            if not isinstance(weights, dict):
                raise ConfigError("Weights must be a dictionary")
            
            weight_sum = sum(weights.values())
            if not (0.99 <= weight_sum <= 1.01):
                raise ConfigError(f"Weights must sum to 1.0 (got {weight_sum})")

            # Validate numeric values
            if not (0 < config["confidence_threshold"] <= 1):
                raise ConfigError("Confidence threshold must be between 0 and 1")
            
            if not (1 <= config["frame_rate"] <= 120):
                raise ConfigError("Frame rate must be between 1 and 120")

            # Validate model paths
            for key in ["vision_model_m", "vision_model_s", "audio_model"]:
                model_path = Path(self._base_path.parent) / config[key]
                if not model_path.exists():
                    logger.warning(f"Model path does not exist: {model_path}")

        except KeyError as e:
            raise ConfigError(f"Missing required configuration key: {e}")
        except Exception as e:
            raise ConfigError(f"Configuration validation failed: {e}")

    def _is_cache_valid(self) -> bool:
        """Check if cached configuration is still valid."""
        if not self._cache_valid or not self._last_modified:
            return False

        # Check if config file has been modified
        current_mtime = self._config_path.stat().st_mtime
        if current_mtime > self._last_modified:
            return False

        # Check if cache has expired
        cache_age = datetime.now().timestamp() - self._last_modified
        return cache_age < self._cache_lifetime

    def _update_cache(self, config: Dict[str, Any]) -> None:
        """Update configuration cache."""
        self._config_cache = config
        self._last_modified = datetime.now().timestamp()
        self._cache_valid = True

    def invalidate_cache(self) -> None:
        """Manually invalidate configuration cache."""
        with self._lock:
            self._config_cache = {}
            self._last_modified = None
            self._cache_valid = False
            logger.debug("Configuration cache invalidated")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with fallback."""
        try:
            config = self.load_config()
            return config.get(key, default)
        except Exception as e:
            logger.error(f"Error retrieving config key '{key}': {e}")
            return default

    def cleanup(self) -> None:
        """Cleanup resources."""
        with self._lock:
            self.invalidate_cache()
            logger.info("Configuration manager cleaned up")

# Global instance
config_manager = ConfigManager()

# Cleanup on module unload
import atexit
atexit.register(config_manager.cleanup)

__all__ = ['config_manager', 'ConfigError']