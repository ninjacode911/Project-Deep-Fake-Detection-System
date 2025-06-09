"""
DeepFake Detection System - Model Tests
Created: 2025-06-07
Author: ninjacode911

This module provides comprehensive testing for all model components with
proper resource management and error handling.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Generator, Dict, Any, List
from unittest.mock import Mock, patch

from ..models.base import BaseModel
from ..models.efficientnet import EfficientNetModel
from ..models.wav2vec2 import Wav2Vec2Model
from ..models.factory import ModelFactory
from ..core.exceptions.backend_exceptions import ModelError
from . import TEST_CONFIG, logger

class TestModelBase:
    """Test suite for base model functionality."""

    @pytest.fixture(autouse=True)
    def setup_base_model(self) -> Generator[BaseModel, None, None]:
        """Provide base model instance for testing."""
        model = None
        try:
            model = BaseModel()
            yield model
        finally:
            if model:
                try:
                    model.cleanup()
                except Exception as e:
                    logger.error(f"Failed to cleanup base model: {e}")

    def test_model_initialization(self, setup_base_model: BaseModel) -> None:
        """Test model initialization."""
        assert isinstance(setup_base_model, BaseModel)
        assert hasattr(setup_base_model, 'cleanup')

class TestEfficientNet:
    """Test suite for EfficientNet model."""

    @pytest.fixture(scope="class")
    def model_config(self) -> Dict[str, Any]:
        """Provide model configuration."""
        return {
            'model_name': 'efficientnetv2_m',
            'num_classes': 2,
            'pretrained': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

    @pytest.fixture(autouse=True)
    def setup_model(self, model_config: Dict[str, Any], temp_dir: Path) -> Generator[EfficientNetModel, None, None]:
        """Provide EfficientNet model instance."""
        model = None
        try:
            model = EfficientNetModel(
                config=model_config,
                cache_dir=temp_dir / "model_cache"
            )
            yield model
        finally:
            if model:
                try:
                    model.cleanup()
                except Exception as e:
                    logger.error(f"Failed to cleanup EfficientNet model: {e}")

    def test_model_predict(self, setup_model: EfficientNetModel) -> None:
        """Test model prediction."""
        try:
            # Create dummy input
            batch_size = 1
            channels = 3
            height = 224
            width = 224
            dummy_input = torch.randn(batch_size, channels, height, width)
            
            # Run prediction
            result = setup_model.predict(dummy_input)
            
            # Verify output
            assert isinstance(result, float)
            assert 0 <= result <= 1
            
        except Exception as e:
            pytest.fail(f"Model prediction failed: {str(e)}")

    @pytest.mark.gpu
    def test_gpu_memory_management(self, setup_model: EfficientNetModel) -> None:
        """Test GPU memory management."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        try:
            # Record initial memory
            initial_memory = torch.cuda.memory_allocated()
            
            # Run multiple predictions
            for _ in range(5):
                dummy_input = torch.randn(1, 3, 224, 224).cuda()
                _ = setup_model.predict(dummy_input)
                
            # Clear cache
            torch.cuda.empty_cache()
            
            # Verify memory cleanup
            final_memory = torch.cuda.memory_allocated()
            assert final_memory <= initial_memory * 1.1  # Allow 10% overhead
            
        except Exception as e:
            pytest.fail(f"GPU memory test failed: {str(e)}")

class TestWav2Vec2:
    """Test suite for Wav2Vec2 model."""

    @pytest.fixture(scope="class")
    def audio_config(self) -> Dict[str, Any]:
        """Provide audio model configuration."""
        return {
            'sample_rate': 16000,
            'model_path': TEST_CONFIG['MODEL_WEIGHTS_DIR'] / "wav2vec2",
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

    @pytest.fixture(autouse=True)
    def setup_model(self, audio_config: Dict[str, Any], temp_dir: Path) -> Generator[Wav2Vec2Model, None, None]:
        """Provide Wav2Vec2 model instance."""
        model = None
        try:
            model = Wav2Vec2Model(
                config=audio_config,
                cache_dir=temp_dir / "audio_cache"
            )
            yield model
        finally:
            if model:
                try:
                    model.cleanup()
                except Exception as e:
                    logger.error(f"Failed to cleanup Wav2Vec2 model: {e}")

    def test_audio_processing(self, setup_model: Wav2Vec2Model) -> None:
        """Test audio processing functionality."""
        try:
            # Create dummy audio input
            sample_rate = 16000
            duration = 3  # seconds
            dummy_audio = torch.randn(sample_rate * duration)
            
            # Process audio
            result = setup_model.process_audio(dummy_audio)
            
            # Verify output
            assert isinstance(result, dict)
            assert 'features' in result
            assert isinstance(result['features'], torch.Tensor)
            
        except Exception as e:
            pytest.fail(f"Audio processing failed: {str(e)}")

class TestModelFactory:
    """Test suite for model factory."""

    @pytest.fixture(autouse=True)
    def setup_factory(self) -> Generator[ModelFactory, None, None]:
        """Provide model factory instance."""
        factory = None
        try:
            factory = ModelFactory()
            yield factory
        finally:
            if factory:
                try:
                    factory.cleanup()
                except Exception as e:
                    logger.error(f"Failed to cleanup model factory: {e}")

    def test_model_creation(self, setup_factory: ModelFactory) -> None:
        """Test model creation through factory."""
        try:
            # Test creating different model types
            vision_model = setup_factory.create_model(
                'efficientnet',
                {'num_classes': 2}
            )
            audio_model = setup_factory.create_model(
                'wav2vec2',
                {'sample_rate': 16000}
            )
            
            # Verify instances
            assert isinstance(vision_model, EfficientNetModel)
            assert isinstance(audio_model, Wav2Vec2Model)
            
        except Exception as e:
            pytest.fail(f"Model creation failed: {str(e)}")

    def test_invalid_model_type(self, setup_factory: ModelFactory) -> None:
        """Test error handling for invalid model type."""
        with pytest.raises(ModelError) as exc_info:
            setup_factory.create_model('invalid_model', {})
        assert "Unknown model type" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main([__file__])