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

    def test_wav2vec2_init_and_inference_with_missing_weights(self, temp_dir: Path):
        """Test Wav2Vec2Model initialization and inference with missing weights."""
        # Path for weights_path that does not exist
        non_existent_path = temp_dir / "non_existent_model_dir_for_wav2vec2"

        # Initialize Wav2Vec2Model with a non-existent weights_path
        # The __init__ of Wav2Vec2Model now expects 'weights_path' and 'device'
        # It no longer takes a 'config' dict directly in the constructor signature shown in existing tests.
        # It seems the existing TestWav2Vec2.setup_model passes a 'config' dict to Wav2Vec2Model,
        # which implies the actual Wav2Vec2Model constructor might be different or the test setup is adapting.
        # Assuming Wav2Vec2Model(weights_path: Union[str, Path], device: str = "cuda")

        # Let's check the actual Wav2Vec2Model constructor from previous steps:
        # Wav2Vec2Model(self, weights_path: Union[str, Path], device: str = "cuda")
        # This is fine.

        model = Wav2Vec2Model(weights_path=non_existent_path, device='cpu')

        assert model.weights_loaded is False, \
            "Wav2Vec2Model.weights_loaded should be False when initialized with a non-existent weights_path."

        # Prepare a dummy audio waveform for inference attempt
        dummy_waveform = np.random.randn(16000).astype(np.float32) # 1 second of audio at 16kHz

        # Assert that calling inference() raises ModelError
        with pytest.raises(ModelError, match="Wav2Vec2Model is not operational due to missing or failed loading of weights."):
            model.inference(dummy_waveform)

        # Ensure cleanup, though model might not have much to clean if weights didn't load
        try:
            model.cleanup()
        except Exception as e:
            logger.warning(f"Error during cleanup in test_wav2vec2_init_and_inference_with_missing_weights: {e}")


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
        with pytest.raises(ModelError) as exc_info: # This seems to be a pre-existing test for an invalid model name, not type.
            # The factory would raise ValueError for unknown model, then ModelFactory catches and re-raises as ModelError
            setup_factory.create_model('invalid_model_name_that_does_not_exist_in_registry', {})
        # The original assertion might be too broad; let's check for a more specific part of the message if possible,
        # or rely on the error code if ModelFactory sets one consistently.
        # For now, assuming "Unknown model" or similar is part of the raised ModelError.
        # Based on ModelFactory code, it raises ValueError(f"Unknown model: {model_name}")
        # which is then wrapped in ModelError.
        assert "Unknown model" in str(exc_info.value)


    def test_create_model_with_missing_weights_path(self, setup_factory: ModelFactory):
        """Test that ModelFactory returns None when a model's weights_path is invalid/missing."""
        # Simulate that 'model_config.json' has an empty/invalid path for 'audio' model
        # ModelFactory._load_weights_config is called during ModelFactory.__init__
        # So, the setup_factory instance is already configured. We need to re-trigger its internal loading
        # or mock what it has loaded. It's easier to mock what it *would* load for a specific model's path.
        # The factory stores paths in self._model_weights.
        # The create_model method then uses self._model_weights.get(model_name).

        # Let's patch the internal _model_weights dictionary of the already initialized factory.
        # This is a bit of white-box testing but necessary given the factory's design.

        setup_factory._model_weights['audio'] = None # Simulate that loading config resulted in None for audio path

        audio_model_config_for_create = {
            "type": "wav2vec2", # This info is usually in model_config.json
            "name": "Deepfake-audio-detection-V2",
            # weights_path here is illustrative; create_model uses factory._model_weights
            "sampling_rate": 16000,
            "enabled": True
        }

        # Attempt to create the 'audio' model which now has a None weights_path in the factory's state
        created_model = setup_factory.create_model(
            model_name='audio',
            config=audio_model_config_for_create
            # device=None # device can be omitted for this test if model init handles None device
        )
        assert created_model is None, "ModelFactory.create_model should return None if weights path was processed as None."

if __name__ == "__main__":
    pytest.main([__file__])