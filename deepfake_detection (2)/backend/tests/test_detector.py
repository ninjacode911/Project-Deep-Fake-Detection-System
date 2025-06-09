"""
DeepFake Detection System - Detector Tests
Created: 2025-06-07
Author: ninjacode911

This module provides comprehensive testing for the detector component with
proper resource management and error handling.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, patch

from ..core.detector import Detector
from ..core.exceptions.backend_exceptions import ModelError, VideoError, AudioError
from ..types.backend_types import DetectionResult
from . import TEST_CONFIG, logger

class TestDetector:
    """Test suite for Detector class."""

    @pytest.fixture(autouse=True)
    def setup_detector(self, detector: Detector) -> Generator[Detector, None, None]:
        """Provide initialized detector for each test."""
        try:
            yield detector
        finally:
            try:
                detector.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup detector: {e}")

    @pytest.mark.parametrize("media_path", [
        "test_videos/real_video.mp4",
        "test_videos/fake_video.mp4"
    ])
    def test_detect_valid_media(
        self, 
        detector: Detector,
        media_path: str,
        temp_dir: Path
    ) -> None:
        """Test detection on valid media files."""
        try:
            # Prepare test file
            test_file = temp_dir / Path(media_path).name
            test_file.write_bytes(b"test content")

            # Run detection
            result = detector.detect(str(test_file))

            # Verify result structure
            assert isinstance(result, DetectionResult)
            assert isinstance(result.is_fake, bool)
            assert 0 <= result.confidence <= 1
            assert isinstance(result.model_scores, dict)
            assert result.processing_time > 0

        except Exception as e:
            pytest.fail(f"Detection failed: {str(e)}")

    def test_detect_invalid_media(self, detector: Detector) -> None:
        """Test detection with invalid media file."""
        with pytest.raises(VideoError) as exc_info:
            detector.detect("nonexistent.mp4")
        assert "File not found" in str(exc_info.value)

    def test_detect_corrupted_media(
        self,
        detector: Detector,
        temp_dir: Path
    ) -> None:
        """Test detection with corrupted media file."""
        try:
            # Create corrupted file
            corrupted_file = temp_dir / "corrupted.mp4"
            corrupted_file.write_bytes(b"corrupted content")

            with pytest.raises((VideoError, AudioError)) as exc_info:
                detector.detect(str(corrupted_file))
            assert "Failed to process" in str(exc_info.value)

        finally:
            # Cleanup
            if corrupted_file.exists():
                corrupted_file.unlink()

    @pytest.mark.gpu
    def test_detect_gpu_optimization(
        self,
        detector: Detector,
        temp_dir: Path
    ) -> None:
        """Test GPU optimization for detection."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        try:
            # Create test file
            test_file = temp_dir / "test_gpu.mp4"
            test_file.write_bytes(b"test content")

            # Monitor GPU memory
            initial_memory = torch.cuda.memory_allocated()
            result = detector.detect(str(test_file))
            peak_memory = torch.cuda.max_memory_allocated()

            # Verify GPU usage
            assert peak_memory > initial_memory
            assert isinstance(result, DetectionResult)

        finally:
            # Cleanup GPU memory
            torch.cuda.empty_cache()

    def test_detect_caching(
        self,
        detector: Detector,
        temp_dir: Path
    ) -> None:
        """Test detection result caching."""
        try:
            # Create test file
            test_file = temp_dir / "test_cache.mp4"
            test_file.write_bytes(b"test content")

            # First detection
            start_time = time.time()
            result1 = detector.detect(str(test_file))
            first_duration = time.time() - start_time

            # Second detection (should use cache)
            start_time = time.time()
            result2 = detector.detect(str(test_file))
            second_duration = time.time() - start_time

            # Verify cache effectiveness
            assert result1.model_scores == result2.model_scores
            assert second_duration < first_duration

        finally:
            # Clear cache
            detector._cache_manager.clear()

    def test_detect_parallel_processing(
        self,
        detector: Detector,
        temp_dir: Path
    ) -> None:
        """Test parallel processing of multiple models."""
        try:
            # Create test file
            test_file = temp_dir / "test_parallel.mp4"
            test_file.write_bytes(b"test content")

            # Mock models for timing verification
            with patch.dict(detector._models, {
                'model1': Mock(return_value=0.5),
                'model2': Mock(return_value=0.7),
                'model3': Mock(return_value=0.3)
            }):
                result = detector.detect(str(test_file))

            # Verify parallel execution
            assert len(result.model_scores) == 3
            for model_name, score in result.model_scores.items():
                assert 0 <= score <= 1

        except Exception as e:
            pytest.fail(f"Parallel processing test failed: {str(e)}")

    def test_cleanup(self, detector: Detector) -> None:
        """Test proper resource cleanup."""
        try:
            # Force some resource allocation
            detector.detect("test_videos/sample.mp4")

            # Cleanup
            detector.cleanup()

            # Verify cleanup
            assert not hasattr(detector, '_executor') or detector._executor._shutdown
            if torch.cuda.is_available():
                assert torch.cuda.memory_allocated() == 0

        except Exception as e:
            pytest.fail(f"Cleanup test failed: {str(e)}")

    @pytest.mark.parametrize("error_condition", [
        "video_error",
        "audio_error",
        "model_error"
    ])
    def test_error_handling(
        self,
        detector: Detector,
        error_condition: str,
        temp_dir: Path
    ) -> None:
        """Test error handling for different failure scenarios."""
        try:
            # Create test file
            test_file = temp_dir / "test_error.mp4"
            test_file.write_bytes(b"test content")

            # Simulate different errors
            with patch.object(detector, '_video_handler') as mock_video:
                if error_condition == "video_error":
                    mock_video.extract_features.side_effect = VideoError("Video error")
                    expected_error = VideoError
                elif error_condition == "audio_error":
                    mock_video.extract_features.side_effect = AudioError("Audio error")
                    expected_error = AudioError
                else:
                    mock_video.extract_features.side_effect = ModelError("Model error")
                    expected_error = ModelError

                with pytest.raises(expected_error) as exc_info:
                    detector.detect(str(test_file))

                assert error_condition in str(exc_info.value).lower()

        finally:
            # Cleanup
            if test_file.exists():
                test_file.unlink()

if __name__ == "__main__":
    pytest.main([__file__])