"""
DeepFake Detection System - Utility Tests
Created: 2025-06-07
Author: ninjacode911

This module provides comprehensive testing for utility functions with proper
resource management and error handling.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Generator, Dict, Any
import tempfile
import shutil
import gc
import time

from ..utils.video_utils import (
    load_video,
    extract_frames,
    resize_frames,
    normalize_frames
)
from ..utils.audio_utils import (
    load_audio,
    extract_audio,
    resample_audio,
    get_audio_features
)
from ..utils.cache_utils import (
    CacheManager,
    cache_result,
    clear_cache
)
from ..core.exceptions.backend_exceptions import (
    VideoError,
    AudioError,
    CacheError
)
from . import TEST_CONFIG, logger

class TestVideoUtils:
    """Test suite for video utility functions."""

    @pytest.fixture(scope="class")
    def test_video(self, temp_dir: Path) -> Generator[Path, None, None]:
        """Provide test video file."""
        video_path = temp_dir / "test_video.mp4"
        try:
            # Create dummy video file
            with open(video_path, 'wb') as f:
                f.write(b'dummy video content')
            yield video_path
        finally:
            try:
                if video_path.exists():
                    video_path.unlink()
            except Exception as e:
                logger.error(f"Failed to cleanup test video: {e}")

    def test_load_video(self, test_video: Path) -> None:
        """Test video loading functionality."""
        try:
            frames = load_video(test_video)
            assert isinstance(frames, np.ndarray)
            assert len(frames.shape) == 4  # [frames, height, width, channels]
        except Exception as e:
            pytest.fail(f"Video loading failed: {str(e)}")

    def test_extract_frames(self, test_video: Path) -> None:
        """Test frame extraction."""
        try:
            frames = extract_frames(test_video, num_frames=10)
            assert isinstance(frames, list)
            assert len(frames) <= 10
            assert all(isinstance(f, np.ndarray) for f in frames)
        except Exception as e:
            pytest.fail(f"Frame extraction failed: {str(e)}")

    @pytest.mark.parametrize("size", [(224, 224), (299, 299)])
    def test_resize_frames(self, test_video: Path, size: tuple) -> None:
        """Test frame resizing."""
        try:
            frames = load_video(test_video)
            resized = resize_frames(frames, size)
            assert resized.shape[1:3] == size
        except Exception as e:
            pytest.fail(f"Frame resizing failed: {str(e)}")

class TestAudioUtils:
    """Test suite for audio utility functions."""

    @pytest.fixture(scope="class")
    def test_audio(self, temp_dir: Path) -> Generator[Path, None, None]:
        """Provide test audio file."""
        audio_path = temp_dir / "test_audio.wav"
        try:
            # Create dummy audio file
            with open(audio_path, 'wb') as f:
                f.write(b'dummy audio content')
            yield audio_path
        finally:
            try:
                if audio_path.exists():
                    audio_path.unlink()
            except Exception as e:
                logger.error(f"Failed to cleanup test audio: {e}")

    def test_load_audio(self, test_audio: Path) -> None:
        """Test audio loading functionality."""
        try:
            waveform, sample_rate = load_audio(test_audio)
            assert isinstance(waveform, np.ndarray)
            assert isinstance(sample_rate, int)
        except Exception as e:
            pytest.fail(f"Audio loading failed: {str(e)}")

    def test_resample_audio(self, test_audio: Path) -> None:
        """Test audio resampling."""
        try:
            waveform, sr = load_audio(test_audio)
            target_sr = 16000
            resampled = resample_audio(waveform, sr, target_sr)
            assert isinstance(resampled, np.ndarray)
        except Exception as e:
            pytest.fail(f"Audio resampling failed: {str(e)}")

class TestCacheUtils:
    """Test suite for caching utilities."""

    @pytest.fixture(autouse=True)
    def setup_cache(self, temp_dir: Path) -> Generator[CacheManager, None, None]:
        """Set up cache manager."""
        cache_dir = temp_dir / "cache"
        manager = None
        try:
            manager = CacheManager(
                cache_dir=cache_dir,
                max_size=1024 * 1024 * 100  # 100MB
            )
            yield manager
        finally:
            try:
                if manager:
                    manager.cleanup()
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
            except Exception as e:
                logger.error(f"Failed to cleanup cache: {e}")

    def test_cache_result(self, setup_cache: CacheManager) -> None:
        """Test result caching."""
        try:
            # Cache some data
            key = "test_key"
            data = np.random.randn(100, 100)
            setup_cache.set(key, data)

            # Retrieve data
            cached = setup_cache.get(key)
            assert np.array_equal(data, cached)

        except Exception as e:
            pytest.fail(f"Cache operation failed: {str(e)}")

    def test_cache_invalidation(self, setup_cache: CacheManager) -> None:
        """Test cache invalidation."""
        try:
            # Fill cache
            for i in range(10):
                key = f"key_{i}"
                data = np.random.randn(1000, 1000)  # Large array
                setup_cache.set(key, data)

            # Verify cache cleanup
            assert setup_cache.current_size <= setup_cache.max_size

        except Exception as e:
            pytest.fail(f"Cache invalidation failed: {str(e)}")

    def test_clear_cache(self, setup_cache: CacheManager) -> None:
        """Test cache clearing."""
        try:
            # Add some data
            setup_cache.set("key1", "value1")
            setup_cache.set("key2", "value2")

            # Clear cache
            setup_cache.clear()

            # Verify cache is empty
            assert setup_cache.get("key1") is None
            assert setup_cache.get("key2") is None

        except Exception as e:
            pytest.fail(f"Cache clearing failed: {str(e)}")

class TestErrorHandling:
    """Test suite for error handling."""

    def test_video_error_handling(self) -> None:
        """Test video error handling."""
        with pytest.raises(VideoError) as exc_info:
            load_video(Path("nonexistent.mp4"))
        assert "Video file not found" in str(exc_info.value)

    def test_audio_error_handling(self) -> None:
        """Test audio error handling."""
        with pytest.raises(AudioError) as exc_info:
            load_audio(Path("nonexistent.wav"))
        assert "Audio file not found" in str(exc_info.value)

    def test_cache_error_handling(self, setup_cache: CacheManager) -> None:
        """Test cache error handling."""
        with pytest.raises(CacheError) as exc_info:
            setup_cache.get("nonexistent_key")
        assert "Cache key not found" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main([__file__])