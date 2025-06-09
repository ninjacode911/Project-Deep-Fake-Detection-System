"""
DeepFake Detection System - Result Aggregator
Created: 2025-06-07
Author: ninjacode911

This module implements result aggregation for deepfake detection
with proper resource management and error handling.
"""

import logging
import time
import traceback
import threading
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np
import torch
from contextlib import contextmanager

from .exceptions.backend_exceptions import AggregationError, ValidationError, ResourceError
from ..config import config_manager

logger = logging.getLogger(__name__)

class ResultAggregator:
    """Aggregator class for deepfake detection results."""

    def __init__(self) -> None:
        """Initialize result aggregator with configuration."""
        try:
            self._lock = threading.RLock()
            self._memory_limit = config_manager.get("aggregator.memory_limit", 1 * 1024 * 1024 * 1024)  # 1GB
            self._confidence_threshold = config_manager.get("aggregator.confidence_threshold", 0.5)
            self._min_evidence = config_manager.get("aggregator.min_evidence", 3)
            
            # Initialize result registry
            self._results: Dict[str, Dict[str, Any]] = {}
            
            logger.info("Result aggregator initialized successfully")
            
        except Exception as e:
            logger.error(f"Result aggregator initialization failed: {e}")
            raise AggregationError(
                message="Failed to initialize result aggregator",
                error_code=7000,
                operation="init",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Validate detection results.
        
        Args:
            results: List of detection results
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if not results:
                raise ValidationError(
                    message="Results list cannot be empty",
                    error_code=7001,
                    details={'results': results}
                )
                
            required_fields = {'prediction', 'confidence', 'timestamp'}
            for i, result in enumerate(results):
                # Check required fields
                missing_fields = required_fields - set(result.keys())
                if missing_fields:
                    raise ValidationError(
                        message="Missing required result fields",
                        error_code=7002,
                        details={
                            'index': i,
                            'missing_fields': list(missing_fields)
                        }
                    )
                    
                # Validate prediction
                if not isinstance(result['prediction'], (int, float)):
                    raise ValidationError(
                        message="Invalid prediction type",
                        error_code=7003,
                        details={
                            'index': i,
                            'type': type(result['prediction'])
                        }
                    )
                    
                # Validate confidence
                if not isinstance(result['confidence'], (int, float)):
                    raise ValidationError(
                        message="Invalid confidence type",
                        error_code=7004,
                        details={
                            'index': i,
                            'type': type(result['confidence'])
                        }
                    )
                    
                if not 0 <= result['confidence'] <= 1:
                    raise ValidationError(
                        message="Confidence must be between 0 and 1",
                        error_code=7005,
                        details={
                            'index': i,
                            'confidence': result['confidence']
                        }
                    )
                    
                # Validate timestamp
                if not isinstance(result['timestamp'], (int, float)):
                    raise ValidationError(
                        message="Invalid timestamp type",
                        error_code=7006,
                        details={
                            'index': i,
                            'type': type(result['timestamp'])
                        }
                    )
                    
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Results validation failed: {e}")
            raise ValidationError(
                message="Results validation failed",
                error_code=7007,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    @contextmanager
    def _memory_context(self):
        """Context manager for memory monitoring."""
        try:
            start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            yield
            current_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            if current_memory - start_memory > self._memory_limit:
                raise ResourceError(
                    message="Memory limit exceeded",
                    error_code=7008,
                    details={
                        'used': current_memory - start_memory,
                        'limit': self._memory_limit
                    }
                )
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            raise ResourceError(
                message="Memory monitoring failed",
                error_code=7009,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def aggregate_results(
        self,
        results: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate detection results.
        
        Args:
            results: List of detection results
            progress_callback: Optional callback for progress updates
            
        Returns:
            Aggregated results
        """
        try:
            # Validate results
            self._validate_results(results)
            
            # Aggregate results with memory monitoring
            with self._memory_context():
                # Calculate basic statistics
                predictions = np.array([r['prediction'] for r in results])
                confidences = np.array([r['confidence'] for r in results])
                timestamps = np.array([r['timestamp'] for r in results])
                
                if progress_callback:
                    progress_callback(0.3)  # Basic statistics calculated
                    
                # Calculate aggregated prediction
                weighted_prediction = np.average(
                    predictions,
                    weights=confidences
                )
                
                # Calculate confidence score
                confidence_score = np.mean(confidences)
                
                # Calculate evidence score
                evidence_score = self._calculate_evidence_score(
                    predictions,
                    confidences
                )
                
                if progress_callback:
                    progress_callback(0.6)  # Scores calculated
                    
                # Calculate temporal statistics
                temporal_stats = self._calculate_temporal_stats(
                    predictions,
                    confidences,
                    timestamps
                )
                
                if progress_callback:
                    progress_callback(0.9)  # Temporal statistics calculated
                    
                # Create aggregated result
                aggregated = {
                    'prediction': float(weighted_prediction),
                    'confidence': float(confidence_score),
                    'evidence_score': float(evidence_score),
                    'temporal_stats': temporal_stats,
                    'metadata': {
                        'num_results': len(results),
                        'time_range': {
                            'start': float(timestamps.min()),
                            'end': float(timestamps.max())
                        }
                    }
                }
                
                if progress_callback:
                    progress_callback(1.0)  # Aggregation complete
                    
                return aggregated
                
        except Exception as e:
            logger.error(f"Result aggregation failed: {e}")
            raise AggregationError(
                message="Failed to aggregate results",
                error_code=7010,
                operation="aggregate_results",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _calculate_evidence_score(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray
    ) -> float:
        """
        Calculate evidence score.
        
        Args:
            predictions: Array of predictions
            confidences: Array of confidence scores
            
        Returns:
            Evidence score (0-1)
        """
        try:
            # Count high confidence predictions
            high_conf_mask = confidences >= self._confidence_threshold
            num_high_conf = np.sum(high_conf_mask)
            
            # Calculate agreement score
            if num_high_conf >= self._min_evidence:
                high_conf_preds = predictions[high_conf_mask]
                agreement = 1.0 - np.std(high_conf_preds)
                return float(agreement)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Evidence score calculation failed: {e}")
            return 0.0

    def _calculate_temporal_stats(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate temporal statistics.
        
        Args:
            predictions: Array of predictions
            confidences: Array of confidence scores
            timestamps: Array of timestamps
            
        Returns:
            Dictionary of temporal statistics
        """
        try:
            # Sort by timestamp
            sort_idx = np.argsort(timestamps)
            sorted_preds = predictions[sort_idx]
            sorted_confs = confidences[sort_idx]
            sorted_times = timestamps[sort_idx]
            
            # Calculate moving average
            window_size = min(5, len(sorted_preds))
            if window_size > 1:
                moving_avg = np.convolve(
                    sorted_preds,
                    np.ones(window_size) / window_size,
                    mode='valid'
                )
            else:
                moving_avg = sorted_preds
                
            # Calculate trend
            if len(sorted_preds) > 1:
                trend = np.polyfit(
                    np.arange(len(sorted_preds)),
                    sorted_preds,
                    deg=1
                )[0]
            else:
                trend = 0.0
                
            # Calculate stability
            stability = 1.0 - np.std(moving_avg) if len(moving_avg) > 1 else 1.0
            
            return {
                'moving_average': moving_avg.tolist(),
                'trend': float(trend),
                'stability': float(stability),
                'time_points': sorted_times.tolist()
            }
            
        except Exception as e:
            logger.error(f"Temporal statistics calculation failed: {e}")
            return {
                'moving_average': [],
                'trend': 0.0,
                'stability': 0.0,
                'time_points': []
            }

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            with self._lock:
                self._results.clear()
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise AggregationError(
                message="Failed to cleanup resources",
                error_code=7011,
                operation="cleanup",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def __del__(self) -> None:
        """Cleanup in destructor."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Result aggregator cleanup in destructor failed: {e}") 