"""
DeepFake Detection System - Report Generator
Created: 2025-06-07
Author: ninjacode911

This module implements report generation for deepfake detection
with proper resource management and error handling.
"""

import logging
import time
import traceback
import threading
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np
import torch
from datetime import datetime
from contextlib import contextmanager

from .exceptions.backend_exceptions import ReportError, ValidationError, ResourceError
from ..config import config_manager

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generator class for deepfake detection reports."""

    def __init__(self) -> None:
        """Initialize report generator with configuration."""
        try:
            self._lock = threading.RLock()
            self._memory_limit = config_manager.get("report.memory_limit", 512 * 1024 * 1024)  # 512MB
            self._reports_dir = Path(config_manager.get("report.directory", "reports"))
            self._reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize report registry
            self._reports: Dict[str, Dict[str, Any]] = {}
            
            logger.info("Report generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Report generator initialization failed: {e}")
            raise ReportError(
                message="Failed to initialize report generator",
                error_code=8000,
                operation="init",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _validate_results(self, results: Dict[str, Any]) -> None:
        """
        Validate detection results.
        
        Args:
            results: Detection results
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            required_fields = {
                'prediction',
                'confidence',
                'evidence_score',
                'temporal_stats',
                'metadata'
            }
            
            # Check required fields
            missing_fields = required_fields - set(results.keys())
            if missing_fields:
                raise ValidationError(
                    message="Missing required result fields",
                    error_code=8001,
                    details={'missing_fields': list(missing_fields)}
                )
                
            # Validate prediction
            if not isinstance(results['prediction'], (int, float)):
                raise ValidationError(
                    message="Invalid prediction type",
                    error_code=8002,
                    details={'type': type(results['prediction'])}
                )
                
            # Validate confidence
            if not isinstance(results['confidence'], (int, float)):
                raise ValidationError(
                    message="Invalid confidence type",
                    error_code=8003,
                    details={'type': type(results['confidence'])}
                )
                
            if not 0 <= results['confidence'] <= 1:
                raise ValidationError(
                    message="Confidence must be between 0 and 1",
                    error_code=8004,
                    details={'confidence': results['confidence']}
                )
                
            # Validate evidence score
            if not isinstance(results['evidence_score'], (int, float)):
                raise ValidationError(
                    message="Invalid evidence score type",
                    error_code=8005,
                    details={'type': type(results['evidence_score'])}
                )
                
            if not 0 <= results['evidence_score'] <= 1:
                raise ValidationError(
                    message="Evidence score must be between 0 and 1",
                    error_code=8006,
                    details={'evidence_score': results['evidence_score']}
                )
                
            # Validate temporal stats
            if not isinstance(results['temporal_stats'], dict):
                raise ValidationError(
                    message="Invalid temporal stats type",
                    error_code=8007,
                    details={'type': type(results['temporal_stats'])}
                )
                
            # Validate metadata
            if not isinstance(results['metadata'], dict):
                raise ValidationError(
                    message="Invalid metadata type",
                    error_code=8008,
                    details={'type': type(results['metadata'])}
                )
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Results validation failed: {e}")
            raise ValidationError(
                message="Results validation failed",
                error_code=8009,
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
                    error_code=8010,
                    details={
                        'used': current_memory - start_memory,
                        'limit': self._memory_limit
                    }
                )
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            raise ResourceError(
                message="Memory monitoring failed",
                error_code=8011,
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def generate_report(
        self,
        results: Dict[str, Any],
        media_path: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Generate detection report.
        
        Args:
            results: Detection results
            media_path: Path to analyzed media
            progress_callback: Optional callback for progress updates
            
        Returns:
            Generated report
        """
        try:
            # Validate results
            self._validate_results(results)
            
            # Generate report with memory monitoring
            with self._memory_context():
                # Calculate report metrics
                metrics = self._calculate_metrics(results)
                
                if progress_callback:
                    progress_callback(0.3)  # Metrics calculated
                    
                # Generate analysis summary
                summary = self._generate_summary(results, metrics)
                
                if progress_callback:
                    progress_callback(0.6)  # Summary generated
                    
                # Generate recommendations
                recommendations = self._generate_recommendations(results, metrics)
                
                if progress_callback:
                    progress_callback(0.9)  # Recommendations generated
                    
                # Create report
                report = {
                    'media_path': media_path,
                    'timestamp': datetime.now().isoformat(),
                    'results': results,
                    'metrics': metrics,
                    'summary': summary,
                    'recommendations': recommendations,
                    'metadata': {
                        'version': config_manager.get("version", "1.0.0"),
                        'generator': 'ReportGenerator'
                    }
                }
                
                if progress_callback:
                    progress_callback(1.0)  # Report generated
                    
                return report
                
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise ReportError(
                message="Failed to generate report",
                error_code=8012,
                operation="generate_report",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def _calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate report metrics.
        
        Args:
            results: Detection results
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Calculate reliability score
            reliability = (
                0.4 * results['confidence'] +
                0.4 * results['evidence_score'] +
                0.2 * results['temporal_stats']['stability']
            )
            
            # Calculate risk level
            if reliability >= 0.8:
                risk_level = "Low"
            elif reliability >= 0.6:
                risk_level = "Medium"
            else:
                risk_level = "High"
                
            # Calculate consistency score
            consistency = 1.0 - abs(
                results['prediction'] -
                np.mean(results['temporal_stats']['moving_average'])
            )
            
            return {
                'reliability': float(reliability),
                'risk_level': risk_level,
                'consistency': float(consistency),
                'confidence': float(results['confidence']),
                'evidence': float(results['evidence_score'])
            }
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return {
                'reliability': 0.0,
                'risk_level': "Unknown",
                'consistency': 0.0,
                'confidence': 0.0,
                'evidence': 0.0
            }

    def _generate_summary(
        self,
        results: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate analysis summary.
        
        Args:
            results: Detection results
            metrics: Report metrics
            
        Returns:
            Dictionary of summary sections
        """
        try:
            # Generate prediction summary
            prediction = results['prediction']
            confidence = results['confidence']
            if prediction >= 0.8:
                prediction_summary = f"Strong evidence of deepfake (confidence: {confidence:.1%})"
            elif prediction >= 0.6:
                prediction_summary = f"Moderate evidence of deepfake (confidence: {confidence:.1%})"
            elif prediction >= 0.4:
                prediction_summary = f"Uncertain (confidence: {confidence:.1%})"
            elif prediction >= 0.2:
                prediction_summary = f"Likely authentic (confidence: {confidence:.1%})"
            else:
                prediction_summary = f"Strong evidence of authenticity (confidence: {confidence:.1%})"
                
            # Generate reliability summary
            reliability = metrics['reliability']
            if reliability >= 0.8:
                reliability_summary = "High reliability analysis"
            elif reliability >= 0.6:
                reliability_summary = "Moderate reliability analysis"
            else:
                reliability_summary = "Low reliability analysis"
                
            # Generate evidence summary
            evidence = results['evidence_score']
            if evidence >= 0.8:
                evidence_summary = "Strong supporting evidence"
            elif evidence >= 0.6:
                evidence_summary = "Moderate supporting evidence"
            else:
                evidence_summary = "Limited supporting evidence"
                
            return {
                'prediction': prediction_summary,
                'reliability': reliability_summary,
                'evidence': evidence_summary
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return {
                'prediction': "Analysis failed",
                'reliability': "Unknown reliability",
                'evidence': "Insufficient evidence"
            }

    def _generate_recommendations(
        self,
        results: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations.
        
        Args:
            results: Detection results
            metrics: Report metrics
            
        Returns:
            List of recommendations
        """
        try:
            recommendations = []
            
            # Add prediction-based recommendations
            prediction = results['prediction']
            if prediction >= 0.8:
                recommendations.append(
                    "Consider additional verification methods"
                )
                recommendations.append(
                    "Review source and context of media"
                )
            elif prediction <= 0.2:
                recommendations.append(
                    "Media appears authentic"
                )
                
            # Add reliability-based recommendations
            reliability = metrics['reliability']
            if reliability < 0.6:
                recommendations.append(
                    "Consider re-analyzing with different parameters"
                )
                recommendations.append(
                    "Gather additional evidence if possible"
                )
                
            # Add evidence-based recommendations
            evidence = results['evidence_score']
            if evidence < 0.6:
                recommendations.append(
                    "Consider manual verification"
                )
                
            # Add temporal-based recommendations
            stability = results['temporal_stats']['stability']
            if stability < 0.6:
                recommendations.append(
                    "Consider analyzing specific segments in detail"
                )
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return ["Unable to generate recommendations"]

    def save_report(
        self,
        report: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Save report to file.
        
        Args:
            report: Generated report
            output_path: Optional output path
            
        Returns:
            Path to saved report
        """
        try:
            # Generate output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self._reports_dir / f"report_{timestamp}.json"
            else:
                output_path = Path(output_path)
                
            # Create parent directories
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Report saving failed: {e}")
            raise ReportError(
                message="Failed to save report",
                error_code=8013,
                operation="save_report",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            with self._lock:
                self._reports.clear()
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise ReportError(
                message="Failed to cleanup resources",
                error_code=8014,
                operation="cleanup",
                details={'error': str(e), 'traceback': traceback.format_exc()}
            )

    def __del__(self) -> None:
        """Cleanup in destructor."""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Report generator cleanup in destructor failed: {e}") 