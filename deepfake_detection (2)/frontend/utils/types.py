from typing import TypedDict, List, Optional
import numpy as np
from datetime import datetime

class VideoMetadata(TypedDict):
    path: str
    frames: int
    fps: float
    duration: float
    resolution: tuple[int, int]

class AnalysisResult(TypedDict):
    prediction: int  # 0-4 for Fake to Real
    confidence: float  # 0-1
    heatmap: Optional[np.ndarray]
    anomalies: List[dict]
    timestamp: datetime

def validate_metadata(metadata: VideoMetadata) -> bool:
    """Validate video metadata."""
    return all([
        isinstance(metadata['frames'], int),
        isinstance(metadata['fps'], (int, float)),
        isinstance(metadata['duration'], (int, float)),
        len(metadata['resolution']) == 2
    ])