# API Documentation

This document describes the API client implementation and available endpoints for the Deepfake Detection system.

## API Client

The `APIClient` class (`api_client.py`) provides a robust interface for communicating with the backend API. It handles authentication, request retries, and response parsing.

### Configuration

```python
API_BASE_URL = "http://localhost:8000"  # Default backend URL
MAX_RETRIES = 3  # Maximum number of retry attempts
TIMEOUT = 30  # Request timeout in seconds
```

### Methods

#### `analyze_video(video_path: str) -> dict`
Analyzes a video file for potential deepfakes.

**Parameters:**
- `video_path`: Path to the video file

**Returns:**
```python
{
    "prediction": int,  # 0-4 (Fake to Real)
    "confidence": float,  # 0-1
    "heatmap": Optional[np.ndarray],
    "anomalies": List[dict],
    "timestamp": datetime
}
```

#### `get_analysis_status(task_id: str) -> dict`
Checks the status of an ongoing analysis.

**Parameters:**
- `task_id`: Unique identifier for the analysis task

**Returns:**
```python
{
    "status": str,  # "pending", "processing", "completed", "failed"
    "progress": float,  # 0-1
    "message": str
}
```

#### `cancel_analysis(task_id: str) -> bool`
Cancels an ongoing analysis.

**Parameters:**
- `task_id`: Unique identifier for the analysis task

**Returns:**
- `bool`: True if cancellation was successful

## Error Handling

The API client implements comprehensive error handling:

1. **Connection Errors:**
   - Automatic retry with exponential backoff
   - Maximum retry limit
   - Connection timeout handling

2. **Response Validation:**
   - Schema validation for all responses
   - Type checking for numeric values
   - Required field verification

3. **Error Types:**
   - `APIError`: Base class for API-related errors
   - `ConnectionError`: Network-related issues
   - `ValidationError`: Response validation failures
   - `TimeoutError`: Request timeout

## Usage Example

```python
from api_client import APIClient

# Initialize client
client = APIClient()

try:
    # Start analysis
    result = client.analyze_video("path/to/video.mp4")
    
    # Check status
    status = client.get_analysis_status(result["task_id"])
    
    # Cancel if needed
    if status["status"] == "processing":
        client.cancel_analysis(result["task_id"])
        
except APIError as e:
    print(f"API Error: {e}")
except ConnectionError as e:
    print(f"Connection Error: {e}")
```

## Best Practices

1. **Resource Management:**
   - Always use context managers or explicit cleanup
   - Monitor memory usage for large responses
   - Implement proper error handling

2. **Performance:**
   - Use connection pooling
   - Implement request caching where appropriate
   - Monitor response times

3. **Security:**
   - Validate all input parameters
   - Handle sensitive data appropriately
   - Implement proper authentication

## Testing

The API client includes comprehensive tests in `tests/test_api_client.py`:

- Unit tests for all methods
- Integration tests with mock server
- Error handling tests
- Performance benchmarks

Run tests using:
```bash
python -m pytest tests/test_api_client.py
```
