"""
DeepFake Detection System - EfficientNet Utilities
Created: 2025-06-07
Author: ninjacode911

This module provides utility functions for EfficientNet model operations with
optimized performance and error handling.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Optional, Tuple, Union, Any, Dict
from pathlib import Path
import logging
from PIL import Image
import timm
import gc

from ..base import ModelConfig
from ...core.exceptions.backend_exceptions import ModelError

# Configure logger
logger = logging.getLogger(__name__)

class GradCAM:
    """Gradient-weighted Class Activation Mapping."""
    
    def __init__(self, model: nn.Module, target_layer: str) -> None:
        """
        Initialize GradCAM.
        
        Args:
            model: Model to analyze
            target_layer: Name of target layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Get target layer
        target = dict([*self.model.named_modules()])[self.target_layer]
        target.register_forward_hook(forward_hook)
        target.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate CAM heatmap.
        
        Args:
            input_tensor: Input image tensor
            
        Returns:
            Heatmap array
        """
        try:
            # Forward pass
            model_output = self.model(input_tensor)
            pred_class = model_output.argmax(dim=1)
            
            # Backward pass
            self.model.zero_grad()
            class_score = model_output[0, pred_class]
            class_score.backward()
            
            # Generate heatmap
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
            for i in range(self.activations.shape[1]):
                self.activations[:, i, :, :] *= pooled_gradients[i]
                
            heatmap = torch.mean(self.activations, dim=1).squeeze()
            heatmap = torch.relu(heatmap)
            heatmap /= torch.max(heatmap)
            
            return heatmap.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
            raise ModelError(
                message="Failed to generate heatmap",
                error_code=4200,
                operation="generate_heatmap",
                details={'error': str(e)}
            )
        finally:
            cleanup_gradients()

def preprocess_image(
    image_path: Union[str, Path], 
    size: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        image_path: Path to input image
        size: Target size (height, width)
        
    Returns:
        Preprocessed image tensor
    """
    try:
        # Load and resize image
        image = Image.open(image_path).convert('RGB')
        image = image.resize(size, Image.LANCZOS)
        
        # Convert to numpy array
        image = np.array(image) / 255.0
        image = image.transpose(2, 0, 1)
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)
        
        return image
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise ModelError(
            message="Failed to preprocess image",
            error_code=4201,
            operation="preprocess_image",
            details={
                'image_path': str(image_path),
                'error': str(e)
            }
        )

def load_efficientnet(
    model_name: str,
    weights_path: Optional[Union[str, Path]] = None,
    config: Optional[ModelConfig] = None
) -> nn.Module:
    try:
        model = timm.create_model(model_name, pretrained=False)
        
        # Update weights path handling for folder structure
        weights_dir = Path(weights_path) if weights_path else Path('models/efficientnet/efficientnetv2_m_finetuned')
        if weights_dir.is_dir():
            # Load from directory structure
            state_dict = torch.load(weights_dir / 'data.pkl')
        else:
            # Fallback to direct .pth loading
            state_dict = torch.load(weights_path)
            
        model.load_state_dict(state_dict)
        return model
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        raise ModelError(
            message="Failed to load EfficientNet model",
            error_code=4202,
            details={'model_name': model_name, 'weights_path': str(weights_path)}
        )

def get_gradcam_heatmap(
    model: nn.Module,
    image: torch.Tensor,
    layer_name: str = 'conv_head'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate GradCAM visualization.
    
    Args:
        model: Model to analyze
        image: Input image tensor
        layer_name: Target layer name
        
    Returns:
        Tuple of (original image, heatmap)
    """
    try:
        # Move image to device
        device = next(model.parameters()).device
        image = image.to(device)
        
        # Initialize GradCAM
        grad_cam = GradCAM(model, layer_name)
        
        # Generate heatmap
        heatmap = grad_cam.generate_heatmap(image)
        
        # Resize heatmap
        heatmap = cv2.resize(
            heatmap,
            (image.shape[3], image.shape[2])
        )
        
        # Convert to RGB heatmap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Get original image
        orig_img = image.cpu().squeeze().permute(1, 2, 0).numpy()
        orig_img = np.uint8(255 * orig_img)
        
        # Blend images
        heatmap = cv2.addWeighted(orig_img, 0.5, heatmap, 0.5, 0)
        
        return orig_img, heatmap
        
    except Exception as e:
        logger.error(f"GradCAM visualization failed: {e}")
        raise ModelError(
            message="Failed to generate GradCAM visualization",
            error_code=4203,
            operation="get_gradcam",
            details={'error': str(e)}
        )
    finally:
        cleanup_gradients()

def cleanup_gradients() -> None:
    """Clean up gradient computations and GPU memory."""
    try:
        # Clear gradients
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
    except Exception as e:
        logger.error(f"Gradient cleanup failed: {e}")
        raise ModelError(
            message="Failed to cleanup gradients",
            error_code=4204,
            operation="cleanup_gradients",
            details={'error': str(e)}
        )