"""Resource management package for DeepFake Detection System."""
try:
    from .theme_manager import theme_manager
    __all__ = ['theme_manager']
except ImportError as e:
    print(f"Failed to initialize resources: {e}")