import importlib
import sys

REQUIRED_PACKAGES = [
    ('torch', 'PyTorch'),
    ('torchvision', 'TorchVision'), 
    ('transformers', 'Transformers'),
    ('PyQt6', 'PyQt6'),
    ('cv2', 'OpenCV'),
    ('numpy', 'NumPy'),
    ('qtawesome', 'QtAwesome'),
    ('librosa', 'Librosa'),  # For audio processing
    ('pydub', 'Pydub'),      # For audio conversion
    ('sqlite3', 'SQLite3'),   # For database
    ('qtawesome', 'QtAwesome') # For icons
]

def check_dependencies():
    missing = []
    for package, name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"✓ {name} found")
        except ImportError:
            missing.append(name)
            print(f"✗ {name} missing")
    
    if missing:
        print("\nMissing dependencies:")
        print("\n".join(f"- {pkg}" for pkg in missing))
        sys.exit(1)
    print("\nAll dependencies satisfied!")

if __name__ == "__main__":
    check_dependencies()