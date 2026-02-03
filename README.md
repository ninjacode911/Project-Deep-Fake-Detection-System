#  DeepGuard AI: Forensic Deepfake Detection System 

![Deepfake Detection](https://img.shields.io/badge/Status-Modernized-neonblue)
![Django](https://img.shields.io/badge/Backend-Django-092e20)
![PyTorch](https://img.shields.io/badge/AI-PyTorch-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

**DeepGuard AI** is a state-of-the-art forensic tool designed to detect deepfake videos using sequence analysis. By combining spatial feature extraction (CNN) with temporal modeling (LSTM), the system identifies subtle artifacts that indicate synthetic manipulation.

> [!IMPORTANT]
> **2026 UI Update**: This project has been modernized with a futuristic Cyberpunk/Glassmorphism interface for a premium forensic experience.

---

##  Features

-  **Sequence Analysis**: Deep learning architecture (ResNext50 + LSTM) for detecting temporal inconsistencies.
-  ** Dashboard**: A modernized, glassmorphism-inspired UI with interactive results.
-  **Drag-and-Drop**: Seamless video upload interface with real-time feedback.
-  **Live Webcam Scanner**: Real-time browser-side face verification using `face-api.js`.
-  **Artifact Heatmaps**: Visualization of region-specific manipulation probabilities (optional).
-  **Docker Ready**: One-command deployment for consistent environment setup.

---

##  Architecture

The system uses a two-stage forensic pipeline:
1. **Spatial Extraction**: A ResNext50 CNN extracts high-dimensional features from individual video frames.
2. **Temporal Modeling**: A Long Short-Term Memory (LSTM) network analyzes these features over time to detect anomalies between frames.

---

##  Getting Started

### Option 1: Using Docker (Recommended)
This is the easiest way to run the project as it handles all complex ML dependencies automatically.

1. Ensure you have [Docker](https://www.docker.com/) installed.
2. Navigate to the `Django Application` folder:
   ```bash
   cd "Django Application"
   ```
3. Start the system:
   ```bash
   docker-compose up --build
   ```
4. Access the dashboard at `http://127.0.0.1:8000`.

### Option 2: Local Installation (Manual)
*Note: Requires Visual Studio C++ Build Tools on Windows.*

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run migrations (if any):
   ```bash
   python manage.py migrate
   ```
3. Start the server:
   ```bash
   python manage.py runserver
   ```

---

##  Project Structure

- `Django Application/` - Core web application (Django).
  - `ml_app/` - Main app logic and ML views.
  - `static/css/styles_2026.css` - Custom modernized styling.
  - `templates/` - Modernized UI artifacts.
- `Model Creation/` - Research notebooks for model training.
- `models/` - Storage for trained `.pt` model files.
- `Documentation/` - Project reports and research papers.

---

##  Developed By

**Navnit** - *AI MLOps Engineer*
[GitHub Profile](https://github.com/ninjacode911)

---

## 📜 License
This project is licensed under the MIT License.
