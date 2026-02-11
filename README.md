# OpenScope 🔬

**Scientific Image Analysis Platform**

![Build Status](https://img.shields.io/github/actions/workflow/status/KULLANICI_ADIN/OpenScope/build.yml?branch=main)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

**OpenScope** is a modular, cross-platform desktop application designed for high-resolution microscopy image analysis. It specializes in detecting bacteria and other micro-objects using advanced computer vision algorithms.

Built with **Python**, **PyQt6**, **OpenCV**, and **NumPy**, OpenScope features a plugin-based architecture that allows researchers to easily integrate custom segmentation logic.

---

## 🌟 Key Features

- **High-Fidelity Imaging:**
  - Supports large 8-bit and 16-bit TIFF files.
  - Lossless loading and saving (LZW compression).
  - Smart Canvas with high-performance Zoom & Pan (via `QGraphicsView`).

- **Advanced Analysis Plugins:**
  - **Basic Blob Detector:** Simple thresholding for high-contrast images.
  - **Advanced Bacteria Detector (v4.5):** Uses a **Consensus Voting** mechanism (CLAHE, LoG, TopHat, Adaptive Threshold) to robustly identify bacteria.
  - **Rescue Bacteria Detector:** Specialized algorithm for detecting faint/ghost objects.

- **ROI (Region of Interest):**
  - Select specific areas for analysis using `Ctrl + Mouse Drag`.
  - Analyze only the ROI or the full image.

- **Overlay & Export:**
  - Real-time green overlay visualization of detected objects.
  - Export results as merged TIFFs (Original + Overlay) without quality loss.

- **Cross-Platform:**
  - Runs natively on Windows, macOS, and Linux.
  - Automated builds via GitHub Actions.

---

## 🛠️ Installation

### Option 1: Download Executable (Recommended for Users)

Go to the [Actions](https://github.com/KULLANICI_ADIN/OpenScope/actions) tab or **Releases** page (if created) to download the latest artifact for your operating system:

- `OpenScope-Windows.zip`
- `OpenScope-Linux.zip`
- `OpenScope-macOS.zip`

> **Note:** On Linux/macOS, you may need to grant execution permissions: `chmod +x OpenScope`

### Option 2: Run from Source (For Developers)

#### Prerequisites

- Python 3.10 or higher
- Git

#### 1. Clone the Repository

```bash
git clone https://github.com/KULLANICI_ADIN/OpenScope.git
cd OpenScope
```

#### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Linux Users Only:** If you encounter PyQt6 errors, install system libs:
>
> ```bash
> sudo apt-get install libxcb-cursor0 libegl1 libgl1
> ```

#### 4. Run the Application

```bash
python main.py
```

---

## 📖 How to Use

1. **Open Image:** Click `File > Open TIFF` (`Ctrl+O`) to load a microscopy image.

2. **Select Algorithm:** Choose an algorithm from the sidebar dropdown (e.g., "Advanced Bacteria Detector").

3. **Select ROI (Optional):**
   - Hold `Ctrl` and `Left Click & Drag` on the image to draw a box.
   - Only this area will be analyzed. Click outside to reset.

4. **Run Analysis:** Click the blue **"Run Analysis"** button.
   - Detected objects will be highlighted in green.
   - Object count will appear in the status bar.

5. **Save Results:** Click the green **"Save Result as TIFF"** button to save the analyzed image with the overlay.

---

## 🧩 Plugin Architecture

OpenScope uses a modular plugin system located in `plugins/`.

| Plugin | Description | Best For |
|---|---|---|
| Basic Blob | Simple Percentile Thresholding | High contrast, clean images |
| Advanced (v4.5) | Multi-method Voting (CLAHE + LoG + Adaptive) | Complex backgrounds, scientific data |
| Rescue (v1) | Aggressive Adaptive Thresholding | Faint, out-of-focus objects |

### Adding a New Plugin

Create a new file in `plugins/` inheriting from `AnalysisPlugin`:

```python
from core.plugin_interface import AnalysisPlugin

class MyCustomPlugin(AnalysisPlugin):
    @property
    def name(self):
        return "My Custom Algo"

    def run(self, image, params):
        # Your OpenCV logic here
        return mask
```

Then register it in `gui/main_window.py`.

---

## 📦 Building Standalone App

To create a standalone `.exe` or binary file locally:

1. **Install PyInstaller:**

   ```bash
   pip install pyinstaller
   ```

2. **Run the build command:**

   ```bash
   # Windows / Linux
   pyinstaller --noconfirm --onedir --windowed --clean --name "OpenScope" main.py

   # macOS
   pyinstaller --noconfirm --onedir --windowed --name "OpenScope" main.py
   ```

3. Check the `dist/` folder for the output.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
