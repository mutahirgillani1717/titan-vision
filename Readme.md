# 👁️ Titan Vision: High-Speed Object Tracking Pipeline

**Developed by:** Mutahir Hussainn  
**Live Demo:** [Link to your Streamlit App] (You will add this link in Phase 4)

## 🚀 Overview
Titan Vision is a high-speed, edge-optimized computer vision pipeline designed for real-time object detection and tracking. Built with **YOLOv10** and enhanced with a modern **Streamlit** dashboard, this application processes video streams dynamically while heavily filtering false positives in chaotic visual environments.

## 🧠 Core Technologies
* **Computer Vision:** YOLOv10 (Nano model for high-FPS edge inference)
* **Tracking Algorithm:** ByteTrack (Multi-object persistent ID tracking)
* **Interface & Telemetry:** Streamlit, Custom CSS Dashboards
* **Processing Framework:** OpenCV, PyTorch

## ⚙️ Key Features
* **Live Telemetry:** Real-time calculation of FPS, frames processed, and active on-screen targets.
* **Dynamic Filtering:** Interactive UI allowing users to isolate specific COCO classes (e.g., tracking only 'Motorcycles' in heavy traffic).
* **Anti-Hallucination:** Strict confidence thresholds to prevent false-positive artifacts in unpredictable lighting.

## 💻 Run Locally
To test this pipeline on your own machine:
1. Clone the repository:
   ```bash
   git clone [https://github.com/mutahirgillani1717/titan-vision.git](https://github.com/mutahirgillani1717/titan-vision.git)
