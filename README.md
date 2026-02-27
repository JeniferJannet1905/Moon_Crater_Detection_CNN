# Lunar Crater Detection using CNN & Streamlit

## Project Overview

The Lunar Crater Detection System is a Deep Learning-based application designed to automatically identify crater regions from Lunar Digital Elevation Model (DEM) images. The system uses a Convolutional Neural Network (CNN) for patch-based classification and provides an interactive Streamlit interface for real-time crater detection and visualization.

This project reduces manual crater inspection effort and enables efficient, scalable planetary surface analysis.

---

## Key Features

- Automated crater detection using CNN  
- Patch-based sliding window scanning (32×32 pixels)  
- Confidence-based crater localization  
- DBSCAN clustering to merge duplicate detections  
- Grad-CAM explainability for model transparency  
- Interactive Streamlit web interface  
- Lightweight CPU-based inference  

---

## System Workflow

1. User uploads a lunar DEM image via Streamlit interface.
2. Image is converted to grayscale and normalized (0–1 scale).
3. Sliding window extracts 32×32 patches across the image.
4. Each patch is passed to the trained CNN model.
5. Patches with confidence ≥ threshold are marked as craters.
6. DBSCAN merges overlapping detections.
7. Final annotated image with crater coordinates and confidence values is displayed.

---

## Model Architecture

The CNN model consists of:

- Convolution Layers (feature extraction)
- MaxPooling Layers (dimensionality reduction)
- Flatten Layer
- Dense Fully Connected Layers
- Output Layer (Sigmoid/Softmax for crater classification)

### Training Configuration:
- Optimizer: Adam  
- Loss Function: Categorical Crossentropy  
- Batch Size: 32  
- Epochs: 15–30  
- Data Augmentation applied  

---

## Explainable AI (Grad-CAM)

Grad-CAM is integrated to:
- Highlight important pixel regions influencing prediction
- Improve model interpretability
- Increase scientific reliability of crater detection

Heatmaps are generated for selected patches to verify that the model focuses on circular depression structures.

---

## Technologies Used

### Backend
- Python  
- TensorFlow / Keras  
- NumPy  
- OpenCV  

### Frontend
- Streamlit  
- Matplotlib  

### AI & Image Processing
- Convolutional Neural Network (CNN)  
- DBSCAN Clustering  
- Grad-CAM Visualization  

---

## Installation & Setup

```bash
git clone https://github.com/your-username/lunar-crater-detection.git
cd lunar-crater-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py   
