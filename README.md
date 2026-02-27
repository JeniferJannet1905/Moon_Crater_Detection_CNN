##Lunar Crater Detection using CNN & Streamlit##
#Project Overview

The Lunar Crater Detection System is a Deep Learning–based application designed to automatically identify crater regions from Lunar Digital Elevation Model (DEM) images. The system uses a Convolutional Neural Network (CNN) for patch-based classification and provides an interactive Streamlit interface for real-time crater detection and visualization.

This project reduces manual crater inspection effort and enables efficient, scalable planetary surface analysis.

🚀 Key Features

✅ Automated crater detection using CNN

✅ Patch-based sliding window scanning (32×32 pixels)

✅ Confidence-based crater localization

✅ DBSCAN clustering to merge duplicate detections

✅ Grad-CAM explainability for model transparency

✅ Interactive Streamlit web interface

✅ Lightweight CPU-based inference

🧠 System Workflow

User uploads a lunar DEM image via Streamlit interface.

Image is converted to grayscale and normalized (0–1 scale).

Sliding window extracts 32×32 patches across the image.

Each patch is passed to the trained CNN model.

Patches with confidence ≥ threshold are marked as craters.

DBSCAN merges overlapping detections.

Final annotated image with crater coordinates and confidence values is displayed.

🏗️ Model Architecture

The CNN model consists of:

Convolution Layers (feature extraction)

MaxPooling Layers (dimensionality reduction)

Flatten Layer

Dense Fully Connected Layers

Output Layer (Sigmoid / Softmax for crater classification)

Training Configuration:

Optimizer: Adam

Loss Function: Categorical Crossentropy

Batch Size: 32

Epochs: 15–30

Data Augmentation applied

🔍 Explainable AI (Grad-CAM)

Grad-CAM is integrated to:

Highlight important pixel regions influencing prediction

Improve model interpretability

Increase scientific reliability of crater detection

Heatmaps are generated for selected patches to verify that the model focuses on circular depression structures.

🛠️ Technologies Used
Backend

Python

TensorFlow / Keras

NumPy

OpenCV

Frontend

Streamlit

Matplotlib

AI & Image Processing

Convolutional Neural Network (CNN)

DBSCAN Clustering

Grad-CAM Visualization

💻 Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/your-username/lunar-crater-detection.git
cd lunar-crater-detection
2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Application
streamlit run app.py

Then open the local server URL shown in the terminal.

📊 Performance Highlights

Efficient crater vs non-crater classification

Reduced duplicate detections using DBSCAN

Fast CPU-based inference (no GPU required)

Stable performance across different lunar surface regions

⚠️ Limitations

Detects crater centers only (not full segmentation)

Cannot measure crater diameter/depth in current version

Model performance depends on DEM resolution

Generalization to other planets requires retraining

🔮 Future Enhancements

Implement U-Net / Mask R-CNN for crater segmentation

Estimate crater diameter and depth from DEM gradients

Extend detection to Mars and Mercury

Cloud deployment (AWS / Azure / GCP)

GIS integration for exporting crater coordinates

📂 Project Structure
├── app.py
├── crater_model.keras
├── requirements.txt
├── README.md
└── sample_images/
📚 References

He et al., Deep Residual Learning for Image Recognition (CVPR 2016)

Grad-CAM: Selvaraju et al., ICCV 2017

TensorFlow Documentation

NASA Planetary Data Archive

OpenCV & Streamlit Documentation
