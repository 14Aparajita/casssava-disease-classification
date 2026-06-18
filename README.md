# Cassava Leaf Health Classifier 🌿

A deep learning-based web application built with **Streamlit** that analyzes cassava leaves to detect diseases and provides actionable crop care suggestions instantly.

## Overview
This project uses a custom CNN architecture with **ECA-Net** Attention to identify and classify cassava leaf conditions into one of 5 categories:
- **Cassava Bacterial Blight (CBB)**
- **Cassava Brown Streak Disease (CBSD)**
- **Cassava Green Mite (CGM)**
- **Cassava Mosaic Disease (CMD)**
- **Healthy Leaf**

The model analyzes uploaded images and provides not only a diagnosis but also a confidence score and recommended management actions to help farmers maintain healthy crops.

## Features
- **Instant Diagnosis:** Upload an image of a cassava leaf (`.jpg`, `.png`, `.jpeg`) and get immediate classification.
- **Actionable Tips:** Detailed descriptions of the identified disease and best practices for disease management and crop care.
- **Dynamic UI:** Features a modern, glassmorphism-inspired design for a clean, user-friendly experience.

## Technology Stack
- **Frontend/Backend:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** TensorFlow, Keras (Custom ECALayer implementation)
- **Image Processing:** PIL (Pillow), NumPy
- **Styling:** Custom CSS (Glassmorphism, Google Fonts: Outfit & Inter)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd cassava-disease-classification
   ```

2. **Install the dependencies:**
   Make sure you have Python installed. Run the following command to install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## Model Architecture
The core model is a Convolutional Neural Network (CNN) integrated with an **ECA-Net** (Efficient Channel Attention) module to improve feature representation without significantly increasing the model's complexity.

## Disclaimer
This tool is intended to assist in plant pathology research and farming practices, but it is not a substitute for professional agricultural diagnosis.
