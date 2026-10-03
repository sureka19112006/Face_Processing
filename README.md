# 👤 FACE PROCESSING

An interactive web-based Face Processing Laboratory developed using Python, OpenCV and Streamlit.

This application provides a visual and interactive way to understand important face processing and computer vision techniques through real-time image processing, calculations, theory and output visualization.

🚀 Live Demo

https://faceprocessing-vnbrhjczv5hfeutxcrtsy4.streamlit.app/



✨ Features

- 👤 Face image processing
- 🖼️ Default face image
- 📤 Upload custom face images
- 🔍 Template Matching
- 🎯 Viola-Jones Algorithm
- 🧠 DeepFace Analysis
- 🧬 FaceNet Embedding
- 📐 Formula-based calculations
- 🔢 Live pixel values
- 🔬 Pixel Lens visualization
- 📊 Cosine similarity
- 📏 Euclidean distance
- 🧮 Embedding calculations
- 📖 Theory and working steps
- 🖼️ Input and output image comparison
- 🎨 Interactive user interface

## 🧪 Operations

# 1. Template Matching

Template Matching searches for a smaller template image inside a larger image.

Formula:

Correlation coefficient is used to measure the similarity between the template and image region.

The application displays:

- Correlation score
- Best matching pixel position
- Template size
- Processed output image

# 2. Viola-Jones Algorithm

Viola-Jones is a classical face detection algorithm based on Haar-like features, integral images, AdaBoost and cascade classifiers.

The application displays:

- Detected face regions
- Integral image calculation
- Haar feature calculation
- Rectangle sums
- Processing time
- Detected output

# 3. DeepFace

DeepFace is a deep-learning based framework for face analysis and representation.

The application provides:

- Face detection
- Feature extraction
- Face analysis
- Numerical feature calculations
- Processed output

# 4. FaceNet

FaceNet represents a face as a numerical embedding vector.

The application displays:

- Embedding dimension
- Embedding values
- Cosine similarity
- Euclidean distance
- Mean
- Standard deviation
- Embedding preview

# 🔬 Pixel Lens

The Pixel Lens provides an interactive view of image pixel values.

It displays:

- Grayscale pixel values
- Pixel intensity
- Mean intensity
- Pixel range
- Face region coordinates

# 🛠️ Technologies Used

- Python
- Streamlit
- OpenCV
- NumPy
- Pillow
- Scikit-learn

# 📁 Project Structure

```text
Face_Processing/
│
├── app.py
├── requirements.txt
└── README.md
