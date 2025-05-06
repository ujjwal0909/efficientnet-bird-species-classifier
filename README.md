## 🐦 EfficientNet-Based Bird Species Classifier
Welcome to our collaborative project on bird species classification using computer vision. This repository showcases our best-performing model, leveraging EfficientNetB0 for accurate bird species identification.

## 📊 Dataset
We utilized various datasets from Kaggle, comprising 200 bird species and images organized into train, validation, and test sets

## 🧠 Model Architecture
Our final model employs EfficientNetB0, a state-of-the-art convolutional neural network known for its efficiency and accuracy. Key features include:
Pretrained on ImageNet: Leveraging transfer learning for better performance
Fine-tuned: Adjusted on our specific dataset for optimal result
Data Augmentation: Techniques like rotation, flipping, and scaling to enhance model robustness

## 🚀 Getting Started
Prerequisites
Ensure you have the following installed:
Python 3.7 or higher
PyTorch
torchvision
matplotlib
numpy
PIL

## Running the Notebook
Open and run the Final Output.ipynb notebook to:

Load and preprocess the dataset
Train the EfficientNetB0 model
Evaluate model performance

## Making Predictions

Use the bird.py script to predict bird species from images:

#### python bird.py --image_path path_to_your_image.jpg

## 📈 Results
Our EfficientNetB0 model achieved:
Training Accuracy: 83.20%
Validation Accuracy: 71.55%
Test Accuracy: 84.12%
These results demonstrate the model's effectiveness in classifying bird species accurately.

## 👥 Team Contributions
This project was a collaborative effort:

Myself : Implemented EfficientNetB0 model and training pipeline
https://github.com/devdalal2002: Trained the MobileNetV2 and one of the earlier EfficientNetB0 models
https://github.com/rushikesh2842: Trained the ResNet models, developed the Gradio UI
