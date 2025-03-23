# skincancerpred
## Overview
This project focuses on comparing the performance of different machine learning models in predicting skin cancer types using the Skin Cancer MNIST: HAM10000 dataset. The dataset contains a large collection of multi-source dermatoscopic images of pigmented lesions. The goal is to perform multiclass classification to identify different types of skin cancer, including Melanocytic nevi, Melanoma, Benign keratosis-like lesions, Basal cell carcinoma, Actinic keratoses, Vascular lesions, and Dermatofibroma.

The project explores three primary models:
- Feedforward Neural Network (FNN)
- Random Forest
- Convolutional Neural Network (CNN)

The models are evaluated based on their accuracy, precision, and recall, with a focus on identifying the strengths and weaknesses of each approach.

## Dataset
The dataset used is the HAM10000 dataset, which includes:
- Metadata: Contains information about the images, such as lesion type, age, sex, and localization.
- Images: Dermatoscopic images of skin lesions in different resolutions (8x8, 28x28, and 128x128 pixels) in both grayscale and RGB formats.
- The dataset is available on Kaggle.

Techniques and Models
1. Feedforward Neural Network (FNN)
- A simple neural network with fully connected layers.
- Hyperparameter tuning was performed to optimize the number of hidden layers, activation functions, and learning rates.
- Achieved a test accuracy of ~66.4%.

2. Random Forest
- A traditional machine learning model used for classification.
- Hyperparameter tuning was performed to optimize the number of estimators, max depth, and other parameters.
- Achieved a test accuracy of ~61.2%.

3. Convolutional Neural Network (CNN)
- A deep learning model specifically designed for image classification.
- Multiple convolutional layers with max pooling and dropout for regularization.
- Achieved the highest test accuracy of ~76.7%.

## Results
Model	Training Accuracy	Validation Accuracy	Test Accuracy
Feedforward Neural Network	67.52%	66.40%	66.40%
Random Forest	65.93%	61.51%	61.16%
Convolutional Neural Network	80.13%	75.74%	76.68%

## Key Findings:
- CNN outperformed both FNN and Random Forest, demonstrating the effectiveness of convolutional layers in image classification tasks.
- The Random Forest model struggled with the high-dimensional image data, resulting in lower accuracy compared to the neural network models.
- Data augmentation and class balancing were critical in improving the performance of the CNN model, especially for minority classes.

## How to Replicate the Results
1. Setup
Clone the repository:
git clone https://github.com/your-username/skin-cancer-classification.git
cd skin-cancer-classification
Install the required dependencies:

```python
pip install -r requirements.txt
```

2. Data Preparation
- Download the HAM10000 dataset from Kaggle.
- Place the dataset in the ./data directory.

3. Training the Models
Run the Jupyter notebook or Python script to train the models:

```python
jupyter notebook skin_cancer_classification.ipynb
```python

Alternatively, run the script directly:
```python
python skin_cancer_classification.py
```python

4. Evaluation
The notebook includes code for evaluating the models on the test set and generating confusion matrices.
You can also visualize the training and validation accuracy curves to understand the model's performance over time.

## Impact
This project demonstrates the potential of machine learning and deep learning in medical image analysis, particularly in the early detection of skin cancer. By comparing different models, we highlight the importance of choosing the right architecture for specific tasks. The CNN model's superior performance suggests that deep learning techniques can be highly effective in medical diagnostics, potentially aiding dermatologists in making more accurate and timely diagnoses.

## Future Work
- Larger Dataset: Experiment with a larger dataset to further improve model generalization.
- Transfer Learning: Explore the use of pre-trained models like ResNet or EfficientNet for better performance.
- Explainability: Incorporate techniques like Grad-CAM to provide insights into the model's decision-making process.
- Deployment: Develop a web or mobile application for real-time skin cancer detection.

## Acknowledgments
The HAM10000 dataset was sourced from Kaggle.
This project was inspired by the need for accessible and accurate tools for skin cancer detection.

Note: This project is for educational and research purposes only. It is not intended for clinical use. Always consult a medical professional for accurate diagnosis and treatment.
