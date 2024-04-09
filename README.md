# MNIST Handwritten Digit Classifier

## Overview
This project implements three different methods for classifying handwritten digits from the MNIST dataset: Deep Neural Network (DNN), Data Augmentation with DNN, and Data Augmentation with Convolutional Neural Network (CNN). The goal is to explore various approaches to improve classification accuracy and evaluate their performance.

## Methods Implemented:

1. **Deep Neural Network (DNN):**
    - A basic feedforward neural network architecture.
   
2. **Data Augmentation with DNN:**
   - Utilizes data augmentation techniques such as rotation, translation, shearing and scaling to artificially increase the size of the training dataset.
   
3. **Data Augmentation with Convolutional Neural Network (CNN):**
   - Incorporates convolutional layers for feature extraction and spatial hierarchies, combined with data augmentation techniques.

## Dataset

The MNIST dataset is a classic benchmark dataset consisting of 28x28 pixel grayscale images of handwritten digits (0-9). Each image is labeled with the corresponding digit it represents.

The dataset images can be downloaded and split into train and test sets simultaneously with the code : 
```python 
(x_train, y_train),(x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
```
## Steps to run the code
- Upload the `MNIST_Classification.ipynb` to Google Colab and run the notebook (As Colab was used originally for training purpose)
- To run the ipynb in local machine, just remove the code: `from google.colab import drive; drive.mount('/content/drive')` and set the paths to save and load the model accordingly.

## Results
The dataset was almost balanced (not biased). 

|Methods used              |Accuracy   |Loss      |
|--------------------------|-----------|----------|
|DNN                       |`97.97%`   |`0.1070`  |
|Data augmentation + DNN   |`98.72%`   |`0.0365`  |
|Data augmentation + CNN   |`99.45%`   |`0.0153`  |
 
    
