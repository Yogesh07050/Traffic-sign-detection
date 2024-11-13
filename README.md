

# Traffic Sign Detection using Machine Learning
This repository contains a traffic sign detection project implemented using Convolutional Neural Networks (CNN) in Python with the Keras and TensorFlow libraries. The project involves the classification of traffic signs into their respective categories.

## Table of Contents
### Introduction
### Dataset
### Installation
### Usage
### Model Architecture
### Results
### License

### Introduction
The goal of this project is to build a machine learning model capable of detecting and classifying traffic signs. This can be useful for autonomous vehicles and driver assistance systems. The model is trained on a dataset of traffic sign images and achieves a good accuracy on the test set.

### Dataset
The dataset used for this project is stored locally in the Dataset folder. Each subfolder within Dataset corresponds to a different traffic sign class, and the images are labeled according to these classes. The labels are also provided in a CSV file labels.csv.

### Installation
To run this project, you'll need to have Python installed along with the following libraries:

##### numpy
##### matplotlib
##### keras
##### tensorflow
##### scikit-learn
##### pandas
##### opencv-python**

You can install these dependencies using pip:
#### pip install numpy matplotlib keras tensorflow scikit-learn pandas opencv-python



Navigate to the project directory:
#### cd traffic-sign-detection

Place your dataset in the Dataset folder and the labels CSV file in the root directory.

Run the main script to train the model:
#### python traffic_sign_detection.py

#### Model Architecture
The CNN model used in this project has the following architecture:

Convolutional layers with ReLU activation
MaxPooling layers
Dropout layers for regularization
Fully connected (Dense) layers with ReLU and softmax activation
The model is compiled with the Adam optimizer and categorical crossentropy loss function. Data augmentation is applied to increase the robustness of the model.

#### Results
After training, the model's performance is evaluated on the test set. The training history is plotted to visualize the loss and accuracy over epochs.

The final model is saved as model.h5.
