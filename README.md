# HackMIT_2022

**ASL**earn Live Project

# About

Our project is a computer vision task to translate ASL gestures into letters. Our goal was to enhance ASL comprehension to bridge the communication gap between individuls with and without hearing and/or speaking disabilities. Future additions to our project would include backtracking to take audio as input and translate to real-time videos stitching together image frames of ASL gestures and letters.

# Input Data

We began training with a dataset of 87,000 images for 29 different classes -- representing each letter of the alphabet along with placeholder symbols such as space, delete, and nothing labels. These images were obtained from the ASL Recognition with CNNs dastaset on Kaggle: https://www.kaggle.com/code/abdul390/asl-recognition-with-convolutional-neural-networks

# How the Model Works

We used a Keras Sequential Model to create our CNN. It is composed of three different layers such as the Convolutional, Max Pooling, and Softmax layers. By adjusting different hyperparameters such as the kernel size, the padding dimensions, the number of layers, and the number of epochs along with adding dropout layers to prevent overfitting, we worked on improving the accuracy of our model. In order for the webcam to read in the ASL letters as accurately as possible, we need to crop the hand so other objects in frame do not confuse the model.

# Requirement
- python 3.8 +
- opencv-python
- tensorflow-cpu

# By:

- Chitra Mukherjee
- Khushi Adukia
- Kevin Ni
- Lucas Chew
