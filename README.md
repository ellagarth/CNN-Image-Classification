# CNN-Image-Classification
Python code for image classification using a convolutional neural network (CNN).

Includes options to easily modify learning rate, epochs, activation functions, etc., and includes numerous additional options including early stopping.

Requires PyTorch, pandas, scikit-learn, and more libraries (see CNN_model.py). 

Running CNN_model.py will print the loss value for every 100 epochs, generate a plot of loss over time/epoch for the training model, and a confusion matrix for the test images.

ALL IMAGES FOLDER CONTAINS: 22 x circles, 22 x squares, 22 x triangles, 22 x crosses.
TRAINING IMAGE FOLDER CONTAINS: 18 x circles, 18 x squares, 18 x triangles, 18 x crosses, 1 x training.csv file for supervised learning
TEST IMAGE FOLDER CONTAINS: 4 x circles, 4 x squares, 4 x triangles, 4 x crosses, 1 x test.csv file
