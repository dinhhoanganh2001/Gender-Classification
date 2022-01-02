import time

import matplotlib
import scipy
from scipy import ndimage
import pathlib
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
from support import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_images = list()
test_images = list()

image_path = 'C:/Users/Admin/PycharmProjects/ML_asm/newMale'
for path in os.listdir(image_path):
    full_path = os.path.join(image_path, path)
    img = Image.open(full_path)
    train_images.append((img, 1))

image_path = 'C:/Users/Admin/PycharmProjects/ML_asm/newFemale'
for path in os.listdir(image_path):
    full_path = os.path.join(image_path, path)
    img = Image.open(full_path)
    train_images.append((img, 0))

image_path = 'C:/Users/Admin/PycharmProjects/ML_asm/newMale_validation'
for path in os.listdir(image_path):
    full_path = os.path.join(image_path, path)
    img = Image.open(full_path)
    test_images.append((img, 1))

image_path = 'C:/Users/Admin/PycharmProjects/ML_asm/newFemale_validation'
for path in os.listdir(image_path):
    full_path = os.path.join(image_path, path)
    img = Image.open(full_path)
    test_images.append((img, 0))

print(len(train_images))
random.shuffle(train_images)
random.shuffle(test_images)


def load_data():
    train_data = [asarray(pair[0]) for pair in train_images]
    test_data = [asarray(pair[0]) for pair in test_images]
    train_label = [pair[1] for pair in train_images]
    test_label = [pair[1] for pair in test_images]

    print(np.shape(train_data))

    train_set_x_orig = np.array(train_data)  # your train set features
    train_set_y_orig = np.array(train_label)  # your train set labels

    print(np.shape(train_set_x_orig))

    test_set_x_orig = np.array(test_data)  # your test set features
    test_set_y_orig = np.array(test_label)  # your test set labels

    classes = np.array((b'Female', b'Male'))  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

index = random.randint(0,238)
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


### CONSTANTS ###
layers_dims = [12288, 20, 7, 5, 1] #  4-layer model


# GRADED FUNCTION: L_layer_model

def L_layer_model(X, Y, layers_dims, learning_rate=0.01, num_iterations=3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    # np.random.seed(1)
    costs = []  # keep track of cost
    parameters = initialize_parameters_deep(layers_dims)


    # Loop (gradient descent)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)

    return parameters, costs

parameters, costs = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)