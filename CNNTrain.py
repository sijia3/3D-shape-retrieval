import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
# from PIL import Image
from scipy import ndimage
# from PIL.Image import core as _imaging
import tensorflow as tf
from tensorflow.python.framework import ops
# from cnn_utils import *
import GetFeature as GF

XTrain, YLabels = GF.readH5File('./datasets/train.h5')
# labels = np.array([[i for i in range(0, 10)]])
YLabels = GF.convert_to_one_hot(YLabels, 10)

# def create_placeholders(n_H0, n_W0, n_C0, n_y):
#     """
#     Creates the placeholders for the tensorflow session.
#
#     Arguments:
#     n_H0 -- scalar, height of an input image
#     n_W0 -- scalar, width of an input image
#     n_C0 -- scalar, number of channels of the input
#     n_y -- scalar, number of classes
#
#     Returns:
#     X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
#     Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
#     """
#
#     ### START CODE HERE ### (≈2 lines)
#     X = tf.placeholder('float', shape=[None, n_H0, n_W0, n_C0])
#     Y = tf.placeholder('float', shape=[None, n_y])
#     ### END CODE HERE ###
#
#     return X, Y
#
# X, Y = create_placeholders(64, 64, 3, 6)
# print ("X = " + str(X))
# print ("Y = " + str(Y))
#
#
# # GRADED FUNCTION: initialize_parameters
#
# def initialize_parameters():
#     """
#     Initializes weight parameters to build a neural network with tensorflow. The shapes are:
#                         W1 : [4, 4, 3, 8]
#                         W2 : [2, 2, 8, 16]
#     Returns:
#     parameters -- a dictionary of tensors containing W1, W2
#     """
#
#     tf.set_random_seed(1)  # so that your "random" numbers match ours
#
#     ### START CODE HERE ### (approx. 2 lines of code)
#     W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#     W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
#     ### END CODE HERE ###
#
#     parameters = {"W1": W1,
#                   "W2": W2}
#
#     return parameters
#
# tf.reset_default_graph()
# with tf.Session() as sess_test:
#     parameters = initialize_parameters()
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#     print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
#     print("W2 = " + str(parameters["W2"].eval()[1,1,1]))
#
#
# # GRADED FUNCTION: forward_propagation
#
# def forward_propagation(X, parameters):
#     """
#     Implements the forward propagation for the model:
#     CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
#
#     Arguments:
#     X -- input dataset placeholder, of shape (input size, number of examples)
#     parameters -- python dictionary containing your parameters "W1", "W2"
#                   the shapes are given in initialize_parameters
#
#     Returns:
#     Z3 -- the output of the last LINEAR unit
#     """
#
#     # Retrieve the parameters from the dictionary "parameters"
#     W1 = parameters['W1']
#     W2 = parameters['W2']
#
#     ### START CODE HERE ###
#     # CONV2D: stride of 1, padding 'SAME'
#     Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
#     # RELU
#     A1 = tf.nn.relu(Z1)
#     # MAXPOOL: window 8x8, sride 8, padding 'SAME'
#     P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
#     # CONV2D: filters W2, stride 1, padding 'SAME'
#     Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
#     # RELU
#     A2 = tf.nn.relu(Z2)
#     # MAXPOOL: window 4x4, stride 4, padding 'SAME'
#     P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
#     # FLATTEN
#     P2 = tf.contrib.layers.flatten(P2)
#     # FULLY-CONNECTED without non-linear activation function (not not call softmax).
#     # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
#     print(P2)
#     Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)
#     ### END CODE HERE ###
#
#     return Z3
# tf.reset_default_graph()
#
# with tf.Session() as sess:
#     np.random.seed(1)
#     X, Y = create_placeholders(64, 64, 3, 6)
# #     print(X.shape)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
#     print("Z3 = " + str(a))