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
import CNNUtils
import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np
import CNNUtils as CU
import CNNTrain as CT
import GetFeature as GF
import GetLabels as GL
import H5FileUtils as h5utils
import Predict as PD





# 创建占位符
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    ### START CODE HERE ### (≈2 lines)
    X = tf.placeholder('float', shape=[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder('float', shape=[None, n_y])
    ### END CODE HERE ###

    return X, Y

# X, Y = create_placeholders(64, 64, 3, 10)
# print ("X = " + str(X))
# print ("Y = " + str(Y))


# GRADED FUNCTION: initialize_parameters

# 初始化参数
def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    # tf.set_random_seed(1)  # so that your "random" numbers match ours

    ### START CODE HERE ### (approx. 2 lines of code)
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters

# tf.reset_default_graph()
# with tf.Session() as sess_test:
#     parameters = initialize_parameters()
#     init = tf.global_variables_initializer()
#     sess_test.run(init)
#     print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
#     print("W2 = " + str(parameters["W2"].eval()[1,1,1]))


# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    print(Z1)
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')
    print(P1)
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    print(Z2)
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')
    print(P2)
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    print(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 10, activation_fn=None)
    # Z3 = tf.contrib.layers.fully_connected(P2, 50)
    #
    # Z4 = tf.contrib.layers.fully_connected(Z3, 10, activation_fn=None)

    return Z3
# tf.reset_default_graph()

# with tf.Session() as sess:
#     np.random.seed(1)
#     X, Y = create_placeholders(64, 64, 3, 10)
# #     print(X.shape)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     a = sess.run(Z3, {X: np.random.randn(2,64,64,3), Y: np.random.randn(2,6)})
#     print("Z3 = " + str(a))


# GRADED FUNCTION: compute_cost

def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost
#
# tf.reset_default_graph()
#
# with tf.Session() as sess:
#     np.random.seed(1)
#     X, Y = create_placeholders(64, 64, 3, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
#     print("cost = " + str(a))
#
#
# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.010,
          num_epochs=500, minibatch_size=64, print_cost=True, save_session= False, save_file='./logs/modelBeta1.ckpt'):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    # tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost

    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            _, temp_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train})
            # minibatch_cost = 0.
            # num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            # seed = seed + 1
            # minibatches = CNNUtils.random_mini_batches(X_train, Y_train, minibatch_size, seed)
            #
            # for minibatch in minibatches:
            #     # Select a minibatch
            #     (minibatch_X, minibatch_Y) = minibatch
            #     # IMPORTANT: The line that runs the graph on a minibatch.
            #     # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
            #     _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
            #     minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, temp_cost))
                predict_op = tf.argmax(Z3, 1)  # 返回每行最大值的索引值
                # print(predict_op.eval({X: X_train}))
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                # Calculate accuracy on the test set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                # print(accuracy.eval())
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
                test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
                print("Train Accuracy:", train_accuracy)
                print("Test Accuracy:", test_accuracy)
                if save_session == True:
                    save_file = './another/model_forloop'+str(epoch)+'.ckpt'
                    saver.save(sess, save_file)
                    print(save_file+"模型保存成功.")
            if print_cost is True and epoch % 1 == 0:
                costs.append(temp_cost)
        return parameters



if __name__ == '__main__':
    trainFile = './datasets/3dModelTrainBeta3.h5'
    testFile = './datasets/3dModelTestBeta3.h5'
    XTrain, YTrain, XTest, YTest = CU.loadDataSets(trainFile, testFile)
    parameters = model(XTrain, YTrain, XTest, YTest, num_epochs=7000, save_session=True,
                          save_file='./logs/modelBeta33.ckpt')