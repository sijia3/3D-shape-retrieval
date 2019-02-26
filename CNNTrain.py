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
    X = tf.placeholder('float', shape=[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder('float', shape=[None, n_y])

    return X, Y


# 初始化参数
def initialize_parameters():
    # tf.set_random_seed(1)  # so that your "random" numbers match ours
    W1 = tf.get_variable("W1", [4, 4, 3, 4], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [4, 4, 4, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W3 = tf.get_variable("W3", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W4 = tf.get_variable("W4", [2, 2, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4
                  }

    return parameters


def forward_propagation(X, parameters, num):
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
    W3 = parameters['W3']
    W4 = parameters['W4']

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    print(Z1)
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(P1)


    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    print(Z2)
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(P2)

    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
    print(Z3)
    # RELU
    A3 = tf.nn.relu(Z3)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P3 = tf.nn.max_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(P3)

    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z4 = tf.nn.conv2d(P3, W4, strides=[1, 1, 1, 1], padding='SAME')
    print(Z4)
    # RELU
    A4 = tf.nn.relu(Z4)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P4 = tf.nn.max_pool(A4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(P4.get_shape().as_list(), num)

    pool_shape = P4.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2]* pool_shape[3]
    reshaped = tf.reshape(P4, [num, nodes])

    fc1_weights = tf.get_variable("weight1", [nodes, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
    fc1_biases = tf.get_variable("bias1", [128], initializer=tf.constant_initializer(0.1))
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights)+fc1_biases)
    # fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_weights = tf.get_variable("weight2", [128, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
    fc2_biases = tf.get_variable("bias2", [10], initializer=tf.constant_initializer(0.1))
    logit = (tf.matmul(fc1, fc2_weights)+fc2_biases)

    # # FLATTEN
    # P5 = tf.contrib.layers.flatten(P4)
    # # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
    # print(P5)
    # # Z3 = tf.contrib.layers.fully_connected(P2, 10, activation_fn=None)
    # # L4_drop = tf.nn.dropout(P5, keep_prob=0.5)           # drop 减少过拟合
    # # print(L4_drop)
    #
    # Z4 = tf.contrib.layers.fully_connected(P5, 128, activation_fn=tf.nn.relu)
    # # L5_drop = tf.nn.dropout(Z4, keep_prob=0.5)
    # # print(L5_drop)
    # Z5 = tf.contrib.layers.fully_connected(Z4, 10, activation_fn=None)
    return logit, fc1_weights, fc2_weights

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

    num = tf.placeholder(tf.int32)
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3, fc1w, fc2w = forward_propagation(X, parameters, num)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    regularizer = tf.contrib.layers.l2_regularizer(0.1)
    regularization = regularizer(fc1w)+regularizer(fc2w)
    loss = cost + regularization
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: X_train, Y: Y_train, num: m})
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
            if print_cost is True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
                predict_op = tf.argmax(Z3, 1)  # 返回每行最大值的索引值
                # print(predict_op.eval({X: X_train}))
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                # Calculate accuracy on the test set
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                # print(accuracy.eval())
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train, num: m})
                test_accuracy = accuracy.eval({X: X_test, Y: Y_test, num: 100})
                print("Train Accuracy:", train_accuracy)
                print("Test Accuracy:", test_accuracy)
                if save_session is True:
                    save_files = './session/model_forloop'+str(epoch)+'.ckpt'
                    saver.save(sess, save_files)
                    print("模型"+save_files+"保存成功.")
            if print_cost is True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        # plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        # if save_session == True:
        #     saver.save(sess, save_file)
        #     print("模型保存成功.")
        return parameters

# def loadDataSets():
#     XTrain = GF.readH5File('./datasets/train_model.h5', 'data')
#     YLabels = GF.readH5File('./datasets/train_labels.h5', 'labels')
#     YLabels = YLabels.reshape(1, len(YLabels)).astype('int64')
#     YLabels = GF.convert_to_one_hot(YLabels, 10).T
#     XTest = GF.readH5File('./datasets/test_model.h5', 'data')
#     YTestLabels = GF.readH5File('./datasets/test_labels.h5', 'labels')
#     YTestLabels = YTestLabels.reshape(1, len(YTestLabels)).astype('int64')
#     YTestLabels = GF.convert_to_one_hot(YTestLabels, 10).T
#     return XTrain, YLabels, XTest, YTestLabels


# if __name__ == '__main__':
    # 三维模型测试
    # X_train, Y_train, X_test, Y_test = loadDataSets()
    # parameters = model(X_train, Y_train, X_test, Y_test, num_epochs=60,save_session=True)
    # learning_rate = 0.010
    # num_epochs = 5
    # minibatch_size = 64
    # print_cost = True
    # ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    # tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    # seed = 3  # to keep results consistent (numpy seed)
    # (m, n_H0, n_W0, n_C0) = X_train.shape
    # n_y = Y_train.shape[1]
    # costs = []  # To keep track of the cost
    #
    # # Create Placeholders of the correct shape
    # X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    #
    # # Initialize parameters
    # parameters = initialize_parameters()
    #
    # # Forward propagation: Build the forward propagation in the tensorflow graph
    # Z3 = forward_propagation(X, parameters)
    #
    # # Cost function: Add cost function to tensorflow graph
    # cost = compute_cost(Z3, Y)
    #
    # # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #
    # # Initialize all the variables globally
    # init = tf.global_variables_initializer()
    #
    # # Start the session to compute the tensorflow graph
    # with tf.Session() as sess:
    #
    #     # Run the initialization
    #     sess.run(init)
    #
    #     # Do the training loop
    #     for epoch in range(num_epochs):
    #
    #         minibatch_cost = 0.
    #         num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
    #         seed = seed + 1
    #         minibatches = CNNUtils.random_mini_batches(X_train, Y_train, minibatch_size, seed)
    #
    #         for minibatch in minibatches:
    #             # Select a minibatch
    #             (minibatch_X, minibatch_Y) = minibatch
    #             # IMPORTANT: The line that runs the graph on a minibatch.
    #             # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
    #             _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
    #             minibatch_cost += temp_cost / num_minibatches
    #
    #         # Print the cost every epoch
    #         if print_cost == True and epoch % 5 == 0:
    #             print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
    #         if print_cost == True and epoch % 1 == 0:
    #             costs.append(minibatch_cost)
    #
    #     # plot the cost
    #     # plt.plot(np.squeeze(costs))
    #     # plt.ylabel('cost')
    #     # plt.xlabel('iterations (per tens)')
    #     # plt.title("Learning rate =" + str(learning_rate))
    #     # plt.show()
    # # init = tf.global_variables_initializer()
    # # with tf.Session() as sesss:
    #     # sesss.run(init)
    #     # Calculate the correct predictions
    #     # print(np.squeeze(costs))
    #     predict_op = tf.argmax(Z3, 1)   # 返回每行最大值的索引值
    #     # print(Z3.eval({X: X_train}))
    #     tt = predict_op.eval({X: X_test})
    #
    #     print(predict_op.eval({X: X_test[1]}))
    #     correct_prediction = tf.equal(predict_op, tf.argmax(Y[1], 1))
    #
    #     # Calculate accuracy on the test set
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #     aa = np.squeeze(accuracy)
    #     print(aa)
    #     train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
    #     test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
    #     print("Train Accuracy:", train_accuracy)
    #     print("Test Accuracy:", test_accuracy)


    # 学习案例测试
    # X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = CNNUtils.load_dataset()
    # X_train = X_train_orig / 255.
    # X_test = X_test_orig / 255.
    # Y_train = CNNUtils.convert_to_one_hot(Y_train_orig, 6).T
    # Y_test = CNNUtils.convert_to_one_hot(Y_test_orig, 6).T
    # _, _, parameters = model(X_train, Y_train, X_test, Y_test)

    # with tf.Session() as sess_test:
    #     init = tf.global_variables_initializer()
    #     sess_test.run(init)
    #     print("W1 = " + str(parameters["W1"].eval()))
    #     print("W2 = " + str(parameters["W2"].eval()))
    #     Z3 = forward_propagation(XTest, parameters)
    #     predict = tf.arg_max(Z3, 1)
    #     correct_prediction = tf.equal(predict_op, tf.argmax(YTestLabels, 1))


