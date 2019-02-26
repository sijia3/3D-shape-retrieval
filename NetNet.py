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
import CNNUtils as CU
import CNNTrain as CT





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
    W1 = tf.get_variable("W1", [5, 5, 3, 6], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2", [5, 5, 6, 8], initializer=tf.contrib.layers.xavier_initializer())
    W3 = tf.get_variable("W3", [5, 5, 8, 16], initializer=tf.contrib.layers.xavier_initializer())
    # W4 = tf.get_variable("W4", [2, 2, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  # "W4": W4
                  }

    return parameters


def forward_propagation(X, parameters, num):
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    # W4 = parameters['W4']

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='VALID')
    print(Z1)
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print(P1)

    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='VALID')
    print(Z2)
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print(P2)

    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='VALID')
    print(Z3)
    # RELU
    A3 = tf.nn.relu(Z3)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P3 = tf.nn.max_pool(A3, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')
    # print(P3)
    print(P3.get_shape().as_list(), num)

    pool_shape = P3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2]* pool_shape[3]
    reshaped = tf.reshape(P3, [num, nodes])

    fc1_weights = tf.get_variable("weight1", [nodes, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
    fc1_biases = tf.get_variable("bias1", [64], initializer=tf.constant_initializer(0.1))
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights)+fc1_biases)
    if num != 100:        # 防止过拟合
        fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_weights = tf.get_variable("weight2", [64, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
    fc2_biases = tf.get_variable("bias2", [10], initializer=tf.constant_initializer(0.1))
    logit = (tf.matmul(fc1, fc2_weights)+fc2_biases)
    return logit, fc1_weights, fc2_weights


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.001,
          num_epochs=500, minibatch_size=64, print_cost=True, save_session= False, save_file='./logs/modelBeta1.ckpt'):
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    # tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost
    num = tf.placeholder(tf.int32)
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters()
    Z3, fc1w, fc2w = forward_propagation(X, parameters, num)
    cost = compute_cost(Z3, Y)
    # regularizer = tf.contrib.layers.l2_regularizer(0.05)
    # regularization = regularizer(fc1w)+regularizer(fc2w)
    # loss = cost + regularization
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
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
                predict_op = tf.argmax(Z3, 1)  # 返回每行最大值的索引
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
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
        return parameters

def cnnTrain():
    trainFile = './datasets/3dModelTrainBeta3.h5'
    testFile = './datasets/3dModelTestBeta3Plus.h5'
    XTrain, YTrain, XTest, YTest = CU.loadDataSets(trainFile, testFile)
    parameters = CT.model(XTrain, YTrain, XTest, YTest, num_epochs=10000, save_session=False,
                          minibatch_size=64)
    return parameters
if __name__ == '__main__':
    # 三维模型测试
    cnnTrain()
