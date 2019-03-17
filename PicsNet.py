# import math
# import numpy as np
# import h5py
import matplotlib.pyplot as plt
# import scipy
# # from PIL import Image
# from scipy import ndimage
# from PIL.Image import core as _imaging
import tensorflow as tf
from tensorflow.python.framework import ops
# from cnn_utils import *
import CNNUtils
import GetFeature as GF
import CNNUtils as CU
import CNNTrain as CT
import H5FileUtils as h5utils


# 创建占位符
def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder('float', shape=[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder('float', shape=[None, n_y])

    return X, Y


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder('float', shape=[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder('float', shape=[None, n_y])

    return X, Y


# 初始化参数
def initialize_parameters(randomSeed=1):
    print("初始化参数的seed为" + str(randomSeed))
    W1 = tf.get_variable("W1", [3, 3, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    W11 = tf.get_variable("W11", [3, 3, 8, 8], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    W2 = tf.get_variable("W2", [3, 3, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    W22 = tf.get_variable("W22", [3, 3, 16, 16], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    W3 = tf.get_variable("W3", [3, 3, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    W33 = tf.get_variable("W33", [3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    W4 = tf.get_variable("W4", [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    W44 = tf.get_variable("W44", [3, 3, 64, 64], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    # W5 = tf.get_variable("W5", [3, 3, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    parameters = {"W1": W1,
                  "W11":W11,
                  "W2": W2,
                  "W22":W22,
                  "W3": W3,
                  "W33":W33,
                  "W4": W4,
                  "W44":W44
                  }

    return parameters


# 前向传播
def forward_propagation(X, parameters, num, isTrain=True, flag=0, randomSeed=1):
    print("全连接层的seed为" + str(randomSeed))
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W11 = parameters['W11']
    W2 = parameters['W2']
    W22 = parameters['W22']
    W3 = parameters['W3']
    W33 = parameters['W33']
    W4 = parameters['W4']
    W44 = parameters['W44']

    # 第一层
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    print(Z1)
    # RELU
    A1 = tf.nn.relu(Z1)

    # 第一一层
    Z11 = tf.nn.conv2d(A1, W11, strides=[1, 1, 1, 1], padding='SAME')
    print(Z11)
    # RELU
    A11 = tf.nn.relu(Z11)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(P1)

    # 第二层
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    print(Z2)
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'

    # 第二二层
    Z22 = tf.nn.conv2d(A2, W22, strides=[1, 1, 1, 1], padding='SAME')
    print(Z22)
    # RELU
    A22 = tf.nn.relu(Z22)
    P2 = tf.nn.max_pool(A22, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(P2)

    # 第三层
    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
    print(Z3)
    # RELU
    A3 = tf.nn.relu(Z3)

    # 第三三层
    Z33 = tf.nn.conv2d(A3, W33, strides=[1, 1, 1, 1], padding='SAME')
    print(Z33)
    # RELU
    A33 = tf.nn.relu(Z33)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P3 = tf.nn.max_pool(A33, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(P3)

    # 第s四层
    Z4 = tf.nn.conv2d(P3, W4, strides=[1, 1, 1, 1], padding='SAME')
    print(Z4)
    # RELU
    A4 = tf.nn.relu(Z4)

    # 第s四四层
    Z44 = tf.nn.conv2d(A4, W44, strides=[1, 1, 1, 1], padding='SAME')
    print(Z44)
    # RELU
    A44 = tf.nn.relu(Z44)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P4 = tf.nn.max_pool(A44, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print(P4)

    p4_shape = P4.get_shape().as_list()         # 4,4,32
    resh = tf.reshape(P4, [num/8, 8, p4_shape[1],p4_shape[2], p4_shape[3]])
    P5 = tf.nn.max_pool3d(resh, ksize=[1, 8, 1, 1, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
    print(P5)
    pool_shape = P5.get_shape().as_list()
    print(pool_shape)

    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]*pool_shape[4]
    reshaped = tf.reshape(P5, [num/8, nodes])     # 展开
    w1 = "weight1_" + str(flag)
    b1 = "bias1_" + str(flag)
    w2 = "weight2_" + str(flag)
    b2 = "bias2_" + str(flag)
    w3 = "weight3_" + str(flag)
    b3 = "bias3_" + str(flag)
    # if isTrain:        # 防止过拟合
    #     reshaped = tf.nn.dropout(reshaped, 0.56)
    fc1_weights = tf.get_variable(w1, [nodes, 512],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    fc1_biases = tf.get_variable(b1, [512], initializer=tf.constant_initializer(0.01))
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
    # if isTrain:        # 防止过拟合
    #     fc1 = tf.nn.dropout(fc1, 0.56)

    fc2_weights = tf.get_variable(w2, [512, 128],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    fc2_biases = tf.get_variable(b2, [128], initializer=tf.constant_initializer(0.01))
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
    # if isTrain:        # 防止过拟合
    #     fc2 = tf.nn.dropout(fc2, 0.76)

    fc3_weights = tf.get_variable(w3, [128, 10],
                                  initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    fc3_biases = tf.get_variable(b3, [10], initializer=tf.constant_initializer(0.01))
    logit = (tf.matmul(fc2, fc3_weights) + fc3_biases)
    return logit, fc1_weights, fc2_weights, fc1_weights


# 计算损失函数
def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0002, l2_rate=0.022,
          num_epochs=500, minibatch_size=64, print_cost=True, save_session=False):
    print("learning_rate=" + str(learning_rate), " and l2_rate=" + str(l2_rate))
    print(X_train.shape)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    a, b, c, d = X_test.shape
    m_test = X_test.shape[0]
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost
    isTrain = tf.placeholder(tf.bool)
    num = tf.placeholder(tf.int32)
    flag = tf.Variable(0, trainable=False)
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    seed = 2
    parameters = initialize_parameters(randomSeed=seed)
    Z0, fc1w0, fc2w0, fc3w0= forward_propagation(X, parameters, num, flag=0, randomSeed=seed)

    cost0 = compute_cost(Z0, Y)

    # 采用L2正则化，避免过拟合
    regularizer = tf.contrib.layers.l2_regularizer(l2_rate)
    regularization0 = regularizer(fc1w0) + regularizer(fc2w0)+regularizer(fc3w0)
    cost0 = cost0 + regularization0

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100, 0.96, staircase=True)
    optimizer0 = tf.train.AdamOptimizer(learning_rate).minimize(cost0, global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        # save_path = saver.restore(sess, 'another/t_82/model_forloop170.ckpt')
        for epoch in range(num_epochs):
            print("第"+str(epoch)+"开始")
            _, minibatch_cost = sess.run([optimizer0, cost0], feed_dict={X: X_train, Y: Y_train, num: m})
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
            #     _, temp_cost = sess.run([optimizer0, cost0], feed_dict={X: minibatch_X, Y: minibatch_Y, num: minibatch_X.shape[0]})
            #     minibatch_cost += temp_cost / num_minibatches

            if print_cost is True and epoch % 5 == 0:
                print("损失函数经过%i次遍历后: %f" % (epoch, minibatch_cost))
                # print("Cost after epoch %i: %f" % (epoch, minibatch_cost0+minibatch_cost1+minibatch_cost2))
                Z = Z0 * 1
                predict_op = tf.argmax(Z, 1)  # 返回每行最大值的索引
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train, num: m})
                test_accuracy = accuracy.eval({X: X_test, Y: Y_test, num: m_test, isTrain: False})
                print("训练集识别率:", train_accuracy)
                print("测试集识别率:", test_accuracy)
                if save_session is True and train_accuracy > 0.985 and test_accuracy > 0.89:
                    save_files = './another/x_82/model_forloop' + str(epoch) + '.ckpt'
                    saver.save(sess, save_files)
                    print("模型" + save_files + "保存成功.")
            if print_cost is True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        return parameters

def cnnTrain():
    trainpicsfile = './logs/3dColorPic64Train82_5.h5'
    testpicsfile = './logs/3dColorPic64Test82_5.h5'
    trainFile = './logs/3dModelTrainSBeta_8_2.h5'
    testFile = './logs/3dModelTestSBeta_8_2.h5'
    _, YTrain, _, YTest = CU.loadDataSets(trainFile, testFile)
    XTrain = h5utils.readData(trainpicsfile)
    XTest = h5utils.readData(testpicsfile)
    print(XTrain.shape[0], XTest.shape[0])
    XTrain = XTrain / 255
    XTest = XTest / 255
    # parameters = model(XTrain, YTrain, XTest, YTest, num_epochs=10000, save_session=True)
    return XTrain, YTrain, XTest, YTest


if __name__ == '__main__':
    # 三维模型测试
    XTrain, YTrain, XTest, YTest = cnnTrain()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(XTest[i,:,:,:])
