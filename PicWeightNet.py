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


# 初始化参数
def initialize_parameters(randomSeed=1):

    print("初始化参数的seed为"+str(randomSeed))
    W1 = tf.get_variable("W1", [5, 5, 8, 6], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    W2 = tf.get_variable("W2", [5, 5, 6, 8], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
    W3 = tf.get_variable("W3", [5, 5, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  # "W4": W4
                  }

    return parameters


# 前向传播
def forward_propagation(X, parameters, num, isTrain=True, flag=0, randomSeed=1):
    print("全连接层的seed为" + str(randomSeed))
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    # W4 = parameters['W4']

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='VALID')
    # print(Z1)
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # print(P1)

    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='VALID')
    # print(Z2)
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # print(P2)

    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='VALID')
    # print(Z3)
    # RELU
    A3 = tf.nn.relu(Z3)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P3 = tf.nn.max_pool(A3, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')
    # print(P3)
    # print(P3.get_shape().as_list())

    pool_shape = P3.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(P3, [num, nodes])
    # if isTrain:        # 防止过拟合
    #     reshaped = tf.nn.dropout(reshaped, 0.80)
    # initializer=tf.truncated_normal_initializer(stddev=0.1)
    # initializer=tf.contrib.layers.xavier_initializer(seed=2)
    w1 = "weight1_" + str(flag)
    b1 = "bias1_" + str(flag)
    w2 = "weight2_" + str(flag)
    b2 = "bias2_" + str(flag)
    fc1_weights = tf.get_variable(w1, [nodes, 64],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, seed=randomSeed))
    fc1_biases = tf.get_variable(b1, [64], initializer=tf.constant_initializer(0.1))
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
    # if isTrain:        # 防止过拟合
    #     fc1 = tf.nn.dropout(fc1, 0.66)

    fc2_weights = tf.get_variable(w2, [64, 10],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, seed=randomSeed))
    fc2_biases = tf.get_variable(b2, [10], initializer=tf.constant_initializer(0.1))
    logit = (tf.matmul(fc1, fc2_weights) + fc2_biases)
    return logit, fc1_weights, fc2_weights


# 计算损失函数
def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.002, l2_rate=0.010,
          num_epochs=500, minibatch_size=64, print_cost=True, save_session=False):
    (m, n_H0, n_W0, n_C0) = X_train.shape
    a, b, c, d = X_test.shape
    weight1, weight2, weight3 = 0.6, 0.5, 0.2
    print("采用权值为", str(weight1), str(weight2), str(weight3))
    m_test = X_test.shape[0]
    n_y = Y_train.shape[1]
    costs = []  # To keep track of the cost
    isTrain = tf.placeholder(tf.bool)
    num = tf.placeholder(tf.int32)
    flag = tf.Variable(0, trainable=False)
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    seed = 5
    parameters = initialize_parameters(randomSeed=seed)
    Z0, fc1w0, fc2w0 = forward_propagation(X, parameters, num, flag=0, randomSeed=seed)

    cost0 = compute_cost(Z0, Y)

    # 采用L2正则化，避免过拟合
    regularizer = tf.contrib.layers.l2_regularizer(l2_rate)
    regularization0 = regularizer(fc1w0) + regularizer(fc2w0)
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
            _, minibatch_cost0 = sess.run([optimizer0, cost0], feed_dict={X: X_train, Y: Y_train, num: m})

            if print_cost is True and epoch % 5 == 0:
                print("损失函数经过%i次遍历后: %f" % (epoch, minibatch_cost0))
                # print("Cost after epoch %i: %f" % (epoch, minibatch_cost0+minibatch_cost1+minibatch_cost2))
                Z = Z0 * 1
                predict_op = tf.argmax(Z, 1)  # 返回每行最大值的索引
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train, num: m})
                test_accuracy = accuracy.eval({X: X_test, Y: Y_test, num: m_test, isTrain: False})
                print("训练集识别率:", train_accuracy)
                print("测试集识别率:", test_accuracy)
                if save_session is True and train_accuracy > 0.985:
                    save_files = './another/v_82/model_forloop' + str(epoch) + '.ckpt'
                    saver.save(sess, save_files)
                    print("模型" + save_files + "保存成功.")
            if print_cost is True and epoch % 1 == 0:
                costs.append(minibatch_cost0)
        return parameters

def cnnTrain():
    trainpicsfile = './logs/3dPicTrain82_1.h5'
    testpicsfile = './logs/3dPicTest82_1.h5'
    trainFile = './logs/3dModelTrainDBeta_8_2.h5'
    testFile = './logs/3dModelTestDBeta_8_2.h5'
    _, YTrain, _, YTest = CU.loadDataSets(trainFile, testFile)
    XTrain = h5utils.readData(trainpicsfile)
    XTest = h5utils.readData(testpicsfile)
    print(XTrain.shape[0], XTest.shape[0])
    XTrain = XTrain/255
    XTest = XTest/255
    parameters = model(XTrain, YTrain, XTest, YTest, num_epochs=10000, save_session=True)
    return XTrain, YTrain, XTest, YTest



if __name__ == '__main__':
    # 三维模型测试
    XTrain, YTrain, XTest, YTest = cnnTrain()
