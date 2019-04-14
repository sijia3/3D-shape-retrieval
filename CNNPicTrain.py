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



def initialize_parameters(randomSeed=1):
    """
    初始化参数
    :param randomSeed: 随机seed种子
    :return:
    """
    print("初始化参数的seed为" + str(randomSeed))
    with tf.variable_scope('', reuse=tf.AUTO_REUSE) as tttt:
        W1 = tf.get_variable("W1", [3, 3, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
        W2 = tf.get_variable("W2", [3, 3, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
        W3 = tf.get_variable("W3", [3, 3, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
        W4 = tf.get_variable("W4", [3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
        parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4
                  }
    return parameters


def forward_propagation(X, parameters, num, isTrain=True, flag=0, randomSeed=1):
    """
    定义模型前向传播
    :param X: 模型特征集
    :param parameters: 学习参数
    :param num: 数据集的模型总数量
    :param isTrain: 是否为训练
    :param flag: 全连接层参数的标志
    :param randomSeed: 随机seed种子
    :return:
    """
    print("全连接层的seed为" + str(randomSeed))
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']

    # 第一层（Conv+Relu+Pool）
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第二层（Conv+Relu+Pool）
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第三层（Conv+Relu+Pool）
    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='SAME')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第四层（Conv+Relu+Pool）
    Z4 = tf.nn.conv2d(P3, W4, strides=[1, 1, 1, 1], padding='SAME')
    A4 = tf.nn.relu(Z4)
    P4 = tf.nn.max_pool(A4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 多视图ViewPooling聚合
    p4_shape = P4.get_shape().as_list()         # 4,4,32
    resh = tf.reshape(P4, [num/8, 8, p4_shape[1],p4_shape[2], p4_shape[3]])
    P5 = tf.nn.max_pool3d(resh, ksize=[1, 8, 1, 1, 1], strides=[1, 1, 1, 1, 1], padding='SAME')
    pool_shape = P5.get_shape().as_list()
    print(pool_shape)

    # 卷积向量展开
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
    with tf.variable_scope('', reuse=tf.AUTO_REUSE) as te:
        fc1_weights = tf.get_variable(w1, [nodes, 512],
                                      initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
        fc1_biases = tf.get_variable(b1, [512], initializer=tf.constant_initializer(0.001))
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
    # if isTrain:        # 防止过拟合
    #     fc1 = tf.nn.dropout(fc1, 0.56)
    with tf.variable_scope('', reuse=tf.AUTO_REUSE) as te:
        fc2_weights = tf.get_variable(w2, [512, 128],
                                    initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
        fc2_biases = tf.get_variable(b2, [128], initializer=tf.constant_initializer(0.001))
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
    # if isTrain:        # 防止过拟合
    #     fc2 = tf.nn.dropout(fc2, 0.76)
    with tf.variable_scope('', reuse=tf.AUTO_REUSE) as te:
        fc3_weights = tf.get_variable(w3, [128, 10],
                                    initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
        fc3_biases = tf.get_variable(b3, [10], initializer=tf.constant_initializer(0.001))
    logit = (tf.matmul(fc2, fc3_weights) + fc3_biases)
    return logit, fc1_weights, fc2_weights, fc3_weights


# 计算损失函数
def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, l2_rate=0.022,
          num_epochs=500, print_cost=True, save_session=False):
    """
    模型训练的主方法
    :param X_train: 训练数据集
    :param Y_train: 测试数据集
    :param X_test: 训练标签集
    :param Y_test: 测试标签集
    :param learning_rate: 学习速率
    :param l2_rate: L2正则化速率
    :param num_epochs: 迭代次数
    :param print_cost: 是否打印cost函数
    :param save_session: 是否保存session
    :return:
    """
    print("learning_rate=" + str(learning_rate), " and l2_rate=" + str(l2_rate))
    print(X_train.shape)
    X_train = X_train / 255
    X_test = X_test / 255
    (n_N0, n_H0, n_W0, n_C0) = X_train.shape
    (n_N1, n_H1, n_W1, n_C1) = X_test.shape
    costs = []
    isTrain = tf.placeholder(tf.bool)
    num = tf.placeholder(tf.int32)
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    seed = 5
    parameters = initialize_parameters(randomSeed=seed)
    Z0, fc1w0, fc2w0, fc3w0= forward_propagation(X, parameters, num, flag=0, randomSeed=seed)
    cost0 = compute_cost(Z0, Y)
    # 采用L2正则化，避免过拟合
    regularizer = tf.contrib.layers.l2_regularizer(l2_rate)
    regularization0 = regularizer(fc1w0) + regularizer(fc2w0)+regularizer(fc3w0)
    cost0 = cost0 + regularization0
    # 设置学习速率
    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100, 0.96, staircase=True)
    # 模型反向传播优化算法
    optimizer0 = tf.train.AdamOptimizer(learning_rate).minimize(cost0, global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # 开始训练
    with tf.Session() as sess:
        sess.run(init)
        # save_path = saver.restore(sess, 'another/t_82/model_forloop170.ckpt')
        for epoch in range(num_epochs):
            print("第"+str(epoch)+"开始")
            _, cost = sess.run([optimizer0, cost0], feed_dict={X: X_train, Y: Y_train, num: n_N0})
            if print_cost is True and epoch % 1 == 0:
                print("损失函数经过%i次遍历后: %f" % (epoch, cost))
                Z = Z0 * 1
                predict_op = tf.argmax(Z, 1)  # 返回每行最大值的索引
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train, num: n_N0})
                test_accuracy = accuracy.eval({X: X_test, Y: Y_test, num: n_N1, isTrain: False})
                print("训练集识别率:", train_accuracy)
                print("测试集识别率:", test_accuracy)
                # 保存session，可用于预测
                # if save_session is True and train_accuracy > 0.985 and test_accuracy > 0.89:
                #     save_files = './another/x_82/model_forloop' + str(epoch) + '.ckpt'
                #     saver.save(sess, save_files)
                #     print("模型" + save_files + "保存成功.")
            if print_cost is True and epoch % 1 == 0:
                costs.append(cost)
        return parameters


def cnnTrain():
    print("多视角视图特征训练")
    trainpicsfile = './logs/3dColorPic64Train_2490.h5'
    testpicsfile = './logs/3dColorPic64Test_2490.h5'
    trainFile = './logs/3dModelTrainDBeta_1.h5'
    testFile = './logs/3dModelTestDBeta_1.h5'
    _, YTrain, _, YTest = CU.loadDataSets(trainFile, testFile)
    XTrain = h5utils.readData(trainpicsfile)
    XTest = h5utils.readData(testpicsfile)
    print(XTrain.shape[0], XTest.shape[0])
    print("数据读取完毕")
    parameters = model(XTrain, YTrain, XTest, YTest, num_epochs=2000, save_session=True)
    return XTrain, YTrain, XTest, YTest


if __name__ == '__main__':
    # 三维模型测试
    XTrain, YTrain, XTest, YTest = cnnTrain()
