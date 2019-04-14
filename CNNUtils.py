import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import H5FileUtils as h5utils



def loadDataSets(trainFile, testFile):
    """
    加载模型数据
    :param trainFile: 训练模型数据
    :param testFile: 测试模型数据
    :return: 加载后的模型特征跟标签数据
    """
    # trainFile = './datasets/train_model.h5'
    XTrain, YTrain= h5utils.readDataAndLabels(trainFile)
    YTrain = YTrain.reshape(1, len(YTrain)).astype('int64')
    YTrain = convert_to_one_hot(YTrain, 10).T

    # testFile = './datasets/test_model.h5'
    XTest, YTest = h5utils.readDataAndLabels(testFile)
    YTest = YTest.reshape(1, len(YTest)).astype('int64')
    YTest = convert_to_one_hot(YTest, 10).T
    return XTrain, YTrain, XTest, YTest


def convert_to_one_hot(Y, C):
    """
    标签格式转换
    :param Y:
    :param C:
    :return:
    """
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    # np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


if __name__ == '__main__':
    load_dataset();