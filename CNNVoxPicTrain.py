import tensorflow as tf
from tensorflow.python.framework import ops
import CNNUtils as CU


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    创建占位符
    :param n_H0:
    :param n_W0:
    :param n_C0:
    :param n_y:
    :return:
    """
    X0 = tf.placeholder('float', shape=[None, n_H0, n_W0, 1])
    X1 = tf.placeholder('float', shape=[None, n_H0, n_W0, 1])
    X2 = tf.placeholder('float', shape=[None, n_H0, n_W0, 1])
    Y = tf.placeholder('float', shape=[None, n_y])
    return X0, X1, X2, Y


def initialize_parameters(randomSeed=1):
    """
    初始化参数
    :param randomSeed: 随机seed种子
    :return:
    """
    print("初始化参数的seed为"+str(randomSeed))
    with tf.variable_scope('', reuse=tf.AUTO_REUSE) as te:
        W1 = tf.get_variable("W1", [5, 5, 1, 6], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
        W2 = tf.get_variable("W2", [5, 5, 6, 8], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
        W3 = tf.get_variable("W3", [5, 5, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=randomSeed))
        # W4 = tf.get_variable("W4", [2, 2, 16, 32], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  # "W4": W4
                  }
    return parameters


def forward_propagation(X, parameters, num, isTrain=True, flag=0, randomSeed=1):
    """
    模型前向传播方法
    :param X: 数据集
    :param parameters: 训练参数
    :param num: 数据集的模型总数量
    :param isTrain: 是否为训练数据
    :param flag: 全连接层参数的标志
    :param randomSeed: 随机seed种子
    :return:
    """
    print("全连接层的seed为"+str(randomSeed))
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    # Conv+Relu+Pool
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='VALID')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Conv+Relu+Pool
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='VALID')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Conv+Relu+Pool
    Z3 = tf.nn.conv2d(P2, W3, strides=[1, 1, 1, 1], padding='VALID')
    A3 = tf.nn.relu(Z3)
    P3 = tf.nn.max_pool(A3, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')

    pool_shape = P3.get_shape().as_list()          # 展开
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(P3, [num, nodes])
    # if isTrain:        # 防止过拟合
    #     reshaped = tf.nn.dropout(reshaped, 0.60)
    w1= "weight1_"+str(flag)
    b1 = "bias1_"+str(flag)
    w2= "weight2_"+str(flag)
    b2 = "bias2_"+str(flag)
    with tf.variable_scope('', reuse=tf.AUTO_REUSE) as fc_1:
        fc1_weights = tf.get_variable(w1, [nodes, 64], initializer=tf.truncated_normal_initializer(stddev=0.1, seed=randomSeed))
        fc1_biases = tf.get_variable(b1, [64], initializer=tf.constant_initializer(0.1))
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights)+fc1_biases)
    # if isTrain:        # 防止过拟合
    #     fc1 = tf.nn.dropout(fc1, 0.7)
    with tf.variable_scope('', reuse=tf.AUTO_REUSE) as fc_2:
        fc2_weights = tf.get_variable(w2, [64, 10], initializer=tf.truncated_normal_initializer(stddev=0.1, seed=randomSeed))
        fc2_biases = tf.get_variable(b2, [10], initializer=tf.constant_initializer(0.1))
    logit = (tf.matmul(fc1, fc2_weights)+fc2_biases)
    return logit, fc1_weights, fc2_weights


def compute_cost(Z3, Y):
    """
    计算损失函数
    :param Z3:
    :param Y:
    :return:
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3, labels=Y))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.05, l2_rate=0.03,
          num_epochs=500, print_cost=True, save_session=False):
    """
    模型训练的主方法
    :param X_train: 训练模型特征集
    :param Y_train: 训练模型标签集
    :param X_test: 测试模型特征集
    :param Y_test: 测试模型标签集
    :param learning_rate:  学习速率
    :param l2_rate: L2正则化速率
    :param num_epochs: 迭代次数
    :param print_cost: 是否打印cost函数
    :param save_session: 是否保存session(session保存参数，可用于预测！)
    :return:
    """
    print("l2_rate=" + str(l2_rate) + " and learning_rate=" + str(learning_rate))
    # Step1 : 数据归一化
    X_train = X_train/64
    X_test = X_test/64

    # Step2 : 初始化各类参数，为训练做准备
    # ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    (n_N0, n_H0, n_W0, n_C0) = X_train.shape
    (n_N1, n_H1, n_W1, n_C1) = X_test.shape
    weight1, weight2, weight3 = 0.5, 0.5, 0.5
    print("采用权值为", str(weight1), str(weight2), str(weight3))
    costs = []
    isTrain = tf.placeholder(tf.bool)
    num = tf.placeholder(tf.int32)
    X0, X1, X2, Y = create_placeholders(n_H0, n_W0, n_C0, n_H0)
    seed = 5
    parameters = initialize_parameters(randomSeed=seed)
    Z0, fc1w0, fc2w0 = forward_propagation(X0, parameters, num, flag=0, randomSeed=seed)
    Z1, fc1w1, fc2w1 = forward_propagation(X1, parameters, num, flag=1, randomSeed=seed)
    Z2, fc1w2, fc2w2 = forward_propagation(X2, parameters, num, flag=2, randomSeed=seed)
    cost0 = compute_cost(Z0, Y)
    cost1 = compute_cost(Z1, Y)
    cost2 = compute_cost(Z2, Y)
    # 采用L2正则化，避免过拟合
    regularizer = tf.contrib.layers.l2_regularizer(l2_rate)
    regularization0 = regularizer(fc1w0)+regularizer(fc2w0)
    cost0 = cost0 + regularization0
    regularization1 = regularizer(fc1w1)+regularizer(fc2w1)
    cost1 = cost1 + regularization1
    regularization2 = regularizer(fc1w2)+regularizer(fc2w2)
    cost2 = cost2 + regularization2

    global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, 100, 0.96, staircase=True)
    optimizer0 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost0, global_step)
    optimizer1 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost1, global_step)
    optimizer2 = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost2, global_step)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Step3: 开始训练
    with tf.Session() as sess:
        sess.run(init)
        # save_path = saver.restore(sess, 'another/t_82/model_forloop170.ckpt')
        for epoch in range(num_epochs):
            x0 = X_train[:,:,:,0].reshape(n_N0, n_H0, n_W0, 1)
            x1 = X_train[:,:,:,1].reshape(n_N0, n_H0, n_W0, 1)
            x2 = X_train[:,:,:,2].reshape(n_N0, n_H0, n_W0, 1)
            X0T = X_test[:,:,:,0].reshape(n_N1, n_H1, n_W1, 1)
            X1T = X_test[:,:,:,1].reshape(n_N1, n_H1, n_W1, 1)
            X2T = X_test[:,:,:,2].reshape(n_N1, n_H1, n_W1, 1)
            _, cost0 = sess.run([optimizer0, cost0], feed_dict={X0: x0, Y: Y_train, num: n_N0})
            _, cost1 = sess.run([optimizer1, cost1], feed_dict={X1: x1, Y: Y_train, num: n_N0})
            _, cost2 = sess.run([optimizer2, cost2], feed_dict={X2: x2, Y: Y_train, num: n_N0})

            if print_cost is True and epoch % 1 == 0:
                print("损失函数经过%i次遍历后: %f, %f, %f" % (epoch, cost0, cost1, cost2))
                Z = Z0*weight1+Z1*weight2+Z2*weight3
                predict_op = tf.argmax(Z, 1)  # 返回每行最大值的索引
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                train_accuracy = accuracy.eval({X0: x0, X1: x1, X2: x2, Y: Y_train, num: n_N0})
                test_accuracy = accuracy.eval({X0: X0T, X1: X1T, X2: X2T, Y: Y_test, num: n_N1, isTrain: False})
                print("训练集识别率:", train_accuracy)
                print("测试集识别率:", test_accuracy)
                # Step4 : 如果条件成立，保存session
                if save_session is True and train_accuracy > 0.97 and test_accuracy > 0.923:
                    save_files = './session/model_forloop'+str(epoch)+'.ckpt'
                    saver.save(sess, save_files)
                    print("模型"+save_files+"保存成功.")
                    save_session = False
                # Step5 : 在此区间，通过改变权值，来得到最优准确率(可以注释掉)
                if epoch > 1000 and epoch < 1200:
                    print("-----------------------------------------------------")
                    for i in range(1, 11):
                        for j in range(11 - i):
                            w1 = i / 10
                            w2 = j / 10
                            w3 = 1 - (w1 + w2)
                            # w1, w2, w3 = 0.6, 0.5, 0.2
                            print("w分别为：", str(w1), str(w2), str(w3))
                            Z = Z0 * w1 + Z1 * w2 + Z2 * w3
                            predict_op = tf.argmax(Z, 1)  # 返回每行最大值的索引
                            correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                            train_accuracy = accuracy.eval({X0: x0, X1: x1, X2: x2, Y: Y_train, num: n_N0})
                            test_accuracy = accuracy.eval({X0: X0T, X1: X1T, X2: X2T, Y: Y_test, num: n_N1, isTrain: False})
                            print("训练集识别率:", train_accuracy,"测试集识别率:", test_accuracy)
                    print("-----------------------------------------------------")
            if print_cost is True and epoch % 1 == 0:
                costs.append(cost0+cost1+cost2)
        return parameters


def cnnTrain():
    print("采用L2正则化的加权深层图像特征")
    trainFile = './logs/3dModelTrainDBeta_1.h5'
    testFile = './logs/3dModelTestDBeta_1.h5'
    # trainFile = 'drive/test/logs/3dModelTrainDBeta_8_2.h5'
    # testFile = 'drive/test/logs/3dModelTestDBeta_8_2.h5'
    XTrain, YTrain, XTest, YTest = CU.loadDataSets(trainFile, testFile)
    print("模型个数分别为：", XTrain.shape[0], XTest.shape[0])
    model(XTrain, YTrain, XTest, YTest, num_epochs=2000, save_session=True)     # 训练模型主方法


if __name__ == '__main__':
    cnnTrain()
