import numpy as np

import GetFeature as GF
# import CNNTrain as CT
import ReadOff
import Tri2Vox
import GetFeature as GF
from ModelList import models
import CNNUtils as CU
import WeightNet as WN
import scipy.io as io
import H5FileUtils as h5utils
import tensorflow as tf


def predict():

    # 加载模型
    trainFile = './logs/3dModelTrainDBeta_8_2.h5'
    testFile = './logs/3dModelTestDBeta_8_2.h5'
    X_train, Y_train, X_test, Y_test = CU.loadDataSets(trainFile, testFile)
    # 声明变量，以便模型加载可以存放
    (m, n_H0, n_W0, n_C0) = X_train.shape
    a, b, c, d = X_test.shape
    # weight1, weight2, weight3 = 0.6, 0.5, 0.2
    # print("采用权值为", str(weight1), str(weight2), str(weight3))
    m_test = X_test.shape[0]
    n_y = Y_train.shape[1]
    X0, X1, X2, Y = WN.create_placeholders(n_H0, n_W0, n_C0, n_y)
    num = tf.placeholder(tf.int32)
    seed = 5
    parameters = WN.initialize_parameters(randomSeed=seed)
    Z0, fc1w0, fc2w0 = WN.forward_propagation(X0, parameters, num, flag=0, randomSeed=seed)
    Z1, fc1w1, fc2w1 = WN.forward_propagation(X1, parameters, num, flag=1, randomSeed=seed)
    Z2, fc1w2, fc2w2 = WN.forward_propagation(X2, parameters, num, flag=2, randomSeed=seed)

    normalize0 = tf.nn.l2_normalize(Z0, axis=1)     # 特征向量归一化
    normalize1 = tf.nn.l2_normalize(Z1, axis=1)     # 特征向量归一化
    normalize2= tf.nn.l2_normalize(Z2, axis=1)     # 特征向量归一化
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载文件中的参数数据，会根据filename加载数据并保存到各个变量中
        save_path = saver.restore(sess, './another/u_82/model_forloop0.ckpt')
        x0 = X_train[:, :, :, 0].reshape(m, n_H0, n_W0, 1)/64
        x1 = X_train[:, :, :, 1].reshape(m, n_H0, n_W0, 1)/64
        x2 = X_train[:, :, :, 2].reshape(m, n_H0, n_W0, 1)/64
        X0T = X_test[:, :, :, 0].reshape(a, b, c, 1)/64
        X1T = X_test[:, :, :, 1].reshape(a, b, c, 1)/64
        X2T = X_test[:, :, :, 2].reshape(a, b, c, 1)/64
        for i in range(1, 11):
            for j in range(11-i):
                w1 = i/10
                w2 = j/10
                w3 = 1-w1-w2
                # w1, w2, w3 = 0.6, 0.5, 0.2
                print("w分别为：", str(w1), str(w2), str(w3))
                Z = Z0 * w1 + Z1 * w2 + Z2 * w3
                predict_op = tf.argmax(Z, 1)  # 返回每行最大值的索引
                correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                train_accuracy = accuracy.eval({X0: x0, X1: x1, X2: x2, Y: Y_train, num: m})
                test_accuracy = accuracy.eval({X0: X0T, X1: X1T, X2: X2T, Y: Y_test, num: m_test})
                print("训练集识别率:", train_accuracy)
                print("测试集识别率:", test_accuracy)


def predictOne():
    # # 特征提取
    verts, faces = ReadOff.readOff('./model/bed_0563.off')
    # 特征提取
    # verts, faces = ReadOff.readOff('./model/bed_0615.off')
    # 体素化
    vox = Tri2Vox.Tri2Vox(verts, faces, 32)
    # 获取三视图
    pics1 = GF.getPics(vox, isInDepth=True)

    test_pics = np.array([pics1])

    # 加载模型
    trainFile = './logs/3dModelTrainDBeta_8_2.h5'
    testFile = './logs/3dModelTestDBeta_8_2.h5'
    # trainFile = './logs/logs/3dModelTrainDBeta_1.h5'
    # testFile = './logs/logs/3dModelTestDBeta_1.h5'
    X_train, Y_train, X_test, Y_test = CU.loadDataSets(trainFile, testFile)
    # 声明变量，以便模型加载可以存放
    (m, n_H0, n_W0, n_C0) = X_train.shape
    a, b, c, d = X_test.shape
    weight1, weight2, weight3 = 0.6, 0.5, 0.2
    print("采用权值为", str(weight1), str(weight2), str(weight3))
    m_test = X_test.shape[0]
    n_y = Y_train.shape[1]
    X0, X1, X2, Y = WN.create_placeholders(n_H0, n_W0, n_C0, n_y)
    num = tf.placeholder(tf.int32)
    seed = 5
    # with tf.variable_scope('', reuse=tf.AUTO_REUSE) as te:
    parameters = WN.initialize_parameters(randomSeed=seed)
    Z0, fc1w0, fc2w0 = WN.forward_propagation(X0, parameters, num, flag=0, randomSeed=seed)
    Z1, fc1w1, fc2w1 = WN.forward_propagation(X1, parameters, num, flag=1, randomSeed=seed)
    Z2, fc1w2, fc2w2 = WN.forward_propagation(X2, parameters, num, flag=2, randomSeed=seed)
    normalize0 = tf.nn.l2_normalize(Z0, axis=1)  # 特征向量归一化
    normalize1 = tf.nn.l2_normalize(Z1, axis=1)  # 特征向量归一化
    normalize2 = tf.nn.l2_normalize(Z2, axis=1)  # 特征向量归一化
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载文件中的参数数据，会根据filename加载数据并保存到各个变量中
        save_path = saver.restore(sess, './another/u_82/model_forloop275.ckpt')
        X_train[:, :, :, 0] = X_train[:, :, :, 0] / 64
        X_train[:, :, :, 1] = X_train[:, :, :, 1] / 64
        X_train[:, :, :, 2] = X_train[:, :, :, 2] / 64
        test_pics[:, :, :, 0] = test_pics[:, :, :, 0] / 64
        test_pics[:, :, :, 1] = test_pics[:, :, :, 1] / 64
        test_pics[:, :, :, 2] = test_pics[:, :, :, 2] / 64
        # X_test[:, :, :, 0] = X_test[:, :, :, 0] / 64
        # X_test[:, :, :, 1] = X_test[:, :, :, 1] / 64
        # X_test[:, :, :, 2] = X_test[:, :, :, 2] / 64
        x0 = X_train[:, :, :, 0].reshape(m, n_H0, n_W0, 1)
        x1 = X_train[:, :, :, 1].reshape(m, n_H0, n_W0, 1)
        x2 = X_train[:, :, :, 2].reshape(m, n_H0, n_W0, 1)
        X0T = test_pics[:, :, :, 0].reshape(1, b, c, 1)
        X1T = test_pics[:, :, :, 1].reshape(1, b, c, 1)
        X2T = test_pics[:, :, :, 2].reshape(1, b, c, 1)
        # X0T = X_test[:, :, :, 0].reshape(a, b, c, 1)
        # X1T = X_test[:, :, :, 1].reshape(a, b, c, 1)
        # X2T = X_test[:, :, :, 2].reshape(a, b, c, 1)
        w1, w2, w3 = 0.6, 0.5, 0.2
        XTrainNorm0, XTrainNorm1, XTrainNorm2 = sess.run([normalize0, normalize1, normalize2], feed_dict={
            X0: x0, X1: x1, X2: x2, num: m})
        h5utils.writeH5File('XTrainNorm0.h5', XTrainNorm0)
        h5utils.writeH5File('XTrainNorm1.h5', XTrainNorm1)
        h5utils.writeH5File('XTrainNorm2.h5', XTrainNorm2)
        ModelNorm0, ModelNorm1, ModelNorm2 = sess.run([normalize0, normalize1, normalize2], feed_dict={
            X0: X0T, X1: X1T, X2: X2T, num: 1})

        # io.savemat('./mat/2.mat', {'XTestNorm0': ModelNorm0, 'XTestNorm1': ModelNorm1, 'XTestNorm2': ModelNorm2})
        # 以上训练好的东西，保存到mat文件中
        # XTrainNorm0, XTrainNorm1, XTrainNorm2 = sess.run([Z0, Z1, Z2], feed_dict={
        #     X0: x0, X1: x1, X2: x2, num: m})
        # ModelNorm0, ModelNorm1, ModelNorm2 = sess.run([Z0, Z1, Z2], feed_dict={
        #     X0: X0T, X1: X1T, X2: X2T, num: 1})
        XTrainNorm = XTrainNorm0 * w1 + XTrainNorm1 * w2 + XTrainNorm2 * w3
        ModelNorm = ModelNorm0 * w1 + ModelNorm0 * w2 + ModelNorm2 * w3
        eig = XTrainNorm - ModelNorm  # A-B
        # XTrainNorm = normalize0.eval({X0: X_train})  # A    训练模型特征向量
        # ModelNorm = normalize.eval({X: test_pics})  # B   预测模型特征向量
        # eig0 = XTrainNorm0 - ModelNorm0  # A-B
        # eig1 = XTrainNorm1 - ModelNorm1  # A-B
        # eig2 = XTrainNorm2 - ModelNorm2  # A-B
        eig = np.linalg.norm(eig, axis=1)  # 相似度比较，采用二范数
        # eig0 = np.linalg.norm(eig0, axis=1)          # 相似度比较，采用二范数
        # eig1 = np.linalg.norm(eig1, axis=1)          # 相似度比较，采用二范数
        # eig2 = np.linalg.norm(eig2, axis=1)          # 相似度比较，采用二范数
        eigIndex = np.argsort(eig)  # 相似度排序，输出相似模型的下标
        # eigIndex150 = eigIndex[0:151]
        # print((eigIndex150 < 151).sum())

        # ((eigIndex[0:80] < 160) & (eigIndex[0:80] > 80)).sum()
        predict_op = tf.argmax(ModelNorm, 1)  # 返回每行最大值的索引值
        predict = predict_op.eval({X0: X0T, X1: X1T, X2: X2T})
        print("预测为" + models[predict[0]])


def searchByMatlab(filename, w1=0.6,w2=0.5,w3=0.2):
    print("权值分别为："+str(w1), str(w2), str(w3))
    # # 特征提取
    verts, faces = ReadOff.readOff(filename)
    # 体素化
    vox = Tri2Vox.Tri2Vox(verts, faces, 32)
    # 获取三视图
    pics1 = GF.getPics(vox, isInDepth=True)
    test_pics = np.array([pics1])

    # 加载模型
    trainFile = 'C://Users/sijia3/Desktop/3D-shape-retrieval/logs/3dModelTrainDBeta_8_2.h5'
    testFile = 'C://Users/sijia3/Desktop/3D-shape-retrieval/logs/3dModelTestDBeta_8_2.h5'
    # trainFile = './logs/logs/3dModelTrainDBeta_1.h5'
    # testFile = './logs/logs/3dModelTestDBeta_1.h5'
    X_train, Y_train, X_test, Y_test = CU.loadDataSets(trainFile, testFile)
    # 声明变量，以便模型加载可以存放
    (m, n_H0, n_W0, n_C0) = X_train.shape
    a, b, c, d = X_test.shape
    m_test = X_test.shape[0]
    n_y = Y_train.shape[1]
    X0, X1, X2, Y = WN.create_placeholders(n_H0, n_W0, n_C0, n_y)
    num = tf.placeholder(tf.int32)
    seed = 5
    parameters = WN.initialize_parameters(randomSeed=seed)
    Z0, fc1w0, fc2w0 = WN.forward_propagation(X0, parameters, num, flag=0, randomSeed=seed)
    Z1, fc1w1, fc2w1 = WN.forward_propagation(X1, parameters, num, flag=1, randomSeed=seed)
    Z2, fc1w2, fc2w2 = WN.forward_propagation(X2, parameters, num, flag=2, randomSeed=seed)
    normalize0 = tf.nn.l2_normalize(Z0, axis=1)  # 特征向量归一化
    normalize1 = tf.nn.l2_normalize(Z1, axis=1)  # 特征向量归一化
    normalize2 = tf.nn.l2_normalize(Z2, axis=1)  # 特征向量归一化
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载文件中的参数数据，会根据filename加载数据并保存到各个变量中
        save_path = saver.restore(sess, 'C://Users/sijia3/Desktop/3D-shape-retrieval/another/u_82/model_forloop275.ckpt')
        X_train = X_train / 64
        test_pics = test_pics / 64
        x0 = X_train[:, :, :, 0].reshape(m, n_H0, n_W0, 1)
        x1 = X_train[:, :, :, 1].reshape(m, n_H0, n_W0, 1)
        x2 = X_train[:, :, :, 2].reshape(m, n_H0, n_W0, 1)
        X0T = test_pics[:, :, :, 0].reshape(1, b, c, 1)
        X1T = test_pics[:, :, :, 1].reshape(1, b, c, 1)
        X2T = test_pics[:, :, :, 2].reshape(1, b, c, 1)
        # XTrainNorm0, XTrainNorm1, XTrainNorm2 = sess.run([normalize0, normalize1, normalize2], feed_dict={
        #     X0: x0, X1: x1, X2: x2, num: m})
        XTrainNorm0 = h5utils.readData('C://Users/sijia3/Desktop/3D-shape-retrieval/datasets/XTrainNorm0.h5')
        XTrainNorm1 = h5utils.readData('C://Users/sijia3/Desktop/3D-shape-retrieval/datasets/XTrainNorm1.h5')
        XTrainNorm2 = h5utils.readData('C://Users/sijia3/Desktop/3D-shape-retrieval/datasets/XTrainNorm2.h5')
        ModelNorm0, ModelNorm1, ModelNorm2 = sess.run([normalize0, normalize1, normalize2], feed_dict={
            X0: X0T, X1: X1T, X2: X2T, num: 1})

        # io.savemat('./mat/2.mat', {'XTestNorm0': ModelNorm0, 'XTestNorm1': ModelNorm1, 'XTestNorm2': ModelNorm2})
        # 以上训练好的东西，保存到mat文件中
        # XTrainNorm0, XTrainNorm1, XTrainNorm2 = sess.run([Z0, Z1, Z2], feed_dict={
        #     X0: x0, X1: x1, X2: x2, num: m})
        # ModelNorm0, ModelNorm1, ModelNorm2 = sess.run([Z0, Z1, Z2], feed_dict={
        #     X0: X0T, X1: X1T, X2: X2T, num: 1})
        XTrainNorm = XTrainNorm0 * w1 + XTrainNorm1 * w2 + XTrainNorm2 * w3
        ModelNorm = ModelNorm0 * w1 + ModelNorm0 * w2 + ModelNorm2 * w3
        eig = XTrainNorm - ModelNorm  # A-B
        # XTrainNorm = normalize0.eval({X0: X_train})  # A    训练模型特征向量
        # ModelNorm = normalize.eval({X: test_pics})  # B   预测模型特征向量
        # eig0 = XTrainNorm0 - ModelNorm0  # A-B
        # eig1 = XTrainNorm1 - ModelNorm1  # A-B
        # eig2 = XTrainNorm2 - ModelNorm2  # A-B
        eig = np.linalg.norm(eig, axis=1)  # 相似度比较，采用二范数
        # eig0 = np.linalg.norm(eig0, axis=1)          # 相似度比较，采用二范数
        # eig1 = np.linalg.norm(eig1, axis=1)          # 相似度比较，采用二范数
        # eig2 = np.linalg.norm(eig2, axis=1)          # 相似度比较，采用二范数
        eigIndex = np.argsort(eig)  # 相似度排序，输出相似模型的下标
        # eigIndex150 = eigIndex[0:151]
        # print((eigIndex150 < 151).sum())

        # ((eigIndex[0:80] < 160) & (eigIndex[0:80] > 80)).sum()
        predict_op = tf.argmax(ModelNorm, 1)  # 返回每行最大值的索引值
        predict = predict_op.eval({X0: X0T, X1: X1T, X2: X2T})
        print("预测为" + models[predict[0]])
        return eigIndex.tolist()


if __name__ == '__main__':
    # predictOne()
    a = searchByMatlab('./model/bed_0563.off')