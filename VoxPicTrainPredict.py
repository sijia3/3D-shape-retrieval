import numpy as np
import ReadOff
import Tri2Vox
import GetFeature as GF
from ModelList import models
import CNNUtils as CU
import CNNVoxPicTrain as WN
import tensorflow as tf


def predict():
    # 加载模型
    trainFile = './logs/3dModelTrainDBeta_8_2.h5'
    testFile = './logs/3dModelTestDBeta_8_2.h5'
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
    # Step1: 特征提取
    verts, faces = ReadOff.readOff('./model/bed_0563.off')
    # Step2 :体素化
    vox = Tri2Vox.Tri2Vox(verts, faces, 64)
    # Step3 : 获取三视图特征
    pics1 = GF.getPics(vox, isInDepth=True)
    test_pics = np.array([pics1])
    # Step4: 加载模型
    trainFile = './logs/3dModelTrainDBeta_1.h5'
    testFile = './logs/3dModelTestDBeta_1.h5'
    X_train, Y_train, X_test, Y_test = CU.loadDataSets(trainFile, testFile)
    # Step5: 声明变量，以便模型加载可以存放
    (n_N0, n_H0, n_W0, n_C0) = X_train.shape
    a, b, c, d = X_test.shape
    n_y = Y_train.shape[1]
    X0, X1, X2, Y = WN.create_placeholders(n_H0, n_W0, n_C0, n_y)
    num = tf.placeholder(tf.int32)
    seed = 2
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
        x0 = X_train[:, :, :, 0].reshape(n_N0, n_H0, n_W0, 1)
        x1 = X_train[:, :, :, 1].reshape(n_N0, n_H0, n_W0, 1)
        x2 = X_train[:, :, :, 2].reshape(n_N0, n_H0, n_W0, 1)
        X0T = test_pics[:, :, :, 0].reshape(1, b, c, 1)
        X1T = test_pics[:, :, :, 1].reshape(1, b, c, 1)
        X2T = test_pics[:, :, :, 2].reshape(1, b, c, 1)
        w1, w2, w3 = 0.5, 0.5, 0.5
        XTrainNorm0, XTrainNorm1, XTrainNorm2 = sess.run([normalize0, normalize1, normalize2], feed_dict={
            X0: x0, X1: x1, X2: x2, num: n_N0})
        ModelNorm0, ModelNorm1, ModelNorm2 = sess.run([normalize0, normalize1, normalize2], feed_dict={
            X0: X0T, X1: X1T, X2: X2T, num: 1})
        XTrainNorm = XTrainNorm0 * w1 + XTrainNorm1 * w2 + XTrainNorm2 * w3
        ModelNorm = ModelNorm0 * w1 + ModelNorm0 * w2 + ModelNorm2 * w3
        eig = XTrainNorm - ModelNorm  # A-B
        eig = np.linalg.norm(eig, axis=1)  # 相似度比较，采用二范数
        eigIndex = np.argsort(eig)  # 相似度排序，输出相似模型的下标
        # eigIndex150 = eigIndex[0:151]
        # print((eigIndex150 < 151).sum())
        # ((eigIndex[0:80] < 160) & (eigIndex[0:80] > 80)).sum()
        predict_op = tf.argmax(ModelNorm, 1)  # 返回每行最大值的索引值
        predict = predict_op.eval({X0: X0T, X1: X1T, X2: X2T})
        print("预测为" + models[predict[0]])


if __name__ == '__main__':
    predictOne()
