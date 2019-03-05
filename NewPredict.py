import tensorflow as tf
import numpy as np
import GetFeature as GF
# import CNNTrain as CT
import ReadOff
import Tri2Vox
import GetFeature as GF
from ModelList import models
import CNNUtils as CU
import WeightNet as WN


def predict(predictFilename):
    # 特征提取
    verts, faces = ReadOff.readOff(predictFilename)
    # 体素化
    vox = Tri2Vox.Tri2Vox(verts, faces, 32)
    # 获取三视图
    pics1 = GF.getPics(vox, isInDepth=True)

    test_pics = np.array([pics1])

    # 加载模型
    trainFile = './logs/3dModelTrainDBeta_8_2.h5'
    testFile = './logs/3dModelTestDBeta_8_2.h5'
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
        save_path = saver.restore(sess, 'session/model_forloop435.ckpt')
        x0 = X_train[:, :, :, 0].reshape(m, n_H0, n_W0, 1)
        x1 = X_train[:, :, :, 1].reshape(m, n_H0, n_W0, 1)
        x2 = X_train[:, :, :, 2].reshape(m, n_H0, n_W0, 1)
        X0T = test_pics[:, :, :, 0].reshape(a, b, c, 1)
        X1T = test_pics[:, :, :, 1].reshape(a, b, c, 1)
        X2T = test_pics[:, :, :, 2].reshape(a, b, c, 1)
        w1, w2, w3 = 0.6, 0.3, 0.5
        XTrainNorm0, XTrainNorm1, XTrainNorm2 = sess.run([normalize0, normalize1, normalize2], feed_dict={X0:x0, X1:x1, X2:x2})
        ModelNorm0, ModelNorm1, ModelNorm2 = sess.run([normalize0, normalize1, normalize2], feed_dict={X0:X0T, X1:X1T, X2:X2T})

        XTrainNorm = XTrainNorm0*w1+ XTrainNorm1*w2+XTrainNorm2*w3
        ModelNorm = ModelNorm0*w1+ModelNorm0*w2+ModelNorm2*w3
        eig = XTrainNorm - ModelNorm  # A-B
        # XTrainNorm = normalize0.eval({X0: X_train})  # A    训练模型特征向量
        # ModelNorm = normalize.eval({X: test_pics})  # B   预测模型特征向量
        # eig0 = XTrainNorm0 - ModelNorm0  # A-B
        # eig1 = XTrainNorm1 - ModelNorm1  # A-B
        # eig2 = XTrainNorm2 - ModelNorm2  # A-B
        eig = np.linalg.norm(eig, axis=1)          # 相似度比较，采用二范数
        # eig0 = np.linalg.norm(eig0, axis=1)          # 相似度比较，采用二范数
        # eig1 = np.linalg.norm(eig1, axis=1)          # 相似度比较，采用二范数
        # eig2 = np.linalg.norm(eig2, axis=1)          # 相似度比较，采用二范数
        eigIndex = np.argsort(eig)                 # 相似度排序，输出相似模型的下标
        # eigIndex150 = eigIndex[0:151]
        # print((eigIndex150 < 151).sum())
        predict_op = tf.argmax(ModelNorm, 1)  # 返回每行最大值的索引值
        predict = predict_op.eval({X0: X0T, X1: X1T, X2:X2T})
        print("预测为" + models[predict[0]])
        return models[predict[0]]

if __name__ == '__main__':
    # # 特征提取
    # verts, faces = ReadOff.readOff('./model/bed_0563.off')
       # 特征提取
    verts, faces = ReadOff.readOff('./model/bed_0563.off')
    # 体素化
    vox = Tri2Vox.Tri2Vox(verts, faces, 32)
    # 获取三视图
    pics1 = GF.getPics(vox, isInDepth=True)

    test_pics = np.array([pics1])

    # 加载模型
    trainFile = './logs/3dModelTrainDBeta_8_2.h5'
    testFile = './logs/3dModelTestDBeta_8_2.h5'
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
    parameters = WN.initialize_parameters(randomSeed=seed)
    Z0, fc1w0, fc2w0 = WN.forward_propagation(X0, parameters, num, flag=0, randomSeed=seed)
    Z1, fc1w1, fc2w1 = WN.forward_propagation(X1, parameters, num, flag=1, randomSeed=seed)
    Z2, fc1w2, fc2w2 = WN.forward_propagation(X2, parameters, num, flag=2, randomSeed=seed)
    normalize0 = tf.nn.l2_normalize(Z0, axis=1)     # 特征向量归一化
    normalize1 = tf.nn.l2_normalize(Z1, axis=1)     # 特征向量归一化
    normalize2 = tf.nn.l2_normalize(Z2, axis=1)     # 特征向量归一化
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载文件中的参数数据，会根据filename加载数据并保存到各个变量中
        save_path = saver.restore(sess, 'session/model_forloop500.ckpt')
        x0 = X_train[:, :, :, 0].reshape(m, n_H0, n_W0, 1)
        x1 = X_train[:, :, :, 1].reshape(m, n_H0, n_W0, 1)
        x2 = X_train[:, :, :, 2].reshape(m, n_H0, n_W0, 1)
        X0T = test_pics[:, :, :, 0].reshape(1, b, c, 1)
        X1T = test_pics[:, :, :, 1].reshape(1, b, c, 1)
        X2T = test_pics[:, :, :, 2].reshape(1, b, c, 1)
        w1, w2, w3 = 0.6, 0.3, 0.5
        XTrainNorm0, XTrainNorm1, XTrainNorm2 = sess.run([normalize0, normalize1, normalize2], feed_dict={
            X0: x0, X1: x1, X2: x2, num: m})
        ModelNorm0, ModelNorm1, ModelNorm2 = sess.run([normalize0, normalize1, normalize2], feed_dict={
            X0: X0T, X1: X1T, X2: X2T, num: 1})
        XTrainNorm = XTrainNorm0*w1 + XTrainNorm1*w2 + XTrainNorm2*w3
        ModelNorm = ModelNorm0*w1 + ModelNorm0*w2 + ModelNorm2*w3
        eig = XTrainNorm - ModelNorm  # A-B
        # XTrainNorm = normalize0.eval({X0: X_train})  # A    训练模型特征向量
        # ModelNorm = normalize.eval({X: test_pics})  # B   预测模型特征向量
        # eig0 = XTrainNorm0 - ModelNorm0  # A-B
        # eig1 = XTrainNorm1 - ModelNorm1  # A-B
        # eig2 = XTrainNorm2 - ModelNorm2  # A-B
        eig = np.linalg.norm(eig, axis=1)          # 相似度比较，采用二范数
        # eig0 = np.linalg.norm(eig0, axis=1)          # 相似度比较，采用二范数
        # eig1 = np.linalg.norm(eig1, axis=1)          # 相似度比较，采用二范数
        # eig2 = np.linalg.norm(eig2, axis=1)          # 相似度比较，采用二范数
        eigIndex = np.argsort(eig)                 # 相似度排序，输出相似模型的下标
        # eigIndex150 = eigIndex[0:151]
        # print((eigIndex150 < 151).sum())
        predict_op = tf.argmax(ModelNorm, 1)  # 返回每行最大值的索引值
        predict = predict_op.eval({X0: X0T, X1: X1T, X2: X2T})
        print("预测为" + models[predict[0]])