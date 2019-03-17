# import math
# import numpy as np
# import h5py
import os

import matplotlib.pyplot as plt
# import scipy
# # from PIL import Image
# from scipy import ndimage
# from PIL.Image import core as _imaging
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.framework import ops
# from cnn_utils import *
import CNNUtils
import GetFeature as GF
import CNNUtils as CU
import CNNTrain as CT
import H5FileUtils as h5utils
import AnoPicNet as WN
from ModelList import models


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


def predictOne(modeldirone):
    # dir = "E:\\3d_Retrival_System_beta2\\3D-shape-retrival\\testmodel\\bathtub\\"
    # filename = "bathtub_0156\\"
    # modeldirone = dir+filename
    allpics = []
    for fileroot, filedirs, filefiles in os.walk(modeldirone):
        # print(files)  # 当前路径下所有子目录
        break;
    for k in range(len(filefiles) - 1):
        filename = filefiles[k]
        fillname = modeldirone + "\\"+filename
        print(fillname)
        img = Image.open(fillname)
        img = img.resize((64, 64)).convert('L')
        # image_arr = np.array(img)
        image_arr = np.array(img).reshape((64, 64, 1))
        # image_arr = np.array(img).reshape((64, 64))
        # allpics.append(image_arr)
        # modelpics.append(image_arr)
        # modelpics = np.array(modelpics).transpose(1,2,0)
        modelpics = np.array(image_arr)
        allpics.append(modelpics)
    allpics = np.array(allpics)
    test_pics = allpics/255

    # 加载模型
    trainpicsfile = 'C://Users/sijia3/Desktop/3D-shape-retrieval/logs/3dNoColorPic64Train82_5.h5'
    testpicsfile = 'C://Users/sijia3/Desktop/3D-shape-retrieval/logs/3dNoColorPic64Test82_5.h5'
    trainFile = 'C://Users/sijia3/Desktop/3D-shape-retrieval/logs/3dModelTrainDBeta_8_2.h5'
    testFile = 'C://Users/sijia3/Desktop/3D-shape-retrieval/logs/3dModelTestDBeta_8_2.h5'
    # trainFile = './logs/logs/3dModelTrainDBeta_1.h5'
    # testFile = './logs/logs/3dModelTestDBeta_1.h5'
    _, Y_train, _, Y_test = CU.loadDataSets(trainFile, testFile)
    X_train = h5utils.readData(trainpicsfile)
    X_test = h5utils.readData(testpicsfile)
    X_train = X_train / 255
    X_test = X_test / 255
    # 声明变量，以便模型加载可以存放
    (m, n_H0, n_W0, n_C0) = X_train.shape
    a, b, c, d = X_test.shape
    m_test = X_test.shape[0]
    n_y = Y_train.shape[1]
    X, Y = WN.create_placeholders(n_H0, n_W0, n_C0, n_y)
    num = tf.placeholder(tf.int32)
    seed = 5

    parameters = WN.initialize_parameters(randomSeed=seed)
    Z, fc1w0, fc2w0,fc3w0 = WN.forward_propagation(X, parameters, num, flag=0, randomSeed=seed)
    normalize = tf.nn.l2_normalize(Z, axis=1)  # 特征向量归一化
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载文件中的参数数据，会根据filename加载数据并保存到各个变量中
        save_path = saver.restore(sess, 'C:\\Users\\sijia3\Desktop\\3D-shape-retrieval/another/x_82/model_forloop765.ckpt')
        # XTrainNorm= sess.run(normalize, feed_dict={X: X_train, num: m})
        # h5utils.writeH5File('XTrainNorm.h5', XTrainNorm)
        XTrainNorm = h5utils.readData('C:\\Users\\sijia3\Desktop\\3D-shape-retrieval\datasets\XTrainNorm.h5')
        ModelNorm= sess.run(normalize, feed_dict={X: test_pics, num: 8})
        # io.savemat('./mat/2.mat', {'XTestNorm0': ModelNorm0, 'XTestNorm1': ModelNorm1, 'XTestNorm2': ModelNorm2})
        # 以上训练好的东西，保存到mat文件中
        # XTrainNorm0, XTrainNorm1, XTrainNorm2 = sess.run([Z0, Z1, Z2], feed_dict={
        #     X0: x0, X1: x1, X2: x2, num: m})
        # ModelNorm0, ModelNorm1, ModelNorm2 = sess.run([Z0, Z1, Z2], feed_dict={
        #     X0: X0T, X1: X1T, X2: X2T, num: 1})
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
        predict = predict_op.eval({X:test_pics})
        # print("预测为" + models[predict[0]])
        return eigIndex.tolist()
if __name__ == '__main__':
    a = predictOne("C:\\Users\\sijia3\Desktop\\3D-shape-retrieval\matlab\\tt\\bed_0551\\")
