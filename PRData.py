import numpy as np
import matplotlib.pyplot as plt
import CNNUtils as CU
import CNNVoxPicTrain as WN
import tensorflow as tf


if __name__ == '__main__':
    """
    描绘模型的PR曲线
    """
    # 加载模型
    trainFile = './datasets/3dModelTrain_PR.h5'
    testFile = './datasets/3dModelTest_PR_other.h5'
    X_train, Y_train, X_test, Y_test = CU.loadDataSets(trainFile, testFile)
    # 声明变量，以便模型加载可以存放
    (n_N0, n_H0, n_W0, n_C0) = X_train.shape
    (n_N1, n_H1, n_W1, n_C1) = X_test.shape
    # m_test = X_test.shape[0]
    n_y = Y_train.shape[1]
    X0, X1, X2, Y = WN.create_placeholders(n_H0, n_W0, n_C0, n_y)
    num = tf.placeholder(tf.int32)
    seed = 2
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
        save_path = saver.restore(sess, './logs/session/model_forloop785.ckpt')
        x0 = X_train[:, :, :, 0].reshape(n_N0, n_H0, n_W0, 1)/64
        x1 = X_train[:, :, :, 1].reshape(n_N0, n_H0, n_W0, 1)/64
        x2 = X_train[:, :, :, 2].reshape(n_N0, n_H0, n_W0, 1)/64
        X0T = X_test[:, :, :, 0].reshape(n_N1, n_H1, n_W1, 1)/64
        X1T = X_test[:, :, :, 1].reshape(n_N1, n_H1, n_W1, 1)/64
        X2T = X_test[:, :, :, 2].reshape(n_N1, n_H1, n_W1, 1)/64
        w1, w2, w3 = 0.5, 0.5, 0.2
        XTrainNorm0, XTrainNorm1, XTrainNorm2 = sess.run([Z0, Z1, Z2], feed_dict={
            X0: x0, X1: x1, X2: x2, num: n_N0})
        ModelNorm0, ModelNorm1, ModelNorm2 = sess.run([Z0, Z1, Z2], feed_dict={
            X0: X0T, X1: X1T, X2: X2T, num: n_N1})
        XTrainNorm = XTrainNorm0 * w1 + XTrainNorm1 * w2 + XTrainNorm2 * w3
        indexSets = []
        for i in range(n_N1):
            testNorm = ModelNorm0[i] * w1 + ModelNorm1[i] * w2 + ModelNorm2[i] * w3
            eig = XTrainNorm - testNorm  # A-B
            eig = np.linalg.norm(eig, axis=1)  # 相似度比较，采用二范数
            eigIndex = np.argsort(eig)  # 相似度排序，输出相似模型的下标
            indexSets.append(eigIndex)
        indexSets = np.array(indexSets)
        print(indexSets.shape)
        print(indexSets[0])
    all = []
    for i in range(int(indexSets.shape[0])):
        rangeL = int(i/10)*100
        index = indexSets[i]-rangeL
        index = ((index[:]<100) & (index[:]>=0))+0
        all.append(index)
    all = np.array(all)
    allcql = []
    allczl = []
    # 计算查全率查准率
    for i in range(all.shape[0]):
        t = 0
        chaquanlv = []
        chazhunlv = []
        for j in range(all.shape[1]):
            t+=all[i][j]
            if (t%10) == 0 and all[i][j]==1:
                tt = t/(j+1)
                chazhunlv.append(tt)
                chaquanlv.append(t/100)
        allczl.append(chazhunlv)
        allcql.append(chaquanlv)
    allczl = np.array(allczl)
    allcql = np.array(allcql)
    a = np.mean(allczl, axis=0).tolist()
    b = np.mean(allcql, axis=0).tolist()
    fa = list([1])
    fb = list([0])
    faa = np.array(fa.extend(a))
    fbb = np.array(fb.extend(b))
    print("mAP:"+str(np.mean(fa)))
    # plot the cost
    plt.plot(fb,fa,'g-s')
    plt.ylim((0,1))
    plt.xlim((0,1))
    plt.ylabel("P")
    plt.xlabel("R")
    new_ticks = np.linspace(0, 1, 11)
    plt.xticks(new_ticks)
    plt.yticks(new_ticks)
    # plt.title("Learning rate =" + str(learning_rate))
    plt.show()
