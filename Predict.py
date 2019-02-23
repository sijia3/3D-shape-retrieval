import tensorflow as tf
import numpy as np
import GetFeature as GF
import CNNTrain as CT
import ReadOff
import Tri2Vox
import GetFeature as GF
from ModelList import model

if __name__ == '__main__':
    # 特征提取
    verts, faces = ReadOff.readOff('./model/sofa_0020.off')
    vox = Tri2Vox.Tri2Vox(verts, faces, 32)
    pics1 = GF.getPics(vox, isInDepth=True)

    # verts, faces = ReadOff.readOff('./model/bed_0459.off')
    # vox = Tri2Vox.Tri2Vox(verts, faces, 32)
    # pics2 = GF.getPics(vox, isInDepth=True)
    test_pics = np.array([pics1])

    X_train, Y_train, X_test, Y_test = CT.loadDataSets()
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    X, Y = CT.create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = CT.initialize_parameters()
    Z3 = CT.forward_propagation(X, parameters)
    normalize = tf.nn.l2_normalize(Z3, axis=1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 加载文件中的参数数据，会根据name加载数据并保存到变量W和b中
        save_path = saver.restore(sess, 'logs/model.ckpt')

        # ZZ = Z3.eval({X: test_pics})
        # print(ZZ)
        # 归一化
        XTrainNorm = normalize.eval({X: X_train})         # A
        ModelNorm = normalize.eval({X: test_pics})        # B
        eig = XTrainNorm-ModelNorm                        # A-B
        eig = np.linalg.norm(eig, axis=1)
        eigIndex = np.argsort(eig)
        # eigIndex150 = eigIndex[0:151]
        # print((eigIndex150 < 151).sum())
        predict_op = tf.argmax(ModelNorm, 1)  # 返回每行最大值的索引值
        tt = predict_op.eval({X: test_pics})
        print("预测为"+model[tt[0]])
        # print("预测为"+model[tt[1]])

        # correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        # print(correct_prediction.eval({X: X_test[1:3,], Y: Y_test[1:3,]}))
        # Calculate accuracy on the test set
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # print(accuracy.eval())
        # train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        # test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        # print("Train Accuracy:", train_accuracy)
        # print("Test Accuracy:", test_accuracy)