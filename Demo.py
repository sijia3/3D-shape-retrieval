import tensorflow as tf
import GetFeature as GF
import CNNTrain as CT
def loadDataSets():
    XTrain = GF.readH5File('./datasets/train_model.h5', 'data')
    YLabels = GF.readH5File('./datasets/train_labels.h5', 'labels')
    YLabels = YLabels.reshape(1, len(YLabels)).astype('int64')
    YLabels = GF.convert_to_one_hot(YLabels, 10).T
    XTest = GF.readH5File('./datasets/test_model.h5', 'data')
    YTestLabels = GF.readH5File('./datasets/test_labels.h5', 'labels')
    YTestLabels = YTestLabels.reshape(1, len(YTestLabels)).astype('int64')
    YTestLabels = GF.convert_to_one_hot(YTestLabels, 10).T
    return XTrain, YLabels, XTest, YTestLabels


if __name__ == '__main__':
    # 特征提取
    # X_train, Y_train, X_test, Y_test = loadDataSets()
    parameters = {}
    # parameters = CT.initialize_parameters()
    # (m, n_H0, n_W0, n_C0) = X_train.shape
    # n_y = Y_train.shape[1]
    # X, Y = CT.create_placeholders(n_H0, n_W0, n_C0, n_y)
    # Z3 = CT.forward_propagation(X, parameters)
    # 直接加载持久化的图
    saver = tf.train.import_meta_graph('logs/tt.ckpt.meta')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, 'logs/tt.ckpt')

        # 通过张量的名称来获取张量
        parameters['W1'] = tf.get_default_graph().get_tensor_by_name('W1:0')
        parameters['W2'] = tf.get_default_graph().get_tensor_by_name('W2:0')
        print(parameters['W1'].eval())
        # parameters["W1"] = tf.get_default_graph().get_tensor_by_name('W1:0')
        # parameters["W2"] = tf.get_default_graph().get_tensor_by_name('W2:0')

        # predict_op = tf.argmax(Z3, 1)  # 返回每行最大值的索引值
        # # print(Z3.eval({X: X_train}))
        # tt = predict_op.eval({X: X_test})
        # print(tt)
        # correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        # print(correct_prediction.eval({X: X_test, Y: Y_test}))
        # Calculate accuracy on the test set
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # # print(accuracy.eval())
        # train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        # test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        # print("Train Accuracy:", train_accuracy)
        # print("Test Accuracy:", test_accuracy)