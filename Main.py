import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np
import CNNUtils as CU
import CNNTrain as CT
import GetFeature as GF
import GetLabels as GL
import H5FileUtils as h5utils
import Predict as PD

def getFea(dir, filename):
    """
    获取特征并保存到h5文件中
    :param dir: 主目录
    :param filename: 要保存的文件名
    :return: void：空
    """
    vox = GF.getFeature(dir)
    labels = GL.getLabels(dir)
    h5utils.writeH5(filename, vox, labels)

def cnnTrain():
    trainFile = './datasets/3dModelTrainBeta1.h5'
    testFile = './datasets/3dModelTestBeta1.h5'
    XTrain, YTrain, XTest, YTest = CU.loadDataSets(trainFile, testFile)
    parameters = CT.model(XTrain, YTrain, XTest, YTest, num_epochs=500, save_session=True,
                          save_file='./logs/modelBeta1.ckpt')
    return parameters


# 开始函数
if __name__ == '__main__':
    # getFea("D:\\trainmodels", './datasets/3dModelTrainBeta1.h5')
    # getFea("D:\\testmodel10", './datasets/3dModelTestBeta1.h5')
    # data, labels = h5utils.readH5File('./datasets/tt.h5')
    # para = cnnTrain()
    PD.predict('./model/bed_0459.off')