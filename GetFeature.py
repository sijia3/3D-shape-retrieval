import os
import time

import h5py  #导入工具包
import numpy as np
import matplotlib.pyplot as plt

import PlotTri
import PlotVoxel
import Tri2Vox
import ReadOff
from GetLabels import getLabels
import H5FileUtils as h5utils


def readH5File(filename, key):       # HDF5的读取
    """
    读取H5文件
    :param file: 文件名字
    :param key: 键
    :return: value: 该键对应的值
    """
    f = h5py.File(filename,'r')   #打开h5文件
    # f.keys()                            #可以查看所有的主键
    value = f[key][:]                    #取出主键为data的所有的键值
    # b = f['labels'][:]
    # print(a, b)
    f.close()
    return value


def getFeature(file_dir):
    """
    获取该文件夹下所有目录下的模型特征
    :param file_dir: 主文件夹目录
    :return: allPics: 模型的三视图特征
    """
    # file_dir = "D:\\Downloads\ModelNet10\ModelNet10\\alltest"
    for root, dirs, files in os.walk(file_dir):
        print(dirs)  # 当前目录路径
        print(files)  # 当前路径下所有子目录
        break;
    allPics = []
    for i in range(len(dirs)):
        modelDir = file_dir+"\\"+dirs[i]
        for mid_root, mid_dirs, mid_files in os.walk(modelDir):
            break;
        for j in range(len(mid_files)):
            print("文件夹："+str(file_dir)+"---第"+str(j+1)+"个开始提取特征")
            print("文件名："+str(mid_files[j]))
            modelFile = modelDir + "\\" + str(mid_files[j])
            verts, faces = ReadOff.readOff(modelFile)
            vox = Tri2Vox.Tri2Vox(verts, faces, 32)
            # PlotVoxel.plot2DVoxel(vox, 64,files[i])
            # PlotVoxel.plotVoxel(vox, 64)
            # PlotTri.plotTri(verts,faces)
            pics = getPics(vox, isInDepth=True)
            # PlotVoxel.plotHotPic(pics, 64, files[i])
            allPics.append(pics)
            print("已完成第"+str(i+1)+"个文件夹的第"+str(j+1)+"个")
    allPics = np.array(allPics)
    return allPics


def getPics(vox, isInDepth = False):
    """
    提取单个模型的三视图特征
    :param vox: 模型的体素化点
    :param isInDepth: 是否提取深层图像
    :return: pics：模型的三视图特征
    """
    pics = []
    for j in range(3):
        index = (j + 1) % 3

        pic = np.zeros((64, 64))
        voxL = vox[:, [j, index]]
        if not isInDepth:
            voxL = np.unique(voxL, axis=0)
        for i in range(voxL.shape[0]):
            x = int(voxL[i][0])
            y = int(voxL[i][1])
            if x>63:
                x = 63
            if y>63:
                y = 63
            pic[x][y] += 1.0
            # pic[x][y] = 0.0           # 浅层图像
        pics.append(pic)
    pics = np.array(pics).transpose(2, 1, 0)
    return pics


def convert_to_one_hot(Y, C):    # 标签转换
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def getFea(dir, filename):
    vox = getFeature(dir)
    labels = getLabels(dir)
    h5utils.writeH5(filename, vox, labels)


if __name__ == '__main__':
    getFea("D:\\ModelNet10\\train", './logs/sijia3/3dModelTrainDBeta_8_2.h5')
    getFea("D:\\ModelNet10\\test", './logs/sijia3/3dModelTestDBeta_8_2.h5')

