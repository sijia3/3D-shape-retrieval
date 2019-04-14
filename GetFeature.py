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
            vox = Tri2Vox.Tri2Vox(verts, faces, 64)
            pics = getPics(vox, isInDepth=True)
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
        pics.append(pic)
    pics = np.array(pics).transpose(2, 1, 0)
    return pics



def getFea(dir, filename):
    """
    提取所有模型的特征和标记，写入到文件中
    :param dir: 模型所在的总文件夹
    :param filename: 特征所保存的文件名字
    :return:
    """
    vox = getFeature(dir)
    labels = getLabels(dir)
    h5utils.writeDataAndLabels(filename, vox, labels)


if __name__ == '__main__':
    # getFea("D:\\ModelNet10_PR\\train", './datasets/3dModelTrain_PR.h5')
    getFea("D:\\ModelNet10_PR\\test", './datasets/3dModelTest_PR_other.h5')

