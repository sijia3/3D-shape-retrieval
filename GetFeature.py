import os
import time

import h5py  #导入工具包
import numpy as np
import matplotlib.pyplot as plt

import PlotTri
import PlotVoxel
import Tri2Vox
import ReadOff


def readH5File(file):       # HDF5的读取
    f = h5py.File(file,'r')   #打开h5文件
    f.keys()                            #可以查看所有的主键
    a = f['data'][:]                    #取出主键为data的所有的键值
    b = f['labels'][:]
    # print(a, b)
    f.close()
    return a, b


def getFeature(file_dir):
    # file_dir = "D:\\Downloads\ModelNet10\ModelNet10\\alltest"
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        print(files)  # 当前路径下所有子目录
        break;
    allPics = []
    for i in range(len(files)):
        file = file_dir + "\\" + files[i]
        verts, faces = ReadOff.readOff(file)
        vox = Tri2Vox.Tri2Vox(verts, faces, 32)
        # PlotVoxel.plot2DVoxel(vox, 64,files[i])
        # PlotVoxel.plotVoxel(vox, 64)
        # PlotTri.plotTri(verts,faces)
        pics = getPics(vox, isInDepth=True)
        # PlotVoxel.plotHotPic(pics, 64, files[i])
        allPics.append(pics)
        print("已完成第"+str(i+1)+"个")
    allPics = np.array(allPics)
    return allPics


def getPics(vox, isInDepth = False):
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


def writeH5(filename, voxs, labels=[]):
    # labels = np.array([[i for i in range(1, 12)]])
    # HDF5的写入：
    # (h, w) = voxs.shape
    # imgData = np.ones((1000,64,64,3))
    print("开始写入文件》》》》》")
    f = h5py.File(filename,'w')   # 创建一个h5文件，文件指针是f
    f['data'] = voxs                 # 将数据写入文件的主键data下面

    f['labels'] = labels           # 将数据写入文件的主键labels下面
    print("写入结束《《《《《《")
    f.close()

def convert_to_one_hot(Y, C):    # 标签转换
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

if __name__ == '__main__':
    feature = getFeature("D:\\testmodel10");
    labels = np.array([[i for i in range(0, 10)]])   # 1* 10
    labels = np.array([[0,1,2,2,3,4,5,6,7,8,9]])
    writeH5('./datasets/train.h5', feature, labels)
    Y = convert_to_one_hot(labels, 10)


    # startTime = time.time()
    # verts, faces = ReadOff.readOff("./model/bathtub_0001.off")
    # vox = Tri2Vox.Tri2Vox(verts, faces, 32)
    # pics = getPics(vox, isInDepth=True)
    # stopTime = time.time()
    # print("需要花费：", stopTime-startTime)
    # plt.imshow(pics[:, :, 0]


