import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np
import os
from matplotlib import cm
import ReadOff
import Tri2Vox
import GetFeature as GF


def plotVoxel(vox, boxSize):
    """
    绘制三维模型散点图
    :param vox: 三维模型体素点集合
    :param boxSize: 模型大小
    :return:
    """
    # plot
    x = vox[:, 0]
    y = vox[:, 1]
    z = vox[:, 2]
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, cmap='Greens')
    ax.set_ylim(0, boxSize)
    ax.set_xlim(0, boxSize)
    plt.show()


def plot2DVoxel(vox, voxSize, name):
    """
    绘制三维模型三视图
    :param vox: 模型点
    :param voxSize: 模型大小
    :param name: 模型名字
    :return:
    """
    plt.subplot(2, 2, 1)
    plt.xlim(0, voxSize)
    plt.ylim(0, voxSize)
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.imshow()
    plt.scatter(vox[:, 0], vox[:, 1])
    #
    plt.subplot(2, 2, 2)
    plt.xlim(0, voxSize)
    plt.ylim(0, voxSize)
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.scatter(vox[:, 1], vox[:, 2])

    plt.subplot(2, 2, 3)
    plt.xlim(0, voxSize)
    plt.ylim(0, voxSize)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.scatter(vox[:, 0], vox[:, 2])
    # plt.savefig('./pic/'+str(name)+'.png')
    plt.show()


def plotHotPic(pic,voxSize,name=""):
    """
    绘制三维模型的三视图特征热力图
    :param pic: 三视图的深度特征图像
    :param voxSize: 模型尺寸
    :param name: 模型名字
    :return:
    """
    plt.subplot(2, 2, 1)
    plt.xlim(0, voxSize)
    plt.ylim(0, voxSize)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.imshow(pic[0])

    plt.subplot(2, 2, 2)
    plt.xlim(0, voxSize)
    plt.ylim(0, voxSize)
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.imshow(np.transpose(pics[1]))

    plt.subplot(2, 2, 3)
    plt.xlim(0, voxSize)
    plt.ylim(0, voxSize)
    plt.xlabel("Z")
    plt.ylabel("X")
    plt.imshow(pic[2])
    # plt.savefig('./pic/'+str(name)+'.png')
    # plt.show()


if __name__ == '__main__':
    file_dir = "./model/airplane_0007.off"
    verts, faces = ReadOff.readOffWithoutPca(file_dir)
    vox = Tri2Vox.Tri2Vox(verts, faces, 64)
    pics = []
    for j in range(3):
        index = (j + 1) % 3
        pic = np.zeros((64, 64))
        voxL = vox[:, [j, index]]
        for i in range(voxL.shape[0]):
            x = int(voxL[i][0])
            y = int(voxL[i][1])
            if x>63:
                x = 63
            if y>63:
                y = 63
            pic[x][y] += 1.0
        pics.append(pic)
    # 绘制以上得到的三个方向的深度图像
    plotHotPic(pics, 64)

    # plotVoxel(vox, 64)
    # plot2DVoxel(vox, 64, file_dir)
