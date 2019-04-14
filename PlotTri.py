import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np
import os
from matplotlib import cm

import ReadOff


def plotTri(planes, faces, filename, size=64):
    """
    三维模型绘制
    :param planes: 点
    :param faces: 面片
    :param filename: 文件名
    :param size: 坐标轴尺寸
    :return: void：空
    """
    # plot
    x = (planes[:, 0]+1)*size/2
    y = (planes[:, 1]+1)*size/2
    z = (planes[:, 2]+1)*size/2
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_ylim(0, size)
    ax.set_xlim(0, size)
    ax.plot_trisurf(x, y, z, triangles=faces, cmap=cm.coolwarm, linewidth=0.2)
    ax.set_title(filename)
    plt.show()

if __name__ == '__main__':
    file_dir = "./model/table_0002.off"
    verts, faces = ReadOff.readOff(file_dir)         # 模型预处理
    plotTri(verts, faces,file_dir)
