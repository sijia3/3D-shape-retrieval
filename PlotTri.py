import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np
import os
from matplotlib import cm

import ReadOff


def plotTri(planes,faces):
    # plot
    # x = (planes[:, 0]+1)*32
    # y = (planes[:, 1]+1)*32
    # z = (planes[:, 2]+1)*32
    x = planes[:, 0]
    y = planes[:, 1]
    z = planes[:, 2]
    fig = plt.figure()

    ax = plt.axes(projection='3d')
    # plt.subplot(2, 2, 1)
    # ax.set_ylim(0, 64)
    # ax.set_xlim(0, 64)
    ax.plot_trisurf(x, y, z, triangles=faces, cmap=cm.coolwarm, linewidth=0.2)

    plt.show()

if __name__ == '__main__':
    # vox = np.array([[1,1,1]])
    # plotVoxel(vox=vox, boxSize=64)
    #
    file_dir = "./$RAXIQBV.off"
    verts, faces = ReadOff.readOff(file_dir)
    plotTri(verts, faces)
