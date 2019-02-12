import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np
import os
from matplotlib import cm


def plotTri(planes,faces):
    # plot
    x = planes[:, 0]
    y = planes[:, 1]
    z = planes[:, 2]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, z, triangles=faces, cmap=cm.coolwarm, linewidth=0.2)
    # plt.show()

