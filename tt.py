import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np
import os
from matplotlib import cm
import ReadOff
import Tri2Vox
import PlotTri

if __name__ == '__main__':
    # vox = np.array([[1,1,1]])
    # plotVoxel(vox=vox, boxSize=64)
    #
    file_dir = "./m0.off"
    verts, faces = ReadOff.readOffWithoutPca(file_dir)
    PlotTri.plotTri(verts, faces, file_dir)
    # vox = Tri2Vox.Tri2Vox(verts, faces, 32)
    # pics = []
    # for j in range(3):
    #     index = (j+1)%3
    #     pic = np.zeros((64,64))
    #     voxL = vox[:,[j,index]]
    #     voxL = np.unique(voxL, axis=0)
    #     for i in range(voxL.shape[0]):
    #         x = int(voxL[i][0])-1
    #         y = int(voxL[i][1])-1
    #         pic[x][y] += 1.0;
    #     pics.append(pic)
