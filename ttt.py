import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np

if __name__ == '__main__':
    # a = np.array([[0, 0]]);
    # b = np.array([[3, 0]]);
    # c = np.array([[3, 5]]);
    # ss = 0.5 * np.linalg.norm(np.cross((b - a), (c - a)), ord=1)
    # p = (3 + 4 + 5) / 2
    # s = np.sqrt(p * (p - 3) * (p - 4) * (p - 5))
    # print(ss,s)
    X = [[-1, 1, 0],
         [-4, 3, 0],
         [1, 0, 2]]
    X = np.matrix(X)
    # print"------------------下面计算原始矩阵的特征值和特征向量-----------------------"
    eigenvalue,featurevector=np.linalg.eig(X)
    #
    #
    # print"原始矩阵的特征值"
    # print"eigenvalue=",eigenvalue
    # print"featurevector=",featurevector
    print(eigenvalue)
    print(featurevector)
    # -*- coding: utf-8 -*-

    import os


    def file_name(file_dir):
        for root, dirs, files in os.walk(file_dir):
            # print(root)  # 当前目录路径
            print(dirs)  # 当前路径下所有子目录
            break;
        for i in range(len(dirs)):
            model_path = file_dir+"\\"+dirs[i]+"\\"+dirs[i]+".off"
            print(model_path)
        # print(dirs[0])
    if __name__ == '__main__':
        dir = "C:\\Users\97933\Downloads\psb_v1\\benchmark\db\\0"
        file_name(dir)
