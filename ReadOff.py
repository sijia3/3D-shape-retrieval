import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np
import os
from matplotlib import cm

def readOffWithoutPca(filename):
    return readOff(filename, isTranspose=False)

def readOff(filename, isTranspose=False):
    """
    函数功能：读取文件内容并预处理模型
    :param filename: 文件名字
    :param isTranspose: 是否使用PCA
    :return: new_verts, faces：处理好后的模型点和面片
    """
    with open(filename, 'r') as file:
        firstLine = file.readline().strip()
        if 'OFF' != firstLine[0:3]:
            raise ('Not a valid OFF header')
        if 'OFF' != firstLine:
            imf = firstLine[3:]
            n_verts, n_faces, n_dontknow = tuple([int(s) for s in imf.split(' ')])
        else:
            n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = np.array([[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)])
        faces = np.array([[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)])

        # Step1: 平移 根据面片权值加权质心
        Area = 0
        S = 0   # 加权面积
        for i in range(faces.shape[0]):    # int(faces.shape[0])
            pointId = faces[i,:]
            dotA = verts[faces[i][0]].reshape(1, 3)
            dotB = verts[faces[i][1]].reshape(1, 3)
            dotC = verts[faces[i][2]].reshape(1, 3)
            # 边长
            AB = np.linalg.norm(dotB - dotA, axis=1, keepdims=True)
            BC = np.linalg.norm(dotC - dotB, axis=1, keepdims=True)
            AC = np.linalg.norm(dotC - dotA, axis=1, keepdims=True)

            p = (AB + BC + AC) / 2
            # print(i,AB,BC,AC,p)
            s = np.sqrt(p * np.abs(p - AB) * np.abs(p - AC) * np.abs(p - AC))  # 该面片的面积
            # todo 计算面片质心
            planeGri = np.mean(verts[pointId], axis=0)
            S += s * planeGri;          # sum（gi * Si）
            Area += s
        p = S/Area
        new_verts = verts-p
        # verts_mean = np.mean(verts, axis=0)
        # new_verts = verts - verts_mean

        # Step2: 旋转(PCA)
        if isTranspose:
            PTP = np.dot(new_verts.T, new_verts)
            Area = 0
            for i in range(faces.shape[0]):
                dotA = new_verts[faces[i][0]].reshape(1, 3)
                dotB = new_verts[faces[i][1]].reshape(1, 3)
                dotC = new_verts[faces[i][2]].reshape(1, 3)
                # 边长
                AB = np.linalg.norm(dotB - dotA, axis=1, keepdims=True)
                BC = np.linalg.norm(dotC - dotB, axis=1, keepdims=True)
                AC = np.linalg.norm(dotC - dotA, axis=1, keepdims=True)
                p = (AB + BC + AC) / 2
                s = np.sqrt(p * (p - AB) * (p - AC) * (p - AC))
                Area += s
            M = PTP / Area
            eigenvalue, featurevector = np.linalg.eig(M)
            param_sort = np.argsort(-eigenvalue)  # 特整向量的排序
            R = featurevector[:, param_sort]
            R0 = R[:, 0] / np.linalg.norm(R[:, 0])
            R1 = R[:, 1] / np.linalg.norm(R[:, 1])
            R2 = R[:, 2] / np.linalg.norm(R[:, 2])
            # 构造旋转矩阵
            R = np.array([R0, R1, R2])
            P = np.dot(R, new_verts.T).T
            new_verts = P

        # Step3: 统一缩放
        s = np.max(np.abs(new_verts))
        new_verts = new_verts / s
        print(str(filename)+"预处理完成。")
        return new_verts, faces



if __name__ == '__main__':
    file_name = "./model/m0.off"
    points, faces = readOff(file_name)