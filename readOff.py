import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np
import os
from matplotlib import cm
def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

if __name__ == '__main__':
    file_dir = "C:\\Users\97933\Downloads\psb_v1\\benchmark\db\\7"
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        print(dirs)  # 当前路径下所有子目录
        break;
    for i in range(len(dirs)):
        if i > 10:
            break
        model_path = file_dir + "\\" + dirs[i] + "\\" + dirs[i] + ".off"
        print(model_path)
        with open(model_path, 'r') as file:
            if 'OFF' != file.readline().strip():
                raise ('Not a valid OFF header')
            n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
            verts = np.array([[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)])
            faces = np.array([[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)])
            x = verts[:, 0]
            y = verts[:, 1]
            z = verts[:, 2]
            verts_mean = np.mean(verts, axis=0)
            new_verts = verts - verts_mean

            tt = new_verts.T
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
            R = np.array([R0,R1,R2])
            # R = R.T
            # P = np.dot(new_verts, R)
            P = np.dot(R, new_verts.T).T
            # 统一缩放
            s = np.max(np.abs(P))
            P = P/s
            # plot
            x = P[:, 0]
            y = P[:, 1]
            z = P[:, 2]
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_trisurf(x, y, z, triangles=faces, cmap=cm.coolwarm, linewidth=0.2)
            plt.show()

            x = verts[:, 0]
            y = verts[:, 1]
            z = verts[:, 2]
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_trisurf(x, y, z, triangles=faces, cmap='viridis', linewidth=0.2)
            plt.show()

