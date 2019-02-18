import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np
import os
from matplotlib import cm
def readOffWithoutPca(file):
    return readOff(file, isTranspose=False)


def readOff(file, isTranspose=False):
    with open(file, 'r') as file:
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

        # 平移 todo 重做（根据面片权值）
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

        if isTranspose:                # 是否做PCA旋转
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
            # 统一缩放
            s = np.max(np.abs(P))
            P = P / s
            return P, faces
        # 统一缩放
        s = np.max(np.abs(new_verts))
        new_verts = new_verts / s
        # P = P / s
        print(str(file)+"预处理完成。。。。")
        return new_verts, faces

def mathArea(face):            # 暂不能使用
    dotA = verts[face[0]].reshape(1, 3)
    dotB = verts[face[1]].reshape(1, 3)
    dotC = verts[face[2]].reshape(1, 3)
    # 边长
    AB = np.linalg.norm(dotB - dotA, axis=1, keepdims=True)
    BC = np.linalg.norm(dotC - dotB, axis=1, keepdims=True)
    AC = np.linalg.norm(dotC - dotA, axis=1, keepdims=True)
    p = (AB + BC + AC) / 2
    s = np.sqrt(p * (p - AB) * (p - AC) * (p - AC))  # 该面片的面积
    return s

if __name__ == '__main__':
    # file_dir = "C:\\Users\97933\Downloads\psb_v1\\benchmark\db\\15"
    file_dir = "C:\\Users\97933\Downloads\ModelNet40\ModelNet40\dresser\\train"
    # file_dir = "D:\\testmodels"
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        print(files)  # 当前路径下所有子目录
        break;
    for i in range(len(files)):
        # if i > 1:
        #     break
        # model_path = file_dir + "\\" + dirs[i] + "\\" + dirs[i] + ".off"
        model_path = file_dir + "\\" + files[i]
        print(model_path)
        with open(model_path, 'r') as file:
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
            x = verts[:, 0]
            y = verts[:, 1]
            z = verts[:, 2]
            verts_mean = np.mean(verts, axis=0)
            new_verts = verts - verts_mean

            # tt = new_verts.T
            # PTP = np.dot(new_verts.T, new_verts)
            # Area = 0
            # for j in range(faces.shape[0]):
            #     dotA = new_verts[faces[j][0]].reshape(1, 3)
            #     dotB = new_verts[faces[j][1]].reshape(1, 3)
            #     dotC = new_verts[faces[j][2]].reshape(1, 3)
            #     # 边长
            #     AB = np.linalg.norm(dotB - dotA, axis=1, keepdims=True)
            #     BC = np.linalg.norm(dotC - dotB, axis=1, keepdims=True)
            #     AC = np.linalg.norm(dotC - dotA, axis=1, keepdims=True)
            #     p = (AB + BC + AC) / 2
            #     s = np.sqrt(p * (p - AB) * (p - AC) * (p - AC))
            #     Area += s
            # M = PTP / Area
            # eigenvalue, featurevector = np.linalg.eig(M)
            # param_sort = np.argsort(-eigenvalue)  # 特整向量的排序
            # R = featurevector[:, param_sort]
            # R0 = R[:, 0] / np.linalg.norm(R[:, 0])
            # R1 = R[:, 1] / np.linalg.norm(R[:, 1])
            # R2 = R[:, 2] / np.linalg.norm(R[:, 2])
            #
            # # 构造旋转矩阵
            # R = np.array([R0,R1,R2])
            # P = np.dot(R, new_verts.T).T
            # 统一缩放
            s = np.max(np.abs(new_verts))
            new_verts = new_verts / s
            # plot
            # x = P[:, 0]
            #             # y = P[:, 1]
            #             # z = P[:, 2]
            #             # fig = plt.figure()
            #             # # ax = plt.subplot(1,11,i+1)
            #             # ax = plt.axes(projection='3d')
            #             # ax.plot_trisurf(x, y, z, triangles=faces, cmap=cm.coolwarm, linewidth=0.2)
            #             # plt.show()

            x = new_verts[:, 0]
            y = new_verts[:, 1]
            z = new_verts[:, 2]
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_trisurf(x, y, z, triangles=faces, cmap='viridis', linewidth=0.2)
            plt.show()

