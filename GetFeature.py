import os

import h5py  #导入工具包
import numpy as np

import PlotTri
import PlotVoxel
import Tri2Vox
import ReadOff
                         #关闭文件
#
#
# #HDF5的读取：
# f = h5py.File('HDF5_FILE.h5','r')   #打开h5文件
# f.keys()                            #可以查看所有的主键
# a = f['data'][:]                    #取出主键为data的所有的键值
# f.close()



def getFeature():
    file_dir = "D:\\Downloads\ModelNet10\ModelNet10\\alltest"
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        print(files)  # 当前路径下所有子目录
        break;
    allPics = []
    for i in range(len(files)):
        file = file_dir + "\\" + files[i]
        verts, faces = ReadOff.readOff(file)
        vox = Tri2Vox.Tri2Vox(verts, faces, 32)
        PlotVoxel.plot2DVoxel(vox, 64,files[i])
        PlotVoxel.plotVoxel(vox, 64)
        # PlotTri.plotTri(verts,faces)
        pics = getPics(vox)
        allPics.append(pics)
        print("已完成第"+str(i+1)+"个")
    allPics = np.array(allPics)
    return allPics


def getPics(vox):
    pics = []
    for j in range(3):
        index = (j + 1) % 3
        pic = np.zeros((64, 64))
        voxL = vox[:, [j, index]]
        voxL = np.unique(voxL, axis=0)
        for i in range(voxL.shape[0]):
            x = int(voxL[i][0])
            y = int(voxL[i][1])
            if x>63:
                x = 63
            if y>63:
                y = 63
            pic[x][y] += 1.0;
        pics.append(pic)
    pics = np.array(pics)
    return pics


def writeH5(filename, voxs, labels):
    # HDF5的写入：
    # (h, w) = voxs.shape
    # imgData = np.ones((1000,64,64,3))
    f = h5py.File(filename,'w')   # 创建一个h5文件，文件指针是f
    f['data'] = voxs                 # 将数据写入文件的主键data下面
    f['labels'] = range(len(labels))            # 将数据写入文件的主键labels下面
    f.close()


if __name__ == '__main__':
    a = getFeature();