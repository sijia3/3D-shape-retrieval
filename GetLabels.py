import numpy as np
import os
import glob
from ModelDict import model


def getLabels(fileDir):
    # dir = "D:\\testmodels"
    YLabels = []
    for root, dirs, files in os.walk(fileDir):
        print(dirs)  # 当前目录路径
        for i in range(len(dirs)):
            model_dir = dir+"\\"+dirs[i]+"\\"+"*.off"
            path_file_number = glob.glob(pathname=model_dir)
            model_num = (len(path_file_number))        # 文件夹下模型的个数
            model_label = np.array((i)).repeat(model_num)
            t = np.append(t, model_label)
        break;
    return YLabels

if __name__ == '__main__':
    dir = "D:\\testmodels"
    t = []
    for root, dirs, files in os.walk(dir):
        print(dirs)  # 当前目录路径
        for i in range(len(dirs)):
            model_dir = dir+"\\"+dirs[i]+"\\"+"*.off"
            path_file_number = glob.glob(pathname=model_dir)
            model_num = (len(path_file_number))        # 文件夹下模型的个数
            model_label = np.array((i)).repeat(model_num)
            t = np.append(t, model_label)
        break;
    # return t

