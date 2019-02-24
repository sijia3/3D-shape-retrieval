import numpy as np
import os
import glob
from ModelList import models
import h5py  #导入工具包


# def writeH5(filename, labels=[]):
#     print("开始写入文件》》》》》")
#     f = h5py.File(filename,'w')   # 创建一个h5文件，文件指针是f
#     f['labels'] = labels           # 将数据写入文件的主键labels下面
#     print("写入结束《《《《《《")
#     f.close()

def getLabels(fileDir):
    """
    获取该文件夹下各目录的模型标签
    :param fileDir: 总文件夹目录
    :return: 总模型标签
    """
    # fileDir = "D:\\trainmodels"
    yLabels = []
    for root, dirs, files in os.walk(fileDir):
        print(dirs)  # 当前目录路径
        for i in range(len(dirs)):
            model_dir = str(fileDir)+"\\"+str(dirs[i])+"\\"+"*.off"
            path_file_number = glob.glob(pathname=model_dir)
            model_num = int(len(path_file_number))        # 文件夹下模型的个数
            model_label = np.array((i)).repeat(model_num)
            yLabels = np.append(yLabels, model_label)
        break;
    # yLabels = yLabels.reshape()
    return yLabels

if __name__ == '__main__':
    dir = "D:\\trainmodels"
    # t = []
    # for root, dirs, files in os.walk(dir):
    #     print(dirs)  # 当前目录路径
    #     for i in range(len(dirs)):      # 遍历模型的目录
    #         model_dir = dir+"\\"+dirs[i]+"\\"+"*.off"
    #         path_file_number = glob.glob(pathname=model_dir)
    #         model_num = (len(path_file_number))        # 文件夹下模型的个数
    #         model_label = np.array((i)).repeat(model_num)
    #         t = np.append(t, model_label)
    #     break;
    YLabels = getLabels(dir)
    # writeH5('./datasets/test_labels.h5', YLabels)

