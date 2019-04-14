import numpy as np
import os
import glob


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
    YLabels = getLabels(dir)

