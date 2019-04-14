import h5py


def readDataAndLabels(filename):       # HDF5的读取
    """
    读取H5文件
    :param filename: 文件名字
    :return: data, labels : 所有模型三视图特征，所有模型的特征
    """
    f = h5py.File(filename,'r')   # 打开h5文件
    data = f['data'][:]           # 取出主键为data的所有的键值 b = f['labels'][:]
    labels = f['labels'][:]
    f.close()
    return data, labels

def writeDataAndLabels(filename, voxs, labels):
    """
    将np数组写进H5文件中
    :param filename: 需写入文件名
    :param voxs: 体素化点
    :param labels: 标签
    :return: 无
    """
    print("开始写入文件》》》》》")
    f = h5py.File(filename,'w')   # 创建一个h5文件，文件指针是f
    f['data'] = voxs                 # 将数据写入文件的主键data下面
    f['labels'] = labels           # 将数据写入文件的主键labels下面
    print("写入结束《《《《《《")
    f.close()

def writeData(filename, voxs):
    """
    将体素点写进H5文件中
    :param filename: 需写入文件名
    :param voxs: 体素化点
    :param labels: 标签
    :return: 无
    """
    print("开始写入文件》》》》》")
    f = h5py.File(filename,'w')   # 创建一个h5文件，文件指针是f
    f['data'] = voxs                 # 将数据写入文件的主键data下面
    print("写入结束《《《《《《")
    f.close()

def readData(filename):
    """
    读取h5文件中的data字段
    :param filename: 需写入文件名
    :return: 无
    """
    f = h5py.File(filename,'r')   #打开h5文件
    data = f['data'][:]                    #取出主键为data的所有的键值 b = f['labels'][:]
    f.close()
    return data
