import numpy as np
from PIL import Image
import os
import CNNUtils as CU
import H5FileUtils as h5utils
import matplotlib.pyplot as plt


def pic2array(path):
    allpics = []
    file_dir = path          # 'C:/Users/97933/Desktop/AA/testmodel/'
    for root, dirs, files in os.walk(file_dir):
        # print(root)  # 当前目录路径
        print(files)  # 当前路径下所有子目录
        break;
    for i in range(len(dirs)):
        modelname = dirs[i]
        modeldir = file_dir+modelname+'/'
        for modelroot, modeldirs, modelfiles in os.walk(modeldir):
            # print(root)  # 当前目录路径
            # print(files)  # 当前路径下所有子目录
            break;
        for j in range(len(modeldirs)):
            modelnameone = modeldirs[j]
            modeldirone = modeldir+modelnameone+'/'
            for fileroot, filedirs, filefiles in os.walk(modeldirone):
                # print(files)  # 当前路径下所有子目录
                break;
            modelpics = []
            for k in range(len(filefiles)-1):
                filename = filefiles[k]
                fillname = modeldirone+filename
                print(fillname)
                img = Image.open(fillname)
                img = img.resize((64, 64)).convert('L')
                # image_arr = np.array(img)
                image_arr = np.array(img).reshape((64, 64, 1))
                # image_arr = np.array(img).reshape((64, 64))
                # allpics.append(image_arr)
                # modelpics.append(image_arr)
            # modelpics = np.array(modelpics).transpose(1,2,0)
                modelpics = np.array(image_arr)
                allpics.append(modelpics)
    allpics = np.array(allpics)
    return allpics




if __name__ == '__main__':

    testarr = pic2array('E:/3d_Retrival_System_beta2/3D-shape-retrival/testmodel/')
    trainarr = pic2array('E:/3d_Retrival_System_beta2/3D-shape-retrival/trainmodel/')
    h5utils.writeH5File('./logs/3dNoColorPic64Train82_5.h5', trainarr)
    h5utils.writeH5File('./logs/3dNoColorPic64Test82_5.h5', testarr)




# img = Image.open('C:/Users/97933/Desktop/AA/testmodel/bathtub/bathtub_0015/bathtub_0015_1.jpg').convert('L')
# img = img.resize((64, 64))
# img.show()
# image_arr = np.array(img).reshape((64,64,1))
