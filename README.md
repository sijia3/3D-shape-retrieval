3D-shape-retrieval(三维模型检索)    
---
##### 如有任何建议或疑问，可与我交流。
##### 联系方式：(QQ)979337189   (微信) qq-979337189

## 快速使用
为方便读者有更直观的认识，已将本人训练好的模型特征数据上传，只需按照以下步骤进行解压   
Step1: 将**logs**下的feaSets.rar中的文件解压出来，！！放到**logs**文件夹中！！   
Step2: 将**session**/**vox**/下的session.rar中的文件解压出来，！！放到**session**/**vox**/文件夹中！！  
Step3: 打开VoxPicTrainPredict.py文件，执行main方法即可得到模型预测结果。
   
---

## 算法模型(简略介绍)
1. **基于三视图加权的卷积神经网络**。针对视图特征的提取，本章通过体素化模型来得到深层的三视图特征，并针对三视图特征的表达能力不同，通过视图加权使模型的检索率得到明显的上升。
2. **基于多视图的卷积神经网络**。考虑到二维图像的深度学习算法更加成熟和稳定，因此从三维模型渲染后，通过摆放不同方位地虚拟摄像机来得到多角度图像出发，并通过ViewPooling聚合使多视图融合，得到新特征向量，再通过深度学习进行构建本章的神经网络。


---
## 项目主要内容：
1. 主要功能介绍(每个文件都有单独的main函数(测试用))    

    2. ReadOff文件。读取.off后缀的模型文件，做模型预处理。   
         
    2. Tri2Vox文件。对模型进行体素化。       
    
    2. GetFeature文件。对体素化后的模型进行提取特征。    
    
    2. GetLabels文件。提取主文件下模型的标签。(监督式学习)    
    
    2. CNNUtils文件。神经网络训练的工具。     
    
    2. CNNVixPicTrain文件。对体素化后的模型三视图进行训练。    
    
    2. H5FileUtils文件。针对特征数据存储读写的h5utils工具。     
    
    2. VoxPicTrainPredict文件。对训练好的模型，针对单个模型，进行预测。     
    
    2. CNNPicTrain文件。对多视角视图特征进行训练。     
    
    2. PlotTri，PlotVoxel文件。对模型或模型特征进行可视化输出。    
    
2. **model**文件夹。用于存放测试模型。模型选自ModelNet或PSB数据集。有兴趣可到官网浏览或下载。链接: [普林斯大学modelnet模型库](http://modelnet.cs.princeton.edu/)
3. **session**文件夹。用于存放模型训练参数的session。可用于保存训练中得到的较好参数，并在模型预测时可加载session得到模型特征。
4. **logs**文件夹。用于存放模型特征数据。


---
### 使用环境
1. 语言：python3.6
2. 框架：tensorflow  
可以选择在本地搭建环境来跑程序  
也可选择Google提供的工具Colaboratory进行搭建学习。教程: [如何使用Colaboratory](https://www.jianshu.com/p/e6f1058614c0?from=timeline&isappinstalled=0)
---


### 实现步骤(介绍基于三视图加权的卷积神经网络)
Step1:  读取模型并进行预处理(使用ReadOff文件)   
Step2:  进行模型体素化(使用Tri2Vox文件)  
Step3:  提取模型特征，并存放到特定位置。(使用GetFeature文件)  
Step4:  训练模型，保存较好的session。 (使用CNNVoxPicTrain文件)   
Step5:  预测模型。加载模型session，并可对未知模型进行预测。(使用VoxPicTrainPredict文件)      
三维模型检索系统原型使用matlab开发，对接python。有兴趣可尝试一下。

---
### 使用本项目构建网络(可集成用户自定义方法)        
 如果需要训练自己的训练集，那需要根据以下步骤:    
 Step1：定义模型的组织结构（文件结构尽量保持一致，代码可复用）。         
![模型组织结构](https://github.com/sijia3/3D-shape-retrieval/blob/master/pic/dir.png)    
 Step2: 执行GetFeature.py文件的main方法，获取训练集和测试集的h5文件。       
 Step3: 执行CNNVoxPicTrain.py文件的mian方法，对Step2获取的h5文件传入，设置参数，执行main函数，进行深度学习。       
 
 PS: (1)对于用户定义的模型组织结构不与本文相同，那需要重写getFeature()函数。    
     (2)模型的类型不是off文件，替换getFeature()函数的verts, faces = ReadOff.readOff(modelFile)即可，只需返回参数内容相同即可。  
     (3)集成自定义体素化方法，替换getFeature()函数的vox = Tri2Vox.Tri2Vox(verts, faces, 64)成个人实现的体素化方法，只需返回参数内容相同即可。
     (4)对于(2)(3)的参数内容可自己在相对应的函数进行测试。不懂可找我哈哈。  

---