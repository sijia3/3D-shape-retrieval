# 3D-shape-retrieval
 问题:三维模型的三视图有点问题，需做调整
 
#
ModelNet选型：
1. airplane(飞机)
2. bed(床)
3. bench(长凳)
4. bookself(书架)
5. bottle(瓶子)
6. bowl(碗)
7. car(车)
8. chair(椅子)
9. cup(杯)
10. door(门)
11. person(人)
12. sofa(沙发)
13. stairs(楼梯)
14. tables(桌子)
15. toilet(盥洗室)
16. bathtub(浴缸)

17. lamp(台灯)  替换stairs

#
2019-02-21
问题：
1. 如何保存训练时的session(已解决)
2. 如何改进目前的模型，提高检索率
3. 如何确定模型检索率（查全率跟查准率）(通过softmax之前的输入，进行归一化，做相似度计算)
4. 是否需要增加训练模型的种类和数量
5. 是否有新的特征提取算法可以进行深度学习
6. 深度学习模型建议使用tensorflow（学习）
7. 原型采用的技术定向（个人偏向于web端系统）
8. 分析作图
9. 程序注释啊啊啊啊啊
10. 精简函数，每个函数都要编写自身的测试用例
11. 分别做modelNet10和modelNet40的测试