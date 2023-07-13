配置环境 conda env create -f fer-master.yaml

训练代码 命令行运行 结果保存在checkpoints\Resnet文件夹下面 包括损失 精度曲线 模型文件，如果训练跑完还会记录Excel文件

python train.py network=Resnet name=Resnet

检验代码 命令行运行 !!!注意里面的权重加载路径要自己修改

python evaluate.py network=Resnet 

训练运行截图![image-20230711221056419](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230711221056419.png)

![image-20230711221132514](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230711221132514.png)

测试运行截图:

![image-20230711221159580](C:\Users\ASUS\AppData\Roaming\Typora\typora-user-images\image-20230711221159580.png)
