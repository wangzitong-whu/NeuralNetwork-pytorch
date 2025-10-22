# DeepLearning复现经典的神经网络算法（NeuralNetwork-pytorch）
**保研完毕，从零开始学习深度学习，此仓库当作学习记录，有时间就复现论文，欢迎讨论**
## 主线1.LeNet网络分类手写数字  
<img width="1055" height="276" alt="image" src="https://github.com/user-attachments/assets/73333ebe-44a1-4618-bad5-aa1763ff495a" />  
LeNet作为首个提出的神经网络，主要由卷积层，池化层，全连接层组成，中间穿插激活函数(本项目使用RELU)，最后实现识别手写数字。  
MNIST数据集的一张图片的数据类型为[1,28,28]，代码中第一层卷积没有改变图片的pixel，只改变了通道数（卷积核数），之后通过池化层（（28-2）/2）+1 == 14，来对图片pixel进行降维，减少参数量，之后就是完全按照LeNet的流程图设计网络结构。  

## 主线2.VGGNet16网络图像分类  
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/7cf1905e-e84c-4d13-9540-d41733f255cf" />  

VGGNet16主要由3X3卷积（步长1，填充1，不会改变pixel，会改变通道数量），池化层（pixel/2，减少参数量）和全连接层组成（VGGNet16代码已完成，停更几天，这几天要提交论文手稿）
## 支线1.Pytorch-lighting<br> 





