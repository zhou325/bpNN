## 项目简介
在不采用开源机器学习包仅采用numpy的前提下用python编写包含两层隐藏层的bp神经网络，采用随机梯度下降方式对MNIST数据进行训练并对比两种不同激励函数的效果。

## 数据文件
MNIST来源于http://yann.lecun.com/exdb/mnist/
其中：
- Training set images: train-images-idx3-ubyte
- Training set labels: train-labels-idx1-ubyte
- Test set images: t10k-images-idx3-ubyte
- Test set labels: t10k-labels-idx1-ubyte

训练样本共60000个，测试样本共10000个。

## 数据预处理
采用网络代码 `data_util.py` 对将每个样本的28*28的像素展开为一维行向量，并生成对应的目标向量。

为方便后续研究，通过 `data_processing.py` 对目标向量作onehot编码处理，并将各数据集分别保存在对应的“.txt”文件中，便于在程序中加载。

## bp神经网络
`bpNN.py` 包含了运用两种不同的激励函数编写bp神经网络的代码。

在运行此代码前，需先运行`python data_processing.py`将数据以正确的形式储存。

此后运行`python bpNN.py`可查看两种激励函数的错误分类率。

## 对比报告
`对比报告.md` 中展示了两种不同激励函数的训练效果.
