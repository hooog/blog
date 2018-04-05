---
title: 使用 scikit-learn 识别手写数字介绍机器学习
date: 2018-03-17 17:05:59
password:
top:
categories:
  - Machine learning
tags:
  - scikit-learn 
---
<!--more-->


## 机器学习简介
一般来说，一个学习问题通常会考虑一系列 n 个 样本 数据，然后尝试预测未知数据的属性。 如果每个样本是 多个属性的数据 （比如说是一个多维记录），就说它有许多“属性”，或称 **features(特征)** 。

我们可以将学习问题分为几大类:

- **监督学习** , 其中数据带有一个附加属性，即我们想要预测的结果值。这个问题可以是:
    - **分类** : 样本属于两个或更多个类，我们想从已经标记的数据中学习如何预测未标记数据的类别。 分类问题的一个例子是手写数字识别，其目的是将每个输入向量分配给有限数目的离散类别之一。 我们通常把分类视作监督学习的一个离散形式（区别于连续形式），从有限的类别中，给每个样本贴上正确的标签。
    - **回归** : 如果期望的输出由一个或多个连续变量组成，则该任务称为 回归 。 回归问题的一个例子是预测鲑鱼的长度是其年龄和体重的函数。
- **无监督学习**, 其中训练数据由没有任何相应目标值的一组输入向量x组成。这种问题的目标可能是在数据中发现彼此类似的示例所聚成的组，这种问题称为 **聚类** , 或者，确定输入空间内的数据分布，称为 密度估计 ，又或从高维数据投影数据空间缩小到二维或三维以进行 可视化。

### 训练集和测试集
机器学习是从数据的属性中学习，并将它们应用到新数据的过程。 这就是为什么机器学习中评估算法的普遍实践是把数据分割成 训练集 （我们从中学习数据的属性）和 测试集 （我们测试这些性质）。

## 机器学习小案例
scikit-learn 提供了一些标准数据集，例如 用于分类的 iris 和 digits 数据集 和 波士顿房价回归数据集 .
在下文中，我们从我们的 shell 启动一个 Python 解释器，然后加载 iris 和 digits 数据集。

数据集是一个类似字典的对象，它保存有关数据的所有数据和一些元数据。 该数据存储在 .data 成员中，它是 n_samples, n_features 数组。 在监督问题的情况下，一个或多个响应变量存储在 .target 成员中。

### 识别手写数字源码


```python
print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
```

    Automatically created module for IPython interactive environment
    Classification report for classifier SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False):
                 precision    recall  f1-score   support
    
              0       1.00      0.99      0.99        88
              1       0.99      0.97      0.98        91
              2       0.99      0.99      0.99        86
              3       0.98      0.87      0.92        91
              4       0.99      0.96      0.97        92
              5       0.95      0.97      0.96        91
              6       0.99      0.99      0.99        91
              7       0.96      0.99      0.97        89
              8       0.94      1.00      0.97        88
              9       0.93      0.98      0.95        92
    
    avg / total       0.97      0.97      0.97       899
    
    
    Confusion matrix:
    [[87  0  0  0  1  0  0  0  0  0]
     [ 0 88  1  0  0  0  0  0  1  1]
     [ 0  0 85  1  0  0  0  0  0  0]
     [ 0  0  0 79  0  3  0  4  5  0]
     [ 0  0  0  0 88  0  0  0  0  4]
     [ 0  0  0  0  0 88  1  0  0  2]
     [ 0  1  0  0  0  0 90  0  0  0]
     [ 0  0  0  0  0  1  0 88  0  0]
     [ 0  0  0  0  0  0  0  0 88  0]
     [ 0  0  0  1  0  1  0  0  0 90]]



![png](/images/output_1_1.png)



```python
### sklearn实现
```


```python
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
digits.data
```




    array([[  0.,   0.,   5., ...,   0.,   0.,   0.],
           [  0.,   0.,   0., ...,  10.,   0.,   0.],
           [  0.,   0.,   0., ...,  16.,   9.,   0.],
           ..., 
           [  0.,   0.,   1., ...,   6.,   0.,   0.],
           [  0.,   0.,   2., ...,  12.,   0.,   0.],
           [  0.,   0.,  10., ...,  12.,   1.,   0.]])



在数字数据集的情况下，digits.data 使我们能够得到一些用于分类的样本特征:
并且 digits.target 表示了数据集内每个数字的真实类别，也就是我们期望从每个手写数字图像中学得的相应的数字标记:


```python
digits = datasets.load_digits()
digits.target
```




    array([0, 1, 2, ..., 8, 9, 8])



数据数组的形状
数据总是二维数组，形状 (n_samples, n_features) ，尽管原始数据可能具有不同的形状。 在数字的情况下，每个原始样本是形状 (8, 8) 的图像，可以使用以下方式访问:


```python
digits.images[-1]
```




    array([[  0.,   0.,  10.,  14.,   8.,   1.,   0.,   0.],
           [  0.,   2.,  16.,  14.,   6.,   1.,   0.,   0.],
           [  0.,   0.,  15.,  15.,   8.,  15.,   0.,   0.],
           [  0.,   0.,   5.,  16.,  16.,  10.,   0.,   0.],
           [  0.,   0.,  12.,  15.,  15.,  12.,   0.,   0.],
           [  0.,   4.,  16.,   6.,   4.,  16.,   6.,   0.],
           [  0.,   8.,  16.,  10.,   8.,  16.,   8.,   0.],
           [  0.,   1.,   8.,  12.,  14.,  12.,   1.,   0.]])



## 学习和预测
在数字数据集的情况下，任务是给出图像来预测其表示的数字。 我们给出了 10 个可能类（数字 0 到 9）中的每一个的样本，我们在这些类上 拟合 一个 估计器 ，以便能够 预测 未知的样本所属的类。
在 scikit-learn 中，分类的估计器是一个 Python 对象，它实现了 fit(X, y) 和 predict(T) 等方法。
估计器的一个例子类 sklearn.svm.SVC ，实现了 支持向量分类 。 估计器的构造函数以相应模型的参数为参数，但目前我们将把估计器视为黑箱即可:

### 选择模型的参数
在这个例子中，我们手动设置 gamma 值。不过，通过使用 网格搜索 及 交叉验证 等工具，可以自动找到参数的良好值。
我们把我们的估计器实例命名为 clf ，因为它是一个分类器（classifier）。它现在必须拟合模型，也就是说，它必须从模型中 learn（学习） 。 这是通过将我们的训练集传递给 fit 方法来完成的。作为一个训练集，让我们使用数据集中除最后一张以外的所有图像。 我们用 [:-1] Python 语法选择这个训练集，它产生一个包含 digits.data 中除最后一个条目（entry）之外的所有条目的新数组


```python
from sklearn import svm

clf = svm.SVC(gamma = 0.001, C=100.) #生成模型（估计器，分类器）
clf.fit(digits.data[:-1], digits.target[:-1])#拟合模型进行学习
clf.predict(digits.data[-1:]) # 对数据进行预测
```




    array([8])



## 模型持久化
可以通过使用 Python 的内置持久化模块（即 pickle ）将模型保存:


```python
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)  

import pickle
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])

y[0]
```




    0



在scikit的具体情况下，使用 joblib 替换 pickle（ joblib.dump & joblib.load ）可能会更有趣，这对大数据更有效，但只能序列化 (pickle) 到磁盘而不是字符串变量:


```python
from sklearn.externals import joblib
joblib.dump(clf, 'filename.pkl') 
```




    ['filename.pkl']



在scikit的具体情况下，使用 joblib 替换 pickle（ joblib.dump & joblib.load ）可能会更有趣，这对大数据更有效，但只能序列化 (pickle) 到磁盘而不是字符串变量:

之后，您可以加载已保存的模型（可能在另一个 Python 进程中）:


```python
from sklearn.externals import joblib
joblib.dump(clf, 'filename.pkl') 
clf = joblib.load('filename.pkl')
```

## 规定
scikit-learn 估计器遵循某些规则，使其行为更可预测。
### 类型转换
默认的float64转换成float32


```python
from sklearn import random_projection

rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
X.dtype
X =np.array(X, dtype='float32')
X.dtype
```

分别使用整数数组和字符串数组类型


```python
from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
clf = SVC()
# 下面模型返回整数数组，因为在 fit 中使用了 iris.target （一个整数数组）
clf.fit(iris.data, iris.target)  
clf.predict(iris.data[:3])
```




    array([0, 0, 0])



### 再次训练和更新参数
估计器的超参数可以通过 sklearn.pipeline.Pipeline.set_params 方法在实例化之后进行更新。 调用 fit() 多次将覆盖以前的 fit() 所学到的参数:


```python
import numpy as np
from sklearn.svm import SVC

rng = np.random.RandomState(0)   #np.random.RandomState函数是机器学习中常用的伪随机数产生器。arrayA = rng.uniform(0,1,(2,3))指在0～1区间产生一个2行3列的随机数。
                                
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)#rng.binomial(1, 0.5, 100)指以数字0为随机数种子，在1～0.5的区间内产生100个随机数。
                            #对于某一个伪随机数发生器，只要该种子（seed）相同，产生的随机数序列就是相同的
X_test = rng.rand(5, 10)

#在这里，估计器被 SVC() 构造之后，默认内核 rbf 首先被改变到 linear ，然后改回到 rbf 重新训练估计器并进行第二次预测。
clf = SVC()
clf.set_params(kernel='linear').fit(X, y)  
clf.predict(X_test)
# 覆盖参数重新学习
clf.set_params(kernel='rbf').fit(X, y)  
clf.predict(X_test)
```




    array([0, 0, 0, 1, 0])



### 多分类与多标签拟合

当使用 多类分类器 时，执行的学习和预测任务取决于参与训练的目标数据的格式:


```python
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]

classif = OneVsRestClassifier(estimator=SVC(random_state=0))
classif.fit(X, y).predict(X)
```




    array([0, 0, 1, 1, 2])


