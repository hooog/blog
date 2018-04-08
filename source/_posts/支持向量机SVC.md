---
title: SVM支持向量机SVC实现
date: 2018-04-9 01:05:59
password:
top:
categories:
  - Machine learning
tags:
  - scikit-learn 
---
<!--more-->


支持向量机(support vector machine)是一种分类算法，但是也可以做回归，根据输入的数据不同可做不同的模型（若输入标签为连续值则做回归，若输入标签为分类值则用SVC()做分类）。通过寻求结构化风险最小来提高学习机泛化能力，实现经验风险和置信范围的最小化，从而达到在统计样本量较少的情况下，亦能获得良好统计规律的目的。通俗来讲，它是一种二类分类模型，其基本模型定义为特征空间上的间隔最大的线性分类器，即支持向量机的学习策略便是间隔最大化，最终可转化为一个凸二次规划问题的求解。
`sklearn`里对SVM的算法实现在包`sklearn.svm`里。

###  `svm.SVC`分类器简介：
- **C**：C-SVC的惩罚参数C?默认值是1.0

    C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
- **kernel** ：核函数，默认是rbf，可以是‘linear’,‘poly’, ‘rbf’

        liner – 线性核函数：u'v

        poly – 多项式核函数：(gamma*u'*v + coef0)^degree

        rbf – RBF高斯核函数：exp(-gamma|u-v|^2)


- **degree** ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。

- **gamma** ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features

- **coef0** ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。

- **probability** ：是否采用概率估计？.默认为False

- **shrinking** ：是否采用shrinking heuristic方法，默认为true

- **tol** ：停止训练的误差值大小，默认为1e-3

- **cache_size** ：核函数cache缓存大小，默认为200

- **class_weight** ：类别的权重，字典形式传递。设置第几类的参数C为weight * C(C-SVC中的C)

- **verbose** ：允许冗余输出？

- **max_iter** ：最大迭代次数。-1为无限制。

- **decision_function_shape** ：‘ovo’, ‘ovr’ or None, default=None3

- **random_state** ：数据洗牌时的种子值，int值



```python
# 利用np.meshgrid()生成一个坐标矩阵，然预测坐标矩阵中每个点所属的类别，最后用contourf()函数
# 为最表矩阵中不同类别填充不同颜色
import numpy as np
def plot_hyperplane(clf, X, y, 
                    h=0.02, 
                    draw_sv=True, 
                    title='hyperplan'):
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)

    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y)
    for label in labels:
        plt.scatter(X[y==label][:, 0], 
                    X[y==label][:, 1], 
                    c=colors[label], 
                    marker=markers[label], s=20)
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='black', marker='x', s=15)
```

### **1、第一个例子：**


```python
%matplotlib inline
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.3)

clf = svm.SVC(C = 1.0, kernel='linear')
clf.fit(X,y)

print(clf.score(X,y))
plt.figure(figsize=(10,3), dpi=100)
plot_hyperplane(clf, X, y, h=0.01, title='Maximiin Margin Hyperplan')
```

![](https://ws4.sinaimg.cn/large/006tKfTcly1fq5iil2vlrj30ma07iwex.jpg)


```python
上图带有X标记点的是支持向量，它保存在模型的support_vectors_里。
```


```python
clf.support_vectors_
```




    array([[0.70993435, 3.70954839],
           [1.65719647, 3.86747763],
           [1.7033305 , 1.48075002]])



### **2、第二个例子**

生成一个有两个特征、三纵类别的数据集，然后分别构造4个SVM算法来拟合数据集，分别是线性和函数、三姐多项式核函数、gamma=0.5的高斯核RBF核函数和gamma=1的高斯核函数。最后把四个算法拟合出来的分割超平面画出来。


```python
from sklearn import svm
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=3, random_state=0, cluster_std=1)

clf_linear = svm.SVC(C=1.0, kernel='linear')
clf_poly = svm.SVC(C=1.0, kernel='poly', degree=3)
clf_rbf = svm.SVC(C=1.0, kernel='rbf', gamma=0.5)
clf_rbf1 = svm.SVC(C=1.0, kernel='rbf', gamma=1)

plt.figure(figsize=(12,10),dpi=140)

clfs = [clf_linear, clf_poly, clf_rbf, clf_rbf1]
titles = ['linearSVC','polySVC_d3','rbfSVC_0.5','rbfSVC_1']

for clf, i in zip(clfs, range(len(clfs))):
    clf.fit(X, y)
    plt.subplot(2, 2, i+1)
    plot_hyperplane(clf, X, y, title=titles[i])
```

![](https://ws4.sinaimg.cn/large/006tKfTcly1fq5iizji9fj31140v2q7c.jpg)

这里需要注意的是坐下和右下高斯函数的图。既然支持向量是离超平面最近的点，那为什么高斯函数的图中离分割超平面很远的点也是支持向量呢？

原因是高斯核函数把输入特征向量映射到了无限维的向量空间里，在高维空间里这些点其实就是最近的点。


### **3、例三：乳腺癌检测**

- **使用RBF高斯核函数**


```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print('data shape:{0}; no. positive:{1}; no. negative:{2}'.format(X.shape, y[y==1].shape[0], y[y==0].shape[0]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
```

    data shape:(569, 30); no. positive:357; no. negative:212



```python
from sklearn.svm import SVC

clf = SVC(C=1, kernel='rbf', gamma=.1)
clf.fit(X_train, y_train)
print("Train_score:{0}\nTest_score:{1}".format(clf.score(X_train, y_train), clf.score(X_test, y_test)))
```

    Train_score:1.0
    Test_score:0.6754385964912281


上例典型过拟合。代码gamma参数为0.1，这个参数说明作用呢？

回忆下RBF 核函数$$K\left( x^{\left( i \right)},x^{\left( j \right)} \right)=\exp \left( -\frac{\left( x^{\left( i \right)}-x^{\left( j \right)} \right)^{2}}{2\sigma ^{2}} \right)$$
$γ$（指本式中的$\sigma$）主要定义了单个样本对整个分类超平面的影响，当$γ$
比较小时，单个样本对整个分类超平面的影响比较小，不容易被选择为支持向量，反之，当$γ$比较大时，单个样本对整个分类超平面的影响比较大，更容易被选择为支持向量，或者说整个模型的支持向量也会多。scikit-learn中默认值是1/n_features（1/样本量）。

因此判断造成过拟合的原因是$γ$太大。下面用`GridSearchCV`来自动选择$γ$的最优参数以及对应的交叉验证评分及召回率和F1得分。


```python
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 

thresholds = np.linspace(0, 0.001, 100)
# Set the parameters by cross-validation
param_grid = {'gamma': thresholds}

clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
clf.fit(X_train, y_train)
print("best param: {0}\nbest score: {1}".format(clf.best_params_, 
                                                clf.best_score_))
y_pred = clf.predict(X_test)

print("查准率：",metrics.precision_score(y_pred, y_test))
print("召回率：",metrics.recall_score(y_pred, y_test))
print("F1：",metrics.f1_score(y_pred, y_test))
```

    best param: {'gamma': 9.090909090909092e-05}
    best score: 0.945054945054945
    查准率： 0.961038961038961
    召回率： 0.925
    F1： 0.9426751592356688



```python
%matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learn_curve(estimator, title, X, y, ylim = None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1., 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("train exs")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_score_mean = np.mean(train_scores, axis=1)
    train_score_std = np.std(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    test_score_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes, train_score_mean - train_score_std, 
                     train_score_mean + train_score_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_score_mean - test_score_std, 
                     test_score_mean + test_score_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_score_mean, 'o-', color='r', label='train score训练得分')
    plt.plot(train_sizes, test_score_mean, 'o-', color='g', label='cross-validation score交叉验证得分')
    
    plt.legend(loc='best')
    return plt
```


```python
import time 
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=10, test_size=.2, random_state=0)
title = 'Learing Curve for SVC-RBF'

gammas=[0.001, 0.0005, 0.0003, 0.0001]
start = time.clock()

plt.figure(figsize=(14,10), dpi=140)
for i in range(len(gammas)):
    plt.subplot(2, 2, i+1)
    plot_learn_curve(SVC(C=1., gamma=gammas[i]), gammas[i], X, y, cv=cv)

# plt.figure(figsize=(8,6), dpi=120)
# plot_learn_curve(SVC(C=1., gamma=gammas[0]), title, X, y, cv=cv)
print("耗时：", time.clock() - start)
```

![](https://ws1.sinaimg.cn/large/006tKfTcly1fq5kp9xbugj319e0wmn3e.jpg)

- **使用多项式核函数**


```python
clf = SVC(C=1., kernel='poly', degree=2)
start = time.clock()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("耗时：",time.clock()-start)
print("Train_score:{0}\nTest_score:{1}".format(clf.score(X_train, y_train), clf.score(X_test, y_test)))
print("查准率：",metrics.precision_score(y_pred, y_test))
print("召回率：",metrics.recall_score(y_pred, y_test))
print("F1：",metrics.f1_score(y_pred, y_test))
```

    耗时： 13.866157000000015
    Train_score:0.978021978021978
    Test_score:0.9736842105263158
    查准率： 0.9866666666666667
    召回率： 0.9736842105263158
    F1： 0.9801324503311258



```python
import time
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=10, test_size=.2, random_state=0)
titles = ['poly_SVC_D=1','ploy_SVC_D=2']
degrees = [1,2]
start = time.clock()
plt.figure(figsize=(12,4), dpi=130)
for i in range(len(degrees)):
    plt.subplot(1, len(degrees), i+1)
    plot_learn_curve(SVC(kernel='poly', degree=degrees[i]), titles[i], X, y, cv=cv)

print("耗时：", time.clock()-start)
```

![](https://ws4.sinaimg.cn/large/006tKfTcly1fq5njno5cbj310k0dxacq.jpg)

可以看出，一阶多项式核函数的拟合效果更好。平均交叉验证数据集评分0.94，最高时可达0.975。当然消耗的时间也更久，一阶多项式计算代价比高斯核函数的SVC运算事件多出了好多倍。 之前笔者使用逻辑回归算法检测乳腺癌检测问题时，使用二项多项式逻辑回归增加特征同时使用L1范数作为正则项的拟合效果比这里的支持向量机效果好，更重要的是逻辑回归算法的运算效率远远高于二阶多项式核函数的支持向量机算法。当然，这里的支持向量机算法的效果还是比使用L2范数作为正则项的逻辑回归准确率高，由此可见木星选择和参数调优对机器学习的重要性。
