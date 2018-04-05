---
title: 支持向量机、学习曲线、查准率和召回率
date: 2018-03-28 17:05:59
password:
top:
categories:
  - Machine learning
tags:
  - scikit-learn 
---
<!--more-->



## 手写字体识别--支持向量机


```python
from sklearn import datasets
digits = datasets.load_digits()
```


```python
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.20, random_state=2) 
```


```python
import sklearn.metrics as sm
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("平均绝对误差（mean absolute error） ："
      , round(sm.mean_absolute_error(y_test, y_pred), 2))

print("均方误差（mean squared error） ："
     , round(sm.mean_squared_error(y_test, y_pred), 2))

print("中位数绝对误差（median absolute error） ："
     , round(sm.median_absolute_error(y_test, y_pred), 2))

print("解释方差分（explained variance score） ："
     , round(sm.explained_variance_score(y_test, y_pred), 2))

print("R方得分（R2 score） ："
     , round(sm.r2_score(y_test, y_pred)))
clf.score(X_test, y_test)
```

    平均绝对误差（mean absolute error） ： 0.08
    均方误差（mean squared error） ： 0.33
    中位数绝对误差（median absolute error） ： 0.0
    解释方差分（explained variance score） ： 0.96
    R方得分（R2 score） ： 1.0





    0.9777777777777777



## 学习曲线


```python
import numpy as np
n = 200
X = np.linspace(0,1,n)
y = np.sqrt(X) + 0.2*np.random.rand(n) -0.1
X = X.reshape(-1,1)
y = y.reshape(-1,1)
```


```python
%matplotlib inline
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def poly_model(degree=1):
    poly_features = PolynomialFeatures(degree = degree, include_bias=False)
    linear_regr = LinearRegression()
    # 这是一个流水线，先增加多项式阶数，再用线性回归算法来拟合数据
    pipeline = Pipeline([
        ('poly_features', poly_features), ('linear_regr', linear_regr)
    ])
    return pipeline
```


```python
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

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
# 为了让学习曲线更平滑，计算10次交叉验证数据的分数
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
titles = ['under fitting', 'learning curves', 'over fitting']

degrees = [1,3,10]

plt.figure(figsize=(18,4), dpi=150)
for i in range(len(degrees)):
    plt.subplot(1, 3, i+1)
    plot_learn_curve(poly_model(degrees[i]), titles[i],
                    X, y, ylim=(0.75, 1.01), cv=cv)
plt.show()
```

![png](/images/sklearn/1.png)

**过拟合**： 模型对训练集的准确性高，其成本比较低；对交叉验证数据集的准确性低，其成本比较高
- 获取更多的训练数据
- 减少输入的特征数量

**欠拟合**： 模型对训练集的准确性低，其成本比较高；对交叉验证数据集的准确性低，其成本比较高
- 增加有价值的特征
- 增加多项式特征
    -如原数据只有$x_1, x_2$，优化后可以增加特征变成，$x_1,x_2,x_1x_2,x_1^2,x_2^2$。这样即可增加模型复杂度

## 查准率和召回率

- 准确率(Precision) =  预测到的相关的 / 预测到的相关的+预测到的不相关的
- 召回率(Recall)      =  预测到的相关的 / 预测到的相关的+没有被预测到的相关的

在`scikit-learn`中，评估模型性能的算法都在`sklearn.metrics`包里。

其中计算查准率和召回率的`API`分别为`sklearn.metrics.precision_score()`和`sklearn.metrics.recall_score()`

如果有一个算法的查准率是0.5，召回率是0.4；另外一个查准率是0.02，召回率是1.0,那么那个算法更好？

为了解决这个问题我们引入了$F_1Score$的概念：$$F_1Score = 2*PR / (P+R)$$
`sklearn中对应的算法包是：sklearn.metrics.f1_score()`

**但是这个方法好想只接受二值化的数据类型**([0,1] 或者[1,2])


```python
from sklearn import metrics 

x = np.random.randint(1,3,100)
y = np.random.randint(1,3,100)

print(metrics.precision_score(x, y))
print(metrics.recall_score(x, y))
print(metrics.f1_score(x, y))
```

    0.6595744680851063
    0.5535714285714286
    0.6019417475728156

