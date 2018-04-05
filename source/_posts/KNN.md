---
title: KNN-糖尿病预测-学习曲线的应用
date: 2018-03-29 17:05:59
password:
top:
categories:
  - Machine learning
tags:
  - scikit-learn 
---
<!--more-->

## 算法原理
KNN算法的核心思想是为预测样本的类别，即使最邻近的k个邻居中类别占比最高的的类别：

假设X_test为未标记的数据样本，X_train为已标记类别的样本，算法原理伪代码如下：
- 遍历X_train中所有样本，计算每个样本与X_test的距离，并保存在Distance数组中
- 对Distance数组进行排序，取距离最近的k个点，记为X_knn
- 在X_knn中统计每个类别的个数
- 代表记得样本的类别，就是在X_knn中样本最多的类别


- KNeighborsClassifier
- KNeighborsRegressor
- RadiusNeighborsClassifier
- RadiusNeighborsRegressor


### 算法优缺点

**优点：**准确性高，对异常值和噪声有较高的容忍度

**缺点：**计算量大，对内存的需求也较大


### 算法参数（$k$）

**$k$越大：模型偏差越大，对噪声越不敏感。过大是造成欠拟合**

**$k$越小：模型的方差就会越大。太小是会造成过拟合**


### 算法的变种

**增加邻居的权重：**默认情况下X_knn的权重相等，我们可以指定算法的`weights`参数调整成距离越近权重越大

**使用一定半径内的点取代距离最近的$k$个点**，`RadiusNeighborsClassifier`类实现了这个算法


## 使用knn进行分类
- `sklearn.neighbors.KNeighborsClassifier`


```python
from sklearn.datasets.samples_generator import make_blobs
# 生成n_samples个训练样本，分布在centers参数指定的中心点周围。 cluster_std为标准差，指定生成的点分布的稀疏程度
centers = [[-2,2], [2,2], [0,4]]
X , y = make_blobs(n_samples=100, centers=centers, random_state=0, cluster_std=0.60)

# 画出数据
%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

plt.figure(figsize=(8,5), dpi=100)
c = np.array(centers)
plt.scatter(X[:,0], X[:,1], c=y, s=10, cmap='cool')
plt.scatter(c[:,0], c[:,1], s=50, marker='^', c='red')
```

![png](/images/sklearn/3.png)


```python
from sklearn.neighbors import KNeighborsClassifier

k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X, y)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=1, n_neighbors=5, p=2,
               weights='uniform')




```python
# X_test = np.array([0, 2]).reshape(1,-1)
X_test = [[0,2]]
y_test = clf.predict(X_test)
neighbors = clf.kneighbors(X_test, return_distance=False)
```


```python
from sklearn.datasets.samples_generator import make_blobs
# 生成n_samples个训练样本，分布在centers参数指定的中心点周围。 cluster_std为标准差，指定生成的点分布的稀疏程度
centers = [[-2,2], [2,2], [0,4]]
X , y = make_blobs(n_samples=100, centers=centers, random_state=0, cluster_std=0.60)

# 画出数据
%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

plt.figure(figsize=(8,5), dpi=100)
c = np.array(centers)
plt.scatter(X[:,0], X[:,1], c=y, s=10, cmap='cool')  # 样本
plt.scatter(c[:,0], c[:,1], s=50, marker='^', c='red') # 中心点
plt.scatter(X_test[0][0], X_test[0][1], marker='x', s=50, c='blue') # 中心点

for i in neighbors[0]:
    plt.plot([X[i][0], X_test[0][0]], 
             [X[i][1], X_test[0][1]],
             'k--', linewidth=0.5)
```

![png](/images/sklearn/2.png)

## KNN 回归拟合

`sklearn.neighbors.KNeighborsRegressor`

### 模型展示


```python
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
%matplotlib inline
from matplotlib import pyplot as plt

n = 50
X = 5 * np.random.rand(n ,1)
y = np.cos(X).ravel()
# 添加一些噪声
y += 0.2 * np.random.rand(n) - 0.1
```


```python
k = 5
knn = KNeighborsRegressor(k)
knn.fit(X, y)
```




    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
              metric_params=None, n_jobs=1, n_neighbors=5, p=2,
              weights='uniform')




```python
T = np.linspace(0,5, 500)[:, np.newaxis]
y_pred = knn.predict(T)
knn.score(X,y)
```




    0.9909058023770559




```python
plt.figure(figsize=(8,5), dpi=100)
plt.scatter(X, y, label='data', s=10)
plt.scatter(T, y_pred, label='prediction', lw=4, s=0.1)
plt.axis('tight')
plt.title("KNeighborsRegressor (k=%i)" % k)
plt.show()
```

![png](/images/sklearn/4.png)

## 糖尿病预测

总共有768个数据、8个特征，其中Outcome为标记值（1表示有糖尿病）


```python
import numpy as np
import pandas as pd
data = pd.read_csv('code/datasets/pima-indians-diabetes/diabetes.csv')
X = data.iloc[:,0:8]
y = data.iloc[:,8]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### 模型比较 --KFold和cross_val_score()

- 分别使用普通KNN，加权重KNN，和指定权重的KNN分别对数据拟合计算评分


```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier

models = []
models.append(("KNN", KNeighborsClassifier(n_neighbors=10)))
models.append(("KNN + weights", KNeighborsClassifier(
    n_neighbors=10, weights="distance")))
models.append(("Radius Neighbors", RadiusNeighborsClassifier(n_neighbors=10, radius=500.0)))
```


```python
results = []
for name, model in models:
    model.fit(X_train, y_train)
    results.append((name, model.score(X_test, y_test)))
for i in range(len(results)):
    print("name:{}; score:{}".format(results[i][0], results[i][1])) 
```

    name:KNN; score:0.7207792207792207
    name:KNN + weights; score:0.6818181818181818
    name:Radius Neighbors; score:0.6558441558441559


- 此时单从得分上看，普通的KNN性能是最好的，但是我们的训练样本和测试样本是随机分配的，不同的训练集、测试集会造成不同得分。
- 为了消除随机样本集对得分结果可能的影响，scikit-learn提供了**`KFold和cross_val_score()`**函数来处理这个问题



```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

results = []
for name , model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, X, y, cv=kfold)  # 这里要给模型全部的样本集
    results.append((name, cv_result))
for i in range(len(results)):
    print("name:{}; cross_val_score:{}".format(results[i][0], results[i][1].mean()))

```

    name:KNN; cross_val_score:0.74865003417635
    name:KNN + weights; cross_val_score:0.7330485304169514
    name:Radius Neighbors; cross_val_score:0.6497265892002735


结果显示还是普通KNN更适合该数据集。

### 用查准率和召回率以及F1对该模型进行评估：


```python
from sklearn.metrics import f1_score, precision_score, recall_score

knn = KNeighborsClassifier(10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("该模型查准率为：", precision_score(y_test, y_pred))
print("该模型召回率为：", recall_score(y_test, y_pred))
print("该模型F1_score为：", f1_score(y_test, y_pred))
```

    该模型查准率为： 0.6086956521739131
    该模型召回率为： 0.5283018867924528
    该模型F1_score为： 0.5656565656565657


### 模型的训练及分析 -- 学习曲线

看起来还是普通KNN更优一些。

下面就选择用普通KNN算法模型对数据集进行训练，并查看训练样本的拟合情况及对策测试样本的预测准确性：

`from sklearn.model_selection import learning_curve
rain_sizes, train_scores, test_scores = learning_curve(`

输入：
    - estimator : 你用的分类器。
    - title : 表格的标题。
    - X : 输入的feature，numpy类型
    - y : 输入的target vector
    - ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    - cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    - n_jobs : 并行的的任务数(默认1))
输出：
    - train_sizes_abs :训练样本数
    - train_scores:训练集上准确率
    - test_scores:交叉验证集上的准确率) 


```python
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
train_score = knn.score(X_train, y_train)
test_score = knn.score(X_test, y_test)
print('训练集得分：',train_score)
print('测试集得分：',test_score)
```

    训练集得分： 0.8517915309446255
    测试集得分： 0.6948051948051948



```python
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
# from common.utils import plot_learning_curve

def plot_learn_curve(estimator, title, X, y, ylim = None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1., 10)):
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
data = pd.read_csv('code/datasets/pima-indians-diabetes/diabetes.csv')
X = data.iloc[:,0:8]
y = data.iloc[:,8]

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plt.figure(figsize=(8,6), dpi=100)
plot_learn_curve(KNeighborsClassifier(2),"KNN score",
                X, y, ylim=(0.5, 1), cv=cv)
plt.show()
```

![png](/images/sklearn/5.png)
当训练集和测试集的误差收敛但却很高时，为高偏差。 
左上角的偏差很高，训练集和验证集的准确率都很低，很可能是欠拟合。 
我们可以增加模型参数，比如，构建更多的特征，减小正则项。 
此时通过增加数据量是不起作用的。

当训练集和测试集的误差之间有大的差距时，为高方差。 
当训练集的准确率比其他独立数据集上的测试结果的准确率要高时，一般都是过拟合。 
右上角方差很高，训练集和验证集的准确率相差太多，应该是过拟合。 
我们可以增大训练集，降低模型复杂度，增大正则项，或者通过特征选择减少特征数。

理想情况是是找到偏差和方差都很小的情况，即收敛且误差较小。

### 特征选择及数据可视化

**使用sklearn.feature_selection.SelectKBest选择相关性最大的两个特征**


```python
from sklearn.feature_selection import SelectKBest
from sklearn.neighborse import KN

selector = SelectKBest(k=2)
X_new = selector.fit_transform(X,y)
X_new[0:5]  #把相关性最大的两个特征放到X_new里并查看前5个数据样本
```




    array([[148. ,  33.6],
           [ 85. ,  26.6],
           [183. ,  23.3],
           [ 89. ,  28.1],
           [137. ,  43.1]])



- 使用相关性最大的两个特征，对3种不同的KNN算法进行检验


```python
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier
models = []
models.append(("KNN", KNeighborsClassifier(n_neighbors=5)))
models.append(("KNN + weights", KNeighborsClassifier(
    n_neighbors=5, weights="distance")))
models.append(("Radius Neighbors", RadiusNeighborsClassifier(n_neighbors=5, radius=500.0)))

results = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, X_new, y, cv=kfold)
    results.append((name, cv_result))
for i in range(len(results)):
    print("name: {}; cross_val_score: {}".format(results[i][0], results[i][1].mean()))
```

    name: KNN; cross_val_score: 0.7369104579630894
    name: KNN + weights; cross_val_score: 0.7199419002050581
    name: Radius Neighbors; cross_val_score: 0.6510252904989747


从输出结果来看，还是普通KNN的准确性更高，与所有特征放到一起训练的准确性差不多，这也侧面证明了SelectKNBest特征选取的准确性。

回到目标上来，我们是想看看为什么KNN不能很好的拟合训练样本。现在我们至于2个特征可以很方便的在二维坐标上画出所有的训练样本，观察这些数据分布情况


```python
%matplotlib inline
from matplotlib import pyplot as plt

plt.figure(figsize=(8,6), dpi=100)
plt.ylabel("BMI")
plt.xlabel("Glucose")

plt.scatter(X_new[y==0][:,0], X_new[y==0][:,1], marker='o', s=10)
plt.scatter(X_new[y==1][:,0], X_new[y==1][:,1], marker='^', s=10)
```

![png](/images/sklearn/6.png)

横坐标是血糖值，纵坐标是BMI值反应身体肥胖情况。在数据密集的区域，代表糖尿病的阴性和阳性的样本几乎重叠到了一起。这样就很直观的看到，KNN在糖尿病预测的这个问题上无法达到很高的预测准确性

## 关于如何特高KNN算法的运算效率
- `K-D Tree 数据结构  ——Bentley，J.L.，Communications of the ACM(1975)`
- `Ball Tree（对K-D Tree的优化） ——Five balltree construction algorithms`