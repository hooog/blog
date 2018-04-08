---
title: sklearn聚合算法整理
date: 2018-04-9 00:19:59
password:
top:
categories:
  - Machine learning
tags:
  - scikit-learn 
---


## 随机森林分类预测泰坦尼尼克号幸存者


```python
import pandas as pd
import numpy as np

def read_dataset(fname):
    data = pd.read_csv(fname, index_col=0)
    data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    lables = data['Sex'].unique().tolist()
    data['Sex'] = [*map(lambda x: lables.index(x) , data['Sex'])]
    lables = data['Embarked'].unique().tolist()
    data['Embarked'] = data['Embarked'].apply(lambda n: lables.index(n))
    data = data.fillna(0)
    return data
train = read_dataset('code/datasets/titanic/train.csv')

from sklearn.model_selection import train_test_split

y = train['Survived'].values
X = train.drop(['Survived'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("X_train_shape:", X_train.shape, " y_train_shape:", y_train.shape)
print("X_test_shape:", X_test.shape,"  y_test_shape:", y_test.shape)
```

    X_train_shape: (712, 7)  y_train_shape: (712,)
    X_test_shape: (179, 7)   y_test_shape: (179,)



```python
****
```


```python
from sklearn.ensemble import RandomForestClassifier
import time

start = time.clock()
entropy_thresholds = np.linspace(0, 1, 50)
gini_thresholds = np.linspace(0, 0.1, 50)
#设置参数矩阵：
param_grid = [{'criterion': ['entropy'], 'min_impurity_decrease': entropy_thresholds},
              {'criterion': ['gini'], 'min_impurity_decrease': gini_thresholds},
              {'max_depth': np.arange(2,10)},
              {'min_samples_split': np.arange(2,20)},
              {'n_estimators':np.arange(2,20)}]
clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
clf.fit(X, y)

print("耗时：",time.clock() - start)
print("best param:{0}\nbest score:{1}".format(clf.best_params_, clf.best_score_))
```

    耗时： 13.397480000000002
    best param:{'min_samples_split': 10}
    best score:0.8406285072951739



```python
clf = RandomForestClassifier(min_samples_split=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("训练集得分:", clf.score(X_train, y_train))
print("测试集得分:", clf.score(X_test, y_test))
print("查准率:", metrics.precision_score(y_test, y_pred))
print("召回率:", metrics.recall_score(y_test, y_pred))
print("F1_score:", metrics.f1_score(y_test, y_pred))
```

    训练集得分: 0.8974719101123596
    测试集得分: 0.7988826815642458
    查准率: 0.8082191780821918
    召回率: 0.7283950617283951
    F1_score: 0.7662337662337663


这次分别对模型的`criterion`,`max_depth`,`min_samples_split`,`n_estimators`四个参数进行了比较。

经过多次执行发现结果仍不是很稳定，最优参数集中在`min_samples_split`分别为8，10，12上

## 自助聚合算法预测泰坦尼尼克号幸存者


```python
from sklearn.ensemble import BaggingClassifier

clf = BaggingClassifier(n_estimators=50)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("训练集得分:", clf.score(X_train, y_train))
print("测试集得分:", clf.score(X_test, y_test))
print("查准率:", metrics.precision_score(y_test, y_pred))
print("召回率:", metrics.recall_score(y_test, y_pred))
print("F1_score:", metrics.f1_score(y_test, y_pred))
```

    训练集得分: 0.9817415730337079
    测试集得分: 0.7877094972067039
    查准率: 0.7792207792207793
    召回率: 0.7407407407407407
    F1_score: 0.7594936708860759


## Boosting正向激励算法预测泰坦尼尼克号幸存者


```python
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("训练集得分:", clf.score(X_train, y_train))
print("测试集得分:", clf.score(X_test, y_test))
print("查准率:", metrics.precision_score(y_test, y_pred))
print("召回率:", metrics.recall_score(y_test, y_pred))
print("F1_score:", metrics.f1_score(y_test, y_pred))
```

    训练集得分: 0.8300561797752809
    测试集得分: 0.8156424581005587
    查准率: 0.8076923076923077
    召回率: 0.7777777777777778
    F1_score: 0.7924528301886792


## Extra Trees算法预测泰坦尼尼克号幸存者


```python
from sklearn.ensemble import RandomForestClassifier
import time

start = time.clock()
entropy_thresholds = np.linspace(0, 1, 50)
gini_thresholds = np.linspace(0, 0.1, 50)
#设置参数矩阵：
param_grid = [{'criterion': ['entropy'], 'min_impurity_decrease': entropy_thresholds},
              {'criterion': ['gini'], 'min_impurity_decrease': gini_thresholds},
              {'max_depth': np.arange(2,10)},
              {'min_samples_split': np.arange(2,20)},
              {'n_estimators':np.arange(2,20)}]
clf = GridSearchCV(ExtraTreesClassifier(), param_grid, cv=5)
clf.fit(X, y)

print("耗时：",time.clock() - start)
print("best param:{0}\nbest score:{1}".format(clf.best_params_, clf.best_score_))
```

    耗时： 16.29516799999999
    best param:{'min_samples_split': 12}
    best score:0.8226711560044894



```python
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier(min_samples_split=12from sklearn.ensemble import RandomForestClassifier
import time

start = time.clock()
entropy_thresholds = np.linspace(0, 1, 50)
gini_thresholds = np.linspace(0, 0.1, 50)
#设置参数矩阵：
param_grid = [{'criterion': ['entropy'], 'min_impurity_decrease': entropy_thresholds},
              {'criterion': ['gini'], 'min_impurity_decrease': gini_thresholds},
              {'max_depth': np.arange(2,10)},
              {'min_samples_split': np.arange(2,20)},
              {'n_estimators':np.arange(2,20)}]
clf = GridSearchCV(ExtraTreesClassifier(), param_grid, cv=5)
clf.fit(X, y)

print("耗时：",time.clock() - start)
print("best param:{0}\nbest score:{1}".format(clf.best_params_, clf.best_score_)))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("训练集得分:", clf.score(X_train, y_train))
print("测试集得分:", clf.score(X_test, y_test))
print("查准率:", metrics.precision_score(y_test, y_pred))
print("召回率:", metrics.recall_score(y_test, y_pred))
print("F1_score:", metrics.f1_score(y_test, y_pred))
```

    训练集得分: 0.8932584269662921
    测试集得分: 0.8100558659217877
    查准率: 0.8405797101449275
    召回率: 0.7160493827160493
    F1_score: 0.7733333333333333


## 结论：
针对此数据集预测泰坦尼克号的结果对比中，Boosting正向激励算法性能最佳最稳定，其次是参数优化后的Extra Trees算法。
