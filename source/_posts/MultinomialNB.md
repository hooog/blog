---
title: 朴素贝叶斯--文档分类
date: 2018-04-9 20:10:59
password:
top:
categories:
  - Machine learning
tags:
  - scikit-learn 
---

### 把文档转换成向量

TF-IDF是一种统计方法，用以评估一个词语对于一份文档的重要程度。

- TF表示词频， 即：词语在一片文档中出现的次数 ÷ 词语总数
- IDF表示一个词的**逆向文档频率指数**， 即：对（总文档数目÷包含该词语的文档的数目）的商取对数  $log(m / m_{i-in-m})$

基础原理：词语的重要性随着它在文档中出现的次数成正比例增加，但同时会随着它在语料库中出现的频率呈反比下降。


sklearn中有包实现了把文档转换成向量的过程，首先把训练用额语料库读入内存：


```python
from time import time 
from sklearn.datasets import load_files


t = time()
news_train = load_files('code/datasets/mlcomp/379/train')
print(len(news_train.data), "\n",len(news_train.target_names))
print("done in {} seconds".format(time() - t))
```

    13180 
     20
    done in 6.034918308258057 seconds


news_train.data是一个数组，包含了所有文档的文本信息。
news_train.target_names也是一个数组，包含了所有文档的属性类别，对应的是读取train文件夹时，train文件夹下所有的子文件夹名称。

该语料库总共有13180个文档，其中分成20个类别，接着需要转换成由TF-IDF表达的权重信息构成向量。

![](https://ws3.sinaimg.cn/large/006tNc79ly1fq6g2bkj9sj30hf09lta4.jpg)


```python
from sklearn.feature_extraction.text import TfidfVectorizer

t = time()
vectorizer  = TfidfVectorizer(encoding = 'latin-1')
X_train = vectorizer.fit_transform((d for  d in news_train.data))
print("文档 [{0}]特征值的非零个数:{1}".format(news_train.filenames[0] , X_train[0].getnnz()))
print("训练集：",X_train.shape)
print("耗时： {0} s.".format(time() - t))
```

    文档 [code/datasets/mlcomp/379/train/talk.politics.misc/17860-178992]特征值的非零个数:108
    训练集： (13180, 130274)
    耗时： 3.740567207336426 s.


**TfidfVectorizer**类是用来把所有的文档转换成矩阵，该矩阵每一行都代表一个文档，一行中的每个元素代表一个对应的词语的重要性，词语的重要性由TF-IDF来表示。其`fit_transform()`方法是`fit()`和`transform()`的结合,`fit()`先完成语料库分析，提取词典等操作`transform()`把每篇文档转换为向量，最终构成一个矩阵，保存在`X_train`里。

程序输出可以看到该词典总共有130274个词语，即每篇文档都可以转换成一个13274维的向量组。第一篇文档中只有108个非零元素，即这篇文档由108个不重复的单词组成，在这篇文档中出现的这108个单词次的**TF-IDF**会被计算出来，保存在向量的指定位置。这里的到X_train是一个纬度为12180 x 130274的系数矩阵。

### 训练模型


```python
from sklearn.naive_bayes import MultinomialNB

t = time()
y_train = news_train.target
clf = MultinomialNB(alpha=0.001)  #alpga表示平滑参数，越小越容易造成过拟合；越大越容易欠拟合。
clf.fit(X_train, y_train)

print("train_score:", clf.score(X_train, y_train))
print("耗时：{0}s".format(time() - t))
```

    train_score: 0.9974203338391502
    耗时：0.23757004737854004s



```python
# 加载测试集检验结果
news_test = load_files('code/datasets/mlcomp/379/test')
print(len(news_test.data))
print(len(news_test.target_names))
```

    5648
    20



```python
# 把测试集文档数学向量化
t = time()
# vectorizer  = TfidfVectorizer(encoding = 'latin-1')  # 这里注意vectorizer这条语句上文已经生成执行，这里不可重复执行
X_test = vectorizer.transform((d for  d in news_test.data))
y_test = news_test.target

print("测试集：",X_test.shape)
print("耗时： {0} s.".format(time() - t))
```

    测试集： (5648, 130274)
    耗时： 1.64164400100708 s.



```python
import numpy as np
from sklearn import metrics 


y_pred = clf.predict(X_test)
print("Train_score:", clf.score(X_train, y_train))
print("Test_score:", clf.score(X_test, y_test))

for i in range(10):
    r = np.random.randint(X_test.shape[0])
    if clf.predict(X_test[r]) == y_test[r]:
        print("√：{0}".format(r))
    else:print("X：{0}".format(r))
```

    Train_score: 0.9974203338391502
    Test_score: 0.9123583569405099
    √：1874
    √：2214
    √：2579
    √：1247
    √：375
    √：5384
    √：5029
    √：1951
    √：4885
    √：1980


### 评价模型：

#### `classification_report()`查看查准率、召回率、F1
使用`classification_report()`函数查看针对每个类别的预测准确性：


```python
from sklearn.metrics import classification_report

print(clf)
print("查看针对每个类别的预测准确性：")
print(classification_report(y_test, y_pred, 
                            target_names = news_test.target_names))
```

    MultinomialNB(alpha=0.001, class_prior=None, fit_prior=True)
    查看针对每个类别的预测准确性：
                              precision    recall  f1-score   support
    
                 alt.atheism       0.90      0.92      0.91       245
               comp.graphics       0.80      0.90      0.84       298
     comp.os.ms-windows.misc       0.85      0.80      0.82       292
    comp.sys.ibm.pc.hardware       0.81      0.82      0.81       301
       comp.sys.mac.hardware       0.90      0.92      0.91       256
              comp.windows.x       0.89      0.88      0.88       297
                misc.forsale       0.88      0.82      0.85       290
                   rec.autos       0.93      0.93      0.93       324
             rec.motorcycles       0.97      0.97      0.97       294
          rec.sport.baseball       0.97      0.96      0.97       315
            rec.sport.hockey       0.97      0.99      0.98       302
                   sci.crypt       0.96      0.95      0.96       297
             sci.electronics       0.91      0.85      0.88       313
                     sci.med       0.96      0.96      0.96       277
                   sci.space       0.95      0.97      0.96       305
      soc.religion.christian       0.93      0.96      0.94       293
          talk.politics.guns       0.90      0.96      0.93       246
       talk.politics.mideast       0.95      0.98      0.97       296
          talk.politics.misc       0.91      0.89      0.90       236
          talk.religion.misc       0.89      0.77      0.82       171
    
                 avg / total       0.91      0.91      0.91      5648
    


#### `confusion_matrix`混淆矩阵

通过`confusion_matrix`函数生成混淆矩阵，观察每种类别别错误分类的情况。例如，这些被错误分类的文档是被错误分类到哪些类别里。


```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

# 第一行表示类别0的文档被正确分类的由255个，其中有2、5、13个错误分类被分到了14、15、19类中了。
```

    [[225   0   0   0   0   0   0   0   0   0   0   0   0   0   2   5   0   0   0  13]
     [  1 267   6   4   2   8   1   1   0   0   0   2   3   2   1   0   0   0   0   0]
     [  1  12 233  26   4   9   3   0   0   0   0   0   2   1   0   0   0   0   1   0]
     [  0   9  16 246   7   3  10   1   0   0   1   0   8   0   0   0   0   0   0   0]
     [  0   2   3   5 236   2   2   1   0   0   0   3   1   0   1   0   0   0   0   0]
     [  0  22   6   3   0 260   0   0   0   2   0   1   0   0   1   0   2   0   0   0]
     [  0   2   5  11   3   1 238   9   2   3   1   0   7   0   1   0   2   2   3   0]
     [  0   1   0   0   1   0   7 302   4   1   0   0   1   2   3   0   2   0   0   0]
     [  0   0   0   0   0   2   2   3 285   0   0   0   1   0   0   0   0   0   0   1]
     [  0   1   0   0   1   1   1   2   0 302   6   0   0   1   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   2   1 299   0   0   0   0   0   0   0   0   0]
     [  0   1   2   1   1   1   2   0   0   0   0 283   1   0   0   0   2   1   2   0]
     [  0  11   2   6   5   2   4   5   1   1   1   3 267   1   3   0   0   0   1   0]
     [  1   1   0   1   1   1   0   0   0   0   0   1   1 265   2   1   0   0   2   0]
     [  0   3   0   0   1   0   0   0   0   0   0   1   1   1 296   0   1   0   1   0]
     [  3   1   0   1   0   0   0   0   0   0   1   0   0   2   0 281   0   1   2   1]
     [  1   0   1   0   0   0   0   0   1   0   0   0   0   0   0   0 237   1   4   1]
     [  1   0   0   0   0   1   0   0   0   0   0   0   0   0   0   3   0 290   1   0]
     [  1   1   0   0   1   1   0   1   0   0   0   0   0   0   0   1  12   7 210   1]
     [ 16   1   0   0   0   0   0   0   0   0   0   0   0   0   0  12   5   2   4 131]]


```python
%matplotlib inline
from matplotlib import pyplot as plt

plt.figure(figsize=(6, 6), dpi=120)
plt.title('Confusion matrix of the classifier')
ax = plt.gca()                                  
ax.spines['right'].set_color('none')            
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.matshow(cm, fignum=1, cmap='gray')
plt.colorbar();

# 除对角线外，颜色越浅说明错误越多
```

![](https://ws4.sinaimg.cn/large/006tNc79ly1fq6kdjano8j30gq0hegm8.jpg)


```python
# 上图不直观，重新画图
import random
from pyecharts import HeatMap

x_axis = np.arange(20)
y_axis = np.arange(20)
data = [[i, j, cm[i][j]] for i in range(20) for j in range(20)]
heatmap = HeatMap()
heatmap.add("混淆矩阵", x_axis, y_axis, data, is_visualmap=True,
            visual_text_color="#fff", visual_orient='horizontal')
# heatmap.render()
# heatmap
```

![](https://ws2.sinaimg.cn/large/006tNc79ly1fq6kc4ryv9j30h00gu0tm.jpg)
