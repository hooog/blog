---
title: 利用Pyecharts连接Mysql并对数据进行可视化
date: 2018-03-18 17:05:59
password:
top:
categories:
  - Python
tags:
  - Pyecharts
---
<!--more-->

## 每日交易额汇总


```python
import pymysql

conn = pymysql.connect(host='192.168.56.111', port=3306, user='hive', passwd='hive', db='test', charset='utf8')
# 创建游标
cursor = conn.cursor()
cursor.execute("select day_date from day_sum order by day_date")
tran_data = cursor.fetchall()

cursor.execute("select day_sum from day_sum order by day_date")
transum = cursor.fetchall()
tran_sum = [*map(lambda x :x[0]/100,list(transum))]

cursor.close()

tran_add = [0]
for i in range (len(tran_sum)):
    if i > 0:
        tran_add.append((tran_sum[i] - tran_sum[i-1]) / tran_sum[i-1] * 100)
```


```python
from pyecharts import Bar, Line, Overlap

attr1 = tran_data
v1 = tran_sum
v2 = tran_add
bar1 = Bar("每日交易信息汇总")
bar1.add("日期 /金额", attr1, v1, mark_line=["average"], mark_point=["max", "min"], is_datazoom_show=True)
line1 = Line()
line1.add('环比增长率(%)',attr1, v2, mark_line=["average"], mark_point=["max", "min"], is_datazoom_show=True )

line2 = Line()
line2.add("日期/ 金额", attr1, v1, mark_line=["average"], mark_point=["max", "min"], is_datazoom_show=True)

overlap = Overlap()
overlap.add(bar1)
# overlap.add(bar2)
overlap.add(line1)
overlap.render()
overlap

#bar.add("evaporation", attr, v2, mark_line=["average"], mark_point=["max", "min"])
#bar.render()

```

![png](/images/pyecharts/1.png)

## 2016-04-29各出口交易金额


```python
import pymysql
conn = pymysql.connect(host='192.168.56.111', port=3306, user='hive', passwd='hive', db='test', charset='utf8')
# 创建游标
cursor = conn.cursor()
#2016-04-29每个出口id交易额信息
cursor.execute("select in_id from in_sum where in_date='429' order by in_sum desc")
in_plazaid = cursor.fetchall()
#2016-04-29每个出口交易额
cursor.execute("select in_sum from in_sum where in_date='429' order by in_sum desc")
in_transum = [*map(lambda x :x[0]/100,list(cursor.fetchall()))]
cursor.close()

from pyecharts import Bar

attr2 = in_plazaid
v2 = in_transum
bar2 = Bar("2016-4-29日各出口交易额汇总","")
bar2.add("出口ID/金额（元）", attr2, v2, mark_line=["average"], mark_point=["max", "min"], is_datazoom_show = True)
#bar.add("evaporation", attr, v2, mark_line=["average"], mark_point=["max", "min"])
bar2.render()
bar2
```

![png](/images/pyecharts/2.png)

## 2016-04-29 入出口交易金额 


```python
import pymysql
conn = pymysql.connect(host='192.168.56.111', port=3306, user='hive', passwd='hive', db='test', charset='utf8')
# 创建游标
cursor = conn.cursor()
cursor.execute("select ent_plazaid from test.tran_ent_plaza_sum where trans_date='2016-04-29' order by trans_sum desc")
ent_plazaid = cursor.fetchall()

cursor.execute("select trans_sum from test.tran_ent_plaza_sum where trans_date='2016-04-29' order by trans_sum desc")
ent_transum = [*map(lambda x :x[0]/100,list(cursor.fetchall()))]

cursor.close()

from pyecharts import Bar

attr3 = ent_plazaid
v3 = ent_transum
bar3 = Bar("2016-4-29日各入口交易额汇总","")
bar3.add("入口ID/金额（元）", attr3, v3, mark_line=["average"], mark_point=["max", "min"], is_datazoom_show = True)
#bar.add("evaporation", attr, v2, mark_line=["average"], mark_point=["max", "min"])
bar3.render()
bar3
```

![png](/images/pyecharts/3.png)


## 某入口交易额汇总


```python
import pymysql
conn = pymysql.connect(host='192.168.56.111', port=3306, user='hive', passwd='hive', db='test', charset='utf8')
# 创建游标
cursor = conn.cursor()
cursor.execute("select trans_date from test.tran_ent_plaza_sum where ent_plazaid=100859")
ent_date = cursor.fetchall()
cursor.execute("select trans_sum from test.tran_ent_plaza_sum where ent_plazaid=100859")
transum_ = [*map(lambda x :x[0]/100,list(cursor.fetchall()))]
cursor.close()

from pyecharts import Bar

attr3 = ent_date
v3 = transum_
bar3 = Bar("某出口每日交易额","")
bar3.add("出口/金额（元）", attr3, v3, mark_line=["average"], mark_point=["max", "min"], is_datazoom_show = True)
#bar.add("evaporation", attr, v2, mark_line=["average"], mark_point=["max", "min"])
bar3.render()
bar3
```

![png](/images/pyecharts/4.png)

## 4-29日金额突增原因分析


```python
from pyecharts import Funnel

#根据2016-4-29出入口及各车型的交易额信息，每个组选取前4个贡献率最高的样本。
#利用除均操作对这三种因素的影响率进行标准化。
#下图可以看出2016-4-29日，这三种影响因素的前四个样本分别占改组贡献率的比例
#100108红门主站出京入口、100861榆垡南出京出口、车型-1对该日的收费的贡献率最大

attr = ['100861榆垡南出京出','100158璃河南出京出','100109红门主站进京出','100135六里桥站进京出','100108红门主站出京入','100134六里桥站出京入','100862榆垡南进京入','100175西红门南桥出京入','车型-1','车型-4','车型-3','车型-2']
value = [6.24, 6.01, 4.72, 4.71, 16.75, 15.52, 10.85, 8.08, 4.46, 0.29, 0.13, 0.1]
funnel = Funnel("")
funnel.add("因素", attr, value, is_label_show=True,
           label_pos="inside", label_text_color="#fff")
funnel.render()
funnel
```

![png](/images/pyecharts/5.png)

## 2016-04-29 各车型交易汇总


```python
import pymysql
conn = pymysql.connect(host='192.168.56.111', port=3306, user='hive', passwd='hive', db='test', charset='utf8')
# 创建游标
cursor = conn.cursor()
cursor.execute("select distinct vehtype from test.tran_vehtype_sum where trans_date = '2016-04-29' order by vehtype")
type_id = cursor.fetchall()

cursor.execute("select distinct trans_sum from test.tran_vehtype_sum where trans_date = '2016-04-29' order by vehtype")
type_sum = [*map(lambda x :x[0]/100,list(cursor.fetchall()))]
cursor.close()
#========================

from pyecharts import Bar

attr4 = type_id
v4 = type_sum
bar4 = Bar("2016-4-29日各车型交易额汇总","")
bar4.add("车型/金额（元）", attr4, v4, mark_line=["average"], mark_point=["max", "min"], is_datazoom_show = True)
#bar.add("evaporation", attr, v2, mark_line=["average"], mark_point=["max", "min"])
bar4.render()
bar4
```

![png](/images/pyecharts/6.png)

## 全日期各车型交易额汇总


```python
import pandas as pd
import numpy as np
import pymysql

conn = pymysql.connect(host='192.168.56.111', port=3306, user='hive', passwd='hive', db='test', charset='utf8')
cursor = conn.cursor()
cursor.execute("select * from tran_vehtype_sum")
typedata = cursor.fetchall()
cursor.close()

typedata = pd.DataFrame(list(typedata))
typedata.columns = ['type','date','value']

data=pd.pivot_table(typedata,index="date",columns="type",values="value",aggfunc=np.sum,fill_value=0)
# data.to_csv("data.csv")

from pyecharts import Bar

y, x1, x2, x3, x4, x5 = [], [], [], [], [], []
for i in range(len(data)):
    y.append(data.index[i])
    x1.append(data[1][i])
    x2.append(data[2][i])
    x3.append(data[3][i])
    x4.append(data[4][i])
    x5.append(data[5][i])

bar = Bar("各车型每日交易额")
bar.add("车型-1", y, x1, is_stack=True, mark_line=["average"], mark_point=["max", "min"], is_datazoom_show = True)
bar.add("车型-2", y, x2, is_stack=True, mark_line=["average"], mark_point=["max", "min"], is_datazoom_show = True)
bar.add("车型-3", y, x3, is_stack=True, mark_line=["average"], mark_point=["max", "min"], is_datazoom_show = True)
bar.add("车型-4", y, x4, is_stack=True, mark_line=["average"], mark_point=["max", "min"], is_datazoom_show = True)
bar.add("车型-5", y, x5, is_stack=True, mark_line=["average"], mark_point=["max", "min"], is_datazoom_show = True)
bar.render()
bar
```

![png](/images/pyecharts/7.png)

## PyMysql使用


```python
import pymysql
#创建连接
conn = pymysql.connect(host='192.168.56.111', port=3306, user='hive', passwd='hive', db='test', charset='utf8')
# 创建游标
cursor = conn.cursor()
  
# 执行SQL，并返回收影响行数
effect_row = cursor.execute("select * from tran_day_sum")
  
# 执行SQL，并返回受影响行数
#effect_row = cursor.execute("update tb7 set pass = '123' where nid = %s", (11,))
  
# 执行SQL，并返回受影响行数,执行多次
#effect_row = cursor.executemany("insert into tb7(user,pass,licnese)values(%s,%s,%s)", [("u1","u1pass","11111"),("u2","u2pass","22222")])
  
  
# 提交，不然无法保存新建或者修改的数据
conn.commit()
  
# 关闭游标
cursor.close()
# 关闭连接
conn.close()

```


```python
#分析每个入口的交易笔数和交易总额
import pymysql

conn = pymysql.connect(host='192.168.56.111', port=3306, user='hive', passwd='hive', db='test', charset='utf8')
cursor = conn.cursor()
cursor.execute("select ent_plazaid from tran_ent_plaza_sum group by ent_plazaid")
tran_id = cursor.fetchall()

cursor.execute("select count(trans_sum) from tran_ent_plaza_sum group by ent_plazaid")
tran_count = [*map(lambda x :x[0],list(cursor.fetchall()))]

cursor.execute("select sum(trans_sum) from tran_ent_plaza_sum group by ent_plazaid")
tran_sum = [*map(lambda x :int(x[0]),list(cursor.fetchall()))]

cursor.close()

```
