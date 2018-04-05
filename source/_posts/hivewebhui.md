---
title: Hive-2.2.0启动webui端口
date: 2018-03-26 00:28:59
password:
top:
categories:
  - Hadoop
tags:
  - Hive
---
<!--more-->
这里Hive-2.2.0启动web端口的前提是启动hiveserver2和HiveMetaStore,这里关于hive.server2.wubui的默认配置在hive-default.xml里面：

```
    <name>hive.server2.webui.host</name>
    <value>0.0.0.0</value>
    <description>The host address the HiveServer2 WebUI will listen on</description>
  </property>
  <property>
    <name>hive.server2.webui.port</name>
    <value>10002</value>
```


输入命令启动hiveserver2和HiveMetaStore服务：
```
hive --service metastore &
hive --service hiveserver2 &

或者直接hiveserver2
```

hiveserver2服务的默认端口号是 `10000`
这时查看下hiveserver2和HiveMetaStore状态：
```
netstat -an | grep 10000
tcp4       0      0  *.10000                *.*                    LISTEN

netstat -an | grep 10002
tcp4       0      0  *.10002                *.*                    LISTEN
```
显示`10000`和`10002`端口均在监听状态
 ps -ef | grep Hive显示也正常
```
org.apache.hadoop.util.RunJar /usr/local/hive/lib/hive-metastore-2.1.1.jar org.apache.hadoop.hive.metastore.HiveMetaStore

org.apache.hadoop.util.RunJar /usr/local/hive/lib/hive-service-2.1.1.jar org.apache.hive.service.server.HiveServer2
  501  1898  1609   0 12:14上午 ttys000    0:00.01 grep --color=auto --exclude-dir=.bzr --exclude-dir=CVS --exclude-dir=.git --exclude-dir=.hg --exclude-dir=.svn Hive
```

此时浏览器输入 `localhost:10002`

成功进入web页面。

总结：网上大多启动webui的方法都只是提到开启hiveserver2服务，但经过多次测试均不能打开web界面，最终误打误撞启动了HiveMetaStore，至此hivewebui才能正常访问。
启动hivewebui的前提是：10000，和10002端口都要启动才行！










