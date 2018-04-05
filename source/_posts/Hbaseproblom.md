---
title: Hbase启动后进程丢失的原因分析
date: 2018-03-21 11:59:21
update: 
comments: true
categories:
  - Hadoop
tags:
  - Hbase
top: 10
---
<!--more-->

### Hbase启动后进程丢失的原因分析

**原理：**
	
- 分布式Apache HBase安装依赖于正在运行的ZooKeeper集群。**Apache HBase默认情况下为您管理ZooKeeper“集群”。**它将启动和停止ZooKeeper集合作为HBase启动/停止过程的一部分。
- 当然也可以独立于HBase管理ZooKeeper集群，只需要在Hbase的配置文件hbase-env.sh中做一些设置即可。要切换ZooKeeper的HBase管理，请使用conf/hbase-env.sh中的HBASE_MANAGES_ZK变量。 此变量默认为true，告诉HBase是否启动/停止ZooKeeper集合服务器作为HBase启动/停止的一部分。如果为true，这Hbase把zookeeper启动，停止作为自身启动和停止的一部分。如果设置为false，则表示独立的Zookeeper管理。

- 既然我们配置的是HBase管理zookeeper，那么zookeeper在给Hbase提供底层支撑的时候需要与Hbase建立通信，这里最直接高效的通信方式就是建立在本地回环上。
- 由于最初虚拟机安装Hbase和zookeeper后启动服务的时候Hbase默认是按照 hosts文件的第二行127.0.1.1的本地回环地址和匹配的主机名建立通信（这里我们默认安装的是hduser）。这时zookeeper会生成一个记录文件把127.0.1.1和主机名保存下来。
- 后来改成集群模式的时候，把hosts文件第二行的127.0.1.1以及对应的主机名屏蔽掉了。但是Hbase还是按照之前的本地回环地址和主机名尝试取联系对应的zookeeper，但是“身份“信息已经在hosts中被我们屏蔽掉了，所以造成最直接的影响就是主节点的HMaster进程启动后很快丢失，并且可能会造成master节点多了个HRegionServer进程以及从节点的进程紊乱。
- 为什么会这样呢？ 启动HBase后我们jps看一下主节点的进程有个HMaster和一个HQ*的进程。 HQ*这个进程是用来寻找zookeeper的，但是它这时找不到zookeeper，所以就和HMaster一起“死掉了”。
- 同理这也是为什么为分布式集群改为分布式集群启动之前需要删除tmp并格式化的原因。否则Apache-hadoop会默认按照之前记录的本地回环地址进行寻址。


这里的解决方案是彻底删除hbase和zookeeper组件重新安装配置后即可正常启动，且进程将不会再莫名丢失。

**主节点进程：**
![Alt text](/images/1.png)

**从节点进程：**
![Alt text](/images/2.png)


