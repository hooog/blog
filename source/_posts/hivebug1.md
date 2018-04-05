---
title: Hive提交MapReduce任务失败The auxService:mapreduce_shuffle does not exist
date: 2018-03-25 17:05:59
password:
top:
categories:
  - Hadoop
tags:
  - Hive
---
<!--more-->

Hive提交MapReduce任务失败The auxService:mapreduce_shuffle does not exist
```
hive> create table t1 as select * from test;
Query ID = hadoop_20180325202454_fb66568f-8299-4367-88a0-3f5cf3bc8374
Total jobs = 3
Launching Job 1 out of 3
Number of reduce tasks is set to 0 since there's no reduce operator
Starting Job = job_1521979558681_0002, Tracking URL = http://bogon:8088/proxy/application_1521979558681_0002/
Kill Command = /usr/local/hadoop/bin/hadoop job  -kill job_1521979558681_0002
Hadoop job information for Stage-1: number of mappers: 1; number of reducers: 0
2018-03-25 20:25:01,203 Stage-1 map = 0%,  reduce = 0%
2018-03-25 20:25:10,531 Stage-1 map = 100%,  reduce = 0%
Ended Job = job_1521979558681_0002 with errors
Error during job, obtaining debugging information...
Examining task ID: task_1521979558681_0002_m_000000 (and more) from job job_1521979558681_0002

Task with the most failures(4):
-----
Task ID:
  task_1521979558681_0002_m_000000

URL:
  http://0.0.0.0:8088/taskdetails.jsp?jobid=job_1521979558681_0002&tipid=task_1521979558681_0002_m_000000
-----
Diagnostic Messages for this Task:
Container launch failed for container_1521979558681_0002_01_000005 : org.apache.hadoop.yarn.exceptions.InvalidAuxServiceException: The auxService:mapreduce_shuffle does not exist
    at sun.reflect.NativeConstructorAccessorImpl.newInstance0(Native Method)
    at sun.reflect.NativeConstructorAccessorImpl.newInstance(NativeConstructorAccessorImpl.java:62)
    at sun.reflect.DelegatingConstructorAccessorImpl.newInstance(DelegatingConstructorAccessorImpl.java:45)
    at java.lang.reflect.Constructor.newInstance(Constructor.java:423)
    at org.apache.hadoop.yarn.api.records.impl.pb.SerializedExceptionPBImpl.instantiateException(SerializedExceptionPBImpl.java:168)
    at org.apache.hadoop.yarn.api.records.impl.pb.SerializedExceptionPBImpl.deSerialize(SerializedExceptionPBImpl.java:106)
    at org.apache.hadoop.mapreduce.v2.app.launcher.ContainerLauncherImpl$Container.launch(ContainerLauncherImpl.java:155)
    at org.apache.hadoop.mapreduce.v2.app.launcher.ContainerLauncherImpl$EventProcessor.run(ContainerLauncherImpl.java:369)
    at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149)
    at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624)
    at java.lang.Thread.run(Thread.java:748)


FAILED: Execution Error, return code 2 from org.apache.hadoop.hive.ql.exec.mr.MapRedTask
MapReduce Jobs Launched:
Stage-Stage-1: Map: 1   HDFS Read: 0 HDFS Write: 0 FAIL
Total MapReduce CPU Time Spent: 0 msec

```
