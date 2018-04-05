---
title: 利用Pyhive实现Python连接Hive数据仓库
date: 2018-03-21 10:05:59
password:
top:
categories:
  - Hadoop
tags:
  - Hive
  - Pyhive
---
<!--more-->


### 安装Pyhive相关依赖
```
sudo apt-get install libsasl2-dev（Ubuntu里需要执行）
pip install sasl
pip install thrift
pip install thrift-sasl
pip install PyHive
```

### 启动hiveserver2服务:
```
hive --service hiveserver2 &
netstat -an | grep 10000
```
### 尝试连接
- **Pyhive同步连接**
```python
# Pyhive同步代码
from pyhive import presto, hive  # or import hive
cursor = presto.connect('localhost', port=10000).cursor()
cursor.execute('SELECT * FROM test LIMIT 10')
print (cursor.fetchone())
print (cursor.fetchall())
```
- 报错：
```python
ConnectionError: ('Connection aborted.', BadStatusLine('\x04\x00\x00\x00\x11Invalid status 80',))
```

- **Pyhive异步连接**
```python
# Phive 异步代码
from pyhive import hive
from TCLIService.ttypes import TOperationState
cursor = hive.connect('localhost').cursor()
cursor.execute('SELECT * FROM test LIMIT 10', async=True)

status = cursor.poll().operationState
while status in (TOperationState.INITIALIZED_STATE, TOperationState.RUNNING_STATE):
    logs = cursor.fetch_logs()
    for message in logs:
        print (message)

    # If needed, an asynchronous query can be cancelled at any time with:
    # cursor.cancel()

    status = cursor.poll().operationState

print (cursor.fetchall())
```
- 报错：
```python
OperationalError: TOpenSessionResp(status=TStatus(statusCode=3, infoMessages=['*org.apache.hive.service.cli.HiveSQLException:Failed to open new session: java.lang.RuntimeException: org.apache.hadoop.ipc.RemoteException(org.apache.hadoop.security.authorize.AuthorizationException): User: hadoop is not allowed to impersonate hadoop:14:13', 'org.apache.hive.service.cli.session.SessionManager:createSession:SessionManager.java:336', 'org.apache.hive.service.cli.session.SessionManager:openSession:SessionManager.java:279', 'org.apache.hive.service.cli.CLIService:openSessionWithImpersonation:CLIService.java:189', 'org.apache.hive.service.cli.thrift.ThriftCLIService:getSessionHandle:ThriftCLIService.java:423', 'org.apache.hive.service.cli.thrift.ThriftCLIService:OpenSession:ThriftCLIService.java:312', 'org.apache.hive.service.rpc.thrift.TCLIService$Processor$OpenSession:getResult:TCLIService.java:1377', 'org.apache.hive.service.rpc.thrift.TCLIService$Processor$OpenSession:getResult:TCLIService.java:1362', 'org.apache.thrift.ProcessFunction:process:ProcessFunction.java:39', 'org.apache.thrift.TBaseProcessor:process:TBaseProcessor.java:39', 'org.apache.hive.service.auth.TSetIpAddressProcessor:process:TSetIpAddressProcessor.java:56', 'org.apache.thrift.server.TThreadPoolServer$WorkerProcess:run:TThreadPoolServer.java:286', 'java.util.concurrent.ThreadPoolExecutor:runWorker:ThreadPoolExecutor.java:1149', 'java.util.concurrent.ThreadPoolExecutor$Worker:run:ThreadPoolExecutor.java:624', 'java.lang.Thread:run:Thread.java:748', '*java.lang.RuntimeException:java.lang.RuntimeException: org.apache.hadoop.ipc.RemoteException(org.apache.hadoop.security.authorize.AuthorizationException): User: hadoop is not allowed to impersonate hadoop:22:8', 'org.apache.hive.service.cli.session.HiveSessionProxy:invoke:HiveSessionProxy.java:89', 'org.apache.hive.service.cli.session.HiveSessionProxy:access$000:HiveSessionProxy.java:36', 'org.apache.hive.service.cli.session.HiveSessionProxy$1:run:HiveSessionProxy.java:63', 'java.security.AccessController:doPrivileged:AccessController.java:-2', 'javax.security.auth.Subject:doAs:Subject.java:422', 'org.apache.hadoop.security.UserGroupInformation:doAs:UserGroupInformation.java:1692', 'org.apache.hive.service.cli.session.HiveSessionProxy:invoke:HiveSessionProxy.java:59', 'com.sun.proxy.$Proxy35:open::-1', 'org.apache.hive.service.cli.session.SessionManager:createSession:SessionManager.java:327', '*java.lang.RuntimeException:org.apache.hadoop.ipc.RemoteException(org.apache.hadoop.security.authorize.AuthorizationException): User: hadoop is not allowed to impersonate hadoop:29:7', 'org.apache.hadoop.hive.ql.session.SessionState:start:SessionState.java:591', 'org.apache.hadoop.hive.ql.session.SessionState:start:SessionState.java:526', 'org.apache.hive.service.cli.session.HiveSessionImpl:open:HiveSessionImpl.java:168', 'sun.reflect.NativeMethodAccessorImpl:invoke0:NativeMethodAccessorImpl.java:-2', 'sun.reflect.NativeMethodAccessorImpl:invoke:NativeMethodAccessorImpl.java:62', 'sun.reflect.DelegatingMethodAccessorImpl:invoke:DelegatingMethodAccessorImpl.java:43', 'java.lang.reflect.Method:invoke:Method.java:498', 'org.apache.hive.service.cli.session.HiveSessionProxy:invoke:HiveSessionProxy.java:78', '*org.apache.hadoop.ipc.RemoteException:User: hadoop is not allowed to impersonate hadoop:49:20', 'org.apache.hadoop.ipc.Client:call:Client.java:1470', 'org.apache.hadoop.ipc.Client:call:Client.java:1401', 'org.apache.hadoop.ipc.ProtobufRpcEngine$Invoker:invoke:ProtobufRpcEngine.java:232', 'com.sun.proxy.$Proxy30:getFileInfo::-1', 'org.apache.hadoop.hdfs.protocolPB.ClientNamenodeProtocolTranslatorPB:getFileInfo:ClientNamenodeProtocolTranslatorPB.java:752', 'sun.reflect.NativeMethodAccessorImpl:invoke0:NativeMethodAccessorImpl.java:-2', 'sun.reflect.NativeMethodAccessorImpl:invoke:NativeMethodAccessorImpl.java:62', 'sun.reflect.DelegatingMethodAccessorImpl:invoke:DelegatingMethodAccessorImpl.java:43', 'java.lang.reflect.Method:invoke:Method.java:498', 'org.apache.hadoop.io.retry.RetryInvocationHandler:invokeMethod:RetryInvocationHandler.java:187', 'org.apache.hadoop.io.retry.RetryInvocationHandler:invoke:RetryInvocationHandler.java:102', 'com.sun.proxy.$Proxy31:getFileInfo::-1', 'org.apache.hadoop.hdfs.DFSClient:getFileInfo:DFSClient.java:1977', 'org.apache.hadoop.hdfs.DistributedFileSystem$18:doCall:DistributedFileSystem.java:1118', 'org.apache.hadoop.hdfs.DistributedFileSystem$18:doCall:DistributedFileSystem.java:1114', 'org.apache.hadoop.fs.FileSystemLinkResolver:resolve:FileSystemLinkResolver.java:81', 'org.apache.hadoop.hdfs.DistributedFileSystem:getFileStatus:DistributedFileSystem.java:1114', 'org.apache.hadoop.fs.FileSystem:exists:FileSystem.java:1400', 'org.apache.hadoop.hive.ql.session.SessionState:createRootHDFSDir:SessionState.java:689', 'org.apache.hadoop.hive.ql.session.SessionState:createSessionDirs:SessionState.java:635', 'org.apache.hadoop.hive.ql.session.SessionState:start:SessionState.java:563'], sqlState=None, errorCode=0, errorMessage='Failed to open new session: java.lang.RuntimeException: org.apache.hadoop.ipc.RemoteException(org.apache.hadoop.security.authorize.AuthorizationException): User: hadoop is not allowed to impersonate hadoop'), serverProtocolVersion=8, sessionHandle=None, configuration=None)
```
**错误信息：**
```
errorMessage='Failed to open new session: java.lang.RuntimeException: org.apache.hadoop.ipc.RemoteException(org.apache.hadoop.security.authorize.AuthorizationException): User: hadoop is not allowed to impersonate hadoop'), serverProtocolVersion=8, sessionHandle=None, configuration=None)
```

### 解决方案
这个报错的原因：
用户代理未生效。增加检查core-site.xml配置。
```
<property>
  <name>hadoop.proxyuser.hadoop.hosts</name>
  <value>*</value>
</property>
<property>
  <name>hadoop.proxyuser.hadoop.groups</name>
 <value>hadoop</value>
</property>
```
重启hadoop、yarn、MR再次尝试报同样错误。

修改hadoop.proxyuser.hadoop.groups的value为*
```
<property>
  <name>hadoop.proxyuser.hadoop.hosts</name>
  <value>*</value>
</property>
<property>
  <name>hadoop.proxyuser.hadoop.groups</name>
 <value>*</value>
</property>
```

重启hadoop、yarn、MR再次尝试：
```
# Phive 异步代码
from pyhive import hive
from TCLIService.ttypes import TOperationState
cursor = hive.connect('localhost').cursor()
cursor.execute('SELECT * FROM test LIMIT 10', async=True)

status = cursor.poll().operationState
while status in (TOperationState.INITIALIZED_STATE, TOperationState.RUNNING_STATE):
    logs = cursor.fetch_logs()
    for message in logs:
        print (message)

    # If needed, an asynchronous query can be cancelled at any time with:
    # cursor.cancel()

    status = cursor.poll().operationState

print (cursor.fetchall())
```
返回结果：
`[('1', 'beijing'), ('2', 'shanghai'), ('3', 'nanjing')]`

至此Pyhive连接Hive数据库成功。