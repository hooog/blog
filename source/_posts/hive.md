---
title: Hive使用Mysql作为元数据库报错的bug解决办法
date: 2018-03-25 17:05:59
password:
top:
categories:
  - Hadoop
tags:
  - Hive
---
<!--more-->

配置好hive-site.xml用mysql作为hive的元数据库之后，使用命令进行格式化：schematool -dbType mysql -initSchema 失败报错：
```
Logging initialized using configuration in jar:file:/hive/apache-hive-1.1.0-bin/lib/hive-common-1.1.0.jar!/hive-log4j.properties
SLF4J: Class path contains multiple SLF4J bindings.
SLF4J: Found binding in [jar:file:/hadoop-2.5.2/share/hadoop/common/lib/slf4j-log4j12-1.7.5.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: Found binding in [jar:file:/hive/apache-hive-1.1.0-bin/lib/hive-jdbc-1.1.0-standalone.jar!/org/slf4j/impl/StaticLoggerBinder.class]
SLF4J: See http://www.slf4j.org/codes.html#multiple_bindings for an explanation.
SLF4J: Actual binding is of type [org.slf4j.impl.Log4jLoggerFactory]
[ERROR] Terminal initialization failed; falling back to unsupported
java.lang.IncompatibleClassChangeError: Found class jline.Terminal, but interface was expected
        at jline.TerminalFactory.create(TerminalFactory.java:101)
        at jline.TerminalFactory.get(TerminalFactory.java:158)
        at jline.console.ConsoleReader.<init>(ConsoleReader.java:229)
        at jline.console.ConsoleReader.<init>(ConsoleReader.java:221)
        at jline.console.ConsoleReader.<init>(ConsoleReader.java:209)
        at org.apache.hadoop.hive.cli.CliDriver.getConsoleReader(CliDriver.java:773)
        at org.apache.hadoop.hive.cli.CliDriver.executeDriver(CliDriver.java:715)
        at org.apache.hadoop.hive.cli.CliDriver.run(CliDriver.java:675)
        at org.apache.hadoop.hive.cli.CliDriver.main(CliDriver.java:615)
        at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
        at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:57)
        at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
        at java.lang.reflect.Method.invoke(Method.java:606)
        at org.apache.hadoop.util.RunJar.main(RunJar.java:212)

```
原因是hadoop目录下存在老版本jline

解决办法：
- 将hive下的新版本jline的JAR包拷贝到hadoop下
```
cp /hive/lib/jline-2.12.jar ./hadoop/share/hadoop/yarn/lib：
```

重新输入schematool -dbType mysql -initSchema ，格式化成功。
这是进入mysql的hive数据库中即可看到hive的元数据结构信息