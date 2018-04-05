---
title: DStream的基本操作
date: 2018-03-13 11:59:21
update: 
comments: true
categories:
  - Machine learning
tags:
  - MLlib
---

<!--more-->
本文参考- Machine learning with Scala by CDA 吴昊天

## 一、转换操作

### 1.概述

DStream转换操作包括无状态转换和有状态转换

- 无状态转换：每个批次的处理不依赖于之前批次的数据；
- 有状态转换：当前批次的处理需要使用之前批次的数据或者中间结果。有状态转换包括基于滑动窗口的转换和追踪状态变化的转换(updateStateByKey)。

### 2.DStream无状态转换操作

下面给出一些无状态转换操作的含义：

- map(func) ：对源DStream的每个元素，采用func函数进行转换，得到一个新的DStream；
- flatMap(func)： 与map相似，但是每个输入项可用被映射为0个或者多个输出项；
- filter(func)： 返回一个新的DStream，仅包含源DStream中满足函数func的项；
- repartition(numPartitions)： 通过创建更多或者更少的分区改变DStream的并行程度；
- union(otherStream)： 返回一个新的DStream，包含源DStream和其他DStream的元素；
- count()：统计源DStream中每个RDD的元素数量；
- reduce(func)：利用函数func聚集源DStream中每个RDD的元素，返回一个包含单元素RDDs的新DStream；
- countByValue()：应用于元素类型为K的DStream上，返回一个（K，V）键值对类型的新DStream，每个键的值是在原DStream的每个RDD中的出现次数；
- reduceByKey(func, [numTasks])：当在一个由(K,V)键值对组成的DStream上执行该操作时，返回一个新的由(K,V)键值对组成的DStream，每一个key的值均由给定的recuce函数（func）聚集起来；
- join(otherStream, [numTasks])：当应用于两个DStream（一个包含（K,V）键值对,一个包含(K,W)键值对），返回一个包含(K, (V, W))键值对的新DStream；
- cogroup(otherStream, [numTasks])：当应用于两个DStream（一个包含（K,V）键值对,一个包含(K,W)键值对），返回一个包含(K, Seq[V], Seq[W])的元组；
- transform(func)：通过对源DStream的每个RDD应用RDD-to-RDD函数，创建一个新的DStream。支持在新的DStream中做任何RDD操作。

无状态转换操作实例：我们之前套接字流部分介绍的词频统计，就是采用无状态转换，每次统计，都是只统计当前批次到达的单词的词频，和之前批次无关，不会进行累计。

### 3.DStream有状态转换操作

#### 3.1 概述

​	对于DStream有状态转换操作而言，当前批次的处理需要使用之前批次的数据或者中间结果。有状态转换包括基于滑动窗口的转换和追踪状态变化(updateStateByKey)的转换。

#### 3.2 滑动窗口转换操作基本概念

​	滑动窗口转换操作的计算过程如下图所示，我们可以事先设定一个滑动窗口的长度（也就是窗口的持续时间），并且设定滑动窗口的时间间隔（每隔多长时间执行一次计算），然后，就可以让窗口按照指定时间间隔在源DStream上滑动，每次窗口停放的位置上，都会有一部分DStream被框入窗口内，形成一个小段的DStream，这时，就可以启动对这个小段DStream的计算。

![sparkstreaming滑动窗口](http://dblab.xmu.edu.cn/blog/wp-content/uploads/2016/11/SparkStreaming%E6%BB%91%E5%8A%A8%E7%AA%97%E5%8F%A3%E8%BD%AC%E6%8D%A2%E6%93%8D%E4%BD%9C.png)

下面给给出一些窗口转换操作的含义：

- window(windowLength, slideInterval) 基于源DStream产生的窗口化的批数据，计算得到一个新的DStream；
- countByWindow(windowLength, slideInterval) 返回流中元素的一个滑动窗口数；
- reduceByWindow(func, windowLength, slideInterval) 返回一个单元素流。利用函数func聚集滑动时间间隔的流的元素创建这个单元素流。函数func必须满足结合律，从而可以支持并行计算；
- reduceByKeyAndWindow(func, windowLength, slideInterval, [numTasks]) 应用到一个(K,V)键值对组成的DStream上时，会返回一个由(K,V)键值对组成的新的DStream。每一个key的值均由给定的reduce函数(func函数)进行聚合计算。注意：在默认情况下，这个算子利用了Spark默认的并发任务数去分组。可以通过numTasks参数的设置来指定不同的任务数；
- reduceByKeyAndWindow(func, invFunc, windowLength, slideInterval, [numTasks]) 更加高效的reduceByKeyAndWindow，每个窗口的reduce值，是基于先前窗口的reduce值进行增量计算得到的；它会对进入滑动窗口的新数据进行reduce操作，并对离开窗口的老数据进行“逆向reduce”操作。但是，只能用于“可逆reduce函数”，即那些reduce函数都有一个对应的“逆向reduce函数”（以InvFunc参数传入）；
- countByValueAndWindow(windowLength, slideInterval, [numTasks]) 当应用到一个(K,V)键值对组成的DStream上，返回一个由(K,V)键值对组成的新的DStream。每个key的值都是它们在滑动窗口中出现的频率。

#### 3.3 窗口转换操作实例：

​	在之前的Apache Kafka作为DStream数据源内容中，在我们已经使用了窗口转换操作，也就是，在KafkaWordCount.scala代码中，你可以找到下面这一行：

```scala
val wordCounts = pair.reduceByKeyAndWindow(_ + _,_ - _,Minutes(2),Seconds(10),2)
```

​	这行代码中就是一个窗口转换操作reduceByKeyAndWindow，其中，Minutes(2)是滑动窗口长度，Seconds(10)是滑动窗口时间间隔（每隔多长时间滑动一次窗口）。reduceByKeyAndWindow中就使用了加法和减法这两个reduce函数，加法和减法这两种reduce函数都是“可逆的reduce函数”，也就是说，当滑动窗口到达一个新的位置时，原来之前被窗口框住的部分数据离开了窗口，又有新的数据被窗口框住，但是，这时计算窗口内单词的词频时，不需要对当前窗口内的所有单词全部重新执行统计，而是只要把窗口内新增进来的元素，增量加入到统计结果中，把离开窗口的元素从统计结果中减去，这样，就大大提高了统计的效率。尤其对于窗口长度较大时，这种“逆函数”带来的效率的提高是很明显的。

#### 3.4 updateStateByKey操作

​	当我们需要在跨批次之间维护状态时，就必须使用updateStateByKey操作。下面我们就给出一个具体实例。我们还是以前面在套接字流部分讲过的NetworkWordCount为例子来介绍，在之前的套接字流的介绍中，我们统计单词词频采用的是无状态转换操作，也就是说，每个批次的单词发送给NetworkWordCount程序处理时，NetworkWordCount只对本批次内的单词进行词频统计，不会考虑之前到达的批次的单词，所以，不同批次的单词词频都是独立统计的。
​	对于有状态转换操作而言，本批次的词频统计，会在之前批次的词频统计结果的基础上进行不断累加，所以，最终统计得到的词频，是所有批次的单词的总的词频统计结果。下面，我们来改造一下在套接字流介绍过的NetworkWordCount程序。在master上执行

```shell
cd ~/mycode                           
mkdir -p ./stateful/src/main/scala                   # 创建项目结构
cd ./stateful/src/main/scala/
vim NetworkWordCountStateful.scala                   # 创建代码文件
```

输入以下代码：

```scala
import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.storage.StorageLevel

object NetworkWordCountStateful {
  def main(args: Array[String]) {
    //定义状态更新函数
    val updateFunc = (values: Seq[Int], state: Option[Int]) => {
      val currentCount = values.foldLeft(0)(_ + _)
      val previousCount = state.getOrElse(0)
      Some(currentCount + previousCount)
    }
        StreamingExamples.setStreamingLogLevels()  //设置log4j日志级别
    val conf = new SparkConf().setMaster("local[2]").setAppName("NetworkWordCountStateful")
    val sc = new StreamingContext(conf, Seconds(5))
    sc.checkpoint("file:///home/hadoop/mycode/stateful/")    //设置检查点，检查点具有容错机制
    val lines = sc.socketTextStream("localhost", 9999)
    val words = lines.flatMap(_.split(" "))
    val wordDstream = words.map(x => (x, 1))
    val stateDstream = wordDstream.updateStateByKey[Int](updateFunc)
    stateDstream.print()
    sc.start()
    sc.awaitTermination()
  }
}
```

保存并退出
​	这里要对这段代码中新增的updataStateByKey稍微解释一下。Spark Streaming的updateStateByKey可以把DStream中的数据按key做reduce操作，然后对各个批次的数据进行累加。注意，wordDstream.updateStateByKey[Int]每次传递给updateFunc函数两个参数，其中，第一个参数是某个key（即某个单词）的当前批次的一系列值的列表（Seq[Int]形式）,updateFunc函数中 val currentCount = values.foldLeft(0)(_ + _)的作用，就是计算这个被传递进来的与某个key对应的当前批次的所有值的总和，也就是当前批次某个单词的出现次数，保存在变量currentCount中。传递给updateFunc函数的第二个参数是某个key的历史状态信息，也就是某个单词历史批次的词频汇总结果。实际上，某个单词的历史词频应该是一个Int类型，这里为什么要采用Option[Int]呢？
​	Option[Int]是类型 Int的容器，更确切地说，你可以把它看作是某种集合，这个特殊的集合要么只包含一个元素（即单词的历史词频），要么就什么元素都没有（这个单词历史上没有出现过，所以没有历史词频信息）。之所以采用 Option[Int]保存历史词频信息，这是因为，历史词频可能不存在，很多时候，在值不存在时，需要进行回退，或者提供一个默认值，Scala 为Option类型提供了getOrElse方法，以应对这种情况。 state.getOrElse(0)的含义是，如果该单词没有历史词频统计汇总结果，那么，就取值为0，如果有历史词频统计结果，就取历史结果，然后赋值给变量previousCount。最后，当前值和历史值进行求和，并包装在Some中返回。

然后，再次使用vim编辑器新建一个StreamingExamples.scala文件，用于设置log4j日志级别，

```shell
vim StreamingExamples.scala
```

代码如下：

```scala
import org.apache.spark.Logging
import org.apache.log4j.{Level, Logger}
/** Utility functions for Spark Streaming examples. */
object StreamingExamples extends Logging {
  /** Set reasonable logging levels for streaming if the user has not configured log4j. */
  def setStreamingLogLevels() {
    val log4jInitialized = Logger.getRootLogger.getAllAppenders.hasMoreElements
    if (!log4jInitialized) {
      // We first log something to initialize Spark's default logging, then we override the
      // logging level.
      logInfo("Setting log level to [WARN] for streaming example." +
        " To override add a custom log4j.properties to the classpath.")
      Logger.getRootLogger.setLevel(Level.WARN)
    }
  }
}
```

退出vim编辑器。然后创建工程文件：

```shell
cd ~/mycode/stateful
vim simple.sbt
```

在simple.sbt中输入以下内容：

```sh
name := "Simple Project"
version := "1.0"
scalaVersion := "2.10.4"
libraryDependencies += "org.apache.spark" %% "spark-streaming" % "1.4.0"
```

保存并退出，然后查看项目结构

```
cd ~/mycode/stateful
find .
```

![26](G:\屏幕截图\Spark Streaming部分\26.jpg)

然后进行打包编译

```shell
cd ~/mycode/stateful
/usr/local/sbt/sbt package
```

打包成功以后，就可以输入以下命令启动这个程序：

```shell
cd ~/mycode/stateful
/usr/local/spark/bin/spark-submit --class "NetworkWordCountStateful" /home/hadoop/mycode/stateful/target/scala-2.10/simple-project_2.10-1.0.jar
```

执行上面命令后，就进入了监听状态

![27](G:\屏幕截图\Spark Streaming部分\27.jpg)

然后再开一个窗口作为nc窗口，启动nc程序：

```shell
nc -lk 9999
```

随机输入一些信息

![28](G:\屏幕截图\Spark Streaming部分\28.jpg)

然后，你切换到刚才的监听窗口，就会发现此时的计数是累计计数

![29](G:\屏幕截图\Spark Streaming部分\29.jpg)

至此，操作成功。

## 二、输出操作

​	在Spark应用中，外部系统经常需要使用到Spark DStream处理后的数据，因此，需要采用输出操作把DStream的数据输出到数据库或者文件系统中，此过程即成为DStream的输出操作。我们仍然以上述转换操作所举的例子进行后续操作。

### 1.把DStream输出到文本文件中

为了不破坏以前的代码，我们将转换操作中涉及代码复制一份，然后针对副本进行修改。在master执行：

```shell
cd ~/mycode
mkdir -p ./dstreamOutput/src/main/scala                                        # 创建项目结构
cp ~/mycode/stateful/src/main/scala/* ~/mycode/dstreamOutput/src/main/scala/
cp ~/mycode/stateful/simple.sbt ~/mycode/dstreamOutput
```

查看项目结构，确认没有出现误操作

```shell
cd ~/mycode/dstreamOutput
find .
```

![30](G:\屏幕截图\Spark Streaming部分\30.jpg)

然后在这个新的dstreamoutput文件夹中进行程序修改。
首先，我们尝试把DStream内容保存到文本文件中，可以使用如下语句：

```scala
stateDstream.saveAsTextFiles("file:///home/hadoop/mycode/dstreamOutput/output.txt")
```

然后修改NetworkWordCountStateful.scala代码文件：

```shell
cd ~/mycode/dstreamOutput/src/main/scala/
vim NetworkWordCountStateful.scala
```

我们要把这条保存数据的语句stateDstream.saveAsTextFiles()放入到NetworkWordCountStateful.scala代码中，修改后的代码如下（也可以把NetworkWordCountStateful.scala原来的代码内容全部删除，直接把下面修改后的代码复制进去）：

```scala
import org.apache.spark._
import org.apache.spark.streaming._
import org.apache.spark.storage.StorageLevel

object NetworkWordCountStateful {
  def main(args: Array[String]) {
    //定义状态更新函数
    val updateFunc = (values: Seq[Int], state: Option[Int]) => {
      val currentCount = values.foldLeft(0)(_ + _)
      val previousCount = state.getOrElse(0)
      Some(currentCount + previousCount)
    }
        StreamingExamples.setStreamingLogLevels()  //设置log4j日志级别
    val conf = new SparkConf().setMaster("local[2]").setAppName("NetworkWordCountStateful")
    val sc = new StreamingContext(conf, Seconds(5))
    sc.checkpoint("file:///home/hadoop/mycode/dstreamOutput/")    //设置检查点，检查点具有容错机制
    val lines = sc.socketTextStream("localhost", 9999)
    val words = lines.flatMap(_.split(" "))
    val wordDstream = words.map(x => (x, 1))
    val stateDstream = wordDstream.updateStateByKey[Int](updateFunc)
    stateDstream.print()
        //下面是新增的语句，把DStream保存到文本文件中
        stateDstream.saveAsTextFiles("file:///home/hadoop/mycode/dstreamOutput/output.txt")
    sc.start()
    sc.awaitTermination()
  }
}

```

然后进行打包编译

```shell
cd ~/mycode/dstreamOutput
/usr/local/sbt/sbt package
```

打包成功以后，请运行程序：

```shell
cd ~/mycode/dstreamOutput
/usr/local/spark/bin/spark-submit --class "NetworkWordCountStateful" /home/hadoop/mycode/dstreamOutput/target/scala-2.10/simple-project_2.10-1.0.jar
```

成功运行即可进入监听状态

![31](G:\屏幕截图\Spark Streaming部分\31.jpg)

然后再打开一个终端作为输入源

```shell
nc -lk 9999
```

![32](G:\屏幕截图\Spark Streaming部分\32.jpg)

然后监听窗口将显示词频累计统计结果

![33](G:\屏幕截图\Spark Streaming部分\33.jpg)

然后停止运行，查看输出结果

```
cd ~/mycode/dstreamOutput/
ls
```

![34](G:\屏幕截图\Spark Streaming部分\34.jpg)

​	由于我们在代码中有一句`val sc = new StreamingContext(conf, Seconds(5))`，也就是说，每隔5秒钟统计一次词频，所以，每隔5秒钟就会生成一次词频统计结果，并输出到`~/mycode/dstreamOutput/output.txt`中，每次生成的output.txt后面会自动被加上时间标记（比如1516790460000）。这里要注意，虽然我们把DStream输出到`~/mycode/dstreamOutput/output.txt`中，output.txt的命名看起来像一个文件，但是，实际上，spark会生成名称为output.txt的目录，而不是文件。
我们可以查看一下某个output.txt下面的内容：

```shell
cd ~/mycode/dstreamOutput
cat output.txt-1516790460000/*
```

![35](G:\屏幕截图\Spark Streaming部分\35.jpg)

和监听时显示结果相同，说明我们已经成功地把DStream输出到文本文件了。

### 2.把DStream写入到MySQL数据库中

接下来，练习将DStream数据写入MySQL数据库中相关操作，首先，打开MySQL服务，创建响应数据库

```shell
service mysql start
mysql -u root -p
```

然后创建数据库及对应的表，用于接下来的数据存储

```mysql
create database spark;
show databases;
use spark;
create table wordcount (word char(20), count int(4));
show tables;
select * from wordcount;
```

可以看到，表wordcount目前为空；

![36](G:\屏幕截图\Spark Streaming部分\36.jpg)

再开一个命令行，创建工程相关代码和文档结构

```shell
cd ~/mycode
mkdir -p ./dstreamOutput1/src/main/scala
cd ./dstreamOutput1/src/main/scala
```

接下来，编辑代码文件

```shell
vim NetworkWordCountStateful.scala
```

输入下述代码：

```scala
import java.sql.{PreparedStatement, Connection, DriverManager}
import java.util.concurrent.atomic.AtomicInteger
import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.storage.StorageLevel

object NetworkWordCountStateful {
  def main(args: Array[String]) {
    //定义状态更新函数
    val updateFunc = (values: Seq[Int], state: Option[Int]) => {
      val currentCount = values.foldLeft(0)(_ + _)
      val previousCount = state.getOrElse(0)
      Some(currentCount + previousCount)
    }
    StreamingExamples.setStreamingLogLevels()  //设置log4j日志级别
    val conf = new SparkConf().setMaster("local[2]").setAppName("NetworkWordCountStateful")
    val sc = new StreamingContext(conf, Seconds(5))
    sc.checkpoint("file:///home/hadoop/mycode/dstreamOutput1/")    //设置检查点，检查点具有容错机制
    val lines = sc.socketTextStream("localhost", 9999)
    val words = lines.flatMap(_.split(" "))
    val wordDstream = words.map(x => (x, 1))
    val stateDstream = wordDstream.updateStateByKey[Int](updateFunc)
    stateDstream.print()
        //下面是新增的语句，把DStream保存到MySQL数据库中     
     stateDstream.foreachRDD(rdd => {
      //内部函数
      def func(records: Iterator[(String,Int)]) {
        var conn: Connection = null
        var stmt: PreparedStatement = null
        try {
          val url = "jdbc:mysql://localhost:3306/spark"
          val user = "root"
          val password = "1"  //输入自己的用户密码
          conn = DriverManager.getConnection(url, user, password)
          records.foreach(p => {
            val sql = "insert into wordcount(word,count) values (?,?)"
            stmt = conn.prepareStatement(sql);
            stmt.setString(1, p._1.trim)
                        stmt.setInt(2,p._2.toInt)
            stmt.executeUpdate()
          })
        } catch {
          case e: Exception => e.printStackTrace()
        } finally {
          if (stmt != null) {
            stmt.close()
          }
          if (conn != null) {
            conn.close()
          }
        }
      }

      val repartitionedRDD = rdd.repartition(3)
      repartitionedRDD.foreachPartition(func)
    })
    sc.start()
    sc.awaitTermination()
  }
}
```

保存并退出，先运行程序，后面我们会对代码进行解释。同时，由于我们在NetworkWordCountStateful.scala代码中加入了Spark SQL的操作，所以在编写工程文件时需要添加Spark SQL依赖包。然后创建StreamingExamples.scala用于设置log4j格式，直接复制之前已经编好的代码

```shell
cd ~/mycode
cp ./dstreamOutput/src/main/scala/StreamingExamples.scala ./dstreamOutput1/src/main/scala/
```

然后创建工程文件

```shell
cd ~/mycode/dstreamOutput1
vim simple.sbt
```

输入下述内容：

```sh
name := "Simple Project"
version := "1.0"
scalaVersion := "2.10.4"
libraryDependencies += "org.apache.spark" %% "spark-core" % "1.4.0"
libraryDependencies += "org.apache.spark" % "spark-streaming_2.10" % "1.4.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "1.4.0"
```

保存并退出，然后执行下面命令打包编译：

```
cd ~/mycode/dstreamOutput1
/usr/local/sbt/sbt package
```

打包编译成功后，就可以执行下面命令运行NetworkWordCountStateful程序进行词频统计，但是，需要注意，因为需要通过JDBC连接MySQL数据库，所以需要在spark-submit命令中提供外部jar包，告诉spark程序可以在哪里找到mysql驱动程序，我们直接从`下载`文件夹中导入jar包，命令如下：

```shell
/usr/local/spark/bin/spark-submit --class "NetworkWordCountStateful" --jars /home/hadoop/下载/mysql-connector-java-5.1.26-bin.jar /home/hadoop/mycode/dstreamOutput1/target/scala-2.10/simple-project_2.10-1.0.jar
```

执行上面命令以后，就进入监听状态

![37](G:\屏幕截图\Spark Streaming部分\37.jpg)

再开另外一个终端，运行下面命令产生单词提供给NetworkWordCountStateful进行词频统计：

```shell
nc -lk 9999
```

![39](G:\屏幕截图\Spark Streaming部分\39.jpg)

输入一些单词以后，就可以按Ctrl+Z停止nc程序。然后切换到刚才运行NetworkWordCountStateful程序的监听窗口，也按Ctrl+Z停止NetworkWordCountStateful程序运行。

![40](G:\屏幕截图\Spark Streaming部分\40.jpg)

```mysql
select * from wordcount;
```

![42](G:\屏幕截图\Spark Streaming部分\42.jpg)

简单解释本段代码含义

```scala
 stateDstream.foreachRDD(rdd => {
      //内部函数
      def func(records: Iterator[(String,Int)]) {
        var conn: Connection = null
        var stmt: PreparedStatement = null
        try {
          val url = "jdbc:mysql://localhost:3306/spark"
          val user = "root"
          val password = "1"
          conn = DriverManager.getConnection(url, user, password)
          records.foreach(p => {
            val sql = "insert into wordcount(word,count) values (?,?)"
            stmt = conn.prepareStatement(sql)
            stmt.setString(1, p._1.trim)
                        stmt.setInt(2,p._2.toInt)
            stmt.executeUpdate()
          })
        } catch {
          case e: Exception => e.printStackTrace()
        } finally {
          if (stmt != null) {
            stmt.close()
          }
          if (conn != null) {
            conn.close()
          }
        }
      }

      val repartitionedRDD = rdd.repartition(3)
      repartitionedRDD.foreachPartition(func)
    })

```

也就是说，对于stateDstream，为了把它保存到MySQL数据库中，我们采用了如下的形式：

```scala
stateDstream.foreachRDD(function)
```

其中，function就是一个RDD[T]=>Unit类型的函数，对于本程序而言，就是RDD[(String,Int)]=>Unit类型的函数，也就是说，stateDstream中的每个RDD都是RDD[(String,Int)]类型（即统计结果的形式是(“hadoop”,3)）。这样，对stateDstream中的每个RDD都会执行function中的操作（即把该RDD保存到MySQL的操作）。

下面解释function的处理逻辑，在function部分，函数体要执行的处理逻辑实际上是下面的形式：

```scala
def func(records: Iterator[(String,Int)]){……}
val repartitionedRDD = rdd.repartition(3)
repartitionedRDD.foreachPartition(func) 
```

也就是说，这里定义了一个内部函数func，它的功能是，接收records，然后把records保存到MySQL中。到这里，你可能会有疑问，为什么不是把stateDstream中的每个RDD直接拿去保存到MySQL中，还要调用rdd.repartition(3)对这些RDD重新设置分区数为3呢。这是因为，每次保存RDD到MySQL中，都需要启动数据库连接，如果RDD分区数量太大，那么就会带来多次数据库连接开销，为了减少开销，就有必要把RDD的分区数量控制在较小的范围内，所以，这里就把RDD的分区数量重新设置为3。然后，对于每个RDD分区，就调用repartitionedRDD.foreachPartition(func)，把每个分区的数据通过func保存到MySQL中，这时，传递给func的输入参数就是Iterator[(String,Int)]类型的records。如果你不好理解下面这种调用形式：

```scala
repartitionedRDD.foreachPartition(func) 
```

这种形式func没有带任何参数，可能不太好理解，不是那么直观，实际上，这句语句和下面的语句是等价的，或许有助于理解

```
repartitionedRDD.foreachPartition(records => func(records)) 
```

上面这种等价的形式比较直观，为func()函数传入了一个records参数，这就正好和 def func(records: Iterator[(String,Int)])定义对应起来了，方便理解。

接下来，解释func()函数的功能，我们再单独把func()函数的代码提取出来

```scala
def func(records: Iterator[(String,Int)]) {
        var conn: Connection = null
        var stmt: PreparedStatement = null
        try {
          val url = "jdbc:mysql://localhost:3306/spark"
          val user = "root"
          val password = "hadoop"
          conn = DriverManager.getConnection(url, user, password)
          records.foreach(p => {
            val sql = "insert into wordcount(word,count) values (?,?)"
            stmt = conn.prepareStatement(sql)
            stmt.setString(1, p._1.trim)
                        stmt.setInt(2,p._2.toInt)
            stmt.executeUpdate()
          })
        } catch {
          case e: Exception => e.printStackTrace()
        } finally {
          if (stmt != null) {
            stmt.close()
          }
          if (conn != null) {
            conn.close()
          }
        }
      }
```

​	可以看出，上面这段代码的功能是，创建数据库连接conn，然后调用records.foreach()，对于records中的每条记录p，都把p插入到MySQL数据库中。这里要注意，数据库连接conn的创建，是在records.foreach()方法之前，这样可以大大减小数据库连接开销。否则，如果把数据库连接conn的创建放在records.foreach()方法之后，那么，每条记录p都需要建立一次数据库连接，这样做会导致开销会增加，对数据库连接资源会造成很大的压力，并不可取。
​	对于prepareStatement中的语句。stmt.setString(1, p.\_1.trim)，是指，为val sql = “insert into wordcount(word,count) values (?,?)” 中的第一个问号设置具体的值，也就是给word字段设置值；stmt.setInt(2,p.\_2.toInt)是指，为第2个问号赋值，也就是给count字段设置值。很显然，每条记录p的形式是类似(“hadoop”,3)这种形式，所以，需要用p.\_1获取到”hadoop”（只是举例，每次获取到的值都会不同），需要用p.\_2获取到3（只是举例，每次获取到的值都会不同）。p.\_1.trim中调用了trim函数，是为了去掉字符串头尾可能存在的空格。p._2.toInt是为了把获取的3，转换成整型。