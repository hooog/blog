---
title: Scala数据类型及基础语法
date: 2018-04-1 00:28:59
password:
top:
categories:
  - Scala
tags:
  - 
---

## scala简介
	2004年，martin ordersky发明，javac的编译器，后来spark,kafka应用广泛，twitter应用推广。它具备面向对象和函数式编程的特点。
	[官网：www.scala-lang.org](www.scala-lang.org)
## Windows环境安装
		a) 安装jdk-7u55-windows-x64.exe
		b) 安装scala-2.10.4.msi
		安装完以上两步，不用做任何修改。
		测试：在控制台下c:/>scala
		c) 安装eclipse-java-juno-SR2-win32-x86_64.zip 解压缩即可
		d) 安装eclipse的scala插件update-site.zip
			解压会有两个目录，features和plugins，分别把内容放到eclispe对应的目录下。	
		e) 重启eclipse
			提示"Upgrade of scala..."，点yes
			提示框"setup Diagnostic"，把Enable JDT weaving...选上
			根据提示重启
## 第一个程序
### 交互式编程
		C:\Users\Administrator>scala
		Welcome to Scala version 2.10.4 (Java HotSpot(TM) 64-Bit Server VM, Java 1.7.0_5
		Type in expressions to have them evaluated.
		Type :help for more information.

		scala> 1+1
		res0: Int = 2

### 脚本形式:通过创建文件来在eclipse/idea中执行代码

	object Test {
		def main(args: Array[String] ) {
			println("Hello world")
		}
	}
- 注1：上面这个是单例对象，这个里面只能存放静态的东西
- 注2：自动导入3个包
		java.lang._
		scala._
		Predef._	
- 注3：语句最后一行的分号不推荐写

## 基础语法
### 变量和常量
- a) 变量var
			格式：
				var 变量名 [:数据类型] = 值
			例：var b :Int = 1
				var c = 2	//类型自动推断
- b) 常量val
			格式：
				val 常量名 [:数据类型] = 值
			例：
				val b :Int = 1
				val b = 1
				b = 2	//报错，常量不能修改

- c) 常量可以用lazy修饰(了解)
				lazy val b :Int = 1	//b用到的时候再赋值

### 数据类型

#### 基本类型
```
Byte	8位有符号值，范围从-128至127
Short	16位有符号值，范围从-32768至32767
Int		32位有符号值，范围从-2147483648至2147483647
Long	64位有符号值，范围从-9223372036854775808至9223372036854775807
Float	32位IEEE 754单精度浮点值
Double	64位IEEE 754双精度浮点值
Char	16位无符号Unicode字符。范围从U+0000到U+FFFF
String	一个Char类型序列
Boolean	文字值true或文字值false
Unit	对应于无值，等价于void类型，只有一个对象叫()
Null	只有一个对象叫null
Nothing	在Scala中处于最底层，比如创建数组时不指定类型，就是Noting。抽象概念
Any		任何类型的超类型; 任何对象的类型为Any
AnyRef	任何引用类型的超类型
```
#### 层次结构

Scala.Any 任何类型的超类型
- AnyVal(值) 任何数值类型的超类型
	Int,Double等，Unit。
- AnyRef（引用）任何引用类型的超类型
	List,Set,Map,Seq,Iterable
	java.lang.String
	Null


#### 重点类型：元组
	格式：(元素1, 元素2, ....)
	访问：变量._N 其中N是元组元素的索引，从1开始
		例：var t = ("a", false, 1)		//t的类型是scala.Tuple3
			var value = t._1	//"a"
			var m,n,(x,y,z) =  ("a", false, 1) 
			m	:("a", false, 1) 
			n	:("a", false, 1) 
			x	: "a"
			y	: false
			z	: 1
#### 重点类型：字符串
			i) 用的是java.lang.String，但是有时候根据需要，会隐式转换到其它类型，比如调用reverse/sorted/sortWith/drop/slice等方法，这些方法定义在IndexedSeqOptimized中
			ii)多行字符串表示，开始和结束用"""
#### 了解：符号类型
			符号字面量： '标识符,是scala.Symbol的实例,像模式匹配，类型判断会比较常用。
			var flag = 'start
			if (flag == 'start) println(1) else println(2)

### 运算符:scala没有运算符，它运算符全部封装成了方法。		
		算术：+ - * / %
		比较: == != > < >= <= 
		逻辑：&& || !
		赋值：= += -= *= /* %=
		位：& | ~ ^ >> << >>> 
		
		注：上面都是方法。例 1+2相当于1.+(2)，其中+是方法，2是参数
### 控制语句
		(1) if，if...else...，if...else if...else...
		(2) scala中的if可以作为表达式用
			var x = if("hello"=="hell") 1 else 0
		(3) switch被模式匹配替换
### 循环语句
		(1)while,do..while和for都有，while和do..while很像，但是for差别很大。
		(2)for循环格式不同
			for(变量 <- 集合 if 条件判断1;if 条件判断2...) {
				所有条件判断都满足才执行
			}

		(3)没有break和continue，用两种方法可以代替
			i) 用for循环的条件判断
			ii)方法2:非写break，要做以下两步
			 //1) 引入一个包scala.util.control.Breaks._
			 //2) 代码块用breakable修饰

			for (value <- 1 to 5) {
				println(value)
			}

### 集合框架(Array，List,Set,Map)
		在scala中，数组Array归到集合的范畴
		包scala.collection，下面有两个分支immutable(不可改变的,默认)和mutable(可变的)
#### 层次结构
```
 Traversable
-Iterable
**(immutable不可改变的)**
-Set
	HashSet,TreeSt
-Map
	HashMap,TreeMap
-Seq
	-IndexedSeq
		Vector,Array,String,Range
	-LinearSeq
		List,Queue,Stack

**(mutable可变的)**
-Set
	HashSet
-Map
	HashMap
-Seq
	-IndexedSeq
		ArraySeq,StringBuilder
	-Buffer
		ArrayBuffer,ListBuffer
	-LinearSeq
		LinkedList,Queue
	Stack
重点记忆几个
不可变			可变
Array			ArrayBuffer
List			ListBuffer
immutable.Map		mutable.Map
immutable.Set		mutable.Set
```
		
#### 数组(非常重要)：数据类型相同的元素，按照一定顺序排序的集合。
不可变数组 scala.Array
可变长数组 scala.collection.mutable.ArrayBuffer

- 1.Array和ArrayBuffer
	Array创建:
	var 变量名 = new Array[类型](长度)	//var arr = new Array[Int](10)
	var 变量名 = Array(元素1,元素2,...) //var arr = Array(1,3,5,7)，实际上是调用Array的方法apply

	取值：变量名(下标)
- 2.ArrayBuffer创建
	var 变量名 = new ArrayBuffer[类型]()	//var arr = new ArrayBuffer[Int]()
	var 变量名 = ArrayBuffer(元素1,元素2,...) //var arr = ArrayBuffer(1,3,5,7)
- 3.共同方法
```
	sum 求和
	max 最大值
	min 最小值
	mkString(分隔符)	例：Array(1,3,5,7).mkString("|")	//结果是1|3|5|7|9
	sorted 排序（从小到大）
	sortBy
	sortWith 自定义排序
	reverse 翻转
	toArray ArrayBuffer转成Array
	toBuffer Array转成ArrayBuffer
	toMap：如果Array内部元素是对偶元组，可以转成map
```
- 4.ArrayBuffer的独有方法
	+= 追加元素
	++= 追加集合
	trimEnd(n):删除末尾n个元素
	insert(index,value1,value2,..valueN):第index的位置，插入value1,value2,...valueN)
	remove(index,n):第index个位置删除n个元素
	clear():清空
- 5.遍历
	i) 通过下标
		for (i <- 0 to 数组.length-1) {
			println(i)) //i表示数组的下标
		}
	ii)直接取值
		for (e <- 数组) {
			println(e)
		}
- 6.使用yield创建新数组
	var arr = for (i <- 数组) yield i * 2
- 7.多维数组
	var arr = Array(Array(元素1...),Array(元素,...)..)
	遍历
	for (i <- arr) {
		for (j <- i) {
			println(j)
		}
	}
		
#### List
#### Set
#### Map
#### Array和List和Tuple都是不可变元素，有什么区别？
## 函数
## 面向对象

