---
title: Java面向对象编程思想+Java内存原理+Java读写优化
date: 2018-03-13 11:59:21
update: 
comments: true
categories:
  - Java 
tags:
  - Java
---
<!--more-->

## Java面向对象编程思想

#### 面向过程编程思想（c、c++）

面向过程的语言也称为结构化程序设计语言，主要设计 问题解决过程中的步骤。它的主要观点是采用自顶向下、逐步求精的程序设计方法，使用三种基本控制结构（控制，选择，循环）构造程序。

#### 面向对象编程思想（Pyhon、Java、Scala）

面向对象编程思想有三大特性：**封装，继承，多态，**封装目的是复用；继承目的是共用；多态目的是多种实现。有了那些封装、多态、继承提升了编程的效率。
*面向对象时相对面向过程而言的*，其实二者从来都是分不开的。例如写一个10行的封装方法，这10行中的线形思想就是面对过程的。
先找到个对象，然后把它的属性，动作全部封装到一起。随时调用。面向对象就是堆和栈的完美结合。其实，面向对象方式的程序与以前结构化的程序在执行上没有任何区别。但是，面向对象的引入，使得对待问题的思考方式发生了改变，而更接近于自然方式的思考。当我们把对象拆开，你会发现，对象的属性其实就是数据，存放在堆中；而对象的行为（方法），就是运行逻辑，放在栈中。我们在编写对象的时候，其实即编写了数据结构，也编写的处理数据的逻辑。
面向对象(Object Oriented)的程序设计语言必须有描述对象及其相互之间关系的语言成分。这些程序设计语言可以归纳为以下几类：系统中一切事物皆为对象；对象是属性及其操作的封装体；对象可按其性质划分为类，对象成为类的实例；实例关系和继承关系是对象之间的静态关系；消息传递是对象之间动态联系的唯一形式，也是计算的唯一形式；方法是消息的序列。
面向对象的编程产生的历史原因：由于面向过程编程在构造系统时，无法解决重用，维护，扩展的问题，而且逻辑过于复杂，代码晦涩难懂，因此，人们开始想能不能让计算机直接模拟现实的环境，以人类解决问题的方法，思路，习惯和步骤来设计相应的应用程序。于是，面向对象的编程思想就产生了。
面向对象的编程的主要思想是把构成问题的各个事物分解成各个对象，建立对象的目的不是为了完成一个步骤，而是为了描述一个事物在解决问题的过程中经历的步骤和行为。对象作为程序的基本单位，将程序和数据封装其中，以提高程序的重用性，灵活性和可扩展性。**类是创建对象的模板，一个类可以创建多个对象。对象是类的实例化。**
```java
class person (name, age){
	String name = name
	int age = age
    String 法术 = 打狗棒法

	def paly (person1,person2, count){

	System.out.print(person1.name +“使用了法术：”+ person1.法术 +“打了person2.name”+ count +"次")
	}
}

new person1 = 孙悟空
new person2 = 白骨精
count = 3

孙悟空.play(孙悟空, 白骨精, count)

# 创建对象，调用成员方法
# 结果实现输出输出 “孙悟空用 打狗棒法 打了白骨精 3次”
```

## Java内存机制

#### 栈内存

栈：（基本数据类型变量、对象的引用变量）基本数据类型的变量（int、short、long、byte、float、double、boolean、char）以及对象的引用变量，其内存分配在栈上，变量出了作用域后就会自动释放。栈内存的主要作用是存放基本数据类型和引用变量。栈的内存管理是通过栈的&quot;后进先出&quot;模式来实现的。
（主要用来执行程序，存取速度快，大小和生存期必须确定，缺乏灵活性）
栈内存在函数中定义的一些基本类型的变量和对象的引用变量都是在函数的栈内存中分配，当在一段代码块定义一个变量时，Java 就在栈中为这个变量分配内存空间，当超过变量的作用域后（比如，在函数A中调用函数B，在函数B中定义变量a，变量a的作用域只是函数B，在函数B运行完以后，变量a会自动被销毁。分配给它的内存会被回收），Java 会自动释放掉为该变量分配的内存空间，该内存空间可以立即被另作它用。

#### 堆内存

**堆**：（对象）引用类型的变量，其内存分配在堆上或者常量池（字符串常量、基本数据类型常量），需要通过new等方式来创建。堆内存主要作用是存放运行时创建(new)的对象。
（主要用于存放对象，存取速度慢，可以运行时动态分配内存，生存期不需要提前确定）
引用参数（类，对象，String）都放在堆内存中，然后把对应堆内存中的物理首地址赋值给引用的他的变量。堆内存的物理空间是有序的，类似Python中的“字典”，每个空间的物理首地址相当于这个空间的ID编号
堆内存用来存放由 new 创建的对象和数组，在堆中分配的内存，由 Java 虚拟机的自动垃圾回收器来管理。在堆中产生了一个数组或者对象之后，还可以在栈中定义一个特殊的变量，让栈中的这个变量的取值等于数组或对象在堆内存中的首地址，栈中的这个变量就成了数组或对象的引用变量，以后就可以在程序中使用栈中的引用变量来访问堆中的数组或者对象，引用变量就相当于是为数组或者对象起的一个名称。


## Java读写优化

#### 1、使用StringBuilder
StingBuilder 应该是在我们的Java代码中默认使用的，应该避免使用 + 操作符。或许你会对 StringBuilder 的语法糖（syntax sugar）持有不同意见，比如：
```java
String x = "a" + args.length + "b";
将会被编译为：

new java.lang.StringBuilder [16]
dup
ldc <String "a"> [18]
invokespecial java.lang.StringBuilder(java.lang.String) [20]
aload_0 [args]
arraylength
invokevirtual java.lang.StringBuilder.append(int) : java.lang.StringBuilder [23]
ldc <String "b"> [27]
invokevirtual java.lang.StringBuilder.append(java.lang.String) : java.lang.StringBuilder [29]
invokevirtual java.lang.StringBuilder.toString() : java.lang.String [32]
astore_1 [x]
```

但究竟发生了什么？接下来是否需要用下面的部分来对 String 进行改善呢？
```java
String x = "a" + args.length + "b";
 
if (args.length == 1)
    x = x + args[0];
```
现在使用到了第二个 StringBuilder，这个 StringBuilder 不会消耗堆中额外的内存，但却给 GC 带来了压力。
```java
StringBuilder x = new StringBuilder("a");
x.append(args.length);
x.append("b");
 
if (args.length == 1);
    x.append(args[0]);

```
**小结:**
在上面的样例中，如果你是依靠Java编译器来隐式生成实例的话，那么编译的效果几乎和是否使用了 StringBuilder 实例毫无关系。请记住：在  N.O.P.E 分支中，每次CPU的循环的时间都白白的耗费在GC或者为 StringBuilder 分配默认空间上了，我们是在浪费 N x O x P 时间。 一般来说，使用 StringBuilder 的效果要优于使用 + 操作符。如果可能的话请在需要跨多个方法传递引用的情况下选择 StringBuilder，因为 String 要消耗额外的资源。JOOQ在生成复杂的SQL语句便使用了这样的方式。在整个抽象语法树（AST Abstract Syntax TreeSQL传递过程中仅使用了一个 StringBuilder 。
更加悲剧的是，如果你仍在使用 StringBuffer 的话，那么用 StringBuilder 代替 StringBuffer 吧，毕竟需要同步字符串的情况真的不多。

#### 2、避免使用正则表达式

正则表达式给人的印象是快捷简便。但是在 N.O.P.E 分支中使用正则表达式将是最糟糕的决定。如果万不得已非要在计算密集型代码中使用正则表达式的话，至少要将 Pattern 缓存下来，避免反复编译Pattern。
```java
static final Pattern HEAVY_REGEX =
    Pattern.compile("(((X)*Y)*Z)*");
```
如果仅使用到了如下这样简单的正则表达式的话：
```java
String[] parts = ipAddress.split("\\.");
```
这是最好还是用普通的 char[] 数组或者是基于索引的操作。比如下面这段可读性比较差的代码其实起到了相同的作用。
```java
int length = ipAddress.length();
int offset = 0;
int part = 0;
for (int i = 0; i < length; i++) {
    if (i == length - 1 ||
            ipAddress.charAt(i + 1) == '.') {
        parts[part] =
            ipAddress.substring(offset, i + 1);
        part++;
        offset = i + 2;
    }
}
```
上面的代码同时表明了过早的优化是没有意义的。虽然与 split() 方法相比较，这段代码的可维护性比较差。

挑战：聪明的小伙伴能想出更快的算法吗？

**小结**
正则表达式是十分有用，但是在使用时也要付出代价。尤其是在 N.O.P.E 分支深处时，要不惜一切代码避免使用正则表达式。还要小心各种使用到正则表达式的JDK字符串方法，比如 String.replaceAll() 或 String.split()。可以选择用比较流行的开发库，比如 Apache Commons Lang 来进行字符串操作。

#### 3、不要使用iterator()方法

这条建议不适用于一般的场合，仅适用于在 N.O.P.E 分支深处的场景。尽管如此也应该有所了解。Java 5格式的循环写法非常的方便，以至于我们可以忘记内部的循环方法，比如：
```java
for (String value : strings) {
    // Do something useful here
}
```
当每次代码运行到这个循环时，如果 strings 变量是一个 Iterable 的话，代码将会自动创建一个Iterator 的实例。如果使用的是 ArrayList 的话，虚拟机会自动在堆上为对象分配3个整数类型大小的内存。
```java
private class Itr implements Iterator<E> {
    int cursor;
    int lastRet = -1;
    int expectedModCount = modCount;
    // ...
```
也可以用下面等价的循环方式来替代上面的 for 循环，仅仅是在栈上“浪费”了区区一个整形，相当划算。
```java
int size = strings.size();
for (int i = 0; i < size; i++) {
    String value : strings.get(i);
    // Do something useful here
}
```
如果循环中字符串的值是不怎么变化，也可用数组来实现循环。
```java
for (String value : stringArray) {
    // Do something useful here
}
```
**小结**
无论是从易读写的角度来说，还是从API设计的角度来说迭代器、Iterable接口和 foreach 循环都是非常好用的。但代价是，使用它们时是会额外在堆上为每个循环子创建一个对象。如果循环要执行很多很多遍，请注意避免生成无意义的实例，最好用基本的指针循环方式来代替上述迭代器、Iterable接口和 foreach 循环。
一些与上述内容持反对意见的看法（尤其是用指针操作替代迭代器）详见Reddit上的讨论。

#### 4、不要调用高开销方法

有些方法的开销很大。以 N.O.P.E 分支为例，我们没有提到叶子的相关方法，不过这个可以有。假设我们的JDBC驱动需要排除万难去计算 ResultSet.wasNull() 方法的返回值。我们自己实现的SQL框架可能像下面这样：
```java
if (type == Integer.class) {
    result = (T) wasNull(rs,
        Integer.valueOf(rs.getInt(index)));
}
 
// And then...
static final <T> T wasNull(ResultSet rs, T value)
throws SQLException {
    return rs.wasNull() ? null : value;
}
```
在上面的逻辑中，每次从结果集中取得 int 值时都要调用 ResultSet.wasNull() 方法，但是 getInt() 的方法定义为：

返回类型：变量值；如果SQL查询结果为NULL，则返回0。

所以一个简单有效的改善方法如下：
```java
static final <T extends Number> T wasNull(
    ResultSet rs, T value
)
throws SQLException {
    return (value == null ||
           (value.intValue() == 0 && rs.wasNull()))
        ? null : value;
}
```
这是轻而易举的事情。

**小结**
将方法调用缓存起来替代在叶子节点的高开销方法，或者在方法约定允许的情况下避免调用高开销方法。

#### 5、使用原始类型和栈

上面介绍了来自 jOOQ的例子中使用了大量的泛型，导致的结果是使用了 byte、 short、 int 和 long 的包装类。但至少泛型在Java 10或者Valhalla项目中被专门化之前，不应该成为代码的限制。因为可以通过下面的方法来进行替换：
```java
//存储在堆上
Integer i = 817598;
……如果这样写的话：

// 存储在栈上
int i = 817598;
在使用数组时情况可能会变得更加糟糕：


//在堆上生成了三个对象
Integer[] i = { 1337, 424242 };
……如果这样写的话：

// 仅在堆上生成了一个对象
int[] i = { 1337, 424242 };
```
**小结**
当我们处于 N.O.P.E. 分支的深处时，应该极力避免使用包装类。这样做的坏处是给GC带来了很大的压力。GC将会为清除包装类生成的对象而忙得不可开交。
 所以一个有效的优化方法是使用基本数据类型、定长数组，并用一系列分割变量来标识对象在数组中所处的位置。
 遵循LGPL协议的 trove4j 是一个Java集合类库，它为我们提供了优于整形数组 int[] 更好的性能实现。

例外
下面的情况对这条规则例外：因为 boolean 和 byte 类型不足以让JDK为其提供缓存方法。我们可以这样写：
```java
Boolean a1 = true; // ... syntax sugar for:
Boolean a2 = Boolean.valueOf(true);
 
Byte b1 = (byte) 123; // ... syntax sugar for:
Byte b2 = Byte.valueOf((byte) 123);

其它整数基本类型也有类似情况，比如 char、short、int、long。
```

不要在调用构造方法时将这些整型基本类型自动装箱或者调用 TheType.valueOf() 方法。

也不要在包装类上调用构造方法，除非你想得到一个不在堆上创建的实例。这样做的好处是为你为同事献上一个巨坑的愚人节笑话。

非堆存储
当然了，如果你还想体验下堆外函数库的话，尽管这可能参杂着不少战略决策，而并非最乐观的本地方案。一篇由Peter Lawrey和 Ben Cotton撰写的关于非堆存储的很有意思文章请点击： OpenJDK与HashMap——让老手安全地掌握（非堆存储！）新技巧。

#### 6、避免递归

现在，类似Scala这样的函数式编程语言都鼓励使用递归。因为递归通常意味着能分解到单独个体优化的尾递归（tail-recursing）。如果你使用的编程语言能够支持那是再好不过。不过即使如此，也要注意对算法的细微调整将会使尾递归变为普通递归。

希望编译器能自动探测到这一点，否则本来我们将为只需使用几个本地变量就能搞定的事情而白白浪费大量的堆栈框架（stack frames）。

**小结**
这节中没什么好说的，除了在 N.O.P.E 分支尽量使用迭代来代替递归。

#### 7、使用entrySet()

当我们想遍历一个用键值对形式保存的 Map 时，必须要为下面的代码找到一个很好的理由：
```java
	for (K key : map.keySet()) {
    V value : map.get(key);
	}
```
更不用说下面的写法：
```java
	for (Entry<K, V> entry : map.entrySet()) {
    K key = entry.getKey();
    V value = entry.getValue();
	}
```
在我们使用 N.O.P.E. 分支应该慎用map。因为很多看似时间复杂度为 O(1) 的访问操作其实是由一系列的操作组成的。而且访问本身也不是免费的。至少，如果不得不使用map的话，那么要用 entrySet() 方法去迭代！这样的话，我们要访问的就仅仅是Map.Entry的实例。

**小结**
在需要迭代键值对形式的Map时一定要用 entrySet() 方法。

#### 8、使用EnumSet或EnumMap

在某些情况下，比如在使用配置map时，我们可能会预先知道保存在map中键值。如果这个键值非常小，我们就应该考虑使用 EnumSet 或 EnumMap，而并非使用我们常用的 HashSet 或 HashMap。下面的代码给出了很清楚的解释：
```java
	private transient Object[] vals;
	public V put(K key, V value) {
    // ...
    int index = key.ordinal();
    vals[index] = maskNull(value);
    // ...
	}
```
上段代码的关键实现在于，我们用数组代替了哈希表。尤其是向map中插入新值时，所要做的仅仅是获得一个由编译器为每个枚举类型生成的常量序列号。如果有一个全局的map配置（例如只有一个实例），在增加访问速度的压力下，EnumMap 会获得比 HashMap 更加杰出的表现。原因在于 EnumMap 使用的堆内存比 HashMap 要少 一位（bit），而且 HashMap 要在每个键值上都要调用 hashCode() 方法和 equals() 方法。

**小结**
Enum 和 EnumMap 是亲密的小伙伴。在我们用到类似枚举（enum-like）结构的键值时，就应该考虑将这些键值用声明为枚举类型，并将之作为 EnumMap 键。

#### 9、优化自定义hasCode()方法和equals()方法

在不能使用EnumMap的情况下，至少也要优化 hashCode() 和 equals() 方法。一个好的 hashCode() 方法是很有必要的，因为它能防止对高开销 equals() 方法多余的调用。

在每个类的继承结构中，需要容易接受的简单对象。让我们看一下jOOQ的 org.jooq.Table 是如何实现的？

最简单、快速的 hashCode() 实现方法如下：
```java
	// AbstractTable一个通用Table的基础实现：
 
	@Override
	public int hashCode() {
 
    // [#1938] 与标准的QueryParts相比，这是一个更加高效的hashCode()实现
    return name.hashCode();
	}
```
name即为表名。我们甚至不需要考虑schema或者其它表属性，因为表名在数据库中通常是唯一的。并且变量 name 是一个字符串，它本身早就已经缓存了一个 hashCode() 值。

这段代码中注释十分重要，因继承自 AbstractQueryPart 的 AbstractTable 是任意抽象语法树元素的基本实现。普通抽象语法树元素并没有任何属性，所以不能对优化 hashCode() 方法实现抱有任何幻想。覆盖后的 hashCode() 方法如下：
```java
	// AbstractQueryPart一个通用抽象语法树基础实现：
 
	@Override
	public int hashCode() {
    // 这是一个可工作的默认实现。
    // 具体实现的子类应当覆盖此方法以提高性能。
    return create().renderInlined(this).hashCode();
	}
```
换句话说，要触发整个SQL渲染工作流程（rendering workflow）来计算一个普通抽象语法树元素的hash代码。

equals() 方法则更加有趣：
```java
	// AbstractTable通用表的基础实现：
	@Override
	public boolean equals(Object that) {
    if (this == that) {
        return true;
    }
 
    // [#2144] 在调用高开销的AbstractQueryPart.equals()方法前，
    // 可以及早知道对象是否不相等。
    if (that instanceof AbstractTable) {
        if (StringUtils.equals(name,
            (((AbstractTable<?>) that).name))) {
            return super.equals(that);
        }
 
        return false;
    }
 
    return false;
	}
```
首先，不要过早使用 equals() 方法（不仅在N.O.P.E.中），如果：
```java
this == argument
this“不兼容：参数
```
**注意**：如果我们过早使用 instanceof 来检验兼容类型的话，后面的条件其实包含了argument == null。我在以前的博客中已经对这一点进行了说明，请参考10个精妙的Java编码最佳实践。

在我们对以上几种情况的比较结束后，应该能得出部分结论。比如jOOQ的 Table.equals() 方法说明是，用来比较两张表是否相同。不论具体实现类型如何，它们必须要有相同的字段名。比如下面两个元素是不可能相同的：
```java
	com.example.generated.Tables.MY_TABLE
	DSL.tableByName(“MY_OTHER_TABLE”)
```
如果我们能方便地判断传入参数是否等于实例本身（this），就可以在返回结果为 false 的情况下放弃操作。如果返回结果为 true，我们还可以进一步对父类（super）实现进行判断。在比较过的大多数对象都不等的情况下，我们可以尽早结束方法来节省CPU的执行时间。

一些对象的相似度比其它对象更高。

在jOOQ中，大多数的表实例是由jOOQ的代码生成器生成的，这些实例的 equals() 方法都经过了深度优化。而数十种其它的表类型（衍生表 （derived tables）、表值函数（table-valued functions）、数组表（array tables）、连接表（joined tables）、数据透视表（pivot tables）、公用表表达式（common table expressions）等，则保持 equals() 方法的基本实现。

#### 10、考虑使用set而并非单个元素

最后，还有一种情况可以适用于所有语言而并非仅仅同Java有关。除此以外，我们以前研究的 N.O.P.E. 分支也会对了解从 O(N3) 到 O(n log n)有所帮助。

不幸的是，很多程序员的用简单的、本地算法来考虑问题。他们习惯按部就班地解决问题。这是命令式（imperative）的“是/或”形式的函数式编程风格。这种编程风格在由纯粹命令式编程向面对象式编程向函数式编程转换时，很容易将“更大的场景（bigger picture）”模型化，但是这些风格都缺少了只有在SQL和R语言中存在的：

声明式编程。

在SQL中，我们可以在不考虑算法影响下声明要求数据库得到的效果。数据库可以根据数据类型，比如约束（constraints）、键（key）、索引（indexes）等不同来采取最佳的算法。

在理论上，我们最初在SQL和关系演算（relational calculus）后就有了基本的想法。在实践中，SQL的供应商们在过去的几十年中已经实现了基于开销的高效优化器CBOs (Cost-Based Optimisers) 。然后到了2010版，我们才终于将SQL的所有潜力全部挖掘出来。

但是我们还不需要用set方式来实现SQL。所有的语言和库都支持Sets、collections、bags、lists。使用set的主要好处是能使我们的代码变的简洁明了。比如下面的写法：
```java
	SomeSet INTERSECT SomeOtherSet
	而不是
	// Java 8以前的写法
	Set result = new HashSet();
	for (Object candidate : someSet)
    if (someOtherSet.contains(candidate))
        result.add(candidate); 
	// 即使采用Java 8也没有很大帮助
	someSet.stream()
       .filter(someOtherSet::contains)
       .collect(Collectors.toSet());
```
有些人可能会对函数式编程和Java 8能帮助我们写出更加简单、简洁的算法持有不同的意见。但这种看法不一定是对的。我们可以把命令式的Java 7循环转换成Java 8的Stream collection，但是我们还是采用了相同的算法。但SQL风格的表达式则是不同的：
```java
	SomeSet INTERSECT SomeOtherSet
```
上面的代码在不同的引擎上可以有1000种不同的实现。我们今天所研究的是，在调用 INTERSECT 操作之前，更加智能地将两个set自动的转化为 EnumSet 。甚至我们可以在不需要调用底层的 Stream.parallel() 方法的情况下进行并行 INTERSECT 操作。

**总结**

在这篇文章中，我们讨论了关于N.O.P.E.分支的优化。比如深入高复杂性的算法。作为jOOQ的开发者，我们很乐于对SQL的生成进行优化。

每条查询都用唯一的StringBuilder来生成。
模板引擎实际上处理的是字符而并非正则表达式。
选择尽可能的使用数组，尤其是在对监听器进行迭代时。
对JDBC的方法敬而远之。
等等。
