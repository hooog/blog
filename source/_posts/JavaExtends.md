---
title: Java的继承、多态、抽象
date: 2018-03-16 10:05:59
password:
top:
categories:
  - Java
tags:
  - Java
---

## Java的继承extends

```java
class SubDemo extends Dome{}
```
- Java 只能继承其父类非私有的属性和参数
- Java 只支持单继承，不支持多继承 

#### **super的用法**关键字和**this**类似
##### **一.super关键字作用**
1、主要存在于子类方法中，用于指向子类对象中父类对象。
2、访问父类的属性
3、访问父类的函数
4、访问父类的构造函数
##### **二.super注意的地方**
this和super很像，this指向的是当前对象的调用，super指向的是当前调用对象的父类。类加载完毕，创建对象，父类的构造方法会被调用（默认自动无参），然后执行子类相应构造创建了一个子类对象，该子类对象还包含了一个父类对象。该父类对象在子类对象内部。this super只能在有对象的前提下使用，不能在静态上下文使用。
##### **三.super关键字的使用**
1、子类的构造函数默认第一行会默认调用父类无参的构造函数，隐式语句 super();这里只可以调用其父类无参的构造函数。
若父类构造函数有参数，则必要要super显示调用。
2、子类显式调用父类构造函数
在子类构造函数第一行通过super关键字调用父类任何构造函数。如果显式调用父类构造函数，编译器自动添加的调用父类无参数的构造就消失。构造函数间的调用只能放在第一行，只能调用一次。super()和this()不能同时存在构造函数第一行。
##### **四、关于构造函数**
1、如果父类中有空参数的构造函数和两个参数的构造函数，那么子类有两个参数的构造函数访问的是父类中哪个构造函数？
2、构造函数的继承只能单次继承
**子类继承父类后，当调用构造函数的时候，都会默认调用父类中的空参数的构造函数访问，原因是：子类的构造函数默认第一行会默认调用父类无参的构造函数，隐式语句。**
**如果父类中删除了空参数的构造函数，那么系统在定义子类的时候就必须加上super关键字并传入参数取显式的调用带有相匹配参数的构造方法**

##### 五、final关键字
final意思是最终版，所以：
1、final修饰的成员方法，可以被继承，但不可以被子类重写的！
2、final修饰的成员变量其实是一个常量！（只读）
3、final修饰的类不能被继承！
4、怎么对类中的私有变量进行赋值：
- 1、调用成员函数中的set方法
- 2、通过调用有参数的构造函数进行赋值

**疑问：**
1、构造方法是否能被继承？
不可以，每个类的构造方法是与自己同名的，唯一的。但是可以引用super（n1,n2,n3）创建子类自己的构造方法）可以调用不可被继承。

2、能否用super.强制调用父类方法（不可以）

3、默认构造方法是否为多态的一种体现（不是）

## Java的多态

**核心的目的是实现一个对象多重身份（状态）相互转换**
#### 方法的重载和覆盖
**方法覆盖** （重写）
- 权限大的覆盖权限小的
- 静态只能覆盖静态

**方法重载**
- 对某一个方法用不同的参数进行重写，传入不同的参数可以实现不同的功能，这就是重载

#### 对象的多态性
**特点**
- 向下转型（子转父）；向上转型（父转子）：目的类似““方法返祖””和“方法进化”。实现：**使用的是自己的躯壳，调用的是别人的灵魂**
- 父子间可以转换，兄弟姐们没有继承关系不能行下或者向上转型
- 编译看左边
- 运行看右边
- 父类不能使用子类特有功能，只能使用子类重写父类后的方法或参数
- 目的是““返祖””，“进化”

总之：方法重写是可以决定多态的，方法重载是决定不了多态的

所以在Java中，“多态体现在方法重载与方法重写”，这句话还正确么？

当子类覆盖父类的属性的时候，我们取出来的值是我们覆盖后的属性值，而如果我们不覆盖，我们取到的还是原来父类中的属性值，但是不管我们是否覆盖，我们在调用getClassName()这个方法的时候，所执行的都是子类中的方法，而不是父类的方法，这就是多态。重载是指方法名相同，参数不同的多个方法，这叫重载，重载首先和继承没有任何关系，更不涉及到覆盖父类的属性或者方法，也和父类动态的去引用子类对象没有任何关系，所以说，重载不是多态！
**结论：重载不是多态！**




## 抽象类，抽象方法

#### **abstract**关键字

2、有抽象方法的类不一定是抽象类
3、抽象类的子类就是抽象类的实例化

```java
public class Test {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Monkey_1 mk = new Monkey_1();
		mk.eat();
	}
}

abstract class Animal{
	public abstract void eat();
}
abstract class Monkey extends Animal{
	public abstract void run();
}
abstract class Human extends Monkey{
	public abstract void think();
}
// 由于eat()和run()方法已经在monkey中被实例化，因此Human这里可以不用重复实例化。若Monkey中没有被实例化则这里必须要实例化！ 
 class Animal_1 extends Animal{
	public void eat(){
		System.out.println("I'm a animal and I can eat!");
	}
}
 class Monkey_1 extends Monkey{
	public void eat(){
		System.out.println("I'm a monkey and i can eat!");
	}
	public void run(){
		System.out.println("I'm a monkey and i can run!");
	}

//@Override
//public void eat() {
//	// TODO Auto-generated method stub
//	System.out.println("我是个外部的eat");
//}
}

 class Human_1 extends Human{
	public void eat(){
		System.out.println("I'm a human and i can eat!");
	}
	public void run(){
		System.out.println("I'm a human and i can run!");
	}
	public void think(){
		System.out.println("I'm a human and i can think!");
	}
}
```

**小结**
1、抽象类里可以包含构造方法，也可以包含常用的一些方法。
2、抽象类里，不能没有抽象方法的定义，虽然编译器可以通过，但是这就失去了抽象类定义的意义。
3、抽象类里不能和Private，Final，Static关键字同时使用。因为Private是私有的性质无法继承或者重写；Final是特别声明不能被其他继承类使用；Static是静态变量，由于已经分配了内存空间，而抽象类是不分配空间的。
*官方解释：*
声明static说明可以直接用类名调用该方法
声明abstract说明需要用抽象类的子类重写该抽象方法（抽象方法实例化）
所以static和abstract不能和static共存

