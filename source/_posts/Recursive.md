---
title: 递归、尾递归、循环的比较及Python尾递归的优化
date: 2018-04-3 03:05:59
password:
top:
categories:
  - Python
tags:
  - 递归
---
<!--more-->


最近在读魏老师推荐的算法图解这本书。该书内容很有趣，把枯燥无味的算法用一个个日常生活的小案例诠释的清晰易懂，很适合作为业余读物。
在读到关于递归算法的时候，本着打破砂锅问到底的精神大致研究了下递归算法和递归进阶 —> **尾递归**。
后来发现Python的解释器不支持尾递归算法优化，而且**Python之父曾经明确表示Python将不会支持尾递归优化！T T...**
不过看我华夏泱泱大国，人才济济，还是有哥们写出了Python上对尾递归爆栈问题的解决办法。

## 什么是递归
一言概之： **递归是一种优雅的问题解决办法。** 它将程序员分为了三个截然不同的阵营：恨它的、爱它的、爱了几年以及恨了几年后又爱上它的。初接触时我对这句话是怀疑的，为什么我感觉递归让解决方案变得更绕了呢？ 直到认真去看了几个用递归思想解决问题的例子之后方才发现此话所言非虚。
粗略的说，递归就是让函数调用自己。它让解决方案更清晰，但并没有性能上的优势。正如Leigh Caldwell在Stack Overflow上说过一句话：“如果使用循环，程序的性能可能更高，如果使用递归，程序可能更容易理解。如何选择要看什么对你来说更重要”（这一点下面会详细说明）。由于递归函数调用自己，因此编写递归函数时，必须告诉它何时停止递归否则会陷入无限循环。所以每个递归函数都必须有两部分：**基准条件和递归条件**

使用递归必须要理解 **调用栈** 这个概念。栈是一种简单的数据结构，遵循先进后出的原则。计算机在内部使用被称为 调用栈 的栈。

在递归的过程中，当计算机调用一个递归函数 A 的时候首先为该函数分配一块内存，并将函数调用涉及到的方法和所有变量的值存储到内存 a 中。递归函数 A 调用自己产生函数 B ，同样计算机会给函数 B 分配一块新的内存 b 用于存储 B 所涉及到的所有变量及方法，并压在 a 上。然后递归函数 B 继续调用自身生成函数 C 并分配内存块 c 压在 内存块 b上。只要递归函数未触发基准条件函数将一直循环调用并生成一个个新的内存块压在上一个内存块上面。这个操作叫 **push（压栈）**。

计算机使用一个 **（栈）** 来表示这些内存块，其中内存块 b 位于 a 内存块上面 c 在  b 上面（栈顶的内存块就是函数的当前活跃部分）。假设函数 C 触发了递归函数的基准条件，在执行完函数 C 后，栈顶的内存块 c 被弹出，这个操作被称作 **pop（弹栈）**

这里在执行函数 B 和 C 时，函数 A 只执行了一部分。这是一个很重要的概念： **调用另一个函数时，当前函数暂停并处于未完成状态**。 该变量的所有变量值都还在内存中。 执行完函数 C 后，会又回到函数 B ，B 执行完毕又返回 A 并从离开的地方接着往下执行。 这个 **栈** 用于存放对多个函数的变量， 被称为 **调用栈**。

使用 **栈** 很方便，但是也要付出代价：存储先进的信息会占用大量的内存。每个函数调用都要占用一定的内存，如果栈很高，就意味着计算机存储了大量函数调用的信息。 而且 **调用栈** 的高度是有限制的（如Python中默认的最高层数时1000）。若递归层数超过一定阀值则会造成 **栈溢出** 也叫 **爆栈** 在这种情况下有三种选择：
- 增大阀值
- 重新编写代码
- 使用 **尾递归**

## 什么是尾递归
关于尾递归，百度百科解释的算是很清晰明了了：

当编译器检测到一个函数调用是尾递归的时候，它就覆盖当前的活动记录而不是在栈中去创建一个新的。编译器可以做到这点，因为递归调用是当前活跃期内最后一条待执行的语句，于是当这个调用返回时栈帧中并没有其他事情可做，因此也就没有保存栈帧的必要了。通过覆盖当前的栈帧而不是在其之上重新添加一个，这样所使用的栈空间就大大缩减了，这使得实际的运行效率会变得更高。

- **尾递归就是把当前的运算结果（或路径）放在参数里传给下层函数**

- **如果在递归函数中，递归调用返回的结果总被直接返回，则称为尾部递归。** 

## 递归、尾递归、循环的比较

下面是我分别用普通递归、尾递归、循环实现阶乘的小程序，这个是我能想到最简洁直观的例子了。


```python
# 设置允许递归的深度
import sys
sys.setrecursionlimit(10000)


# 普通递归阶乘
def fact(n):
    if n == 1:
        return n
    return n * fun(n-1)

# 尾递归阶乘
def fact_iter(n, num):
    if n == 1:
        return num
    return fact_iter(n - 1, n * num)

# 循环阶乘
def fact_loop(n):
    num = 1
    for i in range(1,n):
        num = num * i
    return num        
```

## 尾递归的Python优化

Python的爸爸已经明确表示Python将不会支持尾递归优化了。但是我先是在网上查相关资料的时候发现了两种方法对尾递归的优化：

### 方式一：实现一个 tail_call_optimized 装饰器
[源码：点这里](http://code.activestate.com/recipes/474088/)

为了更清晰的展示开启尾递归优化前、后调用栈的变化和tail_call_optimized装饰器抛异常退出递归调用栈的作用, 一个牛人利用pudb调试工具做了动图。
[动图：点这里](http://python.jobbole.com/86937/)


```python
import sys
 
class TailRecurseException:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

def tail_call_optimized(g):
    """
    This function decorates a function with tail call
    optimization. It does this by throwing an exception
    if it is it's own grandparent, and catching such
    exceptions to fake the tail call optimization.
 
    This function fails if the decorated
    function recurses in a non-tail context.
    """
    def func(*args, **kwargs):
        f = sys._getframe()
        # 为什么是grandparent, 函数默认的第一层递归是父调用,
        # 对于尾递归, 不希望产生新的函数调用(即:祖父调用),
        # 所以这里抛出异常, 拿到参数, 退出被修饰函数的递归调用栈!
        if f.f_back and f.f_back.f_back \
            and f.f_back.f_back.f_code == f.f_code:
            # 抛出异常
            raise TailRecurseException(args, kwargs)
        else:
            while 1:
                try:
                    return g(*args, **kwargs)
                except (TailRecurseException, e):
                    # 捕获异常, 拿到参数, 退出被修饰函数的递归调用栈
                    args = e.args
                    kwargs = e.kwargs
    func.__doc__ = g.__doc__
    return func
 
@tail_call_optimized
def factorial(n, acc=1):
    "calculate a factorial"
    if n == 0:
        return acc
    return factorial(n-1, n*acc)
 
print(factorial(100))
```

因为尾递归没有调用栈的嵌套, 所以Python也不会报RuntimeError: maximum recursion depth exceeded错误了!

但是对于这种发发是否真正意义上实现了完美的优化呢？ 针对这个方法有个知乎大神发话了：

TCO，tail-call optimization，其实有多种解读方式

最常见的解读方式是：对于尾调用的函数调用，不要浪费栈空间，而要复用调用者的栈空间。这样的结果就是一长串尾调用不会爆栈，而没有TCO的话同样的调用就会爆栈。从这个意义上说，如上方法的那个recipe确实达到了TCO的部分目的：

- 通过stack introspection查看调用链上的调用者之中有没有自己
- 有的话，通过抛异常来迫使栈回退（stack unwind）到之前的一个自己的frame
- 在回退到的frame接住异常，拿出后来调用的参数，用新参数再次调用自己

这样就可以让尾递归不爆栈。但这样做性能是没保证的…而且对于完全没递归过的一般尾调用也不起作用。一种对TCO的常见误解是：由编译器或运行时系统把尾调用/尾递归实现得很快。这不是TCO真正要强调的事情——不爆栈才是最重要的。也就是说其实重点不在“优化”，而在于“尾调用不爆栈”这个语义保证。“proper tail-call”的叫法远比“tail-call optimization”来得合适。因而像这种种做法，可以算部分TCO，但算不上“性能优化”意义上的优化。


### 方式二：pattern-matching
关于尾递归实现：

 很简单，永远不要在函数内部去调用自己，甚至不要调用任何东西。每一次调用，都让外层一个agent来做。保证函数栈深永远不超过2或3即可。

这哥们更牛逼了，直接做了个Python的模式匹配库，包含了尾递归的优化，实现了coroutine。

把需要优化的函数的return改成yield，外面套个装饰器，就叫tail_call_opm。装饰器最内层的逻辑是
```
while True:
    try:
        ret=next(ret) 
    except:
        return ret
```
这个应该没有复用释放的空间…但刷题时换了这个就不爆栈了。返回闭包的话情况应该会更复杂一些。
[pattern-matching参考GitHub地址](https://github.com/Xython/pattern-matching)
