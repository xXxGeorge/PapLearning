---
tags:
  - BasicMath
  - LinearAlgebra
relation:
  - "[[Convex optimization]]"
  - "[[Linear Algebra]]"
teacher: 田璐璐
---
**Review Prompt**:

现在我需要你扮演一个数值计算方法的学习搭子的角色, 我们将会学习非线性方程组的数值解法, 线性代数方程组的直接解法, 多项式插值, 数值积分与微分, 线性与非线性方程组的迭代解法, 常微分方程的数值解法几个章节, 现在我处于复习阶段, 请你给出复习的路线, 包括章节复习顺序以及每章节的复习要点,知识等, 然后我会根据你的路径进行复习, 随后当我对你说“向我对...(章节或是知识点)提问”的时候你要对我进行提问, 我会根据你的问题进行回答, 你对我的回答进行评判, 一直重复, 直到这一章的知识点问完了或是我说“结束这一章的学习”为止, 听懂了吗

### 第一章

#### 误差

##### 绝对误差

绝对误差的定义:
$$
\underline{e = x^* - x}
$$
此处需要注意正负号不能弄反, 否则相对误差计算的时候会无法推出近似值

##### 相对误差

相对误差的定义:
$$
e^* = \frac{e}{x}
$$
然后我们可以得到近似算法:
$$
e^* - \frac{e}{x^*}= \frac{e}{x} - \frac{e}{x^*}= \frac{e(x^*-x)}{xx^*} = \frac{e^2}{x^*(x^* - e)} =\frac{(\frac{e}{x^*})^2}{1-\frac{e}{x^*}}
$$
显然对分母的变换依赖于绝对误差的顺序定义.

不过这显然也不太合理, 既然这两个值在数值上应当是对称的.所以如果我们将绝对误差的定义反过来应该也可以写出来:
$$
e^* - \frac{e}{x^*}= \frac{e}{x} - \frac{e}{x^*}= \frac{e(x^*-x)}{xx^*} = \frac{-e^2}{x(x - e)} =\frac{(\frac{e}{x})^2}{\frac{e}{x}-1}
$$


##### 相对误差限

绝对误差限就是误差的绝对值的最大值, 而“不超过某一位的半个单位”是用在有效位数的判定上.

### Chapter2 非线性方程组的数值解法
#### 二分法
  二分法的原理在于对连续函数的零点存在定理, 对其收敛性有$$|x^n - x^*| = \frac{|a-b|}{2^n}$$
程序表达(recursion)
```python
def BiSection(function, a, b, Tol):
    if function(a) * function(b) > 0:
        return False  

    result = (a + b) / 2

    if abs(function(result)) < Tol or abs(a - b) < Tol:
        return result
    elif function(a) * function(result) < 0:
        return BiSection(function, a, result, Tol)
    else:
        return BiSection(function, result, b, Tol)
	
```
这个写法逻辑简单, 然而在递归次数过多时容易栈溢出, 所以可以用循环优化一下
```python
def BiSection(f, a, b, MaxIter, Tol):  
    if(f(a)*f(b) > 0):  
        return False  
    for _ in range(MaxIter):  
        mid = (a + b) / 2  
        if f(a)*f(mid) < 0:  
            b = mid  
        elif f(b)*f(mid) < 0:  
            a = mid  
        else:  
            return mid  
  
        if abs(b-a) < Tol:  
            return (b+a)/2
```
#### 不动点法
##### Definition
$$\displaylines{f(x) \Rightarrow x = \phi(x), \,\, \\ \Rightarrow x_{n+1} = \phi(x_n)}$$
对于不动点法的收敛性, 下面几点是他的充分条件:
- 映内性, 对$x \in (a, b), \phi(x) \in (a,b)$ 
- 满足Lipschitz条件, 即对$\forall x \in (a,b), f'(x) \leq 1$

##### 误差
$$|x^* - x_n| \leq \frac{L}{1-L} |x_n - x_{n-1}| \leq \frac{L^n}{1-L}|x_2 - x_1|$$


#### Newton法
Newto法本质上是一种变形的不动点迭代法, 所以我们可以用不动点法的观点去看待. 
迭代公式:$$x_{n+1} = x_n -  \frac{f(x_n)}{f'(x_n)}$$


### Chapter4 多项式插值

#### 插值问题

插值问题与函数拟合的区别在于函数插值要求得到的函数$\Phi$严格经过已知数据点, 而拟合问题不需要, 只需要大致反映f的变化趋势, 在某意义下与已知数据点最接近

##### 定义
> [!PDF|yellow] [[Ch4.pdf#page=7&selection=10,0,176,1&color=yellow|Ch4, p.7]]
 > 已知定义于区间 $[a, b]$ 上的实值函数$f (x)$ 在 $n + 1$ 个节点 ${\{x_i \}}_{i = 0}^n$ ∈ $[a, b]$ 处的函数值 ${\{f(x_i) \}}_{i = 0}^n$. 若函数集合 $Φ$ 中的函数 $ϕ(x)$ 满足 $$ϕ(xi ) = f (xi ), i = 0, 1, · · · , n,(1.1)$$ 则称 $ϕ(x) 为 f (x)$ 在函数集合 $Φ$ 中关于节点 ${\{x_i \}}_{i = 0}^n$ 的一个插值函数，$f (x)$ 被称为被插值函数，$[a, b]$ 插值区间，${\{x_i \}}_{i = 0}^n$ 插值节点,$(1.1)$ 为插值条件.

##### 误差估计



#### 插值多项式的构造方法
##### Lagrange Interpotation
参考[[Lagrange插值法详解]], 以向量形式完成其程序.
程序实现
```python
import numpy as np  
import matplotlib.pyplot as plt  
def Lagrange(x, x_x, x_y):  
    x_given = np.asarray(x)  
    x_axis = np.asarray(x_x)  
    y_axis = np.asarray(x_y)  
  
    n = len(x_axis)  
    L = 0  
    for i in range(n):  
        p = 1  
        for j in range(n):  
            if i != j:  
                p = p*(x_given - x_axis[j])/(x_axis[i] - x_axis[j])  
        L += y_axis[i]*p  
  
    return L  
  
  
x_x = [1, 2, 3]  
x_y = [1, 4, 9]  
  
x_vals = np.linspace(0.5, 3.5, 200)  
y_vals = Lagrange(x_vals, x_x, x_y)  
plt.plot(x_vals, y_vals, label='Lagrange Interpolation Curve', color='blue')  
plt.scatter(x_x, x_y, color='red', label='Interpolation Points')  
plt.show()
```

此处因为我们输入的x即为自变量. `np.linspace(start, end, number)`可用来生成随机的点列, 用于绘制平滑曲线

##### Newton Interpolation
Newton法需要用到差商, 这个算法也很巧妙. 如果只是在循环内部一个一个写的话我们无法存储那么多差商, 毕竟每一个新的差商都需要低一阶的差商来计算. 所以我们直接对numpy数组进行整体操作:
```python
coef = np.copy(y_data)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x_data[j:n] - x_data[0:n-j])
```
这样就相当于每次都对所有的值进行了差商运算, 并且由于j递增, 所以每次新变换时刚好将上一次得到的有效差商留下, 如图:
![[差商表算法.png]]
程序代码
```python
def Newton(x, x_x, x_y):  
    x_given = np.asarray(x)  
    x_axis = np.asarray(x_x)  
    y_axis = np.asarray(x_y)  
  
    n = len(x_axis)  
    coef = np.copy(y_axis)  
    for i in range(1,n):  
        coef[i:n] = (coef[i:n]-coef[i-1:n-1])/(x_axis[i:n]-x_axis[i-1:n-1])  
  
    N = y_axis[0]  
    polyx = 1  
    for i in range(n-1):  
        polyx *= (x_given - x_axis[i])  
        N += polyx*coef[i+1]  
  
    return N
```


#### Hermite 插值问题
#### 分段插值(piecewise interpolation)
#### 三次样条插值



## 第七章
